# -*- coding: utf-8 -*-
"""P1 公开数据获取层：真实复用原版 public_data.py 的接口，扩展为资产注册表驱动。

复用清单（代码级 import，不是复制粘贴）
--------------------------------------
- v1_reference.public_data.fetch_hs300(caliber)  → 沪深300 全收益/价格（含多源回退与口径保护）
- v1_reference.public_data._fetch_csindex_index(code, name) → 中证官网任意指数
- v1_reference.public_data.fetch_bond_wealth() → 中债财富指数（含 +8h 时区归一）

规则
----
- 不缓存到磁盘、不入库；Streamlit 层用 @st.cache_data(ttl=3600) 做会话级缓存。
- 每个资产单独记录获取日志（来源、成败、原因、行数、区间、耗时）；
  单个资产失败不影响其他资产，但**不生成模拟数据、不静默替换口径**。
- 失败时在 UI 展示失败来源 + 错误原因 + 可行替代（上传/换资产）。
- 请求超时由原版函数控制（40~60s，最多一次重试），不做无限重试、不做并行轰炸。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from core.assets import AssetRegistry
from v1_reference import public_data as v1


@dataclass
class FetchLog:
    asset_id: str
    ok: bool
    source: str = ""
    code: str = ""
    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    elapsed_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id, "成功": self.ok, "来源": self.source,
            "代码": self.code, "行数": self.rows,
            "起始": str(self.start.date()) if self.start is not None else "",
            "结束": str(self.end.date()) if self.end is not None else "",
            "耗时(s)": round(self.elapsed_s, 2), "错误": self.error or "—",
        }


def fetch_asset_series(
    asset_id: str,
    registry: AssetRegistry,
    start_date: str | None = None,
) -> tuple[pd.Series | None, FetchLog]:
    """按注册表的 fetch_key 获取单个资产价格序列。

    `start_date`（YYYY-MM-DD）是可选的**增量**起点，仅对中证官网指数生效；
    默认 None 保持既有的全区间获取行为。中债接口不支持日期参数，
    传入时仍返回全量并在日志中标注（不做静默截断）。
    """
    t0 = time.time()
    try:
        meta = registry.get(asset_id)
        log = FetchLog(asset_id=asset_id, ok=False, source=meta.source, code=meta.code)
    except Exception as exc:  # noqa: BLE001 —— 元数据异常也不能让批量获取整体崩溃
        return None, FetchLog(asset_id=asset_id, ok=False,
                              error=f"资产元数据读取失败：{type(exc).__name__}: {exc}",
                              elapsed_s=time.time() - t0)
    if not meta.fetch_key:
        log.error = "该资产无公开获取接口（上传资产请通过上传通道读取）"
        return None, log
    try:
        kind, _, key = meta.fetch_key.partition(":")
        if kind == "chinabond" and key == "CBA06501":
            ser, _m = v1.fetch_bond_wealth()
            if start_date:
                log.error = "中债接口不支持增量日期参数，已返回全量"
        elif kind == "csindex" and key == "H00300":
            ser, _m = v1.fetch_hs300(v1.CALIBER_TR, start_date=start_date)
        elif kind == "csindex" and key == "000300":
            ser, _m = v1.fetch_hs300(v1.CALIBER_PRICE)   # 保留原版腾讯→中证→新浪回退链
        elif kind == "csindex":
            ser = v1._fetch_csindex_index(key, meta.name, start_date=start_date)
        else:
            raise ValueError(f"未知 fetch_key：{meta.fetch_key}")
        if ser is None or len(ser) < (2 if start_date else 60):
            raise ValueError(f"返回有效数据过少（{0 if ser is None else len(ser)} 条）")
        log.ok = True
        log.rows = int(len(ser))
        log.start, log.end = ser.index[0], ser.index[-1]
        return ser.rename(asset_id), log
    except Exception as exc:  # noqa: BLE001 —— 单资产失败记录后继续
        log.error = f"{type(exc).__name__}: {exc}"
        return None, log
    finally:
        log.elapsed_s = time.time() - t0


def fetch_many(
    asset_ids: list[str], registry: AssetRegistry
) -> tuple[dict[str, pd.Series], list[FetchLog]]:
    """顺序获取多个资产（避免并发请求压垮公开接口）。"""
    out: dict[str, pd.Series] = {}
    logs: list[FetchLog] = []
    for aid in asset_ids:
        # V2.3：逐资产完全隔离。即使 fetch_asset_series 出现未预期的异常
        # （注册表异常、monkeypatch、接口层意外崩溃），也只影响该资产，
        # 不再让整个批量加载失败导致所有资产一起丢失。
        try:
            ser, log = fetch_asset_series(aid, registry)
        except Exception as exc:  # noqa: BLE001
            ser, log = None, FetchLog(
                asset_id=aid, ok=False,
                error=f"批量获取中的单资产异常：{type(exc).__name__}: {exc}")
        if ser is not None:
            out[aid] = ser
        logs.append(log)
        time.sleep(0.2)  # 轻节流
    return out, logs


def logs_frame(logs: list[FetchLog]) -> pd.DataFrame:
    return pd.DataFrame([l.to_dict() for l in logs])
