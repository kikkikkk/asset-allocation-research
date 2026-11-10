# -*- coding: utf-8 -*-
"""P1 数据质量：单资产校验、统一回测日历与重叠区间报告。

关键规则
--------
- **不做全量前向填充**：默认采用「共同交易日交集」作为统一回测日历，
  缺失日期直接排除而非填充；每个资产的被排除天数都会记录。
- 不同市场交易日历不同时，交集日历是唯一对全部资产同时有观测的日期集合，
  估值规则统一为「当日收盘价 / 当日指数点位」，不使用尚未公布的数据。
- 数据频率由相邻日期中位数间隔推断；混合频率显式警告。
- 异常收益（|r| 超阈值、r ≤ −1）只标记、不静默修正（r ≤ −1 视为非法，因 log(1+r) 要求 r > −1）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    asset_id: str
    ok: bool
    rows_raw: int = 0
    rows_valid: int = 0
    duplicate_dates: int = 0
    invalid_prices: int = 0
    missing_values: int = 0
    extreme_returns: int = 0
    illegal_returns: int = 0          # r <= -1（log1p 无定义）
    inferred_frequency: str = "未知"
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id, "ok": self.ok,
            "原始行数": self.rows_raw, "有效行数": self.rows_valid,
            "重复日期": self.duplicate_dates, "非法价格": self.invalid_prices,
            "缺失值": self.missing_values, "异常收益": self.extreme_returns,
            "非法收益(r<=-1)": self.illegal_returns,
            "频率": self.inferred_frequency,
            "起始": str(self.start.date()) if self.start is not None else "",
            "结束": str(self.end.date()) if self.end is not None else "",
            "警告": "；".join(self.warnings) or "—",
            "错误": "；".join(self.errors) or "—",
        }


def validate_price_series(
    ser: pd.Series,
    asset_id: str,
    extreme_threshold: float = 0.5,
    min_obs: int = 60,
) -> tuple[pd.Series, QualityReport]:
    """校验单资产价格序列，返回 (清洗后序列, 质量报告)。不填充缺失值。"""
    rep = QualityReport(asset_id=asset_id, ok=False, rows_raw=int(len(ser)))
    if ser is None or len(ser) == 0:
        rep.errors.append("价格序列为空")
        return pd.Series(dtype=float), rep

    s = pd.Series(ser).copy()
    # 索引日期化
    try:
        s.index = pd.to_datetime(s.index)
    except Exception as exc:  # noqa: BLE001
        rep.errors.append(f"日期索引无法解析：{exc}")
        return pd.Series(dtype=float), rep
    s = s[~s.index.isna()]
    s = pd.to_numeric(s, errors="coerce")
    s.name = asset_id

    rep.missing_values = int(s.isna().sum())
    s = s.dropna()
    rep.invalid_prices = int((s <= 0).sum() + (~np.isfinite(s.to_numpy(dtype=float))).sum())
    s = s[s > 0]
    s = s[np.isfinite(s.to_numpy(dtype=float))]

    dup = int(s.index.duplicated().sum())
    rep.duplicate_dates = dup
    if dup:
        # 同一日期重复时保留最后一条（常见于数据源重复推送），并记录
        s = s[~s.index.duplicated(keep="last")]
        rep.warnings.append(f"发现 {dup} 个重复日期，已保留最后一条")
    s = s.sort_index()

    rep.rows_valid = int(len(s))
    if len(s) >= 2:
        rep.start, rep.end = s.index[0], s.index[-1]
        gaps = np.diff(s.index.values).astype("timedelta64[D]").astype(float)
        med = float(np.median(gaps))
        if med <= 4:
            rep.inferred_frequency = "日频（或高频）"
        elif med <= 10:
            rep.inferred_frequency = "周频"
        elif med <= 45:
            rep.inferred_frequency = "月频"
        else:
            rep.inferred_frequency = "低频/不规则"
        if gaps.max() > med * 8 + 10:
            rep.warnings.append("存在明显长间隔，可能为不同交易日历或数据中断")
        r = s.pct_change().dropna()
        rep.extreme_returns = int((r.abs() > extreme_threshold).sum())
        rep.illegal_returns = int((r <= -1).sum())
        if rep.extreme_returns:
            rep.warnings.append(
                f"{rep.extreme_returns} 个日收益绝对值 > {extreme_threshold:.0%}，"
                "可能是未复权的分红/拆分跳空或数据错误（仅标记，未修正）"
            )
        if rep.illegal_returns:
            rep.errors.append(f"{rep.illegal_returns} 个收益 ≤ −100%，无法取对数，必须核对数据")
    if len(s) < min_obs:
        rep.errors.append(f"有效观测仅 {len(s)} 条，少于最少要求 {min_obs} 条")

    rep.ok = not rep.errors
    return s, rep


def align_price_frame(
    series_map: dict[str, pd.Series],
    policy: str = "intersection",
    min_common: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """按统一回测日历对齐 N 资产价格。默认仅支持交集（不做前向填充）。

    返回 (价格矩阵 DataFrame, 报告 dict)。报告含每资产被排除天数、共同区间与年数。
    """
    if len(series_map) < 2:
        raise ValueError("资产配置分析至少需要 2 个资产")
    if policy != "intersection":
        raise ValueError(
            "当前统一回测日历仅支持 'intersection'（共同交易日交集）。"
            "不支持自动前向填充，以免把未公布数据用于过去的决策。"
        )
    frame = pd.DataFrame(series_map)
    raw_counts = frame.notna().sum().to_dict()
    frame = frame.sort_index()
    frame = frame.dropna(how="any")            # 交集：任一资产无观测则剔除该日
    frame = frame[frame > 0].dropna(how="any")
    if len(frame) < min_common:
        raise ValueError(
            f"共同交易日仅 {len(frame)} 条（要求 ≥ {min_common}）。"
            "可能是交易日历差异过大或数据区间不重叠；请检查资产区间或更换资产。"
        )
    excluded = {k: int(v - len(frame)) for k, v in raw_counts.items()}
    start, end = frame.index[0], frame.index[-1]
    report = {
        "policy": policy,
        "n_common": int(len(frame)),
        "start": start,
        "end": end,
        "span_years": float((end - start).days / 365.25),
        "excluded_days": excluded,
        "assets": list(frame.columns),
        "note": (
            "统一回测日历 = 所有资产同时有有效观测的交易日（交集）。"
            "不同市场非交易日不填充、不外推；估值统一为当日收盘点位；"
            "决策只使用当日收盘后可得的数据。"
        ),
    }
    return frame, report


def returns_from_prices(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """价格矩阵 → 收益矩阵。simple: P_t/P_{t−1}−1；log: ln(1+r)。

    不做填充；第一天为 NaN 并丢弃（收益样本数 = 价格天数 − 1）。
    """
    r = prices.pct_change()
    if method == "log":
        if (r <= -1).any().any():
            raise ValueError("存在 r ≤ −1，log(1+r) 无定义")
        return np.log1p(r).dropna(how="any")
    if method != "simple":
        raise ValueError(f"未知收益方法：{method}")
    return r.dropna(how="any")
