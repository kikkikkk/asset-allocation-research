# -*- coding: utf-8 -*-
"""V2.3 公开数据可靠供给层：接口获取 → 质量校验 → 持久化快照 → 失败回退。

解决的问题
----------
旧实现只有 `@st.cache_data(ttl=3600)` 一层**进程内**缓存：TTL 到期会强制
重新请求接口；一旦接口失败，异常被捕获后 `series = {}`，此前已成功获取
的真实行情毫无保留地丢失，页面显示「可用资产仅 0 个」。服务重启、浏览器
Session 重连、手动刷新失败都走同一条丢失路径。

本模块的规则
------------
1. 每个资产独立获取、独立校验、独立持久化；单资产失败不影响其他资产。
2. 接口成功且通过质量/身份校验才写入快照；空数据、异常结构、错误代码、
   质量失败一律拒绝覆盖健康快照。
3. 接口失败（或 TTL 到期但冷却中）优先恢复最近一次成功的有效快照，
   并明确标注「本地快照（非实时）」，绝不冒充刚获取的数据。
4. 不做日期拼接：每个资产都是完整真实序列，对齐仍由 align_price_frame
   取共同交易日交集；各资产来源/截止/更新时间分别记录。
5. 缓存到期 ≠ 数据失效。自动刷新优先增量；增量失败保留完整历史快照；
   手动刷新走全量，失败同样保留快照。
6. 刷新防抖 + 冷却 + claim 机制，避免多页面/多会话无控制重复请求。
7. 数据新鲜度按工作日判断（含周末/节假日容差）；长期未更新或缓存损坏时
   明确给出 blocked_reason，由上层阻止不可靠计算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from core.assets import AssetRegistry
from core.quality import validate_price_series
from data_sources import persistent_cache as pc
from data_sources.public_api import FetchLog, fetch_asset_series, logs_frame

DEFAULT_TTL = 3600              # 成功结果缓存 1 小时（与原先一致）
FAILURE_COOLDOWN = 60           # 失败后冷却，避免反复打接口
MIN_REFRESH_INTERVAL = 10       # 手动刷新最小间隔（防重复点击）
STALE_WARN_BUSINESS_DAYS = 3    # 超过则提示数据可能滞后
# 不枚举中国法定节假日：春节/国庆长假按「仅排除周末」粗算可能被计为
# 9~11 个工作日，故阈值留足三周余量，避免把正常休市误判为数据长期未更新。
# 超过 15 个工作日（约三周）无新交易日才阻止不可靠计算。
STALE_BLOCK_BUSINESS_DAYS = 15
MAX_END_DIVERGENCE_DAYS = 10    # 两资产截止日差异提示阈值（自然日）
INCREMENTAL_LOOKBACK_DAYS = 14  # 增量请求回溯天数（覆盖重叠校验）


@dataclass
class ProviderResult:
    series: dict[str, pd.Series] = field(default_factory=dict)
    log_rows: list[dict] = field(default_factory=list)
    log_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    states: dict[str, dict] = field(default_factory=dict)
    attempted: bool = False
    skipped_reason: str = ""
    blocked_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    last_attempt_at: float = 0.0
    last_success_at: float = 0.0


# ---------------- 新鲜度（工作日口径，含周末/节假日容差） ----------------
def business_days_between(d0: date, d1: date) -> int:
    """d0 之后到 d1（含）之间的工作日数，粗略排除周末。

    中国法定节假日无法在无外部日历依赖下精确枚举，因此用容差处理：
    lag<=1 视为新鲜，2~3 视为「可能节假日/尚未入库」，>3 才判定滞后。
    """
    if d1 <= d0:
        return 0
    n, cur = 0, d0 + timedelta(days=1)
    while cur <= d1:
        if cur.weekday() < 5:
            n += 1
        cur += timedelta(days=1)
    return n


def expected_latest_trading_day(now: date) -> date:
    """最近一个应为交易日的日期（周末回退到周五，不枚举法定假日）。"""
    d = now
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def freshness_of(end: str | date, now: date) -> tuple[str, int, bool]:
    """返回 (标签, 工作日滞后数, 是否应阻止计算)。"""
    try:
        e = end if isinstance(end, date) else date.fromisoformat(str(end)[:10])
    except Exception:  # noqa: BLE001
        return "日期无法解析", 999, True
    lag = business_days_between(e, expected_latest_trading_day(now))
    if lag <= 1:
        return "新鲜", lag, False
    if lag <= STALE_WARN_BUSINESS_DAYS:
        return "可能缺少最近交易日（周末/节假日或尚未入库）", lag, False
    if lag <= STALE_BLOCK_BUSINESS_DAYS:
        return "数据滞后（历史研究可用，非实时）", lag, False
    return "长期未更新（已阻止不可靠计算）", lag, True


# ---------------- 身份与质量 ----------------
def _asset_meta(aid: str, registry: AssetRegistry) -> dict:
    m = registry.get(aid)
    return {
        "asset_id": aid,
        "name": m.name,
        "code": m.code,
        "source": m.source,
        "source_url": getattr(m, "source_url", "") or "",
        "fetch_key": m.fetch_key,
        "return_basis": m.return_basis,
        "dividend_treatment": m.dividend_treatment,
    }


def _validate_new_series(ser, aid: str, meta: dict, log: FetchLog) -> tuple[pd.Series | None, dict, str]:
    """严格校验接口返回，拒绝空数据/异常结构/错误代码/质量失败。

    必须**在** validate_price_series 之前检查原始序列完整性：该清洗函数会
    dropna / 剔除非法价格并返回 ok=True，若先清洗再判断，含 NaN 或负价格的
    异常返回会被误当成健康数据写入快照，覆盖掉此前有效的真实行情。
    """
    if ser is None or not isinstance(ser, pd.Series) or len(ser) < pc.MIN_OBS:
        return None, {}, f"接口返回无效数据（{0 if ser is None else len(ser)} 条有效观测）"
    if not isinstance(ser.index, pd.DatetimeIndex):
        return None, {}, "收益序列索引不是日期，拒绝覆盖快照"
    if ser.index.has_duplicates or not ser.index.is_monotonic_increasing:
        return None, {}, "日期重复或未升序，拒绝覆盖快照"
    raw = ser.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        return None, {}, "含 NaN/无穷大价格，拒绝覆盖快照"
    if (raw <= 0).any():
        return None, {}, "含非正价格，拒绝覆盖快照"
    if log is not None and not log.ok:
        return None, {}, f"接口返回失败：{log.error}"
    if log is not None and log.code and str(log.code) != str(meta["code"]):
        return None, {}, f"指数代码不一致：期望 {meta['code']}，实际 {log.code}"
    try:
        s2, q = validate_price_series(ser, aid, min_obs=60)
        qd = q.to_dict()
    except Exception as exc:  # noqa: BLE001
        return None, {}, f"质量校验异常：{type(exc).__name__}: {exc}"
    if not q.ok:
        return None, qd, f"质量校验未通过：{qd.get('错误') or qd.get('warnings') or qd}"
    return s2.rename(aid), qd, ""


def _incremental_merge(
    aid: str,
    registry: AssetRegistry,
    meta: dict,
    snap: pd.Series,
    fetcher,
) -> tuple[pd.Series | None, str, FetchLog | None]:
    """尝试只取最近交易日增量；任何异常/不一致都保留完整旧快照。"""
    if not str(meta.get("fetch_key", "")).startswith("csindex:"):
        return None, "该接口不支持增量（返回全量）", None
    start = (snap.index[-1] - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    try:
        ser, log = fetcher(aid, registry, start_date=start)
    except Exception as exc:  # noqa: BLE001
        return None, f"增量请求失败：{type(exc).__name__}: {exc}", None
    if ser is None or len(ser) == 0:
        return None, "增量返回空数据，已保留完整快照", log
    common = ser.index.intersection(snap.index)
    if len(common) == 0:
        return None, "增量与快照无重叠交易日，无法校验一致性，已保留完整快照", log
    try:
        consistent = np.allclose(
            ser.loc[common].to_numpy(dtype=float),
            snap.loc[common].to_numpy(dtype=float), rtol=1e-8, atol=1e-8)
    except Exception:  # noqa: BLE001
        consistent = False
    if not consistent:
        return None, "增量数据与快照重叠区间数值不一致，已保留完整快照", log
    new = ser[~ser.index.isin(snap.index)]
    merged = pd.concat([snap, new]).sort_index()
    if len(merged) < len(snap):
        return None, "增量合并后行数异常，已保留完整快照", log
    return merged.rename(aid), (f"增量更新：新增 {len(new)} 个交易日" if len(new) else
                                "增量更新：无新增交易日，已刷新获取时间"), log


# ---------------- 主入口 ----------------
def get_public_prices(
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
    *,
    version: int = 0,
    force: bool = False,
    now: datetime | None = None,
    fetcher=None,
    cache_dir: Path | str | None = None,
    ttl: int = DEFAULT_TTL,
    cooldown: int = FAILURE_COOLDOWN,
    min_refresh_interval: int = MIN_REFRESH_INTERVAL,
    allow_incremental: bool = True,
) -> ProviderResult:
    """返回可用的真实价格序列 + 完整状态；接口失败时回退最近有效快照。

    参数 `now` / `fetcher` / `cache_dir` / `ttl` 支持可控时间与确定性样例测试。
    """
    d = Path(cache_dir) if cache_dir else pc.CACHE_DIR
    now_dt = pc.beijing_now(now)   # 统一北京时间：云端 UTC 服务器不会误判新鲜度
    now_ts = now_dt.timestamp()
    today = now_dt.date()
    fetcher = fetcher or fetch_asset_series
    ids = [a for a in asset_ids if a and registry.get(a).fetch_key]
    out = ProviderResult()

    # 1) 读持久化状态 + 所有资产的最近有效快照
    state = pc.read_state(d)
    out.last_attempt_at = float(state.get("last_attempt_at", 0.0))
    out.last_success_at = float(state.get("last_success_at", 0.0))
    snaps: dict[str, tuple[pd.Series | None, dict | None, str]] = {}
    for aid in ids:
        snaps[aid] = pc.load_snapshot(aid, _asset_meta(aid, registry), cache_dir=d)

    # 2) 是否需要请求接口（TTL / 冷却 / 刷新防抖 / 并发 claim）
    # version 变化（用户点「刷新公开数据」）视为一次全量刷新请求；
    # 版本记录在持久化状态里，服务重启后也不会误判为「首次」而重复全量下载。
    version_changed = version > int(state.get("refresh_version", 0))
    force = bool(force or version_changed)
    due = out.last_success_at <= 0 or (now_ts - out.last_success_at) >= ttl
    since_attempt = now_ts - out.last_attempt_at if out.last_attempt_at else float("inf")
    want_attempt = False
    if force and since_attempt < min_refresh_interval:
        out.skipped_reason = (f"刷新操作过于频繁（距上次请求不足 {min_refresh_interval} 秒），"
                              "本次未重复请求；已使用现有有效数据")
    elif force:
        want_attempt = True
    elif due and since_attempt >= cooldown:
        want_attempt = True
    elif due:
        out.skipped_reason = (f"数据缓存已到期，但处于失败冷却期（{cooldown} 秒），"
                              "暂不重复请求；已使用现有有效数据")

    if want_attempt:
        # claim：多会话并发时只有一个获得请求权，其余直接读快照
        with pc.cache_lock(d):
            cur = pc.read_state(d)
            if float(cur.get("last_attempt_at", 0.0)) > out.last_attempt_at:
                want_attempt = False
                out.skipped_reason = "另一页面/会话正在刷新数据，本次直接复用现有有效数据"
            else:
                cur["last_attempt_at"] = now_ts
                cur["last_attempt_reason"] = "force" if force else "ttl"
                cur["refresh_version"] = int(version)
                pc.write_state(cur, d)
                out.last_attempt_at = now_ts
        out.attempted = want_attempt

    # 3) 逐资产获取（严格隔离：单个资产异常不影响其他资产）
    fresh: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    origins: dict[str, str] = {}
    fetch_logs: list[FetchLog] = []
    persist_warnings: list[str] = []
    if out.attempted:
        for aid in ids:
            meta = _asset_meta(aid, registry)
            snap_ser, snap_meta, snap_err = snaps[aid]
            log: FetchLog | None = None
            series_new: pd.Series | None = None
            note = ""
            saved_meta: dict = {}
            try:
                use_incremental = allow_incremental and not force and snap_ser is not None
                if use_incremental:
                    # 自动刷新优先增量；失败则保留完整旧快照，不再额外全量重试。
                    # 但「接口不支持增量」不是失败，必须回退到全量获取，
                    # 否则中债这类接口会永远停留在旧快照。
                    candidate, note, log = _incremental_merge(
                        aid, registry, meta, snap_ser, fetcher)
                    if candidate is None and note.startswith("该接口不支持增量"):
                        ser, log = fetcher(aid, registry)
                        series_new, qd, err = _validate_new_series(ser, aid, meta, log)
                        if series_new is None:
                            errors[aid] = err
                        else:
                            note = "全量获取"
                    elif candidate is None:
                        errors[aid] = note
                    else:
                        cand2, qd, err = _validate_new_series(candidate, aid, meta, None)
                        if cand2 is None:
                            errors[aid] = f"增量结果校验失败：{err}"
                        else:
                            series_new = cand2
                else:
                    # 首次获取 / 手动刷新 / 关闭增量 → 走已验证的全量路径
                    ser, log = fetcher(aid, registry)
                    series_new, qd, err = _validate_new_series(ser, aid, meta, log)
                    if series_new is None:
                        errors[aid] = err
                    else:
                        note = "全量获取"
                if series_new is not None:
                    # 只有通过全部校验的数据才写快照；空/异常/错误代码不会覆盖健康缓存
                    saved_meta = pc.save_snapshot(
                        meta, series_new, qd, now=now_dt, cache_dir=d)
                    if saved_meta.get("persisted") is False:
                        persist_warnings.append(
                            f"{meta['name']}：磁盘不可写，本次真实数据仅保留在进程内缓存，"
                            "未写入持久化快照；服务重启后会重新从公开接口获取。")
            except Exception as exc:  # noqa: BLE001 —— 单资产异常完全隔离
                errors[aid] = f"{type(exc).__name__}: {exc}"
                series_new = None

            if series_new is not None:
                fresh[aid] = series_new
                origins[aid] = ("增量更新" if note.startswith("增量") else "本次接口获取")
                snaps[aid] = (series_new, saved_meta, "")
                if log is None:
                    log = FetchLog(asset_id=aid, ok=True, source=meta["source"], code=meta["code"])
                log.ok, log.rows = True, int(len(series_new))
                log.start, log.end = series_new.index[0], series_new.index[-1]
                log.error = ""
                fetch_logs.append(log)
            elif snap_ser is not None:
                origins[aid] = "本地快照"
                fetch_logs.append(FetchLog(
                    asset_id=aid, ok=False, source=meta["source"], code=meta["code"],
                    error=f"接口未更新（{errors.get(aid, '未请求')}）；已保留最近有效快照"))
            else:
                origins[aid] = "不可用"
                fetch_logs.append(FetchLog(
                    asset_id=aid, ok=False, source=meta["source"], code=meta["code"],
                    error=errors.get(aid, "无可用数据")))

        if fresh:
            with pc.cache_lock(d):
                cur = pc.read_state(d)
                cur["last_success_at"] = now_ts
                cur["last_success_assets"] = sorted(fresh)
                pc.write_state(cur, d)
                out.last_success_at = now_ts

    # 4) 组装最终序列：新获取优先，否则最近有效快照
    for aid in ids:
        meta = _asset_meta(aid, registry)
        snap_ser, snap_meta, snap_err = snaps[aid]
        ser = fresh[aid] if aid in fresh else snap_ser
        info = snap_meta or {}
        if ser is not None:
            out.series[aid] = ser
            end = str(info.get("end", ser.index[-1].date()))
            rows = int(info.get("rows", len(ser)))
            fetched = str(info.get("fetched_at", "未知"))
            quality_ok = bool(info.get("quality_ok", True))
            checksum = str(info.get("checksum", ""))
            origin = origins.get(aid, "本地快照")
        else:
            end, rows, fetched = "", 0, ""
            quality_ok, checksum = False, ""
            origin = origins.get(aid, "不可用")
        label, lag, block = freshness_of(end, today) if end else ("无数据", 999, True)
        out.states[aid] = {
            "asset_id": aid, "name": meta["name"], "code": meta["code"],
            "source": meta["source"], "origin": origin,
            "is_snapshot": origin in ("本地快照",),
            "start": str(ser.index[0].date()) if ser is not None else "",
            "end": end, "fetched_at": fetched, "rows": rows,
            "quality_ok": quality_ok, "checksum": checksum[:12],
            "error": errors.get(aid, ""), "freshness": label,
            "business_days_lag": lag, "freshness_block": block,
        }
        if block and origin != "不可用":
            out.blocked_reason = (
                f"{meta['name']}（{meta['code']}）数据末日 {end}，"
                f"距最近应有交易日约 {lag} 个工作日，判定长期未更新；"
                "为避免基于陈旧行情得出误导性结论，已阻止本次计算。请检查网络后手动刷新。")
        if snap_err and snap_ser is None and out.attempted:
            out.warnings.append(f"{meta['name']}：{snap_err}")

    if not out.attempted:
        for aid in ids:
            if not any(x.asset_id == aid for x in fetch_logs):
                st = out.states.get(aid, {})
                fetch_logs.append(FetchLog(
                    asset_id=aid, ok=(aid in out.series), source=st.get("source", ""),
                    code=st.get("code", ""),
                    rows=int(st.get("rows", 0)),
                    start=pd.Timestamp(st["start"]) if st.get("start") else None,
                    end=pd.Timestamp(st["end"]) if st.get("end") else None,
                    error=("使用本地快照（非实时）" if st.get("is_snapshot") else "")))

    # 5) 两资产截止日差异提示（不拼接；对齐仍取共同交易日交集）
    ends = [pd.Timestamp(s["end"]) for s in out.states.values() if s.get("end")]
    if len(ends) >= 2 and (max(ends) - min(ends)).days > MAX_END_DIVERGENCE_DAYS:
        out.warnings.append(
            f"两个资产的数据截止日相差 {(max(ends)-min(ends)).days} 天，"
            "已分别标记更新时间；收益与协方差只在共同有效交易日交集上计算，不做任何填补。")
    if out.skipped_reason:
        out.warnings.append(out.skipped_reason)
    if any(s["origin"] == "本地快照" for s in out.states.values()):
        out.warnings.append(
            "当前包含**本地快照数据（非实时）**：数据来自此前成功获取并通过质量校验的"
            "真实公开行情，页面显示了各自的截止日期与最近成功获取时间；"
            "不得把这些数值表述为刚从接口获取的实时行情。")
    for w in dict.fromkeys(persist_warnings):
        out.warnings.append(w)

    out.log_rows = [l.to_dict() for l in fetch_logs]
    out.log_df = logs_frame(fetch_logs) if fetch_logs else pd.DataFrame()
    if not out.log_df.empty:
        out.log_df["数据状态"] = out.log_df["asset_id"].map(
            {k: v["origin"] for k, v in out.states.items()})
        out.log_df["数据截止"] = out.log_df["asset_id"].map(
            {k: v["end"] for k, v in out.states.items()})
        out.log_df["最近成功获取"] = out.log_df["asset_id"].map(
            {k: v["fetched_at"] for k, v in out.states.items()})
    if len(out.series) < 2 and not out.blocked_reason:
        out.blocked_reason = (
            "可用资产不足 2 个，且本地没有可恢复的有效历史快照。"
            "未生成模拟数据；请检查网络后点击「🔄 刷新公开数据」，"
            "或上传有权使用的真实历史数据。")
    return out
