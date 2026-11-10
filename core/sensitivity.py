# -*- coding: utf-8 -*-
"""P5 参数敏感性分析：窗口长度 × 协方差方法 × 再平衡频率。

**共同样本外评价区间（强制）**
不同估计窗口的回测起点不同（窗口越长，第一次可执行调仓越晚）。
若直接比较各自完整 OOS 的绩效，等于比较不同市场阶段，结论无效。
本模块把所有成功策略的 OOS 日收益对齐后，取
    common_start = max(各策略 OOS 起点)，common_end = min(各策略 OOS 终点)
并**只在该共同区间**重算所有绩效指标。

**避免过拟合警示**：参数敏感性属于探索性分析；在同一段样本外数据上挑出最优参数后，
该段数据已不再是独立的样本外验证。未来应扩展嵌套滚动验证 / 真正保留集。

性能与容错：组合总数上限 max_combinations；单个组合失败只记录错误、不影响其他组合；
支持进度回调；Streamlit 层对结果做会话级缓存。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from core import returns as R
from core.backtest import BacktestConfig, BacktestResult, REBALANCE_PRESETS, run_backtest
from core.models import Constraints

METRICS = {
    "年化收益(实际CAGR)": "实际CAGR(按实际年数)",
    "年化波动率": "年化波动率",
    "夏普比率": "夏普比率",
    "最大回撤": "最大回撤",
    "几何收益(每期)": "几何平均收益(每期)",
}


@dataclass
class SensitivityConfig:
    windows: list[tuple[str, int]] = field(default_factory=lambda: [
        ("3个月（63交易日）", 63), ("6个月（126交易日）", 126),
        ("12个月（252交易日）", 252), ("24个月（504交易日）", 504),
    ])
    cov_methods: list[str] = field(default_factory=lambda: ["sample", "ewma", "ledoit_wolf"])
    rebalance_labels: list[str] = field(default_factory=lambda: ["月度（21交易日）", "季度（63交易日）", "年度（252交易日）"])
    model: str = "最小方差"
    ewma_lambda: float = 0.94
    cost_bps: float = 0.0
    rf_annual: float = 0.0
    constraints: Constraints = field(default_factory=Constraints)
    custom_weights: dict | None = None
    max_combinations: int = 60
    min_obs: int = 60
    allow_truncation: bool = False    # 超过上限时必须显式确认截断，默认禁止静默截断

    def grid(self) -> list[tuple[str, int, str, str, int]]:
        out = []
        for (wl, w), m, rl in product(self.windows, self.cov_methods, self.rebalance_labels):
            out.append((wl, w, m, rl, REBALANCE_PRESETS[rl]))
        if len(out) > self.max_combinations:
            if not self.allow_truncation:
                raise ValueError(
                    f"完整参数组合数 {len(out)} 超过上限 {self.max_combinations}。"
                    "默认不允许静默截断：请减少窗口/方法/频率组合，"
                    f"或显式设置 allow_truncation=True 确认仅运行前 {self.max_combinations} 组"
                    "（结果为**部分组合比较**，不得称为完整参数搜索）。"
                )
            out = out[: self.max_combinations]
        return out


@dataclass
class SensitivityResult:
    ok: bool
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    common_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    curves: pd.DataFrame = field(default_factory=pd.DataFrame)
    cov_compare: pd.DataFrame = field(default_factory=pd.DataFrame)
    rebalance_compare: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_compare: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_by_method: pd.DataFrame = field(default_factory=pd.DataFrame)
    mean_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    common_start: pd.Timestamp | None = None
    common_end: pd.Timestamp | None = None
    n_common: int = 0
    n_total: int = 0
    n_success: int = 0
    n_total_full: int = 0      # 完整参数组合数（截断前）
    n_executed: int = 0        # 实际执行组合数
    n_skipped: int = 0         # 因截断未执行的组合数
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _perf_on_common(r: pd.Series, d0, d1, rf_annual: float) -> dict:
    if len(r) < 2:
        return {}
    p = R.performance_summary(
        r, price_index=pd.DatetimeIndex([pd.Timestamp(d0)] + list(r.index)),
        rf_annual=rf_annual,
    )
    return {
        "共同区间CAGR": p["实际CAGR(按实际年数)"],
        "共同区间年化波动率": p["年化波动率"],
        "共同区间夏普": p["夏普比率"],
        "共同区间最大回撤": p["最大回撤"],
        "共同区间几何收益(每期)": p["几何平均收益(每期)"],
        "共同区间累计收益": p["累计收益"],
        "共同区间期末净值": p["期末净值"],
    }


def run_sensitivity(
    returns_df: pd.DataFrame,
    config: SensitivityConfig,
    progress_cb=None,
) -> SensitivityResult:
    n_full = len(list(product(config.windows, config.cov_methods, config.rebalance_labels)))
    try:
        grid = config.grid()
    except ValueError as exc:   # 截断未经确认：不静默运行部分网格
        return SensitivityResult(ok=False, n_total_full=n_full, n_executed=0,
                                 n_skipped=n_full, truncated=True,
                                 errors=[str(exc)], warnings=[str(exc)])
    truncated = n_full > len(grid)
    total = len(grid)
    n_skipped = n_full - total
    bt_results: dict[str, BacktestResult] = {}
    meta_rows: list[dict] = []
    errors: list[str] = []

    for i, (wl, wdays, method, rlabel, revery) in enumerate(grid):
        key = f"{wl} | {method} | {rlabel}"
        cfg = BacktestConfig(
            model=config.model, cov_method=method, ewma_lambda=config.ewma_lambda,
            window=wdays, window_label=wl, rebalance_every=revery, rebalance_label=rlabel,
            cost_bps=config.cost_bps, rf_annual=config.rf_annual,
            constraints=config.constraints, custom_weights=config.custom_weights,
            min_obs=config.min_obs,
        )
        try:
            bt = run_backtest(returns_df, cfg)
        except Exception as exc:  # noqa: BLE001 —— 单组合失败不影响其余
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
            bt = BacktestResult(ok=False, config=cfg, error=str(exc))
        if bt.ok:
            bt_results[key] = bt
        else:
            errors.append(f"{key}: {bt.error}")
        # 失败调仓（部分成功）不能标为「完全成功」，且失败次数须用引擎的真实计数
        n_failed_trades = bt.n_failed_trades if bt.ok else 0
        failed_reasons = (
            "；".join(str(x) for x in bt.rebalance_log.loc[
                bt.rebalance_log["status"] == "failed", "失败原因"].head(5))
            if bt.ok and {"status", "失败原因"}.issubset(bt.rebalance_log.columns) else ""
        )
        meta_rows.append({
            "key": key, "窗口": wl, "窗口交易日数": wdays, "协方差方法": method,
            "再平衡": rlabel, "调仓间隔": revery,
            "状态": ("失败" if not bt.ok else ("部分成功" if n_failed_trades > 0 else "成功")),
            "错误": "" if bt.ok else bt.error,
            "每组估计窗口": f"{wl} = {wdays} 个交易日",
            "首个有效交易日": bt.oos_start if bt.ok else None,
            "原OOS起点": bt.oos_start if bt.ok else None,
            "原OOS终点": bt.oos_end if bt.ok else None,
            "原OOS天数": bt.n_oos if bt.ok else 0,
            "求解失败次数": (n_failed_trades if bt.ok else 1),
            "失败调仓原因": failed_reasons,
            "调仓次数": bt.n_trade_events if bt.ok else 0,
            "首次建仓次数": bt.n_initial_build if bt.ok else 0,
            "累计换手": bt.turnover_total if bt.ok else 0,
            "累计成本": bt.cost_total if bt.ok else 0,
        })
        if progress_cb:
            progress_cb(i + 1, total, key)

    if not bt_results:
        return SensitivityResult(
            ok=False, n_total=total, truncated=truncated, errors=errors,
            n_total_full=n_full, n_executed=total, n_skipped=n_skipped,
            warnings=["所有参数组合均失败，无敏感性结果（未生成任何虚构数据）"],
        )

    # 共同样本外区间：所有成功策略 OOS 的交集。
    # 独立审查修复：oos_start 是各策略的**首次建仓执行日**——当日资金仍为现金
    # （r_port=0，只扣建仓成本），新权重从其**下一个交易日**起才产生收益。
    # 若直接以 max(oos_start) 作为评价首日，最晚建仓的策略会把「现金建仓日收益」
    # 混入共同区间，与其他策略当日的正常持仓收益口径不一致。
    # 因此把共同区间起点推进到最晚建仓日的下一个交易日，确保区间内所有策略均已持仓。
    raw_start = max(b.oos_start for b in bt_results.values())
    c_end = min(b.oos_end for b in bt_results.values())
    cal = returns_df.index
    pos = cal.get_loc(raw_start)
    if not isinstance(pos, (int, np.integer)):
        pos = int(np.flatnonzero(pos)[0])
    c_start = cal[pos + 1] if pos + 1 < len(cal) else raw_start
    if c_start >= c_end:
        return SensitivityResult(
            ok=False, n_total=total, truncated=truncated, errors=errors,
            n_total_full=n_full, n_executed=total, n_skipped=n_skipped,
            warnings=[f"共同样本外区间为空（{c_start} ~ {c_end}），无法公平比较不同窗口的策略"],
        )
    common_returns = pd.DataFrame({
        k: b.daily["r_net"] for k, b in bt_results.items()
    }).loc[c_start:c_end].dropna(how="any")
    if len(common_returns) < 2:
        return SensitivityResult(
            ok=False, n_total=total, truncated=truncated, errors=errors,
            n_total_full=n_full, n_executed=total, n_skipped=n_skipped,
            warnings=["共同样本外区间不足 2 条收益，无法计算绩效"],
        )

    # V2.3 统一 CAGR 口径：净值 1.0 位于共同区间首个收益日的前一交易日。
    # 用 dropna 后的实际首日定位，避免重复首日或人为增加虚假交易日。
    common_cal = pd.DatetimeIndex(common_returns.index)
    prior = cal[cal < common_cal[0]]
    nav_start = pd.Timestamp(prior[-1]) if len(prior) else pd.Timestamp(common_cal[0])

    rows = []
    for m in meta_rows:
        key = m["key"]
        row = dict(m)
        if key in bt_results:
            b = bt_results[key]
            row.update(_perf_on_common(common_returns[key], nav_start,
                                       c_end, config.rf_annual))
            wlog = b.rebalance_log
            wcols = [c for c in wlog.columns if c.startswith("w_")]
            if wcols:
                row["平均目标权重"] = {c[2:]: float(wlog[c].mean()) for c in wcols}
                row["权重波动(std)"] = float(np.mean([float(wlog[c].std(ddof=0)) for c in wcols]))
            row["预测波动率均值"] = float(wlog["predicted_vol"].mean()) if "predicted_vol" in wlog else np.nan
        rows.append(row)
    results = pd.DataFrame(rows).set_index("key")

    # 曲线数据（窗口长度 × 方法，固定默认再平衡频率取第一档，或按再平衡取均值）
    curves = []
    base_r = config.rebalance_labels[0] if config.rebalance_labels else None
    for (wl, wdays), method in product(config.windows, config.cov_methods):
        sel = results[(results["窗口"] == wl) & (results["协方差方法"] == method)
                      & (results["再平衡"] == base_r) & (results["状态"].isin(["成功", "部分成功"]))]
        if sel.empty:
            continue
        for label, col in METRICS.items():
            k = f"共同区间{label}" if "共同区间" + label in sel.columns else None
            src = k or col
            if src in sel.columns:
                curves.append({"窗口": wl, "窗口交易日数": wdays, "协方差方法": method,
                               "指标": label, "值": float(sel[src].iloc[0])})
    curves_df = pd.DataFrame(curves)

    # 协方差方法比较（固定窗口 = 中间档、固定再平衡 = 第一档）
    mid_w = config.windows[len(config.windows) // 2][0] if config.windows else None
    cov_cmp = results[(results["窗口"] == mid_w) & (results["再平衡"] == base_r)
                      & (results["状态"].isin(["成功", "部分成功"]))].copy()
    rebal_cmp = results[(results["窗口"] == mid_w)
                        & (results["协方差方法"] == config.cov_methods[0])
                        & (results["状态"].isin(["成功", "部分成功"]))].copy()

    mean_w = pd.DataFrame(
        {k: v for k, v in results.get("平均目标权重", pd.Series(dtype=object)).items()
         if isinstance(v, dict)}
    ).T if "平均目标权重" in results.columns else pd.DataFrame()

    # ---- 预测风险 vs 实现风险（**共同区间**，实现风险仅事后评价）----
    fc_rows = []
    meta_by_key = {m["key"]: m for m in meta_rows}
    for key, b in bt_results.items():
        fe = getattr(b, "forecast_eval", pd.DataFrame())
        if fe is None or fe.empty:
            continue
        sub = fe[(fe["execution_date"] >= c_start) & (fe["execution_date"] <= c_end)]
        err = sub["forecast_error"].dropna().to_numpy(dtype=float)
        m = meta_by_key.get(key, {})
        n_obs = sub["eval_n_obs"].dropna().to_numpy(dtype=float)
        err_long = sub.loc[sub["eval_n_obs"] >= 21, "forecast_error"].dropna().to_numpy(dtype=float)
        fc_rows.append({
            "key": key, "窗口": m.get("窗口", ""),
            "协方差方法": m.get("协方差方法", ""), "再平衡": m.get("再平衡", ""),
            "共同区间评价事件数": int(len(err)),
            "评价期限观测数(中位)": float(np.median(n_obs)) if len(n_obs) else np.nan,
            "MAE": float(np.mean(np.abs(err))) if len(err) else np.nan,
            "RMSE": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
            "平均偏差(实现−预测)": float(np.mean(err)) if len(err) else np.nan,
            "有效评价数(排除短样本)": int(len(err_long)),
            "MAE(排除短样本,稳健性)": float(np.mean(np.abs(err_long))) if len(err_long) else np.nan,
            "RMSE(排除短样本,稳健性)": float(np.sqrt(np.mean(err_long ** 2))) if len(err_long) else np.nan,
        })
    forecast_compare = pd.DataFrame(fc_rows)
    forecast_by_method = (
        forecast_compare.groupby("协方差方法")[["MAE", "RMSE", "平均偏差(实现−预测)"]]
        .mean(numeric_only=True)
        if not forecast_compare.empty else pd.DataFrame()
    )
    # 跨再平衡频率平均 MAE/RMSE 会混合不同评价期限，必须显式标注、避免误读为可比结论
    if not forecast_by_method.empty and not forecast_compare.empty:
        grp = forecast_compare.groupby("协方差方法")
        forecast_by_method["包含的再平衡频率数"] = grp["再平衡"].nunique()
        forecast_by_method["评价期限是否一致"] = grp["评价期限观测数(中位)"].nunique().map(
            lambda x: "一致" if x <= 1 else "不一致(不可直接比较)")

    warns = [
        "⚠️ **探索性分析，不是独立样本外验证**：在同一段共同 OOS 数据上挑选最优窗口/方法后，"
        "该段数据已被用于参数选择，不能再称为独立的样本外验证；"
        "**参数搜索后的最优结果不得称为独立验证结果**。建议未来扩展嵌套滚动验证或保留最后一段作为纯保留集。",
        f"所有绩效在**共同样本外区间** {c_start.date()} ~ {c_end.date()}（{len(common_returns)} 个交易日）重算，"
        "避免把不同起止日期的结果直接排名。区间起点已排除各策略的首次建仓日"
        "（当日为现金、新权重次日起生效），确保区间内所有策略均处于持仓状态、收益口径一致。",
        "窗口长度单位为**固定交易日数量**（63/126/252/504/756），不是严格的日历月；"
        "252 个交易日只是近似一年，遇节假日会与 12 个日历月不一致。",
        "交易成本口径统一：所有组合共用同一成本率与同一换手定义"
        "（turnover=Σ|目标权重−交易前权重|，cost=turnover×bp/10000，含首次建仓）。"
        "不同再平衡频率的**累计成本差异来自调仓次数本身**，不是口径不一致；"
        "比较收益时须把换手/成本一并纳入。",
        "⚠️ 预测风险 MAE/RMSE 的评价期限由再平衡频率决定（月度≈21、季度≈63、年度≈252 个交易日），"
        "不同频率下每事件观测数不同，**MAE/RMSE 不可直接跨再平衡频率比较**；"
        "请在相同频率、相近评价期限内比较，或使用「排除短样本」稳健性指标。",
    ]
    partial = [k for k, b in bt_results.items() if b.n_failed_trades > 0]
    if partial:
        warns.append(
            f"⚠️ {len(partial)} 个参数组合存在失败调仓（状态标记为**部分成功**，失败时继续持有漂移权重）："
            + "；".join(partial[:3]) + ("…" if len(partial) > 3 else "")
            + "。这些组合与其他组合的**执行路径不同**，横向比较公平性受影响，结论需谨慎；"
            "失败次数与原因见结果表「求解失败次数 / 失败调仓原因」两列。"
        )
    if truncated:
        warns.append(
            f"⚠️ 参数组合数超过上限 {config.max_combinations}：完整 {n_full} 组，"
            f"实际执行 {total} 组、未执行 {n_skipped} 组（经显式确认截断）。"
            "**本次结果是部分组合比较，不是完整参数搜索**；未执行组合不参与任何结论，"
            "结论不得表述为「该参数区间内全局最优」。"
        )
    if config.cost_bps == 0:
        warns.append("当前交易成本为 0，未计入费用影响。")
    if errors:
        warns.append(f"{len(errors)} 个参数组合失败（已记录错误，不影响其余组合）。")

    return SensitivityResult(
        ok=True, results=results, common_returns=common_returns, curves=curves_df,
        cov_compare=cov_cmp, rebalance_compare=rebal_cmp, mean_weights=mean_w,
        forecast_compare=forecast_compare, forecast_by_method=forecast_by_method,
        common_start=c_start, common_end=c_end, n_common=len(common_returns),
        n_total=total, n_success=len(bt_results), truncated=truncated,
        n_total_full=n_full, n_executed=total, n_skipped=n_skipped,
        warnings=warns, errors=errors,
    )
