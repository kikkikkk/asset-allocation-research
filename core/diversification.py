# -*- coding: utf-8 -*-
"""P2/P4 分散化收益（DR）模块：理论近似与实际再平衡对照严格分离。

理论 DR（二阶近似，年化**对数增长空间**）
-----------------------------------------
    DR ≈ 0.5 × [ Σ_i w_i σ_i² − σ_p² ],     σ_p² = wᵀΣw

- w：固定**目标**权重（Σw = 1，默认 long-only）；
- σ_i²：各资产年化方差（Σ 的对角线）；σ_p²：组合年化方差；
- 时间单位：Σ 为期间协方差 × K（日频 K=252），故 DR 为**年化**量；
- 再平衡假设：**每日按目标权重再平衡**，此时 r_p,t = Σ w_i r_i,t，
  样本组合方差恒等于 wᵀΣw（方差部分无近似）；
- 近似条件：DR 来自 ln(1+r) 的二阶 Taylor 展开，忽略三阶及以上高阶矩；
  |r| 较小、无极端跳空时更准确；
- long-only 下 DR = 0.5·Σ_{i<j} w_i w_j (σ_i−σ_j)² + … ≥ 0（数值可校验）；
- ⚠️ **DR 不是实际再平衡超额收益**：后者是简单收益口径下
  「定期再平衡组合 vs 买入持有组合」的复合收益差，两者不同空间、不可混用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import returns as R
from core.backtest import buy_and_hold, compare_rebalance_frequencies, simulate_portfolio

TRADING_DAYS = 252


def dr_analysis(
    weights: dict | pd.Series,
    cov_ann: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """DR 与方差分解。cov_ann 为年化协方差矩阵（index/columns = asset_id）。"""
    ids = list(cov_ann.columns)
    w = pd.Series({a: float(weights.get(a, 0.0)) for a in ids}, index=ids)
    if abs(float(w.sum()) - 1) > 1e-8:
        raise ValueError(f"权重之和必须等于 1（当前 {w.sum():.8f}）")
    if (w < -1e-10).any():
        raise ValueError("DR 模块默认 long-only（权重非负）")
    C = cov_ann.to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    variances = np.diag(C)
    weighted_var = float(wv @ variances)          # Σ w_i σ_i²
    port_var = float(wv @ C @ wv)                  # wᵀΣw
    dr = 0.5 * (weighted_var - port_var)
    marginal = C @ wv
    contrib = pd.DataFrame({
        "目标权重": wv,
        "年化方差σ_i²": variances,
        "加权方差贡献w_iσ_i²": wv * variances,
        "边际风险(Σw)_i": marginal,
        "总方差贡献w_i(Σw)_i": wv * marginal,
    }, index=ids)
    return {
        "dr_annual": dr,
        "weighted_var": weighted_var,
        "port_var": port_var,
        "port_vol": float(np.sqrt(max(port_var, 0))),
        "covariance_effect": port_var - weighted_var,   # = −2DR
        "contributions": contrib,
        "corr_free_lower_bound": dr >= -1e-12,          # long-only 下应恒为 True
        "n_assets": len(ids),
        "periods_per_year": periods_per_year,
        "definition": {
            "权重": "固定目标权重，Σw=1，long-only",
            "资产方差": "年化协方差矩阵对角线 σ_i² = var(r_i)×K",
            "协方差": "Σ 为年化协方差矩阵（样本/EWMA/LW 由用户选择）",
            "组合方差": "σ_p² = wᵀΣw（每日固定权重再平衡下与样本组合方差一致）",
            "时间单位": f"年化（K={periods_per_year} 期/年）",
            "再平衡假设": "每日收盘后按目标权重再平衡；非每日再平衡时 DR 仅为理论参照",
            "近似条件": "ln(1+r) 二阶 Taylor 展开；忽略三阶及以上高阶矩，|r| 越小越准",
            "注意": "DR 是对数空间二阶近似量，不等于实际再平衡超额收益",
        },
    }


def dr_weight_curve(
    asset_id: str,
    base_weights: dict,
    cov_ann: pd.DataFrame,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """改变单一资产目标权重（其余资产按原比例缩放），给出 DR 曲线。"""
    if grid is None:
        grid = np.linspace(0, 1, 51)
    ids = [a for a in cov_ann.columns]
    other = [a for a in ids if a != asset_id]
    base_other = sum(float(base_weights.get(a, 0.0)) for a in other)
    rows = []
    for w0 in grid:
        w = {asset_id: float(w0)}
        rest = 1 - float(w0)
        for a in other:
            w[a] = rest * (float(base_weights.get(a, 0.0)) / base_other) if base_other > 0 else rest / max(len(other), 1)
        try:
            d = dr_analysis(w, cov_ann)
            rows.append({"股票/资产权重": float(w0), "理论DR(年化)": d["dr_annual"],
                         "加权方差": d["weighted_var"], "组合方差": d["port_var"]})
        except ValueError:
            continue
    return pd.DataFrame(rows)


def dr_vs_actual(
    returns_df: pd.DataFrame,
    weights: dict | pd.Series,
    cov_ann: pd.DataFrame,
    cost_bps: float = 0.0,
) -> dict:
    """理论 DR vs 实际再平衡策略对照（三个量并列，禁止混用）。

    - 理论 DR：年化对数空间二阶近似；
    - 实际对数增长率差：L(每日固定权重再平衡) − Σ w_i L_i（精确，与 DR 同空间）；
    - 再平衡溢价：CAGR(定期再平衡) − CAGR(买入持有)（简单收益口径，实际复合收益差）。
    """
    ids = list(returns_df.columns)
    w = pd.Series({a: float(weights.get(a, 0.0)) for a in ids}, index=ids)
    w = w / w.sum()
    theory = dr_analysis(w, cov_ann)
    r_rebal_daily = pd.Series(returns_df.to_numpy(dtype=float) @ w.to_numpy(dtype=float),
                              index=returns_df.index, name="daily_rebalanced")
    L_p = float(np.log1p(r_rebal_daily).mean() * TRADING_DAYS)
    L_assets = {a: float(np.log1p(returns_df[a]).mean() * TRADING_DAYS) for a in ids}
    L_weighted = float(sum(w[a] * L_assets[a] for a in ids))
    actual_log_diff = L_p - L_weighted

    # 定期再平衡 vs 买入持有（同区间、同成本假设）
    freq_table = compare_rebalance_frequencies(returns_df, w, cost_bps=cost_bps)
    bh = buy_and_hold(returns_df, w)
    # "每日固定权重再平衡"= 每个交易日都恢复目标权重；理论 DR 对照按零成本计算
    sched = [(j, w.to_numpy(dtype=float).copy(), {}) for j in range(0, len(returns_df) - 1)]
    daily_reb = simulate_portfolio(returns_df, sched, cost_bps=0.0)
    cagr_reb = R.cagr_from_nav(float(daily_reb["nav"].iloc[-1]), 1.0,
                               returns_df.index[0], returns_df.index[-1])
    cagr_bh = R.cagr_from_nav(float(bh["nav"].iloc[-1]), 1.0,
                              returns_df.index[0], returns_df.index[-1])
    return {
        "theory": theory,
        "L_portfolio": L_p,
        "L_weighted_assets": L_weighted,
        "actual_log_growth_diff": actual_log_diff,
        "dr_vs_log_diff_gap": theory["dr_annual"] - actual_log_diff,
        "cagr_daily_rebalanced": cagr_reb,
        "cagr_buy_and_hold": cagr_bh,
        "rebalancing_premium_simple": cagr_reb - cagr_bh,
        "rebalance_frequency_table": freq_table,
        "note": (
            "理论 DR（对数空间二阶近似）≈ 实际对数增长率差，但与简单收益口径的"
            "再平衡溢价（每日再平衡 CAGR − 买入持有 CAGR）是不同空间的量，数值可明显不同，三者不可混用。"
        ),
    }
