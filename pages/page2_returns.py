# -*- coding: utf-8 -*-
"""页面2：收益与风险（统一指标、相关性、算术/几何近似、DR、再平衡频率对照）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import returns as R
from core.covariance import estimate_covariance
from core.diversification import dr_analysis, dr_weight_curve, dr_vs_actual
from pages.common import (bar_weights, fmt_dataframe, grouped_bar, heatmap_corr,
                          line_chart, section, show_dict, xy_line)


def render(ctx: dict):
    st.header("2️⃣ 收益与风险")
    rets = ctx["returns"]
    assets = list(rets.columns)
    K = ctx["periods_per_year"]
    rf = ctx["rf_annual"]
    st.caption(
        f"全部指标由 `core/returns.py` 统一计算：{len(rets)} 个收益样本、K={K} 期/年、"
        f"无风险利率年化 {rf:.3%}{'（零基准假设）' if rf == 0 else ''}。"
        "CAGR 按**实际经过年数**，与按频率年化的几何收益分列展示。"
    )

    # ---------- 资产/等权组合指标 ----------
    section("收益与风险指标（按资产 + 等权组合）")
    series_map = {a: rets[a] for a in assets}
    eq_w = {a: 1.0 / len(assets) for a in assets}
    series_map["等权组合（每日再平衡）"] = pd.Series(
        rets.to_numpy() @ np.array(list(eq_w.values())), index=rets.index
    )
    perf = {}
    for name, r in series_map.items():
        try:
            perf[name] = R.performance_summary(r, price_index=ctx["prices"].index,
                                               rf_annual=rf, periods_per_year=K)
        except ValueError as exc:
            st.warning(f"{name}: {exc}")
    perf_df = pd.DataFrame(perf).T
    cols = ["累计收益", "算术平均收益(每期)", "年化算术平均收益", "平均对数收益(每期)",
            "几何平均收益(每期)", "年化几何收益(按频率)", "实际CAGR(按实际年数)",
            "年化波动率", "夏普比率", "最大回撤"]
    show = fmt_dataframe(perf_df[[c for c in cols if c in perf_df.columns]],
                         pct_cols=[c for c in cols if c in perf_df.columns and c != "夏普比率"],
                         num_cols=["夏普比率"], digits=3)
    st.dataframe(show, width="stretch")
    st.caption(
        "单位：收益率/波动率/回撤为百分比；夏普比率为无量纲。"
        "「几何平均收益(每期)」是单期量，「年化几何收益(按频率)」= (1+G)^K−1，"
        "「实际CAGR」按起止日期实际年数 Y 计算；当实际跨度 ≠ N/K 时后两者不同，**不得混用**。"
    )

    # ---------- 相关 / 协方差 ----------
    c1, c2 = st.columns(2)
    with c1:
        section("相关系数矩阵")
        corr = R.correlation_matrix(rets)
        corr.index = [ctx["registry"].get(a).name for a in corr.index]
        corr.columns = [ctx["registry"].get(a).name for a in corr.columns]
        heatmap_corr(corr)
        st.caption(f"区间 {rets.index[0].date()} ~ {rets.index[-1].date()}；单位：无量纲 ρ ∈ [−1,1]。")
    with c2:
        section("年化协方差矩阵（样本估计）")
        cov = estimate_covariance(rets, method="sample", periods_per_year=K)
        cov_show = cov.matrix.copy()
        nm = [ctx["registry"].get(a).name for a in cov_show.columns]
        cov_show.index, cov_show.columns = nm, nm
        st.dataframe(cov_show.map(lambda x: f"{x:.6f}"), width="stretch")
        st.caption(f"估计窗口 {cov.window_start.date()} ~ {cov.window_end.date()}，"
                   f"n={cov.n_obs}，ddof=1，×{K} 年化；单位为「年化方差」。")

    # ---------- 算术 / 几何近似 ----------
    section("波动拖累验证：g ≈ μ − 0.5σ²（A）与 g ≈ μ − 0.5·mean(r²)（B）",
            "两种均为**近似**、非恒等式；g 是平均对数收益率，年化复利必须用 exp(Kg)−1。")
    checks = {}
    for name, r in series_map.items():
        checks[name] = R.arithmetic_geometric_check(r, K)
    chk = pd.DataFrame(checks).T
    tbl = pd.DataFrame({
        "算术平均μ(每期)": chk["mu_arith_period"],
        "精确平均对数收益g": chk["g_exact_period"],
        "近似A：μ−0.5σ²": chk["g_approx_A_period"],
        "近似B：μ−0.5mean(r²)": chk["g_approx_B_period"],
        "几何平均G=exp(g)−1": chk["geom_mean_period"],
        "年化几何exp(Kg)−1": chk["annualized_geom_exact"],
        "A误差(每期, bp)": chk["error_period_bp"],
        "A年化线性误差(bp)": chk["error_annual_linear_bp"],
        "A年化复利误差(bp)": chk["error_annual_compound_bp"],
        "B误差(每期, bp)": chk["error_B_period_bp"],
        "B年化复利误差(bp)": chk["error_B_annual_compound_bp"],
        "A−B差异(bp)": chk["approx_A_minus_B"] * 10000,
        "误差符号": chk["error_sign"],
        "A绝对误差(每期, bp)": chk["abs_error_period"] * 10000,
    })
    pct = ["算术平均μ(每期)", "精确平均对数收益g", "近似A：μ−0.5σ²", "近似B：μ−0.5mean(r²)",
           "几何平均G=exp(g)−1", "年化几何exp(Kg)−1"]
    st.dataframe(fmt_dataframe(tbl, pct_cols=pct, digits=5), width="stretch")
    err_long = tbl[["A误差(每期, bp)", "A年化线性误差(bp)", "A年化复利误差(bp)",
                    "B年化复利误差(bp)"]].reset_index()
    err_long.columns = ["对象", "A每期(bp)", "A年化线性(bp)", "A年化复利(bp)", "B年化复利(bp)"]
    err_melt = err_long.melt(id_vars="对象", var_name="误差类型", value_name="误差bp")
    grouped_bar(err_melt, "对象", "误差bp", "误差类型",
                "算术—几何近似误差（重点：年化后的误差，A/B 对照）", "误差（基点 bp）",
                y_format=".2f", height=320)
    st.caption(
        "**单位：bp（基点）= 收益率差 × 10000，1bp = 0.01pp。**\n\n"
        "- `g = mean(ln(1+r))` 是平均对数收益率；`G = exp(g)−1` 为每期几何收益；"
        "**按频率年化几何收益 = exp(K·g)−1 = (1+G)^K−1**（用 np.expm1 计算）；"
        "它 ≠ 按实际经过年数计算的 CAGR，两者口径不同、数值可不同。\n"
        "- **A（主展示）** `g_A = μ − 0.5σ²`（σ² 为 ddof=1 样本方差，金融惯用波动拖累近似）；"
        "**B（展开细节）** `g_B = μ − 0.5·mean(r²)` 更贴近 ln(1+r)=r−r²/2+r³/3−… 的二阶 Taylor 展开"
        "（因为 E[r²]=Var(r)+μ²）。**A 用 ddof=1 样本方差，不是严格的二阶 Taylor 恒等式**，两者都含高阶项误差。\n"
        "- 年化复利误差 = `expm1(K·g_approx) − expm1(K·g_exact)`，是**收益率之差**；"
        "线性年化误差 = 每期误差 × K，是对数率之差；两者不可混用。\n"
        "- 适用条件：|r| 较小、无极端厚尾/跳空、无杠杆放大；误差符号取决于偏度，"
        "**不保证为正，也不保证 σ 越大误差越大**。\n"
        "- 误差符号与绝对误差列用于检验上述论断，避免把近似当精确。"
    )

    # ---------- DR ----------
    section("分散化收益（理论二阶近似）与实际再平衡对照")
    weights = ctx["shared"].get("target_weights") or eq_w
    weights = {a: float(weights.get(a, 0.0)) for a in assets}
    s = sum(weights.values())
    weights = {a: v / s for a, v in weights.items()} if s > 0 else eq_w
    dr = dr_analysis(weights, cov.matrix)
    show_dict({
        "权重定义": dr["definition"]["权重"], "时间单位": dr["definition"]["时间单位"],
        "再平衡假设": dr["definition"]["再平衡假设"], "近似条件": dr["definition"]["近似条件"],
        "加权资产方差 Σwσ²": f"{dr['weighted_var']*100:.4f}%（年化方差）",
        "组合方差 wᵀΣw": f"{dr['port_var']*100:.4f}%（年化方差）",
        "理论 DR（对数空间近似）": f"{dr['dr_annual']*100:.4f}%/年",
        "long-only 下 DR≥0": dr["corr_free_lower_bound"],
    })
    contrib = dr["contributions"].copy()
    contrib.index = [ctx["registry"].get(a).name for a in contrib.index]
    bar_weights(pd.Series(contrib["加权方差贡献w_iσ_i²"].to_numpy(),
                          index=contrib.index),
                "各资产方差贡献 wᵢσᵢ² / DR 分解", "年化方差贡献")
    cmp_df = pd.DataFrame({
        "项目": ["加权资产方差 Σwσ²", "组合方差 wᵀΣw", "理论 DR×2"],
        "值": [dr["weighted_var"], dr["port_var"], 2 * dr["dr_annual"]],
    })
    bar_weights(pd.Series(cmp_df["值"].to_numpy(), index=cmp_df["项目"]),
                "DR 分解（年化方差）", "年化方差")

    focus = st.selectbox("DR 权重曲线：调整哪个资产的权重", assets,
                         format_func=lambda a: ctx["registry"].get(a).name)
    curve = dr_weight_curve(focus, weights, cov.matrix)
    xy_line(curve, "股票/资产权重", "理论DR(年化)", None,
            "不同目标权重下的理论 DR", "所选资产目标权重", "理论 DR（年化）", height=280)
    st.caption("横轴为所选资产的目标权重（其余资产按原比例缩放），纵轴为年化 DR；单一资产（0% 或 100%）DR=0。")

    actual = dr_vs_actual(rets, weights, cov.matrix, cost_bps=ctx["shared"].get("cost_bps", 0.0))
    show_dict({
        "理论 DR（年化对数空间近似）": f"{dr['dr_annual']*100:.4f}%",
        "实际对数增长率差（精确）": f"{actual['actual_log_growth_diff']*100:.4f}%",
        "两者之差（近似误差）": f"{actual['dr_vs_log_diff_gap']*100:.4f}pp",
        "每日再平衡 CAGR": f"{actual['cagr_daily_rebalanced']*100:.4f}%",
        "买入持有 CAGR": f"{actual['cagr_buy_and_hold']*100:.4f}%",
        "再平衡溢价（简单收益口径）": f"{actual['rebalancing_premium_simple']*100:.4f}pp",
    }, "三量对照：**理论 DR ≠ 实际再平衡超额收益**")
    st.caption(actual["note"])

    freq = actual["rebalance_frequency_table"]
    if not freq.empty and "年化收益(实际CAGR)" in freq.columns:
        fl = freq.reset_index().rename(columns={"index": "再平衡频率"})
        flong = fl.melt(id_vars="再平衡频率",
                        value_vars=["年化收益(实际CAGR)", "年化波动率", "最大回撤", "累计换手"],
                        var_name="指标", value_name="值")
        grouped_bar(flong, "再平衡频率", "值", "指标",
                    "不同再平衡频率的表现（同一区间、同一成本假设）", "数值", y_format=".2f", height=300)
        st.dataframe(fmt_dataframe(freq, pct_cols=["年化收益(实际CAGR)", "年化波动率", "最大回撤"],
                                   num_cols=["夏普比率", "累计换手", "调仓次数", "期末净值"]),
                     width="stretch")

    # ---------- 滚动指标 ----------
    section("滚动收益与波动率（等权组合）")
    win = st.slider("滚动窗口（交易日）", 21, min(504, max(63, len(rets) // 2)), 126, step=21)
    roll = R.rolling_metrics(series_map["等权组合（每日再平衡）"], win, K, rf_annual=ctx["rf_annual"])
    line_chart(roll.dropna()[["滚动年化波动率", "滚动年化几何收益"]],
               "滚动年化波动率 / 滚动年化几何收益", "比率（年化）", y_format="%", height=280)
    line_chart(roll.dropna()[["滚动夏普比率"]], "滚动夏普比率（rf 频率一致）",
               "夏普（无量纲）", y_format=".2f", height=240)
    st.caption(
        f"窗口 {win} 个交易日；收益/波动单位为百分比/年，夏普无量纲；"
        f"rf_annual={ctx['rf_annual']:.3%}{'（零基准假设，非真实超额收益）' if ctx['rf_annual'] == 0 else ''}；"
        f"区间 {rets.index[0].date()} ~ {rets.index[-1].date()}。"
    )
