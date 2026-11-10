# -*- coding: utf-8 -*-
"""页面5：参数敏感性（窗口 × 协方差方法 × 再平衡频率，共同样本外区间）
+ V2.3 估计窗口诊断与分年度研究。"""
from __future__ import annotations

import streamlit as st

from core.backtest import REBALANCE_PRESETS, WINDOW_PRESETS
from core.sensitivity import SensitivityConfig, run_sensitivity
from pages.common import (ensure_v23_diagnosis, fmt_dataframe, grouped_bar,
                          line_chart, section, show_dict, xy_line)


def render(ctx: dict):
    st.header("5️⃣ 参数敏感性")
    P = ctx["params"]
    st.caption(
        "所有策略在**共同样本外区间**上重算绩效；不同窗口的回测起点不同，"
        "不取交集直接排名是不公平的。参数敏感性是探索性分析，**不是独立的样本外验证**。"
    )

    # ============ V2.3：1/2/3 年估计窗口诊断 + 分年度研究 ============
    _render_v23_window_annual(ctx)

    c1, c2, c3 = st.columns(3)
    win_labels = c1.multiselect("历史窗口", list(WINDOW_PRESETS),
                               default=list(WINDOW_PRESETS)[:3])
    methods = c2.multiselect("协方差方法", ["sample", "ewma", "ledoit_wolf"],
                             default=["sample", "ewma", "ledoit_wolf"])
    rebals = c3.multiselect("再平衡频率", [k for k in REBALANCE_PRESETS if k != "不再平衡（买入持有）"],
                            default=["月度（21交易日）", "季度（63交易日）", "年度（252交易日）"][:2])
    n_comb = len(win_labels) * len(methods) * len(rebals)
    if not (win_labels and methods and rebals):
        st.info("请至少各选一项以开始分析。")
        return

    MAX_COMBOS = 60
    confirm_truncate = False
    if n_comb > MAX_COMBOS:
        st.warning(
            f"完整参数组合数为 **{n_comb}**，超过运行上限 {MAX_COMBOS}。"
            "平台**不会静默截断**。请二选一：① 减少窗口/方法/频率组合；"
            "② 勾选下方确认框，仅运行前 60 组（结果为**部分组合比较**，"
            "不得称为完整参数搜索）。"
        )
        confirm_truncate = st.checkbox(
            "我确认只运行前 60 组参数组合（部分组合比较，非完整参数搜索）", value=False)
        if not confirm_truncate:
            st.stop()
    st.caption(f"参数组合数：**{n_comb}**（上限 60）；"
               f"模型 **{P['model']}**｜成本 **{P['cost_bps']}bp**｜λ={P['ewma_lambda']}")

    cfg = SensitivityConfig(
        windows=[(w, WINDOW_PRESETS[w]) for w in win_labels],
        cov_methods=methods, rebalance_labels=rebals,
        model=P["model"], ewma_lambda=P["ewma_lambda"],
        cost_bps=P["cost_bps"], rf_annual=ctx["rf_annual"],
        constraints=ctx["constraints"],
        custom_weights=ctx["shared"].get("custom_weights"),
        max_combinations=MAX_COMBOS, min_obs=60,
        allow_truncation=confirm_truncate,   # 截断必须经用户显式确认
    )
    key = repr((tuple(ctx["asset_ids"]), n_comb, tuple(win_labels), tuple(methods),
                tuple(rebals), P["model"], P["cost_bps"], P["ewma_lambda"],
                confirm_truncate))
    cache = ctx["shared"].setdefault("sens_cache", {})
    if st.button("🔁 重新运行敏感性分析") or key not in cache:
        bar = st.progress(0.0, text="准备参数网格…")

        def cb(i, total, label):
            bar.progress(i / total, text=f"{i}/{total} {label}")

        cache[key] = run_sensitivity(ctx["returns"], cfg, progress_cb=cb)
        bar.empty()
    res = cache[key]
    ctx["shared"]["sensitivity_result"] = res

    if not res.ok:
        st.error("**敏感性分析未能产出结果：**" + "；".join(res.warnings + res.errors[:3]))
        return

    section("共同样本外评价区间")
    show_dict({
        "共同区间": f"{res.common_start.date()} ~ {res.common_end.date()}",
        "共同交易日数": res.n_common,
        "有效收益观测数": res.n_common,
        "成功组合": f"{res.n_success} / {res.n_total}",
        "失败组合数": len(res.errors),
        "求解失败次数合计": int(res.results["求解失败次数"].sum()) if "求解失败次数" in res.results else 0,
        "完整参数组合数": res.n_total_full,
        "实际执行组合数": res.n_executed,
        "未执行组合数(截断)": res.n_skipped,
        "是否截断(部分比较)": res.truncated,
    })
    for w in res.warnings:
        st.warning(w)
    st.info(
        "**参数选择 ≠ 独立样本外验证**：训练/估计窗口用于求权重，滚动样本外回测用于检验，"
        "参数敏感性是在同一段共同样本外数据上比较参数。若据此挑选最优参数，该段已不再独立；"
        "**本版本未实现嵌套滚动验证或独立保留集**，不得把挑参结果称为独立验证。"
    )
    if res.errors:
        with st.expander("失败组合明细（不影响其他组合）"):
            for e in res.errors[:20]:
                st.caption("• " + e)

    # ---------- 8 类图表用标签页组织，避免页面拥挤 ----------
    curves = res.curves
    tabs = st.tabs([
        "① 窗口—收益", "② 窗口—波动", "③ 窗口—夏普", "④ 协方差方法",
        "⑤ 再平衡频率", "⑥ 权重稳定性", "⑦ 换手与成本", "⑧ 预测风险误差",
    ])

    def _window_tab(tab, metric, title, yfmt, unit):
        with tab:
            sub = curves[curves["指标"] == metric]
            if not sub.empty:
                xy_line(sub, "窗口交易日数", "值", "协方差方法", title,
                        "历史估计窗口（固定交易日数量）", metric, y_format=yfmt, height=300)
            st.caption(
                f"单位：{unit}；横轴为固定交易日数量（63/126/252/504/756 近似 3/6/12/24/36 个月，"
                "252 个交易日**不严格等于** 12 个日历月）；曲线按协方差方法分色；"
                f"纵轴来自共同样本外区间 {res.common_start.date()} ~ {res.common_end.date()} 重算的绩效。"
                + ("当前再平衡频率取第一档：" + rebals[0] if rebals else "")
            )

    _window_tab(tabs[0], "年化收益(实际CAGR)", "窗口长度—年化收益率（按协方差方法）", "%", "年化收益 %")
    _window_tab(tabs[1], "年化波动率", "窗口长度—年化波动率（按协方差方法）", "%", "年化波动率 %")
    _window_tab(tabs[2], "夏普比率", "窗口长度—夏普比率（按协方差方法）", ".2f", "夏普（无量纲）")

    with tabs[3]:
        cc = res.cov_compare
        if not cc.empty and "共同区间CAGR" in cc.columns:
            long = cc.reset_index()[["协方差方法", "共同区间CAGR", "共同区间年化波动率",
                                     "共同区间夏普", "共同区间最大回撤"]].melt(
                id_vars="协方差方法", var_name="指标", value_name="值")
            grouped_bar(long, "协方差方法", "值", "指标",
                        "不同协方差估计方法的样本外表现（共同区间）", "数值", y_format=".2f")
            st.dataframe(fmt_dataframe(
                cc[["窗口", "协方差方法", "再平衡", "共同区间CAGR", "共同区间夏普",
                    "共同区间最大回撤", "调仓次数", "累计换手"]],
                pct_cols=["共同区间CAGR", "共同区间最大回撤"],
                num_cols=["共同区间夏普", "调仓次数", "累计换手"]), width="stretch")
        st.caption("三种方法并列比较，**没有默认最优**：μI 收缩目标在异方差资产池会拉平方差，"
                   "EWMA 的有效样本量远小于窗口长度，不能因预测波动率更高就判定方法更差。")

    with tabs[4]:
        rc = res.rebalance_compare
        if not rc.empty and "共同区间CAGR" in rc.columns:
            long2 = rc.reset_index()[["再平衡", "共同区间CAGR", "共同区间年化波动率",
                                      "共同区间夏普", "共同区间最大回撤", "累计换手"]].melt(
                id_vars="再平衡", var_name="指标", value_name="值")
            grouped_bar(long2, "再平衡", "值", "指标",
                        "不同再平衡频率的表现（收益/风险/回撤/换手）", "数值", y_format=".2f")
            st.dataframe(fmt_dataframe(
                rc[["窗口", "协方差方法", "再平衡", "共同区间CAGR", "共同区间夏普",
                    "共同区间最大回撤", "累计换手", "调仓次数"]],
                pct_cols=["共同区间CAGR", "共同区间最大回撤"],
                num_cols=["共同区间夏普", "累计换手", "调仓次数"]), width="stretch")
        st.caption("月/季/半年度/年度/不再平衡均可实现；所有频率从同一首日现金建仓开始，"
                   "使用相同区间、相同成本口径。")

    with tabs[5]:
        if not res.mean_weights.empty:
            st.dataframe(fmt_dataframe(res.mean_weights, pct_cols=list(res.mean_weights.columns)),
                         width="stretch")
        if "权重波动(std)" in res.results.columns:
            st.dataframe(fmt_dataframe(
                res.results[["窗口", "协方差方法", "再平衡", "权重波动(std)", "预测波动率均值"]],
                num_cols=["权重波动(std)", "预测波动率均值"]), width="stretch")
        st.caption("权重稳定性 = 各资产目标权重在调仓序列上的标准差均值；越小越稳定。")

    with tabs[6]:
        if "累计成本" in res.results.columns:
            st.dataframe(fmt_dataframe(
                res.results[["窗口", "协方差方法", "再平衡", "累计换手", "累计成本", "调仓次数"]],
                num_cols=["累计换手", "累计成本", "调仓次数"]), width="stretch")
            grouped_bar(
                res.results.reset_index()[["再平衡", "累计换手", "累计成本"]].melt(
                    id_vars="再平衡", var_name="指标", value_name="值"),
                "再平衡", "值", "指标", "累计换手与累计成本对比", "数值", y_format=".3f")
        st.caption("换手主口径 = Σ|目标权重 − 交易前权重|；成本 = 换手 × bp/10000，含初始建仓成本。")

    with tabs[7]:
        if not res.forecast_by_method.empty:
            st.dataframe(fmt_dataframe(res.forecast_by_method,
                                       pct_cols=["MAE", "RMSE", "平均偏差(实现−预测)"],
                                       num_cols=["包含的再平衡频率数"]),
                         width="stretch")
        if not res.forecast_compare.empty:
            fc = res.forecast_compare
            fc_cols = ["key", "协方差方法", "窗口", "再平衡",
                       "共同区间评价事件数", "评价期限观测数(中位)", "MAE", "RMSE",
                       "有效评价数(排除短样本)", "MAE(排除短样本,稳健性)",
                       "RMSE(排除短样本,稳健性)"]
            st.dataframe(fmt_dataframe(
                fc[[c for c in fc_cols if c in fc.columns]],
                pct_cols=["MAE", "RMSE", "MAE(排除短样本,稳健性)", "RMSE(排除短样本,稳健性)"],
                num_cols=["共同区间评价事件数", "评价期限观测数(中位)",
                          "有效评价数(排除短样本)"]), width="stretch")
        st.caption(
            "预测风险 = 估计期协方差与目标权重推出的年化波动率；"
            "实现风险（主评价口径）= 持有期**固定目标权重**假设收益波动率（与预测口径一致），"
            "区别于实际漂移权重的净收益波动率。实现风险**只用于事后评价**，不参与该时点权重估计。\n\n"
            "⚠️ **可比性**：评价期限 = 单次持有期长度，由再平衡频率决定（月度≈21、季度≈63、年度≈252 个交易日）。"
            "表中同时给出「评价期限观测数(中位)」与「排除短样本」稳健性 MAE/RMSE；"
            "**不同再平衡频率的 MAE/RMSE 不可直接比较**。上方按方法汇总表若标注"
            "「评价期限是否一致 = 不一致」，说明该行 MAE 混合了不同期限，仅作参考、不得用于排名。"
        )

    with st.expander("全部参数结果（含每组估计窗口与首个有效交易日）"):
        cols = [c for c in res.results.columns if c != "平均目标权重"]
        st.dataframe(fmt_dataframe(
            res.results[cols],
            pct_cols=[c for c in cols if c.startswith("共同区间") and
                      any(k in c for k in ("CAGR", "波动率", "回撤", "几何", "累计收益"))],
            num_cols=[c for c in cols if c in ("共同区间夏普", "原OOS天数", "调仓次数",
                                               "累计换手", "累计成本", "求解失败次数")],
        ), width="stretch")


# ==================== V2.3：估计窗口诊断 + 分年度研究 ====================
def _render_v23_window_annual(ctx: dict) -> None:
    diag, err = ensure_v23_diagnosis(ctx)
    if diag is None:
        st.error(f"V2.3 诊断补算失败（**未用模拟数据补齐**）：{err}")
        return

    win = diag["window"]
    ann = diag["annual"]
    wf = win["frame"]
    c0, c1 = diag["common_start"], diag["common_end"]

    section("V2.3 估计窗口诊断：1 年 / 2 年 / 3 年",
            "共同样本外区间一致，不按历史最高收益自动选窗口")
    show_dict({
        "共同样本外区间": f"{win['common_start'].date()} ~ {win['common_end'].date()}"
                          f"（{int(wf['样本外交易日'].iloc[0])} 个交易日）",
        "共同起点依据": "以最长窗口（756 交易日）能开始回测的日期为准，"
                        "并排除首次建仓现金日；三个窗口在该区间上重算绩效",
        "模型/成本": f"最小方差｜{diag['config'].cov_method} 协方差｜"
                     f"{diag['config'].cost_bps}bp｜{diag['config'].rebalance_label}",
        "重要提示": "**不根据历史最高收益自动选择最优窗口**；短窗口与长窗口是"
                    "估计噪声与反应速度的权衡，单区间差异可能来自样本特定性",
    })
    st.dataframe(fmt_dataframe(
        wf, pct_cols=["共同区间CAGR", "年化波动率", "最大回撤", "累计收益", "累计成本",
                      "平均目标股票权重", "股票权重最小值", "股票权重最大值"],
        num_cols=["夏普比率", "累计换手", "再平衡次数", "股票权重标准差",
                  "估计窗口股债相关(均值)", "估计窗口股债相关(标准差)",
                  "相关性估计次数", "实际年数Y"]),
        width="stretch")
    st.caption(
        "「估计窗口股债相关」按每个调仓决策点**严格向前取 window 条历史收益**复算，"
        "与回测的估计窗口一致、无前视；恒定收益导致相关无定义时记为 NaN，不伪造数值。"
        "权重标准差与累计换手反映窗口越短、权重调整越频繁，交易成本随之上升。"
    )

    curves = win["stock_weight_curves"]
    if not curves.empty:
        line_chart(curves.dropna(how="all"),
                   "不同估计窗口下的最小方差目标股票权重（共同区间）",
                   "股票权重", y_format="%", height=280)

    section("分年度研究：收益 / 波动 / 回撤 / 权重 / 股债相关性")
    corr = ann["correlation"]
    bad_years = ann["incomplete_years"]
    st.caption(
        f"共同区间 {c0.date()} ~ {c1.date()}；**不完整年度：{bad_years}**。"
        "这些年度的收益是「部分年度累计收益」，"
        "**不得当作完整年度收益，也不得直接年化**。"
        "「该年数据可得交易日」是该年在实际数据范围内的交易日数，不代表自然年全部交易日。"
    )
    st.dataframe(fmt_dataframe(
        corr,
        num_cols=["股债已实现相关(年度)", "股票年化波动率", "债券年化波动率"]),
        width="stretch")

    ap = ann["annual"]
    if not ap.empty:
        for col, title, pct in [
            ("年度累计收益", "各策略年度累计收益", True),
            ("年化波动率", "各策略年度年化波动率", True),
            ("最大回撤", "各策略年度最大回撤", True),
            ("平均实际股票权重", "各策略平均实际股票权重", True),
        ]:
            piv = ap.pivot(index="年度", columns="策略", values=col)
            st.caption(f"{title}（含不完整年度，按自然年切分）")
            st.dataframe(fmt_dataframe(
                piv, pct_cols=list(piv.columns) if pct else []), width="stretch")
    st.caption(
        "年度股债相关为该年**已实现日收益**的描述性统计（非预测值）；"
        "股票与债券年化波动率按 K=252 计算。观察相关性由负转正的年份"
        "（如本数据的 2021、2026）时最小方差权重是否随之调整，属于诊断而非因果结论。"
    )
