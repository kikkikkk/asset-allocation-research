# -*- coding: utf-8 -*-
"""页面6：研究结果汇总与 Excel 导出（静态 + 回测 + 敏感性 + V2.3 诊断）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import returns as R
from core.covariance import estimate_covariance
from core.diagnosis import common_window, performance_on_window
from core.models import IMPLEMENTED_MODELS, solve_model
from pages.common import (ensure_v23_diagnosis, excel_download, fmt_dataframe,
                          risk_return_scatter, section, show_dict)


def _static(ctx: dict) -> tuple[dict, dict, object]:
    rets = ctx["returns"]
    P = ctx["params"]
    win = min(P["window_days"], len(rets) - 2)
    est = rets.tail(win)
    cov = estimate_covariance(est, method=P["cov_method"], ewma_lambda=P["ewma_lambda"],
                              periods_per_year=ctx["periods_per_year"], min_obs=60)
    eq = {a: 1.0 / len(rets.columns) for a in rets.columns}
    custom = ctx["shared"].get("custom_weights") or eq
    weights = {"等权（1/N）": eq, "自定义目标权重": custom}
    mv = solve_model("最小方差", est, cov, ctx["constraints"], weights=custom)
    weights["最小方差"] = mv.weights.to_dict() if mv.status == "success" else eq
    perf, wrows = {}, []
    for name, w in weights.items():
        arr = np.array([float(w.get(a, 0.0)) for a in rets.columns])
        arr = arr / arr.sum()
        rp = pd.Series(rets.to_numpy() @ arr, index=rets.index)
        perf[name] = R.performance_summary(rp, price_index=ctx["prices"].index,
                                           rf_annual=ctx["rf_annual"],
                                           periods_per_year=ctx["periods_per_year"])
        wrows.append({"模型": name, **{a: w.get(a, 0.0) for a in rets.columns}, "权重和": float(arr.sum())})
    return perf, pd.DataFrame(wrows).set_index("模型"), cov


def render(ctx: dict):
    st.header("6️⃣ 研究结果")
    rets = ctx["returns"]
    P = ctx["params"]
    st.caption(
        "本页汇总静态样本内分析、滚动样本外回测与参数敏感性，并统一导出 Excel。"
        "所有指标由 core 模块计算，与页面2/3/4/5 完全一致。"
    )

    static_perf, wdf, cov = _static(ctx)

    def ensure_results(force: bool = False):
        """页面6 可被独立打开：用同一套引擎补算回测与敏感性，保证页面与导出完整一致。"""
        if force:
            ctx["shared"].pop("backtest_results", None)
            ctx["shared"].pop("sensitivity_result", None)
            ctx["shared"].pop("v22_baseline", None)
            ctx["shared"].pop("v23_diagnosis", None)
            ctx["shared"].pop("v23_cache", None)
        if not ctx["shared"].get("backtest_results"):
            from core.backtest import BacktestConfig, run_backtest
            with st.spinner("正在用统一回测引擎补算三模型结果…"):
                out = {}
                for m in IMPLEMENTED_MODELS:
                    cfg = BacktestConfig(
                        model=m, cov_method=P["cov_method"], ewma_lambda=P["ewma_lambda"],
                        window=P["window_days"], window_label=P["window_label"],
                        rebalance_every=P["rebalance_every"], rebalance_label=P["rebalance_label"],
                        cost_bps=P["cost_bps"], rf_annual=ctx["rf_annual"],
                        periods_per_year=ctx["periods_per_year"],
                        constraints=ctx["constraints"],
                        custom_weights=ctx["shared"].get("custom_weights"), min_obs=60)
                    out[m] = run_backtest(rets, cfg)
                ctx["shared"]["backtest_results"] = out
                ctx["shared"]["primary_backtest"] = P["model"]
        if ctx["shared"].get("sensitivity_result") is None:
            from core.sensitivity import SensitivityConfig, run_sensitivity
            from core.backtest import WINDOW_PRESETS, REBALANCE_PRESETS
            with st.spinner("正在补算默认参数敏感性网格…"):
                sc = SensitivityConfig(
                    windows=list(WINDOW_PRESETS.items())[:2],
                    cov_methods=["sample", "ewma"],
                    rebalance_labels=list(REBALANCE_PRESETS)[1:3],
                    model=P["model"], ewma_lambda=P["ewma_lambda"],
                    cost_bps=P["cost_bps"], rf_annual=ctx["rf_annual"],
                    constraints=ctx["constraints"],
                    custom_weights=ctx["shared"].get("custom_weights"), max_combinations=12)
                ctx["shared"]["sensitivity_result"] = run_sensitivity(rets, sc)
        if ctx["shared"].get("v22_baseline") is None:
            # V2.2：页面6 可被独立打开，用同一套策略/引擎补算双策略基线，保证导出完整
            from core.backtest import BacktestConfig
            from core.strategies import comparison_frame, run_baseline_comparison
            try:
                base_cfg = BacktestConfig(
                    model="最小方差", cov_method=P["cov_method"], ewma_lambda=P["ewma_lambda"],
                    window=P["window_days"], window_label=P["window_label"],
                    rebalance_every=P["rebalance_every"], rebalance_label=P["rebalance_label"],
                    cost_bps=P["cost_bps"], rf_annual=ctx["rf_annual"],
                    periods_per_year=ctx["periods_per_year"],
                    constraints=ctx["constraints"], min_obs=60)
                with st.spinner("正在补算 V2.2 双策略基线…"):
                    bl = run_baseline_comparison(
                        rets, base_cfg, ctx["asset_ids"], ctx["registry"],
                        float(P.get("cm_stock_weight", 0.30)))
                    fr, c0, c1 = comparison_frame(
                        bl, ctx["rf_annual"], ctx["periods_per_year"])
                ctx["shared"]["v22_baseline"] = {
                    "results": bl, "frame": fr, "start": c0, "end": c1, "config": base_cfg}
            except Exception as exc:  # noqa: BLE001 —— 补算失败不阻塞原有导出，明确提示
                st.warning(f"双策略基线补算失败（不影响其他工作表）：{type(exc).__name__}: {exc}")

    ensure_results()
    bt_map = ctx["shared"].get("backtest_results", {})
    primary = ctx["shared"].get("primary_backtest") or P["model"]
    sens = ctx["shared"].get("sensitivity_result")

    # ---------- 摘要 ----------
    section("研究概览")
    show_dict({
        "资产数": len(ctx["asset_ids"]),
        "资产": "、".join(ctx["registry"].get(a).name for a in ctx["asset_ids"]),
        "数据区间": f"{ctx['prices'].index[0].date()} ~ {ctx['prices'].index[-1].date()}",
        "共同交易日": ctx["align_report"].get("n_common"),
        "收益样本": len(rets),
        "默认模型": P["model"],
        "协方差方法": P["cov_method"],
        "估计窗口": f"{P['window_label']}（{P['window_days']} 交易日）",
        "再平衡": P["rebalance_label"],
        "交易成本": f"{P['cost_bps']} bp",
        "无风险利率": f"{ctx['rf_annual']:.3%}" + ("（零基准假设）" if ctx["rf_annual"] == 0 else ""),
        "滚动回测结果": "已生成" if bt_map else "尚未在页面4生成",
        "敏感性结果": "已生成" if sens is not None else "尚未在页面5生成",
    })

    # ---------- 模型比较 ----------
    section("模型比较（静态·样本内）", "同一资产集合、收益口径、统计区间、每日固定权重再平衡、交易成本=0。")
    sp = pd.DataFrame(static_perf).T
    pcols = ["累计收益", "实际CAGR(按实际年数)", "年化波动率", "夏普比率", "最大回撤", "几何平均收益(每期)"]
    st.dataframe(fmt_dataframe(sp[[c for c in pcols if c in sp.columns]],
                               pct_cols=[c for c in pcols if c in sp.columns and c != "夏普比率"],
                               num_cols=["夏普比率"]), width="stretch")
    st.dataframe(fmt_dataframe(wdf, pct_cols=list(wdf.columns)), width="stretch")
    scatter = [{"strategy": m, "vol": p["年化波动率"],
                "ret": p["实际CAGR(按实际年数)"], "sharpe": p["夏普比率"]}
               for m, p in static_perf.items()]
    risk_return_scatter(pd.DataFrame(scatter), "风险—收益散点（静态样本内）")

    section("模型比较（滚动样本外回测）")
    if bt_map:
        rows, sc2 = {}, []
        ok = {m: b for m, b in bt_map.items() if b.ok}
        if ok:
            # V2.3 统一口径：排除首次建仓现金日，CAGR 年数以净值=1 的前一交易日为起点
            c0, c1, nav_start = common_window(ok, rets.index)
            for m, b in ok.items():
                rr = b.daily["r_net"].loc[c0:c1]
                p = performance_on_window(rr, nav_start, ctx["rf_annual"],
                                          ctx["periods_per_year"])
                rows[m] = {"样本外CAGR": p["实际CAGR(按实际年数)"], "年化波动率": p["年化波动率"],
                           "夏普比率": p["夏普比率"], "最大回撤": p["最大回撤"],
                           "调仓次数": int(b.daily["is_rebalance"].loc[c0:c1].sum()),
                           "累计换手": float(b.daily["turnover"].loc[c0:c1].sum()),
                           "累计成本": float(b.daily["cost"].loc[c0:c1].sum())}
                sc2.append({"strategy": m, "vol": p["年化波动率"],
                            "ret": p["实际CAGR(按实际年数)"], "sharpe": p["夏普比率"]})
            oos_df = pd.DataFrame(rows).T
            st.dataframe(fmt_dataframe(oos_df,
                                       pct_cols=["样本外CAGR", "年化波动率", "最大回撤"],
                                       num_cols=["夏普比率", "调仓次数", "累计换手", "累计成本"]),
                         width="stretch")
            risk_return_scatter(pd.DataFrame(sc2), "风险—收益散点（共同样本外区间）")
            st.caption(f"共同样本外区间 {c0.date()} ~ {c1.date()}（已排除首次建仓现金日）；"
                       f"CAGR 实际年数以净值起点 {nav_start.date()} 计算，"
                       "换手/成本均按该共同区间求和，与页面4/5 口径一致。")
    else:
        st.info("尚未生成滚动回测结果。请访问「4️⃣ 回测分析」页，或点击下方按钮补算。")

    section("参数敏感性（探索性分析）")
    if sens is not None and sens.ok:
        st.dataframe(fmt_dataframe(
            sens.results[["窗口", "协方差方法", "再平衡", "共同区间CAGR", "共同区间夏普",
                          "共同区间最大回撤", "状态"]],
            pct_cols=["共同区间CAGR", "共同区间最大回撤"], num_cols=["共同区间夏普"]),
            width="stretch")
        st.caption(f"共同评价区间 {sens.common_start.date()} ~ {sens.common_end.date()}，"
                   f"{sens.n_success}/{sens.n_total} 组合成功。")
    else:
        st.info("尚未生成敏感性结果。请访问「5️⃣ 参数敏感性」页，或点击下方按钮补算。")

    # ---------- V2.3 金融诊断结论（不只列数字） ----------
    section("V2.3 策略有效性诊断结论",
            "四策略统一回测 + 估计窗口 / 分年度诊断，基于真实样本外结果")
    diag23, err23 = ensure_v23_diagnosis(ctx)
    if diag23 is None:
        st.warning(f"V2.3 诊断未完成（未用模拟数据补齐）：{err23}")
    else:
        f23 = diag23["frame"]
        st.dataframe(fmt_dataframe(
            f23, pct_cols=["样本外CAGR", "年化波动率", "最大回撤", "累计收益",
                           "累计成本", "平均实际股票权重"],
            num_cols=["夏普比率", "累计换手", "期末净值", "实际年数Y"]),
            width="stretch")
        st.caption(
            f"共同区间 {diag23['common_start'].date()} ~ {diag23['common_end'].date()}；"
            f"rf={diag23['config'].rf_annual:.4f}"
            + ("（**零基准假设**，夏普非真实超额收益）"
               if diag23["config"].rf_annual == 0 else "") +
            f"；成本 {diag23['config'].cost_bps}bp。四策略共用同一引擎与绩效函数。"
        )
        for line in diag23["interpretation"]:
            st.markdown("• " + line)
        st.caption(
            "详细四策略对比与净值/权重曲线见页面4；窗口敏感性与分年度"
            "（含不完整年度标注）见页面5；完整数值随 Excel 导出。"
        )

    # ---------- 手动重算（配置变化时 app.py 已自动清空缓存） ----------
    if st.button("⚙️ 重新计算回测与敏感性结果"):
        ensure_results(force=True)
        st.rerun()

    # ---------- 假设与来源 ----------
    section("最小方差模型诊断 / 主要发现 / 局限性")
    mdiag = ctx["shared"].get("model_diagnostics")
    if isinstance(mdiag, dict) and mdiag.get("状态") == "success":
        show_dict({
            "求解器": mdiag["求解器"],
            "预测年化波动率": f"{mdiag['预测年化波动率']*100:.4f}%",
            "最大单资产权重": f"{mdiag['集中度']['最大单资产权重']*100:.2f}%",
            "有效资产数(1/HHI)": f"{mdiag['集中度']['有效资产数']:.2f} / {mdiag['集中度']['资产数']}",
            "ΣRC=σ_p 校验": mdiag["风险贡献"]["identity_RC"],
            "ΣVC=σ_p² 校验": mdiag["风险贡献"]["identity_VC"],
            "估计窗口（样本内）": mdiag["估计窗口"],
        })
    else:
        st.caption("最小方差模型诊断：请先访问「3️⃣ 配置模型」页生成结果。")
    findings = [
        f"本研究使用 {len(ctx['asset_ids'])} 个资产、{ctx['align_report'].get('n_common')} 个共同交易日"
        f"（{ctx['prices'].index[0].date()} ~ {ctx['prices'].index[-1].date()}）。",
        "静态三组合比较为**样本内**结果（每日固定权重再平衡、零成本）；"
        "滚动回测为**样本外**结果（含首次现金建仓与交易成本），两者执行假设不同，不可直接排名。",
        "最小方差在本样本中权重可能高度集中于低波动资产，导致零基准夏普偏高；"
        "这是协方差结构与约束的结果，**不构成投资建议**。",
    ]
    if sens is not None and sens.ok:
        findings.append(
            f"参数敏感性 {sens.n_success}/{sens.n_total} 组在共同区间 "
            f"{sens.common_start.date()} ~ {sens.common_end.date()} 比较；"
            "结果属探索性分析，不是独立样本外验证。"
        )
    for f in findings:
        st.caption("• " + f)
    st.warning(
        "**局限性**：只有日收盘价（成交价近似）、零收益现金假设、线性换手成本；"
        "rf=0 为零基准假设；三种协方差方法无默认最优（LW 为 μI 目标简化基准）；"
        "仅完整验证日频；参数敏感性未做嵌套滚动验证；真实浏览器交互未自动化验收。"
        "详见下方「计算假设」与 Excel 的「研究局限性与假设」工作表。"
    )

    section("计算假设、数据来源与口径")
    assumptions = {
        "简单收益率": "r_t = P_t / P_{t−1} − 1",
        "对数收益率": "l_t = ln(1+r_t)，要求 r_t > −1",
        "算术平均": "μ = mean(r_t)；年化 μ×K",
        "平均对数收益": "g = mean(ln(1+r))；几何平均 G = exp(g) − 1（每期）",
        "年化几何": "exp(K·g) − 1 = (1+G)^K − 1（按频率，**用 expm1**）；与按实际年数 CAGR 不同口径",
        "二阶近似A": "g_A = μ − 0.5σ²（σ² 为 ddof=1 样本方差，金融惯用近似）",
        "二阶近似B": "g_B = μ − 0.5mean(r²)（更贴近二阶 Taylor；两者均为近似、非恒等式）",
        "年化复利误差": "expm1(K·g_approx) − expm1(K·g_exact)，收益率之差；bp = 差 × 10000",
        "实际CAGR": "(NAV_end/NAV_start)^(1/Y) − 1，Y = 实际天数/365.25",
        "年化波动率": "σ_period(ddof=1) × √K",
        "夏普": "mean(r−rf_period)/std(r−rf_period)×√K；rf_period=(1+rf_annual)^(1/K)−1；rf=0 为零基准假设",
        "最大回撤": "min(NAV_t / cummax(NAV_t) − 1)",
        "最小方差": "min wᵀΣw，Σw=1，w≥0（SLSQP + 主动集精确解互验；失败明确报错不返回等权）",
        "风险贡献": "MRC_i=(Σw)_i/σ_p，RC_i=w_i·MRC_i（ΣRC=σ_p），VC_i=w_i(Σw)_i（ΣVC=σ_p²）",
        "DR": "0.5×(Σwᵢσᵢ² − wᵀΣw)，年化对数空间二阶近似；≠ 实际再平衡超额收益",
        "再平衡假设（静态主表）": "每日收盘后按目标权重再平衡、成本 0、全样本内区间",
        "回测执行": "t 日收盘后信号 → t+1 日按收盘价近似成交 → t+2 日起新权重产生收益；首日现金建仓并计初始成本",
        "现金假设": "零收益现金假设（现金收益率 0，不代表真实无风险利率）",
        "权重漂移": "w_i,t = w_i,t−1(1+r_i,t)/(1+r_p,t)（持有区间无交易无现金流）",
        "成本主口径": "turnover=Σ|目标权重−交易前权重|（双边绝对权重变化）；cost=turnover×bp/10000，买卖同费率；"
                      "buy_amount/sell_amount 为参考口径（建仓 buy=1、sell=0；再平衡完全投资时 buy=sell）",
        "自融资假设": "成本按比例从净值扣除（每执行时点只扣一次），成交后权重仍精确等于目标权重"
                      "（成本视为从组合外部现金支付，不消耗本金）；真实成本会降低成交后权重",
        "失败调仓处理": "失败时点继续持有漂移权重并记录于调仓日志（status=failed），存在失败的回测不视为完全成功",
        "实现风险口径": "固定目标权重（协方差预测误差主评价）/漂移毛收益/净收益三种，逐层区分漂移、再平衡与成本影响",
        "评价区间": "[新权重生效首日, 下一成功执行日]；该执行日收益归上一持有期，相邻区间首尾相接不重不漏；"
                    "失败调仓时持有期延伸至下一成功执行日，不留无人评价断档",
        "预测误差稳健性": "MAE/RMSE 基于固定目标权重口径（全有效样本，保留不变）；另列排除 3≤观测<21 短样本后的"
                          "「MAE/RMSE(排除短样本,稳健性)」供对照；无剩余事件记 NaN，不以 0 代替",
        "预测误差可比性": "评价期限由再平衡频率决定（月度≈21、季度≈63、年度≈252 交易日），"
                          "不同频率的 MAE/RMSE 不可直接比较；请在相同频率、相近期限内比较",
        "共同样本外区间": "起点取各策略最晚首次建仓日的**下一个交易日**，排除现金/建仓日收益，"
                          "确保区间内所有策略均已持仓、口径一致；终点取最早 OOS 终点",
        "参数敏感性状态": "部分成功（存在失败调仓）的组合单独标记，失败次数与原因随结果导出；"
                          "参数搜索后的最优结果不得称为独立验证结果",
        "统一日历": "所有资产共同交易日交集，不填充、不外推",
        "年化频率K": ctx["periods_per_year"],
        "无风险利率": f"{ctx['rf_annual']:.4f}" + ("（零基准）" if ctx["rf_annual"] == 0 else ""),
        "交易成本": f"{P['cost_bps']} bp / 单位双边绝对权重变化",
        "数据频率限制": "当前仅完整验证**日频**；周频/月频自动转换未实现，请勿据此解释非日频结果",
        **{k: str(v) for k, v in P.items()},
    }
    show_dict(assumptions)
    for w in ctx["warnings"]:
        st.warning(w)

    # ---------- Excel ----------
    # ---------- Excel：与页面计算完全一致的工作表 ----------
    section("Excel 导出", "仅导出衍生计算结果与元数据；不导出原始行情，不公开用户上传数据。")
    from core.risk import concentration_metrics, risk_contributions
    mv_w = wdf.loc["最小方差", ctx["asset_ids"]].astype(float) if "最小方差" in wdf.index else None
    rc = risk_contributions(mv_w, cov.matrix) if mv_w is not None else {"frame": pd.DataFrame()}
    cm = concentration_metrics(mv_w) if mv_w is not None else {}
    asset_perf = {}
    for a in ctx["asset_ids"]:
        try:
            asset_perf[ctx["registry"].get(a).name] = R.performance_summary(
                rets[a], price_index=ctx["prices"].index, rf_annual=ctx["rf_annual"],
                periods_per_year=ctx["periods_per_year"])
        except ValueError:
            continue
    limitations = pd.DataFrame({"项目": [
        "样本内 vs 样本外", "参数选择", "执行价格", "现金假设", "交易成本模型",
        "协方差方法", "数据频率", "收益口径", "数据授权", "浏览器验证", "未实现模型",
    ], "说明": [
        "静态主表为样本内（每日再平衡、零成本、全区间），滚动回测为样本外；两者执行假设不同，不可直接排名",
        "参数敏感性是探索性分析，未实现嵌套滚动验证/独立保留集；挑参后的同一段数据不再是独立验证",
        "只有日收盘数据，以执行日收盘价近似成交，无开盘价/滑点/冲击成本",
        "零收益现金假设，现金收益率 0，不代表真实无风险利率",
        "线性换手成本，买卖同费率，未含税费与最小交易单位",
        "三种方法并列无默认最优；ledoit_wolf 为 μI 目标简化 δ 的可复现基准，异方差下会拉平方差",
        "当前仅完整验证日频；周频/月频自动转换未实现",
        "混用含息/不含息口径会系统性扭曲结果，页面已给出口径警告",
        "内置公开数据仅运行时获取，不得作为原始行情再分发；用户上传数据授权由用户自审",
        "真实浏览器点击/滚动/上传/下载后打开 Excel 未做自动化验证（仅 AppTest 渲染与文件可写性）",
        "风险平价/最大夏普/最大分散度仅预留接口，未实现，不可用于研究结论",
    ]})
    sheets: dict[str, pd.DataFrame] = {
        "研究参数": pd.DataFrame({"项目": list(P), "内容": [str(v) for v in P.values()]}),
        "资产元数据": ctx["registry"].info_frame(ctx["asset_ids"]),
        "数据来源": ctx["fetch_logs"],
        "收益口径": pd.DataFrame({
            "资产": [ctx["registry"].get(a).name for a in ctx["asset_ids"]],
            "代码": [ctx["registry"].get(a).code for a in ctx["asset_ids"]],
            "收益口径": [ctx["registry"].get(a).return_basis for a in ctx["asset_ids"]],
            "分红处理": [ctx["registry"].get(a).dividend_treatment for a in ctx["asset_ids"]],
            "再分发许可": [ctx["registry"].get(a).redistribution_permission for a in ctx["asset_ids"]],
        }),
        "数据质量": ctx["quality"],
        "对齐结果": pd.DataFrame({"项目": list(ctx["align_report"]),
                                  "内容": [str(v) for v in ctx["align_report"].values()]}),
        "收益风险指标": pd.DataFrame(asset_perf).T if asset_perf else pd.DataFrame(),
        "日度收益序列": rets,
        "协方差矩阵": cov.matrix,
        "模型目标权重": wdf,
        "风险贡献": rc["frame"],
        "权重集中度": pd.DataFrame({"指标": list(cm), "值": [str(v) for v in cm.values()]}) if cm else pd.DataFrame(),
        "约束状态": pd.DataFrame({"项目": ["完全投资", "不做空", "无杠杆", "单资产下限", "单资产上限",
                                           "类别约束", "资产类别映射"],
                                  "内容": ["Σw=1", True, True, ctx["constraints"].min_weight,
                                           ctx["constraints"].max_weight,
                                           str(ctx["constraints"].class_caps),
                                           str(ctx["constraints"].asset_class_map)]}),
        "组合对比统计": sp,
        "计算假设": pd.DataFrame({"项目": list(assumptions), "内容": [str(v) for v in assumptions.values()]}),
        "研究局限性与假设": limitations,
    }
    # ensure_results() 已在页面开头用同一套引擎补算，导出直接取共享结果
    bt_map_all = ctx["shared"].get("backtest_results") or {}
    primary = ctx["shared"].get("primary_backtest") or P["model"]
    bt = bt_map_all.get(primary)
    if bt is not None and bt.ok:
        sheets["每日组合收益"] = bt.daily[["r_port", "r_net", "drawdown"]]
        sheets["每日净值"] = bt.daily[["nav", "drawdown"]]
        sheets["回测每期收益与净值"] = bt.daily[["r_port", "r_net", "nav", "drawdown"]]
        sheets["调仓日志"] = bt.rebalance_log
        sheets["目标与实际权重"] = pd.concat(
            [bt.weights_target().add_prefix("目标_"), bt.weights_actual().add_prefix("实际_")], axis=1)
        sheets["交易成本"] = bt.daily[["turnover", "one_side_turnover", "traded_notional",
                                       "buy_amount", "sell_amount",
                                       "cost", "cumulative_cost", "cumulative_turnover",
                                       "is_initial_build", "is_rebalance", "is_trade"]]
        sheets["回测完整绩效"] = pd.Series(bt.performance, name="值").to_frame()
        sheets["预测风险误差"] = bt.forecast_eval
        sheets["预测风险统计"] = pd.DataFrame({"指标": list(bt.forecast_summary),
                                              "值": [str(v) for v in bt.forecast_summary.values()]})
        sheets["模型比较结果"] = pd.DataFrame({
            m: b.performance for m, b in ctx["shared"]["backtest_results"].items() if b.ok
        })
    # V2.2：双策略基线统一回测对比（与页面4 主线同一份结果）
    v22 = ctx["shared"].get("v22_baseline")
    if v22 is not None:
        sheets["双策略基线对比"] = v22["frame"].reset_index(names="策略")
        bcfg = v22["config"]
        sheets["双策略参数设置"] = pd.DataFrame({
            "项目": ["共同评价起点", "共同评价终点", "估计窗口(交易日)", "协方差方法",
                     "再平衡频率", "交易成本(bp)", "无风险利率(年化)", "年化频率K",
                     "恒定比例股票目标权重", "执行假设"],
            "值": [str(v22["start"].date()), str(v22["end"].date()), bcfg.window,
                   bcfg.cov_method, bcfg.rebalance_label, bcfg.cost_bps, bcfg.rf_annual,
                   bcfg.periods_per_year, P.get("cm_stock_weight", 0.30),
                   "t日收盘后信号→t+1日按收盘价近似成交→t+2日起新权重产生收益"],
        })
        for label, bt in v22["results"].items():
            if bt.ok:
                safe = "恒定比例" if "恒定" in label else "最小方差"
                sheets[f"{safe}每日净值"] = bt.daily[["nav", "drawdown"]].loc[v22["start"]:v22["end"]]
                sheets[f"{safe}调仓日志"] = bt.rebalance_log
    if sens is not None and sens.ok:
        sheets["参数敏感性"] = sens.results.drop(
            columns=[c for c in ["平均目标权重"] if c in sens.results.columns], errors="ignore")
        sheets["共同区间日收益"] = sens.common_returns
        if not sens.forecast_compare.empty:
            sheets["敏感性预测风险误差"] = sens.forecast_compare

    # ---------- V2.3：四策略 / 窗口 / 分年度 / 金融诊断（与页面4/5 同一份结果） ----------
    d23, e23 = ensure_v23_diagnosis(ctx)
    if d23 is None:
        st.warning(f"V2.3 诊断结果未能生成，Excel 中不含 V2.3 工作表（未用模拟数据补齐）：{e23}")
    else:
        s0, s1 = d23["common_start"], d23["common_end"]
        cfg23 = d23["config"]
        sheets["V23四策略对比"] = d23["frame"].reset_index(names="策略")
        sheets["V23参数设置"] = pd.DataFrame({
            "项目": ["共同评价起点", "共同评价终点", "净值起点(CAGR年数起点)",
                     "共同交易日", "估计窗口(交易日)", "协方差方法", "再平衡频率",
                     "交易成本(bp)", "无风险利率(年化)", "年化频率K",
                     "恒定比例股票目标权重", "纯债券基准", "执行假设",
                     "CAGR口径", "累计换手/成本口径", "夏普口径"],
            "值": [str(s0.date()), str(s1.date()), str(d23["nav_start"].date()),
                   len(ctx["returns"].loc[s0:s1]), cfg23.window, cfg23.cov_method,
                   cfg23.rebalance_label, cfg23.cost_bps, cfg23.rf_annual,
                   cfg23.periods_per_year, P.get("cm_stock_weight", 0.30),
                   "rebalance_every=0：仅首次建仓，不做周期性再平衡",
                   "t日收盘后信号→t+1日按收盘价近似成交→t+2日起新权重产生收益",
                   "(NAV_end/1)^(1/Y)−1，Y=实际天数/365.25，起点为净值=1 的前一交易日",
                   "按共同区间求和，不含区间外首次建仓的 1.0 换手（四策略一致）",
                   "rf=0 为零基准假设，非真实无风险利率超额收益"
                   if cfg23.rf_annual == 0 else "使用统一侧边栏设定的 rf"],
        })
        sheets["V23四策略每日净值"] = pd.DataFrame(
            {k: v.daily["nav"].loc[s0:s1] for k, v in d23["four"].items()})
        wcols = [a for a in ctx["asset_ids"]
                 if ctx["registry"].get(a).asset_class == "股票"]
        if wcols:
            sheets["V23四策略股票权重"] = pd.DataFrame({
                k: v.weights_actual()[wcols].sum(axis=1).loc[s0:s1]
                for k, v in d23["four"].items()})
        w23, a23 = d23["window"], d23["annual"]
        sheets["V23窗口敏感性"] = w23["frame"].reset_index(names="估计窗口")
        sheets["V23窗口股票权重"] = w23["stock_weight_curves"]
        sheets["V23分年度表现"] = a23["annual"]
        sheets["V23年度股债相关"] = a23["correlation"]
        sheets["V23金融诊断结论"] = pd.DataFrame({
            "序号": range(1, len(d23["interpretation"]) + 1),
            "诊断结论": d23["interpretation"],
        })
    excel_download(sheets, "⬇️ 下载完整研究结果 Excel（多工作表）")
    st.caption(f"当前导出 **{sum(1 for f in sheets.values() if f is not None and len(f) > 0)}** 个工作表，"
               "数据均来自本页面同一套计算上下文；未生成任何未经数据支持的投资建议。")
