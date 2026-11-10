# -*- coding: utf-8 -*-
"""页面3：配置模型（统一模型接口、协方差、窗口、约束、最优权重、预测风险）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import returns as R
from core.covariance import estimate_covariance
from core.models import RESERVED_MODELS, solve_model
from core.strategies import constant_mix_weights
from pages.common import bar_weights, fmt_dataframe, risk_return_scatter, section, show_dict


def _static_performance(rets: pd.DataFrame, weights: dict, ctx: dict) -> dict:
    w = np.array([float(weights.get(a, 0.0)) for a in rets.columns])
    w = w / w.sum()
    rp = pd.Series(rets.to_numpy() @ w, index=rets.index)
    return R.performance_summary(rp, price_index=ctx["prices"].index,
                                 rf_annual=ctx["rf_annual"],
                                 periods_per_year=ctx["periods_per_year"])


def render(ctx: dict):
    st.header("3️⃣ 配置模型")
    rets = ctx["returns"]
    assets = list(rets.columns)
    K = ctx["periods_per_year"]
    st.caption(
        "所有模型通过 `core/models.py` 统一接口求解：输入收益矩阵 + 协方差 + 约束，"
        "输出权重、求解状态、预测风险与诊断。**静态结果均为样本内分析**（权重用了区间内/末尾数据）。"
    )

    P = ctx["params"]
    model_name = P["model"]
    cov_method = P["cov_method"]
    ewma_lambda = P["ewma_lambda"]
    win_label = P["window_label"]
    win_days = P["window_days"]
    constraints = ctx["constraints"]
    st.caption(
        f"当前全局参数：模型 **{model_name}**｜协方差 **{cov_method}**"
        f"（λ={ewma_lambda}）｜窗口 **{win_label} = {win_days} 交易日**｜"
        f"再平衡 **{P['rebalance_label']}**｜成本 **{P['cost_bps']}bp**。在左侧全局设置中修改。"
    )

    # 自定义权重输入（V2.2：默认即恒定比例策略的股票 30% / 债券 70%）
    custom_weights = {}
    if model_name == "自定义目标权重":
        st.subheader("恒定比例策略 · 目标权重（可手动调整）")
        st.caption("这是 Constant-Mix 基线：仅在再平衡日恢复目标比例，非调仓日由回测引擎按收益自然漂移。")
        try:
            defaults = constant_mix_weights(assets, ctx["registry"],
                                            float(P.get("cm_stock_weight", 0.30)))
        except ValueError:
            defaults = {a: 1.0 / len(assets) for a in assets}
        cc = st.columns(min(len(assets), 5))
        raw = {}
        for i, a in enumerate(assets):
            raw[a] = cc[i % len(cc)].number_input(
                ctx["registry"].get(a).name, 0.0, 1.0,
                float(round(defaults.get(a, 1.0 / len(assets)), 2)), 0.01, key=f"cw_{a}",
            )
        total = sum(raw.values())
        if abs(total - 1) > 1e-8:
            st.warning(f"权重合计 {total:.4f}，将**自动归一化**到 1 后送入模型（归一化结果见下表）。")
            custom_weights = {a: v / total if total > 0 else 1 / len(assets) for a, v in raw.items()}
        else:
            custom_weights = raw
    else:
        custom_weights = {a: 1.0 / len(assets) for a in assets}

    # 估计窗口（静态样本内）：取末尾 win_days 条收益
    if win_days >= len(rets):
        st.error(f"估计窗口 {win_days} 条 ≥ 可用收益样本 {len(rets)} 条，无法计算（**不会偷偷缩短窗口**）。")
        return
    est = rets.tail(int(win_days))
    st.caption(
        f"估计窗口（样本内）：{est.index[0].date()} ~ {est.index[-1].date()}，"
        f"{len(est)} 条日收益，信息截止日 {est.index[-1].date()}。"
    )

    try:
        cov = estimate_covariance(est, method=cov_method, ewma_lambda=float(ewma_lambda),
                                  periods_per_year=K, min_obs=60)
    except ValueError as exc:
        st.error(f"**协方差估计失败（未做任何替代）：**{exc}")
        return

    res = solve_model(model_name, est, cov, constraints,
                      weights=custom_weights, solver="auto",
                      asset_class_map={a: ctx["registry"].get(a).asset_class for a in assets})

    section("求解结果")
    with st.expander("未实现模型（本阶段不纳入研究，点击展开）", expanded=False):
        st.caption(f"预留接口、调用会明确报错：{'、'.join(RESERVED_MODELS)}。"
                   "V2.2 阶段聚焦恒定比例与最小方差两个基础策略。")
    if res.status != "success":
        st.error(
            f"**模型求解失败（{res.failure_category or '未分类'}）：{res.message}**\n\n"
            "未返回等权权重、未显示正常结果；请检查约束可行性或协方差数据。"
        )
        st.json({k: str(v) for k, v in res.diagnostics.items()})
        return
    wdf = pd.DataFrame({
        "目标权重": res.weights,
        "资产名称": [ctx["registry"].get(a).name for a in res.weights.index],
        "资产类别": [ctx["registry"].get(a).asset_class for a in res.weights.index],
        "收益口径": [ctx["registry"].get(a).return_basis for a in res.weights.index],
    })
    st.dataframe(fmt_dataframe(wdf, pct_cols=["目标权重"]), width="stretch")
    bar_weights(pd.Series(res.weights.to_numpy(),
                          index=[ctx["registry"].get(a).name for a in res.weights.index]),
                "最优/目标权重（约束：Σw=1、不做空、无杠杆）")
    show_dict({
        "模型": res.model_name, "求解状态": f"{res.status} — {res.message}",
        "求解器": res.diagnostics.get("求解器", "—"),
        "预测组合方差（年化）": f"{res.predicted_variance:.8f}",
        "预测组合波动率（年化）": f"{res.predicted_vol*100:.4f}%",
        "估计窗口": f"{res.est_start.date()} ~ {res.est_end.date()}",
        "信息截止日": str(res.as_of.date()),
        "协方差方法": cov.message,
        "约束检查": str(res.diagnostics.get("约束检查", {})),
        "与精确主动集解方差差": str(res.diagnostics.get("与精确解方差差", "—")),
    })

    # ---- V2.1：权重集中度、风险贡献、上限情景、约束状态 ----
    from core.risk import model_diagnostics, weight_cap_curve
    diag = model_diagnostics(res, cov, constraints)
    section("权重集中度 / 风险贡献 / 上限情景（仅诊断，不做风险平价优化）")
    cc1, cc2 = st.columns(2)
    with cc1:
        cm = diag["集中度"]
        show_dict({
            "最大单资产权重": f"{cm['最大单资产权重']*100:.2f}%",
            "HHI = Σw²": f"{cm['HHI']:.4f}",
            "有效资产数 1/HHI": f"{cm['有效资产数']:.2f} / {cm['资产数']}",
            "非零权重资产数": cm["非零权重资产数"],
            "口径": cm["口径"],
        })
    with cc2:
        rc = diag["风险贡献"]
        show_dict({
            "组合年化波动率 σ_p": f"{rc['sigma_p']*100:.4f}%",
            "ΣRC = σ_p 校验": rc["identity_RC"],
            "ΣVC = σ_p² 校验": rc["identity_VC"],
            "ΣRC": f"{rc['sum_RC']:.6f}",
            "ΣVC": f"{rc['sum_VC']:.8f}",
            "提示": "资金权重 ≠ 风险贡献；RC_i=w_i(Σw)_i/σ_p，VC_i=w_i(Σw)_i",
        })
    if not rc["frame"].empty:
        rc_show = rc["frame"].copy()
        rc_show.index = [ctx["registry"].get(a).name for a in rc_show.index]
        st.dataframe(fmt_dataframe(rc_show,
                                   pct_cols=["资金权重", "风险贡献占比", "方差贡献占比"],
                                   num_cols=["边际风险贡献MRC", "风险贡献RC", "方差贡献VC"],
                                   digits=5), width="stretch")
    with st.expander("权重上限情景（50% / 70% / 90% / 100%）"):
        cap_curve = weight_cap_curve(est, cov, constraints, caps=(0.5, 0.7, 0.9, 1.0))
        st.dataframe(fmt_dataframe(cap_curve,
                                   pct_cols=["单资产上限", "预测年化波动率", "最大单资产权重"],
                                   num_cols=["最小方差目标", "有效资产数"], digits=5),
                     width="stretch")
        st.caption("资产数不足导致某上限不可行时明确显示「不可行」，不强行输出结果；上限越宽松最优目标不劣于更严格上限。")
    with st.expander("统一约束状态与完整诊断信息"):
        show_dict({
            "约束来源": "app.py 统一 Constraints 对象（页面3/4/5/6 与回测/敏感性/导出共用）",
            "单资产下限": constraints.min_weight, "单资产上限": constraints.max_weight,
            "类别约束": str(constraints.class_caps),
            "资产类别映射": str(constraints.asset_class_map),
            "协方差条件数": f"{diag['协方差条件数']:.2e}",
            "协方差最小特征值": f"{diag['协方差最小特征值']:.3e}",
            "PSD 投影": str(res.diagnostics.get("PSD投影", "")),
        })
        st.json({k: str(v) for k, v in res.diagnostics.items()})
        st.json({k: str(v) for k, v in cov.to_dict().items()})
    ctx["shared"]["model_diagnostics"] = diag

    # 保存共享目标权重（页面2/4/6 使用）
    ctx["shared"]["target_weights"] = res.weights.to_dict()
    ctx["shared"]["custom_weights"] = custom_weights
    ctx["shared"]["cost_bps"] = P["cost_bps"]
    ctx["shared"]["model_name"] = model_name
    ctx["shared"]["cov_method"] = cov_method
    ctx["shared"]["ewma_lambda"] = float(ewma_lambda)
    ctx["shared"]["window_days"] = int(win_days)
    ctx["shared"]["window_label"] = win_label
    ctx["shared"]["constraints"] = constraints
    ctx["shared"]["last_cov_result"] = cov
    ctx["shared"]["last_model_result"] = res

    # ---------- 三组合静态比较（同区间、同假设） ----------
    section("三组合比较（静态·样本内）",
            "资产集合、收益口径、统计区间、每日固定权重再平衡、交易成本=0 均一致；"
            "权重来自样本内数据，结果**不是样本外验证**（样本外见页面4/5）。")
    eq_w = {a: 1.0 / len(assets) for a in assets}
    cmp_w = {"等权（1/N）": eq_w,
             "自定义目标权重": custom_weights,
             "最小方差": res.weights.to_dict()}
    perfs, scatter = {}, []
    for name, w in cmp_w.items():
        try:
            p = _static_performance(rets, w, ctx)
            perfs[name] = p
            scatter.append({"strategy": name, "vol": p["年化波动率"],
                            "ret": p["实际CAGR(按实际年数)"], "sharpe": p["夏普比率"]})
        except ValueError as exc:
            st.warning(f"{name}: {exc}")
    pdf = pd.DataFrame(perfs).T
    pcols = ["累计收益", "实际CAGR(按实际年数)", "年化波动率", "最大回撤", "夏普比率", "期末净值"]
    st.dataframe(fmt_dataframe(pdf[[c for c in pcols if c in pdf.columns]],
                               pct_cols=[c for c in pcols if c in pdf.columns and c not in ("夏普比率", "期末净值")],
                               num_cols=["夏普比率", "期末净值"]), width="stretch")
    risk_return_scatter(pd.DataFrame(scatter), "风险—收益散点（静态样本内，同区间）")

    # 协方差矩阵
    section("估计得到的协方差矩阵")
    cm = cov.matrix.copy()
    nm = [ctx["registry"].get(a).name for a in cm.columns]
    cm.index, cm.columns = nm, nm
    st.dataframe(cm.map(lambda x: f"{x:.6f}"), width="stretch")
    st.caption(
        f"单位：年化方差/协方差（×{K}）；{cov.message}；"
        f"n={cov.n_obs}，窗口 {cov.window_start.date()} ~ {cov.window_end.date()}。"
        + ("EWMA 的有效样本量见诊断；延长窗口未必改变估计。" if cov_method == "ewma" else "")
        + ("⚠️ μI 收缩目标会把异方差资产的方差向均值拉近，不保证优于样本协方差。" if cov_method == "ledoit_wolf" else "")
    )
