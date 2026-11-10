# -*- coding: utf-8 -*-
"""页面4：回测分析（V2.3 四策略诊断主线 + V2.2 双策略基线 + 三模型详细诊断）。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.backtest import BacktestConfig, run_backtest
from core.diagnosis import BH_LABEL, CM10_LABEL, build_diagnosis
from core.models import IMPLEMENTED_MODELS
from core.strategies import (CM_LABEL, MV_LABEL, baseline_configs, comparison_frame,
                             constant_mix_weights, interpretation, run_baseline_comparison)
from pages.common import fmt_dataframe, line_chart, risk_return_scatter, section, show_dict


def _cfg(ctx: dict, model: str) -> BacktestConfig:
    P = ctx["params"]
    cw = ctx["shared"].get("custom_weights")
    if model == "自定义目标权重" and not cw:
        # V2.2：恒定比例策略默认 30/70，避免未访问页面3 时整条回测失败
        cw = constant_mix_weights(ctx["asset_ids"], ctx["registry"],
                                  P.get("cm_stock_weight", 0.30))
    return BacktestConfig(
        model=model, cov_method=P["cov_method"], ewma_lambda=P["ewma_lambda"],
        window=int(P["window_days"]), window_label=P["window_label"],
        rebalance_every=int(P["rebalance_every"]), rebalance_label=P["rebalance_label"],
        cost_bps=float(P["cost_bps"]), rf_annual=ctx["rf_annual"],
        periods_per_year=ctx["periods_per_year"],
        constraints=ctx["constraints"],
        custom_weights=cw,
        min_obs=60,
    )


def _cache_key(ctx: dict, cfg: BacktestConfig) -> str:
    return repr((tuple(ctx["asset_ids"]), cfg.to_dict()))


def render(ctx: dict):
    st.header("4️⃣ 回测分析")
    P = ctx["params"]
    st.caption(
        "统一回测引擎 `core/backtest.py`：收盘后生成信号 → 下一交易日按收盘价近似成交 → "
        "再下一期开始持有收益；每个调仓点**只用严格早于执行日的数据**估计权重。"
    )

    # ============ V2.3 主线：四策略（含两个简单对照基准）统一诊断 ============
    _render_v23_diagnosis(ctx)

    # ==================== V2.2：双策略统一回测对比（保留） ====================
    _render_baseline(ctx)

    cache = ctx["shared"].setdefault("bt_cache", {})
    primary = P["model"]
    rerun = st.button("🔁 重新运行回测", use_container_width=False)

    models = IMPLEMENTED_MODELS
    progress = st.progress(0.0, text="正在运行滚动回测…")
    results = {}
    for i, m in enumerate(models):
        cfg = _cfg(ctx, m)
        key = _cache_key(ctx, cfg)
        if rerun or key not in cache:
            cache[key] = run_backtest(ctx["returns"], cfg)
        results[m] = cache[key]
        progress.progress((i + 1) / len(models), text=f"已完成 {m}")
    progress.empty()

    bt = results[primary]
    if not bt.ok:
        st.error(f"**回测失败（未生成任何结果）：**{bt.error}")
        st.info(
            "常见原因：估计窗口过长导致样本外为空、共同交易日不足、或模型求解全部失败。"
            "请缩短窗口、增加资产历史区间或调整约束。页面不会用模拟数据填补。"
        )
        return

    section("样本外区间与唯一时间规则")
    show_dict({
        "模型": primary, "样本外起点": str(bt.oos_start.date()),
        "样本外终点": str(bt.oos_end.date()), "样本外交易日数": bt.n_oos,
        "首次建仓次数": bt.n_initial_build,
        "后续再平衡次数": bt.n_rebalance,
        "总交易事件次数": bt.n_trade_events,
        "计划交易事件数": bt.n_planned_trades,
        "失败交易事件数": bt.n_failed_trades,
        "  └ 首次建仓失败": bt.n_failed_build,
        "  └ 后续调仓失败": bt.n_failed_rebalance,
        "累计换手(主口径 Σ|Δw|)": f"{bt.turnover_total:.4f}",
        "累计成本（占净值）": f"{bt.cost_total*10000:.2f} bp",
        **{k: str(v) for k, v in bt.config.to_dict().items() if k not in ("模型",)},
    })
    st.info(
        "**唯一时间规则**：t 日收盘后生成信号（信息截止 = t 日）→ t+1 日按收盘价近似成交 → "
        "t+2 日起新权重开始产生收益。只有日收盘数据，不虚构次日开盘价。\n\n"
        "**首次建仓**：资金起始为现金（**零收益现金假设**，现金收益率 0，不代表真实无风险利率），"
        "建仓日不享有目标资产收益，收盘按目标权重全额买入并**计提初始交易成本**；"
        "新权重从下一期起生效。\n\n"
        "**成本主口径**：turnover = Σ|目标权重 − 交易前权重|（双边绝对权重变化），"
        "cost = turnover × 成本率(bp)/10000，买卖同费率、每个执行时点只扣一次；"
        "one_side_turnover 与 traded_notional 仅为参考口径，不参与成本计算。"
    )
    log = bt.rebalance_log
    no_lookahead = bool(
        (log["info_cutoff"] < log["execution_date"]).all()
        and (log["execution_date"] < log["holding_start"]).all()
        and (log["est_end"] <= log["info_cutoff"]).all()
    )
    st.success(
        f"**无前视检查：{'通过 ✅' if no_lookahead else '未通过 ❌'}** — "
        "每次交易事件（含首次建仓）均满足 "
        "`est_end ≤ info_cutoff(signal_date) < execution_date < holding_start`；"
        f"三类事件计数一致：{bt.n_initial_build} 建仓 + {bt.n_rebalance} 再平衡 = "
        f"{bt.n_trade_events} 总交易事件。"
    )
    # 失败交易事件：明确警示，不标记为完全成功
    failed_rows = log[log["status"] == "failed"] if "status" in log.columns else log.iloc[0:0]
    if bt.n_failed_trades > 0:
        st.warning(
            f"⚠️ **本次回测非完全成功**：计划 {bt.n_planned_trades} 次交易事件，"
            f"失败 {bt.n_failed_trades} 次（建仓 {bt.n_failed_build} / 调仓 {bt.n_failed_rebalance}）。"
            "失败时点策略继续持有漂移权重。与其他策略比较时，请先确认各策略失败次数是否相近，"
            "否则横向比较的公平性受影响。"
        )
        if not failed_rows.empty:
            fshow = failed_rows[["execution_date", "event_kind", "失败原因"]].copy()
            fshow["execution_date"] = fshow["execution_date"].map(lambda x: str(x.date()))
            st.dataframe(fshow, hide_index=True, width="stretch")
    # 预测风险 vs 实现风险（事后评价）
    fs = bt.forecast_summary
    with st.expander("预测风险 vs 实现风险（实现风险仅事后评价）"):
        # 长文本口径单独展示，数值/短标签进指标表，避免一栏塞满长段落
        long_keys = {"口径", "可比性限制"}
        show_dict({k: (f"{v:.6f}" if isinstance(v, float) else str(v))
                   for k, v in fs.items() if k not in long_keys})
        st.markdown("**稳健性检验（排除短样本）**")
        st.dataframe(fmt_dataframe(pd.DataFrame({
            "指标": ["MAE", "RMSE"],
            "全样本": [fs.get("MAE"), fs.get("RMSE")],
            "排除短样本(<21观测)": [
                fs.get("MAE(排除短样本,稳健性)"),
                fs.get("RMSE(排除短样本,稳健性)")],
        }).set_index("指标"), pct_cols=["全样本", "排除短样本(<21观测)"]),
            width="stretch")
        st.caption(
            f"有效评价数：全样本 {fs.get('有效评价数')} → 排除短样本后 "
            f"{fs.get('有效评价数(排除短样本)')}（剔除 3≤观测<21 的事件 "
            f"{fs.get('排除的短样本事件数(3≤观测<21)')} 个；"
            f"另有 {fs.get('未达最小估计样本事件数(观测<3)')} 个观测<3 的事件无有效实现波动率）。"
        )
        if not bt.forecast_eval.empty:
            line_chart(bt.forecast_eval.set_index("execution_date")[
                           ["predicted_vol", "realized_vol_fixed_weight",
                            "realized_vol_net"]].dropna(),
                       "各决策点预测波动率 vs 实现波动率（固定权重 / 净收益）", "年化波动率",
                       y_format="%", height=260)
        st.caption(
            "**口径**：预测波动率来自估计期协方差与目标权重（wᵀΣw 开方）；"
            "实现波动率主评价口径 = 持有期**固定目标权重**组合收益波动率（与预测口径一致，"
            "MAE/RMSE 基于该口径）；realized_vol_gross_drift（漂移权重毛收益，含漂移/再平衡影响）"
            "与 realized_vol_net（净收益，再含成本影响）仅作策略表现参考。"
            "实现风险只在成交后用于事后评价，不参与该时点的权重估计。"
        )
        st.info("**评价区间**：起点 = 新权重生效首日、终点 = 下一成功执行日（该日收益归上一持有期），"
                "相邻区间首尾相接不重不漏；失败调仓时持有期延伸至下一成功执行日。"
                "全样本 MAE/RMSE 保留不变，另列排除 3≤观测<21 短样本后的稳健性结果；"
                "若无剩余事件则记 NaN（不伪造 0）。")
        st.warning(fs.get("可比性限制", ""))
        with st.expander("完整口径文本"):
            st.caption(fs.get("口径", ""))
    if bt.warnings:
        with st.expander("回测告警 / 假设"):
            for w in bt.warnings:
                st.caption("• " + w)

    # ---------- 图表 ----------
    section("组合净值曲线（期初 = 1）")
    nav_df = pd.DataFrame({m: r.daily["nav"] for m, r in results.items() if r.ok})
    line_chart(nav_df, "三模型滚动回测净值（共同样本外区间）", "净值（倍）", y_format=".2f", height=360)
    st.caption(
        f"区间 {nav_df.index[0].date()} ~ {nav_df.index[-1].date()}；单位：倍（无量纲）；"
        f"三模型使用相同资产、收益口径、估计窗口、调仓日历与 {P['cost_bps']}bp 成本假设。"
    )

    section("回撤曲线")
    dd_df = pd.DataFrame({m: r.daily["drawdown"] for m, r in results.items() if r.ok})
    line_chart(dd_df, "组合回撤", "回撤", y_format="%", height=280)

    c1, c2 = st.columns(2)
    with c1:
        section("滚动目标权重")
        tw = bt.weights_target()
        tw.columns = [ctx["registry"].get(a).name for a in tw.columns]
        line_chart(tw, "目标权重（调仓日更新）", "权重", height=280)
    with c2:
        section("滚动实际权重（含漂移）")
        aw = bt.weights_actual()
        aw.columns = [ctx["registry"].get(a).name for a in aw.columns]
        line_chart(aw, "实际持仓权重（非调仓日按收益漂移）", "权重", height=280)
    st.caption(
        "权重漂移：w_i,t = w_i,t−1 × (1+r_i,t)/(1+r_p,t)，仅适用于无交易、无外部现金流的持有区间；"
        "默认不每天恢复目标权重。"
    )

    # ---------- 绩效 ----------
    section("样本外绩效（三模型共同区间）")
    from core.diagnosis import common_window, performance_on_window
    perf_rows, scatter = {}, []
    ok_results = {m: r for m, r in results.items() if r.ok}
    # V2.3 统一口径：排除首次建仓现金日，以净值=1 的前一交易日为 CAGR 年数起点
    common_start, common_end, nav_start = common_window(ok_results, ctx["returns"].index)
    for m, r in ok_results.items():
        rr = r.daily["r_net"].loc[common_start:common_end]
        p = performance_on_window(
            rr, nav_start, ctx["rf_annual"], ctx["periods_per_year"])
        perf_rows[m] = {
            "样本外CAGR": p["实际CAGR(按实际年数)"],
            "年化波动率": p["年化波动率"],
            "夏普比率": p["夏普比率"],
            "最大回撤": p["最大回撤"],
            "累计收益": p["累计收益"],
            "实际年数Y": p["实际年数Y"],
            "累计换手": float(r.daily["turnover"].loc[common_start:common_end].sum()),
            "累计成本": float(r.daily["cost"].loc[common_start:common_end].sum()),
            "建仓次数": int(r.daily["is_initial_build"].loc[common_start:common_end].sum()),
            "再平衡次数": int(r.daily["is_rebalance"].loc[common_start:common_end].sum()),
            "总交易事件": int(r.daily["is_trade"].loc[common_start:common_end].sum()),
        }
        scatter.append({"strategy": m, "vol": p["年化波动率"],
                        "ret": p["实际CAGR(按实际年数)"], "sharpe": p["夏普比率"]})
    pf = pd.DataFrame(perf_rows).T
    st.dataframe(fmt_dataframe(pf, pct_cols=["样本外CAGR", "年化波动率", "最大回撤", "累计收益"],
                               num_cols=["夏普比率", "实际年数Y", "累计换手", "累计成本",
                                         "建仓次数", "再平衡次数", "总交易事件"]), width="stretch")
    risk_return_scatter(pd.DataFrame(scatter))
    st.caption(
        f"共同评价区间 {common_start.date()} ~ {common_end.date()}（已排除首次建仓现金日）；"
        f"CAGR 实际年数以净值起点 {nav_start.date()} 计算，与页面5/6 及 Excel 口径一致；"
        "全部指标由 `core/returns.py::performance_summary` 统一计算，"
        "夏普使用频率一致的期间超额收益口径"
        + ("（rf=0 为零基准假设，不是真实超额收益表现）" if ctx["rf_annual"] == 0 else "")
        + "。建仓/再平衡/总交易事件三计数分离，避免同一页面数字相差 1。"
    )

    section("当前模型完整绩效（引擎统一实现）")
    full = pd.Series(bt.performance, name="值")
    st.dataframe(full.map(lambda v: f"{v:.6f}" if isinstance(v, (int, float)) else str(v)), width="stretch")

    section("交易事件记录（时间信息完整，首次建仓单独标注）")
    log_show = log.drop(columns=[c for c in ["constraint_check"] if c in log.columns]).copy()
    date_cols = ["est_start", "est_end", "info_cutoff", "signal_date",
                 "exec_date", "execution_date", "hold_return_start", "holding_start", "hold_end"]
    for c in date_cols:
        if c in log_show.columns:
            log_show[c] = log_show[c].map(lambda x: str(x.date()) if pd.notna(x) else "")
    st.dataframe(log_show.head(50), hide_index=True, width="stretch")
    st.caption(
        f"共 {len(log)} 次交易事件（{bt.n_initial_build} 次建仓 + {bt.n_rebalance} 次再平衡）；"
        "字段含 is_initial_build / is_rebalance / is_trade / signal_date / info_cutoff / "
        "execution_date / holding_start / target_weights / pre_trade_weights / "
        "post_trade_weights / turnover / transaction_cost；完整记录见页面6 Excel 导出。"
    )
    ctx["shared"]["backtest_results"] = results
    ctx["shared"]["primary_backtest"] = primary


def _render_baseline(ctx: dict) -> None:
    """V2.2 主线：两个基础策略在同一引擎、同一区间、同一成本口径下对比。"""
    P = ctx["params"]
    st.subheader("双策略基线：统一滚动样本外回测")
    st.caption(
        "数据 → 收益风险 → 两种基础配置 → 滚动回测 → 结果比较。"
        "两策略共用同一收益矩阵、估计窗口、调仓日历、成本与绩效函数；"
        "恒定比例仅在再平衡日恢复目标权重，其余时间允许自然漂移。"
    )
    stock_w = float(P.get("cm_stock_weight", 0.30))
    try:
        base_cfg = _cfg(ctx, "最小方差")
        cfg_map = baseline_configs(base_cfg, ctx["asset_ids"], ctx["registry"], stock_w)
        w_cm = cfg_map[CM_LABEL].custom_weights
    except Exception as exc:  # noqa: BLE001 —— 明确报错，不回落模拟数据
        st.error(f"双策略参数校验失败：{type(exc).__name__}: {exc}")
        return

    show_dict({
        "恒定比例目标权重": "、".join(f"{ctx['registry'].get(a).name} {v:.0%}" for a, v in w_cm.items()),
        "最小方差估计窗口": f"{base_cfg.window} 交易日（{base_cfg.window_label}）",
        "协方差方法": base_cfg.cov_method,
        "再平衡频率": base_cfg.rebalance_label,
        "交易成本": f"{base_cfg.cost_bps} bp / 单位双边换手",
        "无风险利率": f"{base_cfg.rf_annual:.4f}" +
                     ("（零基准假设）" if base_cfg.rf_annual == 0 else ""),
        "执行假设": "t日收盘后信号 → t+1日按收盘价近似成交 → t+2日起新权重产生收益",
    })

    key = repr(("v22_baseline", tuple(ctx["asset_ids"]), stock_w,
                _cache_key(ctx, base_cfg)))
    cache = ctx["shared"].setdefault("bt_cache", {})
    if key not in cache:
        with st.spinner("正在运行两个策略的统一滚动回测…"):
            cache[key] = run_baseline_comparison(
                ctx["returns"], base_cfg, ctx["asset_ids"], ctx["registry"], stock_w)
    results: dict = cache[key]

    failed = {k: v.error for k, v in results.items() if not v.ok}
    if failed:
        st.error("以下策略回测失败（**未用模拟数据补齐**）：" +
                 "；".join(f"{k}: {v[:120]}" for k, v in failed.items()))
        return
    try:
        frame, c0, c1 = comparison_frame(results, base_cfg.rf_annual, base_cfg.periods_per_year)
    except ValueError as exc:
        st.error(f"双策略对比失败：{exc}")
        return
    ctx["shared"]["v22_baseline"] = {"results": results, "frame": frame,
                                     "start": c0, "end": c1, "config": base_cfg}

    nav = pd.DataFrame({k: v.daily["nav"].loc[c0:c1] for k, v in results.items()})
    line_chart(nav, "双策略累计净值（共同样本外区间，期初=1）", "净值（倍）",
               y_format=".2f", height=340)
    st.caption(f"共同评价区间 {c0.date()} ~ {c1.date()}（{len(nav)} 个交易日）；"
               f"两策略使用相同数据与 {base_cfg.cost_bps}bp 成本假设。")

    st.dataframe(fmt_dataframe(
        frame, pct_cols=["样本外CAGR", "年化波动率", "最大回撤", "累计收益", "累计成本"],
        num_cols=["夏普比率", "累计换手", "期末净值", "建仓次数", "再平衡次数", "总交易事件"]),
        width="stretch")

    c1c, c2c = st.columns(2)
    with c1c:
        tw = pd.DataFrame({k: v.weights_target().mean(axis=0) for k, v in results.items()})
        st.caption("两策略平均目标权重（调仓日）")
        st.dataframe(tw.round(4), width="stretch")
    with c2c:
        maxw = {k: float(v.weights_target().to_numpy().max()) for k, v in results.items()}
        st.caption("单资产最大目标权重（集中度诊断）")
        st.dataframe(pd.DataFrame({"最大权重": maxw}).round(4), width="stretch")

    weights = pd.DataFrame({k: v.weights_actual()[ctx["asset_ids"][0]].loc[c0:c1]
                            for k, v in results.items()})
    line_chart(weights, f"股票类实际持仓权重（含非调仓日漂移）：{ctx['asset_ids'][0]}",
               "权重", y_format="%", height=260)

    section("结果解释（基于样本外程序计算，非投资建议）")
    for line in interpretation(frame, base_cfg.rf_annual):
        st.markdown("• " + line)
    st.caption(
        "样本内静态分析见页面2/3；本区为滚动样本外表现，两者分开展示。"
        "程序计算正确不等于策略有效；参数优化需避免在同一段样本外数据上反复挑选（过拟合）。"
    )


# ==================== V2.3 四策略有效性诊断主线 ====================
def _render_v23_diagnosis(ctx: dict) -> None:
    """四策略统一回测：30/70、最小方差、纯债券买入持有、10/90 恒定比例。

    全部进入同一 `run_backtest` 引擎；共同区间排除首次建仓现金日；
    纯债券 B&H 通过 `rebalance_every=0` 实现，不做任何周期性再平衡交易。
    """
    from dataclasses import replace as _replace

    P = ctx["params"]
    ids = ctx["asset_ids"]
    reg = ctx["registry"]
    st.subheader("V2.3 四策略有效性诊断：统一回测区间")
    st.caption(
        "在原有恒定比例 30/70 与最小方差之外，加入两个**不依赖优化**的简单基准："
        "「100% 债券买入持有」与「股票 10% / 债券 90% 恒定比例」，"
        "用于判断最小方差在平均约 96% 债券仓位下是否具备额外价值。"
    )
    try:
        base = _replace(_cfg(ctx, "最小方差"), custom_weights=None)
        key = repr(("v23_diagnosis", tuple(ids), base.to_dict(),
                    float(P.get("cm_stock_weight", 0.30))))
        cache = ctx["shared"].setdefault("v23_cache", {})
        if key not in cache:
            with st.spinner("正在运行四策略统一回测与窗口/分年度诊断…"):
                cache[key] = build_diagnosis(ctx["returns"], base, ids, reg)
        diag = cache[key]
    except Exception as exc:  # noqa: BLE001 —— 明确报错，不回落模拟数据
        st.error(f"V2.3 诊断未能完成（**未用模拟数据补齐**）：{type(exc).__name__}: {exc}")
        return

    ctx["shared"]["v23_diagnosis"] = diag
    frame, c0, c1, nav_start = (diag["frame"], diag["common_start"],
                                diag["common_end"], diag["nav_start"])
    four = diag["four"]

    show_dict({
        "共同样本外区间": f"{c0.date()} ~ {c1.date()}（{len(ctx['returns'].loc[c0:c1])} 个交易日）",
        "净值起点（CAGR 年数起点）": str(nav_start.date()),
        "估计窗口": f"{base.window} 交易日（{base.window_label}）",
        "协方差方法": base.cov_method,
        "再平衡频率": base.rebalance_label,
        "交易成本": f"{base.cost_bps} bp / 单位双边换手",
        "无风险利率": f"{base.rf_annual:.4f}" +
                     ("（**零基准假设**，夏普非真实超额收益）" if base.rf_annual == 0 else ""),
        "纯债券基准": "rebalance_every=0：仅首次建仓，之后权重自然漂移，不产生周期调仓成本",
        "执行假设": "t日收盘后信号 → t+1日按收盘价近似成交 → t+2日起新权重产生收益",
    })

    nav = pd.DataFrame({k: v.daily["nav"].loc[c0:c1] for k, v in four.items()})
    line_chart(nav, "四策略累计净值（共同样本外区间，期初=1）", "净值（倍）",
               y_format=".2f", height=340)

    section("四策略绩效对比（同区间 / 同成本 / 同绩效函数）")
    st.dataframe(fmt_dataframe(
        frame, pct_cols=["样本外CAGR", "年化波动率", "最大回撤", "累计收益", "累计成本",
                         "平均实际股票权重", "平均目标股票权重"],
        num_cols=["夏普比率", "累计换手", "期末净值", "实际年数Y",
                  "建仓次数", "再平衡次数", "总交易事件"]),
        width="stretch")
    st.caption(
        "**口径说明**：共同区间已排除所有策略共同的首次建仓日（当日为现金、r_port=0、"
        "仅计提初始成本），因此「累计换手/累计成本」不含初始建仓的 1.0 换手，四策略完全一致；"
        f"CAGR 的实际年数以 {nav_start.date()}（净值=1）为起点计算，"
        "最大回撤的净值序列包含起点 1.0。"
    )

    wcols = [a for a in ids if reg.get(a).asset_class == "股票"]
    if wcols:
        weights = pd.DataFrame({k: v.weights_actual()[wcols].sum(axis=1).loc[c0:c1]
                                for k, v in four.items()})
        line_chart(weights, "四策略股票类实际持仓权重（含非调仓日自然漂移）",
                   "股票权重", y_format="%", height=280)

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("最小方差 vs 两个简单基准（关键差异）")
        if MV_LABEL in frame.index and BH_LABEL in frame.index and CM10_LABEL in frame.index:
            mv = frame.loc[MV_LABEL]
            st.dataframe(pd.DataFrame({
                "纯债券买入持有": [
                    mv["年化波动率"] - frame.loc[BH_LABEL, "年化波动率"],
                    mv["样本外CAGR"] - frame.loc[BH_LABEL, "样本外CAGR"],
                    mv["最大回撤"] - frame.loc[BH_LABEL, "最大回撤"]],
                "10/90 恒定比例": [
                    mv["年化波动率"] - frame.loc[CM10_LABEL, "年化波动率"],
                    mv["样本外CAGR"] - frame.loc[CM10_LABEL, "样本外CAGR"],
                    mv["最大回撤"] - frame.loc[CM10_LABEL, "最大回撤"]],
            }, index=["波动率差", "CAGR差", "最大回撤差"]).round(6), width="stretch")
        st.caption("正数表示最小方差该指标数值更高；回撤为负值，差为正表示回撤更浅。")
    with col_b:
        st.caption("权重集中度")
        st.dataframe(pd.DataFrame({
            "平均实际股票权重": frame.get("平均实际股票权重"),
            "再平衡次数": frame["再平衡次数"],
            "累计换手": frame["累计换手"],
        }).round(4), width="stretch")

    section("金融意义诊断（基于真实样本外结果，不预设最小方差更优）")
    for line in diag["interpretation"]:
        st.markdown("• " + line)
    st.caption(
        "分年度收益/波动/回撤/权重与股债相关性见页面5；"
        "年度累计收益在 2019、2026 等不完整年度不可当作全年收益或年化收益。"
        "本区为程序计算结果，不等于策略有效性证明，也未进行参数优化。"
    )
