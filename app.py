# -*- coding: utf-8 -*-
"""资产配置研究平台 V2 —— Streamlit 入口（六页面共用同一套数据与计算逻辑）。

基于原版 `deploy_streamlit_cloud` 升级：原版代码完整保存在 `v1_reference/`，
数据接口通过 `data_sources/public_api.py` 代码级复用；原界面中来源展示、
口径说明、Altair 图表、Excel 多表导出等组件按逻辑移植到 V2 页面。

本地运行：
    .venv/bin/streamlit run app.py --server.port 8503
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from core.assets import AssetRegistry
from core.covariance import COV_METHODS, COV_METHOD_LABELS
from core.models import Constraints, IMPLEMENTED_MODELS
from core.quality import align_price_frame, returns_from_prices, validate_price_series
from data_sources.data_provider import get_public_prices
from core.backtest import REBALANCE_PRESETS, WINDOW_PRESETS
from pages import page1_data, page2_returns, page3_models, page4_backtest, page5_sensitivity, page6_results

st.set_page_config(page_title="资产配置研究平台", page_icon="📊", layout="wide")

# 隐藏 Streamlit 依据 pages/ 目录自动生成的英文页面导航（app / common / page1 data …）。
# 六个中文研究页面由下方侧边栏 radio 控制，不受此影响。
st.markdown(
    "<style>[data-testid='stSidebarNav']{display:none !important;}</style>",
    unsafe_allow_html=True,
)

PAGES = [
    "1️⃣ 资产与数据",
    "2️⃣ 收益与风险",
    "3️⃣ 配置模型",
    "4️⃣ 回测分析",
    "5️⃣ 参数敏感性",
    "6️⃣ 研究结果",
]

st.session_state.setdefault("v2_uploads", {})      # asset_id -> (series, AssetMeta)
st.session_state.setdefault("v2_shared", {})       # 页面间共享的计算结果
st.session_state.setdefault("v2_selected", None)
st.session_state.setdefault("v2_data_version", 0)


def get_registry() -> AssetRegistry:
    """内置注册表 + 会话内上传资产（上传数据不落盘、不进部署包）。"""
    reg = AssetRegistry()
    for aid, (_ser, meta) in st.session_state["v2_uploads"].items():
        if aid not in [a.asset_id for a in reg.all()]:
            reg.add_upload(meta)
    return reg


# st.cache_data 只作**进程内快速缓存**（TTL 1 小时）。
# 真正的可靠性来自 data_provider 的本地持久化快照：
# TTL 到期 / 接口失败 / 服务重启时，都从最近一次通过校验的真实快照恢复，
# 不再因为一次请求失败而把已成功获取的数据清空。
SUCCESS_CACHE_TTL = 3600


class PublicDataUnavailable(RuntimeError):
    """零个资产可用且无本地有效快照时抛出；st.cache_data 不缓存异常。"""

    def __init__(self, log_rows: list[dict], log_df, status: dict):
        super().__init__("公开数据全部获取失败，且无本地有效快照")
        self.log_rows, self.log_df, self.status = log_rows, log_df, status


@st.cache_data(ttl=SUCCESS_CACHE_TTL, show_spinner=False)
def cached_public_prices(asset_ids: tuple[str, ...], version: int = 0):
    """进程内快速缓存；返回 (序列, 日志行, 日志表, 数据状态)。

    - 接口失败但**有有效快照**：正常返回快照数据（明确标注非实时），缓存有效结果；
    - 零资产可用且**无快照**：抛 PublicDataUnavailable，失败结果不进 1 小时缓存，
      接口恢复后可立即重试；页面显示明确错误，绝不生成模拟数据。
    """
    reg = AssetRegistry()
    builtin = [a for a in asset_ids
               if a in [m.asset_id for m in reg.all()] and reg.get(a).fetch_key]
    res = get_public_prices(builtin, reg, version=version)
    status = {
        "states": res.states,
        "warnings": res.warnings,
        "blocked_reason": res.blocked_reason,
        "attempted": res.attempted,
        "skipped_reason": res.skipped_reason,
        "last_success_at": res.last_success_at,
    }
    if not res.series and builtin:
        raise PublicDataUnavailable(list(res.log_rows), res.log_df, status)
    return dict(res.series), list(res.log_rows), res.log_df, status


def build_context(selected: list[str], rf_annual: float, periods_per_year: int) -> dict:
    """构建所有页面共用的上下文：单次数据获取 → 质量校验 → 统一日历 → 收益矩阵。"""
    reg = get_registry()
    uploads = st.session_state["v2_uploads"]
    ver = st.session_state["v2_data_version"]
    status: dict = {}
    try:
        series, log_rows, log_df, status = cached_public_prices(tuple(selected), ver)
    except PublicDataUnavailable as exc:
        # 失败结果不进 st.cache_data（异常不被缓存），接口恢复后立即可重试
        series = {}
        log_rows, log_df, status = exc.log_rows, exc.log_df, exc.status
    except Exception as exc:  # noqa: BLE001 —— 缓存层异常也要明确展示，绝不静默丢弃数据
        series = {}
        log_rows = [{"asset_id": "（全部公开资产）", "成功": False, "来源": "公开数据接口",
                     "代码": "", "行数": 0, "起始": "", "结束": "", "耗时(s)": 0,
                     "错误": f"数据供给层异常：{type(exc).__name__}: {exc}"}]
        log_df = pd.DataFrame(log_rows)
        status = {"states": {}, "warnings": [str(exc)],
                  "blocked_reason": "数据供给层异常，且未能恢复本地快照。",
                  "attempted": False, "skipped_reason": "", "last_success_at": 0.0}
    series = dict(series)
    for aid in selected:
        if aid in uploads:
            series[aid] = uploads[aid][0]
    # 只保留实际选择且成功获取的资产
    series = {k: v for k, v in series.items() if k in selected}

    cleaned: dict[str, pd.Series] = {}
    quality_rows = []
    for aid, ser in series.items():
        s2, q = validate_price_series(ser, aid, min_obs=60)
        quality_rows.append(q.to_dict())
        if q.ok:
            cleaned[aid] = s2
            reg.update_quality(aid, "warning" if q.warnings else "ok", q.start, q.end)
        else:
            reg.update_quality(aid, "error", q.start, q.end)

    ctx = {
        "registry": reg,
        "selected": selected,
        "requested": list(selected),
        "prices": None,
        "returns": None,
        "quality": pd.DataFrame(quality_rows),
        "fetch_logs": log_df,
        "fetch_log_rows": log_rows,
        "data_status": status,
        "align_report": {},
        "warnings": [],
        "rf_annual": rf_annual,
        "periods_per_year": periods_per_year,
        "shared": st.session_state["v2_shared"],
        "data_ready": False,
        "error": "",
    }
    if len(cleaned) < 2:
        missing = [a for a in selected if a not in cleaned]
        ctx["error"] = (
            f"可用资产仅 {len(cleaned)} 个（至少需要 2 个）。"
            f"失败/被排除资产：{missing}。"
            "未生成模拟数据；已尝试从本地有效快照恢复，请查看下方获取日志与缓存状态。"
        )
        return ctx
    # 长期未更新 / 缓存损坏 / 身份不一致：明确阻止不可靠计算
    blocked = str(status.get("blocked_reason") or "")
    if blocked and len(cleaned) >= 2 and "可用资产不足" in blocked:
        blocked = ""
    if blocked:
        ctx["error"] = blocked
        return ctx
    try:
        prices, align_report = align_price_frame(cleaned, min_common=60)
    except ValueError as exc:
        ctx["error"] = str(exc)
        return ctx
    ctx["prices"] = prices
    ctx["returns"] = returns_from_prices(prices)
    ctx["align_report"] = align_report
    ctx["asset_ids"] = list(prices.columns)
    ctx["warnings"] = reg.caliber_warnings(ctx["asset_ids"])
    ctx["data_ready"] = True
    return ctx


# ---------------- 侧边栏（全局共享设置） ----------------
st.title("资产配置研究平台")
st.caption(
    "N 资产 · 统一模型接口 · 最小方差 · 滚动样本外回测 · 参数敏感性研究｜"
    "基于原「资产配置分析」网页升级，原版代码保留在 v1_reference/"
)

with st.sidebar:
    st.header("研究页面")
    page = st.radio("切换页面", PAGES, index=0, label_visibility="collapsed")
    st.divider()
    st.header("全局设置")
    registry_preview = get_registry()
    all_ids = [a.asset_id for a in registry_preview.all()]
    labels = {a.asset_id: f"{a.name}（{a.code}）" for a in registry_preview.all()}
    default_sel = st.session_state["v2_selected"] or [
        a for a in registry_preview.defaults() if a in all_ids
    ]
    selected = st.multiselect(
        "资产池（可多选，≥2 个）", all_ids, default=default_sel,
        format_func=lambda x: labels.get(x, x),
        help="默认示例为沪深300全收益 + 中债7-10年财富；可加入其他中证全收益指数或上传自有资产",
    )
    st.session_state["v2_selected"] = selected
    rf_annual = st.number_input(
        "无风险利率（年化，小数）", min_value=0.0, max_value=0.20, value=0.0, step=0.005,
        format="%.3f", help="0 表示零基准假设；平台会按 (1+rf)^(1/K)−1 转换为与收益一致的期间利率",
    )
    periods_per_year = st.selectbox(
        "年化频率 K", [252, 52, 12], index=0,
        help="日频 252 / 周频 52 / 月频 12；σ_annual = σ_period × √K",
    )
    if st.button("🔄 刷新公开数据", use_container_width=True):
        st.session_state["v2_data_version"] += 1
        st.session_state["v2_shared"].clear()   # 只清计算结果，不动真实行情快照
        st.rerun()
    st.caption(
        "真实行情成功获取并通过质量校验后写入**本地持久化快照**（原子写入）；"
        "1 小时缓存到期或接口暂时失败时自动回退最近一次有效快照，页面会明确标注"
        "「本地快照（非实时）」及数据截止/最近成功获取时间。"
        "刷新按钮有最小间隔防抖，失败不会清除已有数据。上传资产仅保留在当前会话内存。")

    st.divider()
    st.header("模型 / 回测参数（页面3–5共用）")
    st.caption(
        "V2.2 双策略基线：**恒定比例（30/70）** 与 **最小方差** 使用同一套数据、"
        "调仓日历、成本与绩效口径；页面4 提供统一回测对比。"
    )
    p_model = st.selectbox("配置模型", IMPLEMENTED_MODELS,
                           index=IMPLEMENTED_MODELS.index("最小方差"))
    p_cm_stock = st.slider(
        "恒定比例策略：股票类目标权重", 0.0, 1.0, 0.30, 0.05,
        help="其余权重分配给非股票类（债券等），同类内等权；仅影响恒定比例策略。"
             "到达再平衡日才恢复目标比例，其余时间允许权重自然漂移。",
    )
    p_cov = st.selectbox("协方差估计方法", COV_METHODS, index=0,
                         format_func=lambda x: COV_METHOD_LABELS[x],
                         help="三种方法并列，没有默认最优；Ledoit–Wolf 为 μI 目标的简化 δ 可复现基准")
    p_lambda = st.slider("EWMA 衰减系数 λ", 0.80, 0.995, 0.94, 0.005,
                         disabled=p_cov != "ewma",
                         help="日频 0.94 只是示例默认，不是所有市场的最优参数")
    # V2.2 基线默认：3 年（756 交易日）估计窗口；仍可手动切换到其他窗口
    p_win = st.selectbox("历史估计窗口", list(WINDOW_PRESETS) + ["自定义交易日数"],
                         index=list(WINDOW_PRESETS).index("36个月（756交易日）"))
    p_win_days = st.number_input("自定义窗口（交易日）", 60, 2000, 252, step=21,
                                 disabled=p_win != "自定义交易日数")
    p_minw = st.number_input("单资产权重下限", 0.0, 0.5, 0.0, 0.01)
    p_maxw = st.number_input("单资产权重上限", 0.2, 1.0, 1.0, 0.05)
    p_classcap = st.checkbox("启用股票类权重上限", value=False)
    p_cap = st.slider("股票类上限", 0.2, 1.0, 0.6, 0.05, disabled=not p_classcap)
    p_rebal = st.selectbox("再平衡频率", list(REBALANCE_PRESETS), index=1)
    p_cost = st.number_input("交易成本（单边 bp / 单位换手）", 0.0, 50.0, 0.0, 1.0)
    st.caption(
        "窗口单位为**固定交易日数量**（63/126/252/504/756），不是严格日历月；"
        "252 个交易日只是近似一年。样本不足会明确报错，不会偷偷缩短窗口。"
    )

params = {
    "model": p_model, "cov_method": p_cov, "ewma_lambda": float(p_lambda),
    "cm_stock_weight": float(p_cm_stock),
    "window_label": p_win,
    "window_days": WINDOW_PRESETS[p_win] if p_win in WINDOW_PRESETS else int(p_win_days),
    "min_weight": float(p_minw), "max_weight": float(p_maxw),
    "class_cap": float(p_cap) if p_classcap else None,
    "rebalance_label": p_rebal, "rebalance_every": REBALANCE_PRESETS[p_rebal],
    "cost_bps": float(p_cost),
}

if len(selected) < 2:
    st.info("请在左侧资产池中**至少选择 2 个资产**，然后开始研究。")
    st.stop()

with st.spinner("正在准备与校验资产数据…"):
    ctx = build_context(selected, rf_annual, periods_per_year)
ctx["params"] = params
constraints = Constraints(min_weight=params["min_weight"], max_weight=params["max_weight"])
if params["class_cap"] is not None and ctx.get("data_ready"):
    stock_ids = [a for a in ctx["asset_ids"] if ctx["registry"].get(a).asset_class == "股票"]
    if stock_ids:
        constraints.class_caps = {"股票": (0.0, params["class_cap"])}
# V2.1：资产类别映射属于统一约束对象，所有页面/回测/敏感性/导出共用，
# 不再依赖某个页面临时注入 asset_class_map
if ctx.get("data_ready"):
    constraints.asset_class_map = {
        a: ctx["registry"].get(a).asset_class for a in ctx["asset_ids"]
    }
ctx["constraints"] = constraints
# 配置签名：任何参数变化都使各页面的缓存结果失效，避免显示旧结果
ctx["config_signature"] = repr((
    tuple(ctx.get("asset_ids", [])), params, constraints.describe(),
    rf_annual, periods_per_year,
))
# V2.1：参数变化时清空所有页面共享的计算结果缓存，避免显示旧配置下的结果
if st.session_state.get("v2_config_signature") != ctx["config_signature"]:
    st.session_state["v2_config_signature"] = ctx["config_signature"]
    st.session_state["v2_shared"].clear()   # 原地清空，保持 ctx["shared"] 引用
ctx["shared"] = st.session_state["v2_shared"]

if not ctx["data_ready"]:
    st.error("**数据未就绪，未做任何替代（不会自动生成模拟行情）。**\n\n" + ctx["error"])
    if not ctx["fetch_logs"].empty:
        st.subheader("数据获取日志")
        st.dataframe(ctx["fetch_logs"], hide_index=True, width="stretch")
    st.info(
        "可行替代：① 在「1️⃣ 资产与数据」页上传你有权使用的 CSV/Excel；"
        "② 取消勾选失败的资产、保留至少 2 个成功资产；③ 稍后重试。"
    )
    st.stop()

# 全局口径/假设提示（所有页面共用）
for w in ctx["warnings"]:
    st.warning(w)
if rf_annual == 0:
    st.caption("ℹ️ 无风险利率 = 0（**零基准假设**）：夏普比率按超额收益 = 组合收益计算。")

st.divider()

ROUTES = {
    PAGES[0]: page1_data,
    PAGES[1]: page2_returns,
    PAGES[2]: page3_models,
    PAGES[3]: page4_backtest,
    PAGES[4]: page5_sensitivity,
    PAGES[5]: page6_results,
}
ROUTES[page].render(ctx)
