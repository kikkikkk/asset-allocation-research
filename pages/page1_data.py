# -*- coding: utf-8 -*-
"""页面1：资产与数据（资产池、上传、来源日志、质量、口径、原始行情图）。"""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from core.assets import CALIBER_NOTES, KNOWN_BASES, make_upload_meta
from data_sources.upload import parse_uploaded_prices, upload_template_frame
from pages.common import line_chart, section, show_dict


def render(ctx: dict):
    st.header("1️⃣ 资产与数据")
    st.caption(
        "同一套数据供全部六个页面使用（进程内快速缓存 1 小时 + 本地持久化真实行情快照）。"
        "资产池不绑定具体指数；内置资产仅为默认示例，可替换、扩展或上传自有数据。"
    )
    _render_cache_status(ctx)

    # ---------- 资产池 ----------
    section("资产池与元数据", "元数据字段对资产类型不敏感；未知字段显式标注为「未知」，不会假定所有资产字段相同。")
    info = ctx["registry"].info_frame()
    st.dataframe(info, width="stretch")
    st.caption(
        f"当前分析使用 **{len(ctx['asset_ids'])}** 个资产："
        + "、".join(f"{ctx['registry'].get(a).name}（{ctx['registry'].get(a).code}）" for a in ctx["asset_ids"])
    )

    # ---------- 上传 ----------
    with st.expander("➕ 上传自有资产（CSV / Excel）", expanded=False):
        st.dataframe(upload_template_frame(), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        up_name = c1.text_input("资产名称", value="自定义资产")
        up_code = c2.text_input("代码（可留空）", value="")
        c3, c4, c5 = st.columns(3)
        up_class = c3.selectbox("资产类别", ["股票", "债券", "商品", "基金", "REITs", "其他"])
        up_inst = c4.selectbox("工具类型", ["指数", "ETF", "公募基金", "现货", "期货", "其他"])
        up_freq = c5.selectbox("数据频率", ["D", "W", "M"])
        c6, c7 = st.columns(2)
        up_basis = c6.selectbox(
            "收益口径", sorted(KNOWN_BASES), index=sorted(KNOWN_BASES).index("未知"),
            help="口径无法确认请选「未知」，平台会持续提示；不要重复计入分红",
        )
        up_div = c7.selectbox("分红/票息处理", ["未知", "不含股息", "股息再投资", "票息再投资", "不适用"])
        up_file = st.file_uploader("上传文件（时间 + 收盘价(元) 两列）", type=["xlsx", "xls", "csv"],
                                   key="v2_upload_file")
        if st.button("添加为资产", disabled=up_file is None):
            try:
                ser, meta = parse_uploaded_prices(up_file)
                aid = "UP_" + re.sub(r"\W+", "_", up_name or "asset")[:20] + f"_{len(st.session_state['v2_uploads']) + 1}"
                m = make_upload_meta(
                    aid, up_name, up_code or aid, up_class, up_inst,
                    frequency=up_freq, return_basis=up_basis, dividend_treatment=up_div,
                    notes=f"解析元信息：{meta}",
                )
                st.session_state["v2_uploads"][aid] = (ser, m)
                st.session_state["v2_selected"] = list(st.session_state["v2_selected"]) + [aid]
                st.session_state["v2_asset_multiselect"] = st.session_state["v2_selected"]
                st.success(f"已添加 {up_name}（{len(ser)} 条价格记录）。数据仅保留在当前会话内存。")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"上传解析失败，未添加资产：{type(exc).__name__}: {exc}")

        if st.session_state["v2_uploads"]:
            st.caption("已上传资产（仅当前会话可见）：" + "、".join(st.session_state["v2_uploads"].keys()))
            if st.button("清除全部上传资产"):
                st.session_state["v2_uploads"].clear()
                st.rerun()

    # ---------- 数据获取日志 ----------
    section("数据获取日志", "每个资产单独记录来源、成败、错误原因与耗时；单资产失败不影响其他资产。")
    logs = ctx["fetch_logs"]
    st.dataframe(logs, hide_index=True, width="stretch")
    failed = logs[~logs["成功"]] if "成功" in logs.columns else pd.DataFrame()
    if not failed.empty:
        st.error(
            "以下资产获取失败（**未生成模拟数据、未静默替换口径**）："
            + "；".join(f"{r['asset_id']}（{r['错误']}）" for _, r in failed.iterrows())
        )
    uploaded = [a for a in ctx["asset_ids"] if a in st.session_state["v2_uploads"]]
    if uploaded:
        st.info(f"会话内上传资产：{uploaded}。请自行确认数据使用与再分发授权；平台不会公开上传数据。")

    # ---------- 数据质量 ----------
    section("数据质量检查", "不前向填充：统一回测日历取所有资产同时有观测的交易日（交集）。")
    st.dataframe(ctx["quality"], hide_index=True, width="stretch")
    rep = ctx["align_report"]
    show_dict({
        "统一回测日历": rep.get("policy", ""),
        "共同交易日数": rep.get("n_common", ""),
        "共同区间": f"{rep.get('start').date()} ~ {rep.get('end').date()}" if rep.get("start") is not None else "",
        "实际跨度(年)": round(float(rep.get("span_years", 0)), 2),
        "各资产被排除天数": str(rep.get("excluded_days", {})),
        "估值规则": rep.get("note", ""),
        "收益样本数": len(ctx["returns"]),
    }, "对齐与日历说明")

    # ---------- 口径 ----------
    section("收益口径与分红处理", "口径混用会系统性扭曲组合收益；无法统一时页面会持续提示。")
    cal = pd.DataFrame({
        "资产": [ctx["registry"].get(a).name for a in ctx["asset_ids"]],
        "代码": [ctx["registry"].get(a).code for a in ctx["asset_ids"]],
        "收益口径": [ctx["registry"].get(a).return_basis for a in ctx["asset_ids"]],
        "分红/票息处理": [ctx["registry"].get(a).dividend_treatment for a in ctx["asset_ids"]],
        "来源": [ctx["registry"].get(a).source for a in ctx["asset_ids"]],
    })
    st.dataframe(cal, hide_index=True, width="stretch")
    with st.expander("收益口径字典（避免重复计入分红 / 混用口径）"):
        st.dataframe(pd.DataFrame({"口径": list(CALIBER_NOTES), "说明": list(CALIBER_NOTES.values())}),
                     hide_index=True, width="stretch")
        st.markdown(
            "- **前复权/后复权是价格调整方式**，不等同于所有类型的总收益计算；前复权历史价格会随新分红变化，不适合长周期回测。\n"
            "- **累计单位净值 ≠ 复权单位净值**：前者未计分红再投资复利，不能直接当总回报序列。\n"
            "- 债券净价指数不含应计利息；财富指数含票息再投资；不可混用。\n"
            "- 口径无法统一时，本平台只提示、不自动换算，也不把不同口径视为经济含义可比。"
        )

    # ---------- 原始行情图 ----------
    section("原始行情图（标准化净值，期初 = 1）", "仅展示形态，不改变底层价格；区间为统一回测日历。")
    norm = ctx["prices"] / ctx["prices"].iloc[0]
    norm.columns = [ctx["registry"].get(a).name for a in norm.columns]
    line_chart(norm, "各资产标准化净值", "净值（期初=1）", y_format=".2f", height=360)
    st.caption(
        f"区间 {norm.index[0].date()} ~ {norm.index[-1].date()}，{len(norm)} 个共同交易日；"
        f"单位：倍（无量纲）。数据口径见上表，若为不同口径仅供形态对照。"
    )


# ---------------- V2.3：数据来源、缓存状态与新鲜度 ----------------
def _render_cache_status(ctx: dict) -> None:
    """明确区分「本次接口获取」与「本地快照」，绝不把旧数据冒充实时行情。"""
    status = ctx.get("data_status") or {}
    states = status.get("states") or {}
    if not states:
        return
    section("数据来源与缓存状态",
            "缓存到期 ≠ 数据失效；接口失败时自动回退最近一次通过校验的真实快照。")

    rows = []
    for aid, s in states.items():
        rows.append({
            "资产": s.get("name", aid),
            "代码": s.get("code", ""),
            "数据状态": s.get("origin", ""),
            "实际数据来源": s.get("source", ""),
            "数据起始": s.get("start", ""),
            "数据截止": s.get("end", ""),
            "最近成功获取": s.get("fetched_at", ""),
            "有效行数": s.get("rows", 0),
            "质量校验": "通过" if s.get("quality_ok") else "未通过",
            "新鲜度": s.get("freshness", ""),
            "滞后工作日": s.get("business_days_lag", ""),
            "校验和": s.get("checksum", ""),
            "本次接口错误": s.get("error", "") or "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    snapshot_assets = [s["name"] for s in states.values() if s.get("is_snapshot")]
    if snapshot_assets:
        st.warning(
            "**正在使用本地持久化快照（非实时数据）**："
            + "、".join(snapshot_assets)
            + "。这些数据是此前从公开接口成功获取、通过质量校验后保存的真实历史行情，"
              "上表列出了各自的数据截止日期与最近成功获取时间；"
              "**不得表述为刚从接口获取的实时行情**。请查看下方获取日志了解本次失败原因。"
        )
    else:
        st.success("当前所有资产均来自本次接口获取或进程内快速缓存，数据来源与截止时间见上表。")

    if status.get("attempted") is False and status.get("skipped_reason"):
        st.info(str(status["skipped_reason"]))
    for w in status.get("warnings") or []:
        st.caption("• " + str(w))
    if status.get("blocked_reason"):
        st.error("**已阻止不可靠计算**：" + str(status["blocked_reason"]))
    st.caption(
        "快照按资产独立保存（原子替换 + 校验和 + 指数身份/收益口径核验），"
        "一个资产更新失败不会污染另一个；收益与协方差只在共同有效交易日交集上计算，"
        "不拼接不一致的日期、不前向填充、不生成模拟行情。"
    )
