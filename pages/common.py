# -*- coding: utf-8 -*-
"""P6 页面共享组件：图表、格式化、Excel 导出（所有页面复用，避免公式/样式分裂）。"""
from __future__ import annotations

import io
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st


def _pct_axis(fmt: str = "%") -> alt.Axis:
    return alt.Axis(format=fmt)


def line_chart(
    df: pd.DataFrame,
    title: str,
    y_title: str,
    y_format: str = "%",
    height: int = 320,
):
    """宽表（index=日期，列=系列）→ Altair 折线。字段名为 ASCII，避免中文/括号解析问题。"""
    d = df.copy()
    d.index.name = "date"
    long = d.reset_index().melt(id_vars="date", var_name="series", value_name="value")
    ch = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="日期"),
            y=alt.Y("value:Q", title=y_title, axis=_pct_axis(y_format)),
            color=alt.Color("series:N", legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(title=title, height=height)
    )
    st.altair_chart(ch, width="stretch")


def bar_weights(weights: pd.Series, title: str, y_title: str = "目标权重", height: int = 280):
    d = pd.DataFrame({"asset": list(weights.index), "value": weights.astype(float).to_numpy()})
    ch = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("asset:N", axis=alt.Axis(labelAngle=0), title="资产"),
            y=alt.Y("value:Q", title=y_title, axis=_pct_axis("%"), scale=alt.Scale(zero=True)),
            tooltip=["asset:N", alt.Tooltip("value:Q", format=".2%")],
        )
        .properties(title=title, height=height)
    )
    st.altair_chart(ch, width="stretch")


def grouped_bar(df_long: pd.DataFrame, x: str, y: str, group: str, title: str,
                y_title: str, y_format: str = "%", height: int = 300):
    ch = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(labelAngle=0), title=None),
            xOffset=f"{group}:N",
            y=alt.Y(f"{y}:Q", title=y_title, axis=_pct_axis(y_format), scale=alt.Scale(zero=True)),
            color=alt.Color(f"{group}:N", legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[f"{x}:N", f"{group}:N", alt.Tooltip(f"{y}:Q", format=".4f")],
        )
        .properties(title=title, height=height)
    )
    st.altair_chart(ch, width="stretch")


def xy_line(df: pd.DataFrame, x: str, y: str, color: str | None, title: str,
            x_title: str, y_title: str, y_format: str = "%", x_type: str = "Q",
            height: int = 300):
    """通用折线（横轴可为数值型，如窗口长度/目标权重）。"""
    enc = {
        "x": alt.X(f"{x}:{x_type}", title=x_title),
        "y": alt.Y(f"{y}:Q", title=y_title, axis=_pct_axis(y_format)),
        "tooltip": [f"{x}:{x_type}", alt.Tooltip(f"{y}:Q", format=".4f")],
    }
    if color:
        enc["color"] = alt.Color(f"{color}:N", legend=alt.Legend(orient="bottom", title=None))
        enc["tooltip"].insert(1, f"{color}:N")
    ch = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(**enc).properties(title=title, height=height)
    st.altair_chart(ch, width="stretch")


def heatmap_corr(corr: pd.DataFrame, title: str = "资产相关系数矩阵"):
    d = corr.copy()
    d.index.name = "a"
    long = d.reset_index().melt(id_vars="a", var_name="b", value_name="corr")
    ch = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("a:N", title="资产"),
            y=alt.Y("b:N", title="资产"),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                            legend=alt.Legend(title="ρ")),
            tooltip=["a:N", "b:N", alt.Tooltip("corr:Q", format="+.3f")],
        )
        .properties(title=title, height=330)
    )
    st.altair_chart(ch, width="stretch")


def risk_return_scatter(df: pd.DataFrame, title: str = "风险—收益散点（共同样本外区间）"):
    """df 需含 ASCII 列：strategy / vol / ret / sharpe（值可中文）。"""
    ch = (
        alt.Chart(df)
        .mark_circle(size=180)
        .encode(
            x=alt.X("vol:Q", title="年化波动率", axis=_pct_axis("%")),
            y=alt.Y("ret:Q", title="年化收益(实际CAGR)", axis=_pct_axis("%")),
            color=alt.Color("strategy:N", legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["strategy:N", alt.Tooltip("vol:Q", format=".2%"),
                     alt.Tooltip("ret:Q", format=".2%"), alt.Tooltip("sharpe:Q", format=".3f")],
        )
        .properties(title=title, height=320)
    )
    st.altair_chart(ch, width="stretch")


def fmt_dataframe(df: pd.DataFrame, pct_cols=(), bp_cols=(), num_cols=(), digits: int = 3) -> pd.DataFrame:
    """显示层格式化（不改底层数值，导出仍用原始 float）。"""
    out = df.copy()
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{float(x)*100:.{digits}f}%" if pd.notna(x) else "—")
    for c in bp_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{float(x)*10000:+.2f} bp" if pd.notna(x) else "—")
    for c in num_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{float(x):.{digits}f}" if pd.notna(x) else "—")
    return out


def excel_download(sheets: dict[str, pd.DataFrame], label: str, prefix: str = "资产配置研究V2"):
    """多工作表 Excel 导出（原版导出结构扩展；只导出有权使用的数据与衍生结果）。"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, frame in sheets.items():
            if frame is None or len(frame) == 0:
                continue
            f = frame.copy()
            if not isinstance(f.index, pd.RangeIndex):
                f = f.reset_index()
            # Excel 工作表名 ≤31 字符且不含非法字符
            safe = str(name)[:31].replace("/", "_").replace("\\", "_").replace("*", "_")
            f.to_excel(writer, sheet_name=safe, index=False)
    st.download_button(
        label or "⬇️ 下载 Excel",
        data=buf.getvalue(),
        file_name=f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption(
        "导出内容仅包含本次会话的**衍生计算结果**（收益序列、权重、绩效、回测记录等）。"
        "内置公开行情按来源授权**仅运行时获取，不得作为原始行情再分发**；"
        "用户上传数据的使用与再分发授权由用户自行确认，平台不存储、不公开上传文件。"
    )


def section(title: str, help_text: str = ""):
    st.subheader(title)
    if help_text:
        st.caption(help_text)


def show_dict(d: dict, title: str = ""):
    if title:
        st.caption(title)
    st.dataframe(pd.DataFrame({"项目": list(d.keys()), "内容": [str(v) for v in d.values()]}),
                 hide_index=True, width="stretch")


def ensure_v23_diagnosis(ctx: dict) -> tuple[dict | None, str | None]:
    """页面5/6 可被独立打开：用同一套引擎补算 V2.3 四策略诊断。

    返回 `(诊断结果, 错误信息)`；任一失败都返回明确原因，不回落模拟数据。
    页面4 已计算时直接复用其缓存，避免重复回测。
    """
    cached = ctx["shared"].get("v23_diagnosis")
    if cached is not None:
        return cached, None
    try:
        from core.backtest import BacktestConfig
        from core.diagnosis import build_diagnosis
        P = ctx["params"]
        cfg = BacktestConfig(
            model="最小方差", cov_method=P["cov_method"], ewma_lambda=P["ewma_lambda"],
            window=int(P["window_days"]), window_label=P["window_label"],
            rebalance_every=int(P["rebalance_every"]), rebalance_label=P["rebalance_label"],
            cost_bps=float(P["cost_bps"]), rf_annual=ctx["rf_annual"],
            periods_per_year=ctx["periods_per_year"],
            constraints=ctx["constraints"], min_obs=60,
        )
        with st.spinner("正在补算 V2.3 四策略 / 窗口 / 分年度诊断…"):
            diag = build_diagnosis(ctx["returns"], cfg, ctx["asset_ids"], ctx["registry"])
        ctx["shared"]["v23_diagnosis"] = diag
        return diag, None
    except Exception as exc:  # noqa: BLE001 —— 明确报错，不静默降级
        return None, f"{type(exc).__name__}: {exc}"
