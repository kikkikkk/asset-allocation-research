# -*- coding: utf-8 -*-
"""V2.2 双策略基线：恒定比例（Constant-Mix） + 最小方差。

设计原则（本次为最小增量，不重构现有引擎）
------------------------------------------
- **不修改** `core/models.py` 的求解器、`core/backtest.py` 的回测引擎、
  `core/returns.py` 的绩效公式。本模块只是把它们按同一口径组装起来。
- 恒定比例策略复用已实现的 `CustomWeightModel`，由本模块生成默认
  股票 30% / 债券 70% 的目标权重；非调仓日仍由回测引擎自然漂移，
  **不是每天强行保持固定比例**。
- 两个策略通过 `dataclasses.replace` 从同一个 `BacktestConfig` 派生，
  保证资产、估计窗口、调仓日历、成本、无风险利率、年化频率完全一致。
- 绩效一律调用 `core/returns.py::performance_summary`，不重复实现公式。
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from core import returns as R
from core.assets import AssetRegistry
from core.backtest import BacktestConfig, BacktestResult, run_backtest

# 双策略基线的稳定标识（页面展示用；映射到现有模型名，不新增注册表项）
CM_LABEL = "恒定比例（30/70）"
MV_LABEL = "最小方差"
BASELINE_STRATEGIES = (CM_LABEL, MV_LABEL)

# 映射到现有模型名：恒定比例 = 用户指定目标权重（不参与优化、无估计窗口依赖）
_CM_MODEL = "自定义目标权重"
_MV_MODEL = "最小方差"

DEFAULT_STOCK_WEIGHT = 0.30
DEFAULT_WINDOW = 756          # 3 年（约 252×3 交易日）
DEFAULT_REBALANCE = 21        # 月度
DEFAULT_COV_METHOD = "sample"


def constant_mix_weights(
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
    stock_weight: float = DEFAULT_STOCK_WEIGHT,
) -> dict[str, float]:
    """生成恒定比例策略的目标权重：股票类合计 stock_weight，其余合计 1-stock_weight。

    - 同类内等权，天然支持 N 资产扩展。
    - 类别缺失（全部股票或全部非股票）时**明确报错**，不静默改成等权。
    - 权重和归一化到 1（浮点修正落在最后一个资产上），满足 long-only 与上下限约束。
    """
    ids = list(asset_ids)
    if len(ids) < 2:
        raise ValueError("恒定比例策略至少需要 2 个资产")
    if not 0.0 <= stock_weight <= 1.0:
        raise ValueError(f"股票目标权重必须在 [0,1]，收到 {stock_weight}")
    stocks = [a for a in ids if registry.get(a).asset_class == "股票"]
    others = [a for a in ids if a not in stocks]
    if not stocks or not others:
        raise ValueError(
            "恒定比例策略需要至少一个股票类与一个非股票类资产；"
            f"当前股票类={stocks or '无'}，其他={others or '无'}。"
            "请检查资产元数据中的资产类别，不要改用其他指数口径顶替。"
        )
    sw = stock_weight / len(stocks)
    ow = (1.0 - stock_weight) / len(others)
    w = {a: sw for a in stocks}
    w.update({a: ow for a in others})
    # 浮点修正：保证严格等于 1（CustomWeightModel 校验 abs(sum-1)<=1e-6）
    last = ids[-1]
    w[last] += 1.0 - sum(w.values())
    return w


def baseline_configs(
    base: BacktestConfig,
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
    stock_weight: float = DEFAULT_STOCK_WEIGHT,
) -> dict[str, BacktestConfig]:
    """从同一配置派生两个策略配置；除 model / custom_weights 外**逐字段相同**。"""
    cm_w = constant_mix_weights(asset_ids, registry, stock_weight)
    cm = replace(base, model=_CM_MODEL, custom_weights=cm_w, window_label=base.window_label,
                 rebalance_label=base.rebalance_label)
    mv = replace(base, model=_MV_MODEL, custom_weights=None)
    return {CM_LABEL: cm, MV_LABEL: mv}


def run_baseline_comparison(
    returns_df: pd.DataFrame,
    base: BacktestConfig,
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
    stock_weight: float = DEFAULT_STOCK_WEIGHT,
) -> dict[str, BacktestResult]:
    """两个策略进入**同一** `run_backtest` 引擎，返回各自结果。"""
    return {
        label: run_backtest(returns_df, cfg)
        for label, cfg in baseline_configs(base, asset_ids, registry, stock_weight).items()
    }


def comparison_frame(
    results: dict[str, BacktestResult],
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
    calendar: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    """在共同样本外区间上，用统一绩效函数生成对比表（支持任意 ≥2 个策略）。

    返回 (对比表, 共同起点, 共同终点)；任一策略失败则 ValueError，不回落模拟数据。

    V2.3 口径修正（CAGR / MDD）
    --------------------------
    旧实现把 `price_index` 构造成 `[c_start] + rr.index`，而 `c_start` 恰等于
    `rr.index[0]`，首日重复；且净值 1.0 隐含在首个收益日**之前**，导致
    `performance_summary` 用 `(last - first)` 计算实际经过年数，少约一个交易日。
    同时 `max_drawdown(nav_from_returns(rr))` 的净值序列缺少起点 1.0，
    会略微低估建仓初期回撤。
    现改为：
    - 共同起点推进到**最晚首次建仓日的下一交易日**（建仓日为现金、
      r_port=0，仅计提初始成本，不应计入绩效评价）；
    - 以该起点在交易日历中的**前一交易日**作为净值 1.0 的日期，
      用它构造 price_index 与实际年数，并让 MDD 净值序列包含起点。
    """
    ok = {k: v for k, v in results.items() if v.ok}
    if len(ok) < 2:
        bad = {k: v.error for k, v in results.items() if not v.ok}
        raise ValueError(f"对比至少需要两个策略都成功；失败原因：{bad}")
    if calendar is None:
        calendar = pd.DatetimeIndex(next(iter(ok.values())).daily.index)
    calendar = pd.DatetimeIndex(calendar)

    c_start = pd.Timestamp(max(v.oos_start for v in ok.values()))
    c_end = pd.Timestamp(min(v.oos_end for v in ok.values()))
    builds: list[pd.Timestamp] = []
    for v in ok.values():
        flag = v.daily["is_initial_build"].astype(bool)
        if flag.any():
            builds.append(pd.Timestamp(v.daily.index[flag][0]))
    if builds:
        nxt = calendar[calendar > max(builds)]
        if len(nxt):
            c_start = max(c_start, pd.Timestamp(nxt[0]))
    if c_start >= c_end:
        raise ValueError(f"共同样本外区间为空（{c_start} ~ {c_end}）")
    prior = calendar[calendar < c_start]
    nav_start = (pd.Timestamp(prior[-1]) if len(prior)
                 else c_start - pd.Timedelta(days=1))

    rows = {}
    for label, bt in ok.items():
        rr = bt.daily["r_net"].loc[c_start:c_end].dropna()
        p = R.performance_summary(
            rr, price_index=pd.DatetimeIndex([nav_start] + list(rr.index)),
            rf_annual=rf_annual, periods_per_year=periods_per_year,
        )
        nav = R.nav_from_returns(rr, start_date=nav_start)
        rows[label] = {
            "样本外CAGR": p["实际CAGR(按实际年数)"],
            "年化波动率": p["年化波动率"],
            "夏普比率": p["夏普比率"],
            "最大回撤": R.max_drawdown(nav),
            "累计收益": p["累计收益"],
            "实际年数Y": p["实际年数Y"],
            "累计换手": float(bt.daily["turnover"].loc[c_start:c_end].sum()),
            "累计成本": float(bt.daily["cost"].loc[c_start:c_end].sum()),
            "建仓次数": int(bt.daily["is_initial_build"].loc[c_start:c_end].sum()),
            "再平衡次数": int(bt.daily["is_rebalance"].loc[c_start:c_end].sum()),
            "总交易事件": int(bt.daily["is_trade"].loc[c_start:c_end].sum()),
            "期末净值": float(nav.iloc[-1]),
        }
    return pd.DataFrame(rows).T, c_start, c_end


def interpretation(frame: pd.DataFrame, rf_annual: float = 0.0) -> list[str]:
    """基于样本外结果生成可解释结论（不预设最小方差更优）。"""
    if CM_LABEL not in frame.index or MV_LABEL not in frame.index:
        return ["双策略结果不完整，暂不生成解释。"]
    cm, mv = frame.loc[CM_LABEL], frame.loc[MV_LABEL]
    out = []
    dvol = mv["年化波动率"] - cm["年化波动率"]
    out.append(
        f"**波动率**：最小方差 {mv['年化波动率']:.2%}，恒定比例 {cm['年化波动率']:.2%}，"
        f"差 {dvol:+.2%} —— " +
        ("样本外确实降低了实现波动率。" if dvol < 0 else "样本外**未**降低实现波动率（不应预设优化一定更优）。")
    )
    dret = mv["样本外CAGR"] - cm["样本外CAGR"]
    out.append(
        f"**收益**：最小方差 CAGR {mv['样本外CAGR']:.2%}，恒定比例 {cm['样本外CAGR']:.2%}，"
        f"差 {dret:+.2%} —— " +
        ("降风险伴随收益下降。" if (dvol < 0 and dret < 0) else
         "降风险未伴随收益下降。" if dvol < 0 else "需结合区间判断，不做因果断言。")
    )
    out.append(
        f"**交易**：恒定比例累计换手 {cm['累计换手']:.3f} / 成本 {cm['累计成本']*10000:.2f}bp；"
        f"最小方差 {mv['累计换手']:.3f} / {mv['累计成本']*10000:.2f}bp。"
        "再平衡恢复目标权重会产生换手，成本直接从净值扣除；提高调仓频率未必改善净收益。"
    )
    out.append(
        f"**夏普口径**：rf={rf_annual:.3f} " +
        ("（零基准假设，不是真实超额收益表现）" if rf_annual == 0 else "（使用统一侧边栏设定）")
    )
    out.append(
        "**局限**：以上为同区间样本外回测的程序计算结果，不等于策略有效性证明；"
        "最小方差权重可能高度集中于低波动资产，需结合权重曲线与不同市场阶段分段观察。"
    )
    return out
