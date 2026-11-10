# -*- coding: utf-8 -*-
"""V2.3 策略有效性诊断：四策略对照 / 估计窗口敏感性 / 分年度分析。

严格增量设计（不重构、不改动任何已有计算引擎）
------------------------------------------------
- **不修改** `core/backtest.py`、`core/models.py`、`core/returns.py`、
  `core/covariance.py`、`core/sensitivity.py` 的任何函数。
- 四个策略全部进入**同一个** `run_backtest`，共用同一收益矩阵、调仓日历、
  成本假设、无风险利率与 `performance_summary` 绩效函数。
- 纯债券 Buy-and-Hold 通过 `rebalance_every=0`（引擎原生支持的「仅建仓、
  不再平衡」）实现，**不单独写一套模拟器**，避免口径分叉。
- 所有共同区间比较都排除首次建仓日（当日为现金、r_port=0），
  并以「净值=1 的前一交易日」作为 CAGR 年数起点。
- 不使用模拟数据；任一策略失败即明确报错，不做静默降级。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from core import returns as R
from core.assets import AssetRegistry
from core.backtest import BacktestConfig, BacktestResult, run_backtest
from core.strategies import (
    CM_LABEL, MV_LABEL, DEFAULT_STOCK_WEIGHT, constant_mix_weights,
    _CM_MODEL, _MV_MODEL,
)

# ---------------- 稳定标识 ----------------
BH_LABEL = "纯债券买入持有"
CM10_LABEL = "恒定比例（10/90）"
FOUR_LABELS: tuple[str, ...] = (CM_LABEL, MV_LABEL, BH_LABEL, CM10_LABEL)
LOW_STOCK_WEIGHT = 0.10
DIAG_WINDOWS: tuple[tuple[int, str], ...] = (
    (252, "1年（252交易日）"),
    (504, "2年（504交易日）"),
    (756, "3年（756交易日）"),
)


# ---------------- 1. 四策略统一配置 ----------------
def bond_only_weights(
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
) -> dict[str, float]:
    """纯债券组合：股票类权重 0，其余（债券类）等权，合计 1。

    明确要求至少一个非股票类资产；不做任何静默等权兜底。
    """
    ids = list(asset_ids)
    if len(ids) < 2:
        raise ValueError("纯债券对照基准至少需要 2 个资产")
    bonds = [a for a in ids if registry.get(a).asset_class != "股票"]
    if not bonds:
        raise ValueError("资产池中没有任何非股票类资产，无法构建纯债券对照基准")
    w = {a: 0.0 for a in ids}
    for a in bonds:
        w[a] = 1.0 / len(bonds)
    last = bonds[-1]
    w[last] += 1.0 - sum(w.values())
    return w


def four_strategy_configs(
    base: BacktestConfig,
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
    stock_weight: float = DEFAULT_STOCK_WEIGHT,
    low_stock_weight: float = LOW_STOCK_WEIGHT,
) -> dict[str, BacktestConfig]:
    """从同一配置派生四个策略；除策略定义必需字段外**逐字段相同**。

    差异仅允许：
    - `model` / `custom_weights`：策略定义；
    - 纯债券 B&H 的 `rebalance_every=0` / `rebalance_label`：为「不做不必要的
      周期再平衡交易」而设，属策略定义而非口径不一致。
    其余（window、cov_method、cost_bps、rf_annual、periods_per_year、
    constraints、min_obs、expanding…）完全一致。
    """
    cm30 = constant_mix_weights(asset_ids, registry, stock_weight)
    cm10 = constant_mix_weights(asset_ids, registry, low_stock_weight)
    bh = bond_only_weights(asset_ids, registry)
    return {
        CM_LABEL: replace(base, model=_CM_MODEL, custom_weights=cm30),
        MV_LABEL: replace(base, model=_MV_MODEL, custom_weights=None),
        BH_LABEL: replace(base, model=_CM_MODEL, custom_weights=bh,
                          rebalance_every=0,
                          rebalance_label="不再平衡（买入持有）"),
        CM10_LABEL: replace(base, model=_CM_MODEL, custom_weights=cm10),
    }


def run_four_strategies(
    returns_df: pd.DataFrame,
    base: BacktestConfig,
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
    stock_weight: float = DEFAULT_STOCK_WEIGHT,
    low_stock_weight: float = LOW_STOCK_WEIGHT,
) -> dict[str, BacktestResult]:
    """四个策略进入同一个 `run_backtest` 引擎。"""
    return {
        label: run_backtest(returns_df, cfg)
        for label, cfg in four_strategy_configs(
            base, asset_ids, registry, stock_weight, low_stock_weight
        ).items()
    }


def common_window(
    results: dict[str, BacktestResult],
    calendar: pd.DatetimeIndex | None = None,
    min_results: int = 2,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """共同样本外区间（排除首次建仓日）+ 净值起点日期。

    返回 `(c_start, c_end, nav_start)`：
    - `c_start = max(各策略 oos_start)`，并推进到**最晚首次建仓日的下一交易日**，
      避免把「现金日 r_port=0 / 仅计提初始成本」混入绩效评价；
    - `c_end = min(各策略 oos_end)`；
    - `nav_start` = `c_start` 在交易日历中的**前一交易日**，对应净值 1.0，
      用于 `performance_summary` 的实际经过年数（CAGR 分母）。
    """
    ok = [v for v in results.values() if v.ok and v.oos_start is not None]
    if len(ok) < min_results:
        bad = {k: v.error for k, v in results.items() if not v.ok}
        raise ValueError(
            f"共同区间比较至少需要{min_results}个策略都成功（实际 {len(ok)}）；"
            f"失败原因：{bad}")
    if calendar is None:
        calendar = ok[0].daily.index
    calendar = pd.DatetimeIndex(calendar)

    c_start = pd.Timestamp(max(v.oos_start for v in ok))
    c_end = pd.Timestamp(min(v.oos_end for v in ok))

    builds: list[pd.Timestamp] = []
    for v in ok:
        flag = v.daily["is_initial_build"].astype(bool)
        if flag.any():
            builds.append(pd.Timestamp(v.daily.index[flag][0]))
    if builds:
        last_build = max(builds)
        nxt = calendar[calendar > last_build]
        if len(nxt):
            c_start = max(c_start, pd.Timestamp(nxt[0]))

    if c_start >= c_end:
        raise ValueError(f"共同样本外区间为空（{c_start} ~ {c_end}）")
    prior = calendar[calendar < c_start]
    nav_start = (pd.Timestamp(prior[-1]) if len(prior)
                 else c_start - pd.Timedelta(days=1))
    return c_start, c_end, nav_start


def performance_on_window(
    r: pd.Series,
    nav_start: pd.Timestamp,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """统一绩效：以净值起点作为 CAGR 年数起点，MDD 包含净值 1.0 起点。

    这是本次核查修正的口径（详见 V2_3_STRATEGY_DIAGNOSIS.md）：
    旧实现把 `price_index` 构造成 `[c_start] + rr.index`，首日重复且
    NAV=1 隐含在 `c_start` 之前，导致 CAGR 年数少约 1 个交易日、
    MDD 净值序列缺少起点 1.0。
    """
    x = pd.Series(r).dropna()
    if len(x) < 2:
        raise ValueError("绩效计算至少需要 2 条收益")
    p = R.performance_summary(
        x,
        price_index=pd.DatetimeIndex([pd.Timestamp(nav_start)] + list(x.index)),
        rf_annual=rf_annual, periods_per_year=periods_per_year,
    )
    nav = R.nav_from_returns(x, start_date=nav_start)
    p["最大回撤"] = R.max_drawdown(nav)
    p["净值起点日期"] = pd.Timestamp(nav_start)
    return p


def four_strategy_comparison(
    results: dict[str, BacktestResult],
    calendar: pd.DatetimeIndex | None = None,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
    asset_ids: list[str] | tuple[str, ...] | None = None,
    registry: AssetRegistry | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """四策略在共同区间上的统一绩效对比表。返回 (frame, c_start, c_end, nav_start)。

    任一策略失败都**明确报错**，不静默用剩余策略出表（避免比较失真）。
    """
    failed = {k: v.error for k, v in results.items() if not v.ok}
    if failed:
        raise ValueError(f"对比要求全部策略回测成功、不做静默降级；失败：{failed}")
    c_start, c_end, nav_start = common_window(
        results, calendar, min_results=len(results))
    rows: dict[str, dict] = {}
    for label, bt in results.items():
        if not bt.ok:
            continue
        d = bt.daily.loc[c_start:c_end]
        p = performance_on_window(d["r_net"], nav_start, rf_annual, periods_per_year)
        row = {
            "样本外CAGR": p["实际CAGR(按实际年数)"],
            "年化波动率": p["年化波动率"],
            "夏普比率": p["夏普比率"],
            "最大回撤": p["最大回撤"],
            "累计收益": p["累计收益"],
            "实际年数Y": p["实际年数Y"],
            "累计换手": float(d["turnover"].sum()),
            "累计成本": float(d["cost"].sum()),
            "建仓次数": int(d["is_initial_build"].sum()),
            "再平衡次数": int(d["is_rebalance"].sum()),
            "总交易事件": int(d["is_trade"].sum()),
            "期末净值": float(R.nav_from_returns(d["r_net"].dropna(),
                                                  start_date=nav_start).iloc[-1]),
        }
        if asset_ids and registry is not None:
            stocks = [a for a in asset_ids if registry.get(a).asset_class == "股票"]
            actual = bt.weights_actual().loc[c_start:c_end]
            target = bt.weights_target().loc[c_start:c_end]
            if len(stocks) and all(a in actual.columns for a in stocks):
                row["平均实际股票权重"] = float(actual[stocks].sum(axis=1).mean())
            if len(stocks) and all(a in target.columns for a in stocks):
                row["平均目标股票权重"] = float(target[stocks].sum(axis=1).mean())
                row["目标股票权重标准差"] = float(target[stocks].sum(axis=1).std(ddof=1))
        rows[label] = row
    return pd.DataFrame(rows).T, c_start, c_end, nav_start


# ---------------- 2. 估计窗口敏感性 ----------------
def _rolling_estimated_corr(
    returns_df: pd.DataFrame,
    window: int,
    execution_dates: pd.DatetimeIndex,
    stock_cols: list[str],
    bond_cols: list[str],
) -> tuple[float, float, int]:
    """复算每个调仓决策点**估计窗口内**的股债相关系数（严格只用过去数据）。

    与 `run_backtest` 的 `est = returns_df.iloc[lo:j]` 一致，无前视。
    返回 (均值, 标准差, 有效估计次数)。
    """
    vals: list[float] = []
    cal = returns_df.index
    for dt in execution_dates:
        if dt not in cal:
            continue
        pos = cal.get_loc(dt)
        if pos < 2:
            continue
        lo = max(0, pos - window)
        est = returns_df.iloc[lo:pos]
        if len(est) < 20 or len(stock_cols) == 0 or len(bond_cols) == 0:
            continue
        a = est[stock_cols].mean(axis=1)
        b = est[bond_cols].mean(axis=1)
        c = float(pd.Series(a).corr(pd.Series(b)))
        if np.isfinite(c):
            vals.append(c)
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)


def window_analysis(
    returns_df: pd.DataFrame,
    base: BacktestConfig,
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
    windows: tuple[tuple[int, str], ...] = DIAG_WINDOWS,
) -> dict:
    """1年/2年/3年估计窗口对照，**必须采用相同样本外区间**。

    共同起点 = 最长窗口（默认 756）能开始回测的日期（并排除建仓现金日），
    即 `max(各窗口 oos_start)`；终点 = `min(oos_end)`。
    不自动挑选收益最高的窗口。
    """
    results: dict[str, BacktestResult] = {}
    for w, label in windows:
        cfg = replace(base, model=_MV_MODEL, custom_weights=None,
                      window=w, window_label=label)
        results[label] = run_backtest(returns_df, cfg)
    failed = {k: v.error for k, v in results.items() if not v.ok}
    if failed:
        raise ValueError(f"窗口敏感性回测存在失败（不做静默降级）：{failed}")

    c_start, c_end, nav_start = common_window(
        results, returns_df.index, min_results=1)
    stocks = [a for a in asset_ids if registry.get(a).asset_class == "股票"]
    bonds = [a for a in asset_ids if a not in stocks]

    rows: dict[str, dict] = {}
    weight_curves: dict[str, pd.Series] = {}
    for (w, label), bt in zip(windows, results.values()):
        d = bt.daily.loc[c_start:c_end]
        p = performance_on_window(d["r_net"], nav_start,
                                  base.rf_annual, base.periods_per_year)
        tw = bt.weights_target().loc[c_start:c_end]
        stock_w = tw[stocks].sum(axis=1) if stocks and all(a in tw.columns for a in stocks) \
            else pd.Series(0.0, index=tw.index)
        ex = pd.DatetimeIndex(bt.rebalance_log["execution_date"]) \
            if "execution_date" in bt.rebalance_log.columns else pd.DatetimeIndex([])
        ex = ex[(ex >= c_start) & (ex <= c_end)]
        corr_mean, corr_std, corr_n = _rolling_estimated_corr(
            returns_df, w, ex, stocks, bonds)
        rows[label] = {
            "估计窗口(交易日)": w,
            "共同区间CAGR": p["实际CAGR(按实际年数)"],
            "年化波动率": p["年化波动率"],
            "夏普比率": p["夏普比率"],
            "最大回撤": p["最大回撤"],
            "累计收益": p["累计收益"],
            "累计换手": float(d["turnover"].sum()),
            "累计成本": float(d["cost"].sum()),
            "再平衡次数": int(d["is_rebalance"].sum()),
            "平均目标股票权重": float(stock_w.mean()),
            "股票权重标准差": float(stock_w.std(ddof=1)),
            "股票权重最小值": float(stock_w.min()),
            "股票权重最大值": float(stock_w.max()),
            "估计窗口股债相关(均值)": corr_mean,
            "估计窗口股债相关(标准差)": corr_std,
            "相关性估计次数": corr_n,
            "样本外交易日": len(d),
            "实际年数Y": p["实际年数Y"],
        }
        weight_curves[label] = stock_w
    return {
        "frame": pd.DataFrame(rows).T,
        "results": results,
        "common_start": c_start,
        "common_end": c_end,
        "nav_start": nav_start,
        "stock_weight_curves": pd.DataFrame(weight_curves),
        "windows": windows,
    }


# ---------------- 3. 分年度诊断 ----------------
def annual_analysis(
    results: dict[str, BacktestResult],
    returns_df: pd.DataFrame,
    c_start: pd.Timestamp,
    c_end: pd.Timestamp,
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
    periods_per_year: int = 252,
) -> dict:
    """按自然年给出各策略收益/波动/回撤/平均权重，并标注不完整年度。

    - 年度收益为**区间累计收益**（不是年化收益），不完整年度明确标注，
      不得当作完整年度收益或年化收益解读。
    - 股债相关性用该年**已实现日收益**计算（描述性，非预测）。
    - `完整年度` 判定：按**自然年边界**判断（首日落在 1 月上旬、末日落在
      12 月下旬）。不能只用「覆盖交易日 = 全样本该年交易日」，因为当数据
      本身在 11 月结束时（如 2026-11-09），该错误口径会把不完整的年份
      误判为完整年度。
    """
    cal = pd.DatetimeIndex(returns_df.index)
    sub = cal[(cal >= c_start) & (cal <= c_end)]
    years = sorted(set(sub.year))
    stocks = [a for a in asset_ids if registry.get(a).asset_class == "股票"]
    all_assets = list(asset_ids)

    rows: list[dict] = []
    corr_rows: list[dict] = []
    for y in years:
        yi = cal[cal.year == y]
        ci = sub[sub.year == y]
        n_full, n_cov = len(yi), len(ci)
        if len(ci) == 0:
            continue
        first_day, last_day = pd.Timestamp(ci[0]), pd.Timestamp(ci[-1])
        # 自然年边界：A 股 1 月上旬开市、12 月下旬收官
        starts_early = first_day.month == 1 and first_day.day <= 10
        ends_late = last_day.month == 12 and last_day.day >= 20
        complete = bool(starts_early and ends_late)
        note = "完整年度" if complete else (
            f"**不完整年度**：覆盖 {first_day.date()} ~ {last_day.date()}"
            f"（共同区间 {n_cov} 日 / 该年数据可得 {n_full} 日），"
            "该年收益为部分年度累计收益，不可当作全年收益或年化收益")
        # 年度股债相关（已实现）
        yr = returns_df.loc[ci]
        corr = float("nan")
        if len(stocks) and len(yr) > 2:
            others = [a for a in all_assets if a not in stocks]
            if others:
                corr = float(pd.Series(yr[stocks].mean(axis=1)).corr(
                    pd.Series(yr[others].mean(axis=1))))
        corr_rows.append({
            "年度": y, "共同区间覆盖交易日": n_cov, "该年数据可得交易日": n_full,
            "年度是否完整": complete,
            "股债已实现相关(年度)": corr,
            "股票年化波动率": R.annualized_vol(yr[stocks].mean(axis=1), periods_per_year)
            if len(stocks) else float("nan"),
            "债券年化波动率": R.annualized_vol(
                yr[[a for a in all_assets if a not in stocks]].mean(axis=1),
                periods_per_year) if len(all_assets) > len(stocks) else float("nan"),
            "说明": note,
        })
        for label, bt in results.items():
            if not bt.ok:
                continue
            d = bt.daily.loc[ci[0]:ci[-1]]
            r = d["r_net"].dropna()
            if len(r) < 2:
                continue
            prior = cal[cal < ci[0]]
            nav_start = prior[-1] if len(prior) else ci[0] - pd.Timedelta(days=1)
            nav = R.nav_from_returns(r, start_date=nav_start)
            aw = bt.weights_actual().loc[ci[0]:ci[-1]]
            sw = (float(aw[stocks].sum(axis=1).mean())
                  if stocks and all(a in aw.columns for a in stocks) else float("nan"))
            rows.append({
                "年度": y,
                "策略": label,
                "年度累计收益": float((1.0 + r).prod() - 1.0),
                "年化波动率": R.annualized_vol(r, periods_per_year),
                "最大回撤": R.max_drawdown(nav),
                "平均实际股票权重": sw,
                "平均实际债券权重": 1.0 - sw if np.isfinite(sw) else float("nan"),
                "共同区间覆盖交易日": n_cov,
                "该年数据可得交易日": n_full,
                "年度是否完整": complete,
                "说明": note,
            })
    return {
        "annual": pd.DataFrame(rows),
        "correlation": pd.DataFrame(corr_rows),
        "incomplete_years": [int(c["年度"]) for c in corr_rows if not c["年度是否完整"]],
        "common_start": c_start,
        "common_end": c_end,
    }


# ---------------- 4. 页面共用的单一构建入口 ----------------
def build_diagnosis(
    returns_df: pd.DataFrame,
    base: BacktestConfig,
    asset_ids: list[str] | tuple[str, ...],
    registry: AssetRegistry,
    windows: tuple[tuple[int, str], ...] = DIAG_WINDOWS,
) -> dict:
    """页面4/5/6 共用的诊断构建入口（结果缓存到 session_state）。

    一次性产出：四策略结果/对比表、窗口敏感性、分年度分析、金融解释。
    任一环节失败都抛出明确异常，不回落模拟数据、不静默降级。
    """
    four = run_four_strategies(returns_df, base, asset_ids, registry)
    failed = {k: v.error for k, v in four.items() if not v.ok}
    if failed:
        raise ValueError(f"四策略回测存在失败（未用模拟数据补齐）：{failed}")
    frame, c0, c1, nav_start = four_strategy_comparison(
        four, returns_df.index, base.rf_annual, base.periods_per_year,
        asset_ids, registry)
    win = window_analysis(returns_df, base, asset_ids, registry, windows)
    ann = annual_analysis(four, returns_df, c0, c1, asset_ids, registry,
                          base.periods_per_year)
    return {
        "four": four,
        "frame": frame,
        "common_start": c0,
        "common_end": c1,
        "nav_start": nav_start,
        "window": win,
        "annual": ann,
        "interpretation": financial_interpretation(
            frame, win, ann, base.rf_annual),
        "config": base,
    }


# ---------------- 5. 金融解释（不预设结论） ----------------
def financial_interpretation(
    four: pd.DataFrame,
    window: dict,
    annual: dict,
    rf_annual: float = 0.0,
) -> list[str]:
    """基于真实数值回答五个诊断问题；只描述结果，不做策略优劣的因果断言。"""
    out: list[str] = []

    def g(label: str, col: str):
        return float(four.loc[label, col]) if label in four.index and col in four.columns else float("nan")

    # 1) 最小方差 vs 纯债券
    if all(k in four.index for k in (MV_LABEL, BH_LABEL)):
        v, b = g(MV_LABEL, "年化波动率"), g(BH_LABEL, "年化波动率")
        rc, rb = g(MV_LABEL, "样本外CAGR"), g(BH_LABEL, "样本外CAGR")
        out.append(
            f"**① 最小方差 vs 纯债券买入持有**：波动率 {v:.2%} vs {b:.2%}（差 {v-b:+.2%}），"
            f"CAGR {rc:.2%} vs {rb:.2%}（差 {rc-rb:+.2%}）。" +
            ("最小方差波动率更低，但接近纯债券时需检查其额外股票仓位是否带来相应补偿。"
             if v < b else "最小方差**未**实现比纯债券更低的波动率。")
        )
    # 2) vs 10/90
    if all(k in four.index for k in (MV_LABEL, CM10_LABEL)):
        v, t = g(MV_LABEL, "年化波动率"), g(CM10_LABEL, "年化波动率")
        rc, rt = g(MV_LABEL, "样本外CAGR"), g(CM10_LABEL, "样本外CAGR")
        out.append(
            f"**② 最小方差 vs 10/90 恒定比例**：波动率 {v:.2%} vs {t:.2%}（差 {v-t:+.2%}），"
            f"CAGR {rc:.2%} vs {rt:.2%}（差 {rc-rt:+.2%}），"
            f"夏普 {g(MV_LABEL,'夏普比率'):.2f} vs {g(CM10_LABEL,'夏普比率'):.2f}。" +
            ("若差异很小，说明优化带来的增量有限，简单低股票仓位已接近同等风险水平。"
             if abs(v - t) < 0.01 else "两者风险水平存在可观察差异，需结合权重与交易成本判断来源。")
        )
    # 3) 窗口差异
    wf = window.get("frame", pd.DataFrame())
    if len(wf) >= 2 and "股票权重标准差" in wf.columns:
        first, last = wf.iloc[0], wf.iloc[-1]
        out.append(
            f"**③ 1年 vs 3年窗口**：共同样本外区间 "
            f"{window['common_start'].date()} ~ {window['common_end'].date()} 一致。"
            f"平均股票权重 {first['平均目标股票权重']:.2%} → {last['平均目标股票权重']:.2%}，"
            f"权重标准差 {first['股票权重标准差']:.4f} → {last['股票权重标准差']:.4f}，"
            f"累计换手 {first['累计换手']:.3f} → {last['累计换手']:.3f}。"
            "短窗口对近期波动更敏感、权重调整更频繁；长窗口估计更平滑、反应更慢。"
            "这是**估计噪声与反应速度的权衡**，不是孰优孰劣；不以历史最高收益选窗口。"
        )
    # 4) 债券集中原因
    if BH_LABEL in four.index and MV_LABEL in four.index:
        sw = g(MV_LABEL, "平均实际股票权重")
        out.append(
            f"**④ 最小方差为何集中于债券**：实测平均实际股票权重 {sw:.2%}。"
            "最小方差的目标函数只最小化组合方差、完全不含预期收益项；在 long-only、"
            "权重和为 1 的约束下，当债券波动率显著低于股票且股债相关不高时，"
            "解析最优解会持续把权重推向低波动资产。可参考「股债年度相关性」与"
            "各资产年化波动率核对；这是**目标函数的必然结果**，不是求解器故障。"
        )
    # 5) 下一步
    inc = annual.get("incomplete_years", [])
    out.append(
        "**⑤ 下一步改进方向**（本阶段不执行）：① 引入权重上限/类别约束或目标波动，"
        "避免退化为单资产持有；② 用真实无风险利率替换 rf=0 零基准；"
        "③ 分组稳健性检验（不同市场阶段、不同起始日），避免在同一段样本外反复挑参；"
        "④ 扩展到更多资产并检验协方差估计方法（EWMA / Ledoit–Wolf）；"
        "⑤ 显式计入换手惩罚与更真实的成交假设。" +
        (f" 分年度结果中 {inc} 为不完整年度，已标注，不可当作全年表现。" if inc else "")
    )
    out.append(
        f"**夏普口径**：rf={rf_annual:.4f} " +
        ("为**零基准假设**，夏普比率不是相对真实无风险利率的超额收益表现。"
         if rf_annual == 0 else "使用侧边栏统一设定，四个策略完全一致。")
    )
    out.append(
        "**结论边界**：以上均为同一区间样本外回测的**程序计算结果**，"
        "通过测试只说明计算正确，不等于策略在未来有效；单区间、单参数组合"
        "不足以证明最小方差相对简单基准具有稳定优势。"
    )
    return out
