# -*- coding: utf-8 -*-
"""P2 收益与风险统一计算模块（所有页面共用，禁止在页面里重复实现公式）。

严格区分三个容易混淆的量
------------------------
1. **每期几何平均收益**  G = exp(mean(ln(1+r))) − 1 —— 单个期间的复利收益；
2. **年化几何收益（按频率换算）** (1+G)^K − 1，K = 年化频率参数（日频 252）；
3. **按实际经过时间计算的 CAGR** (NAV_end/NAV_start)^(1/Y) − 1，
   Y = (end_date − start_date).days / 365.25。

当样本实际跨度不等于 N/K（节假日、缺失日、非整年、多资产交集日历）时，
2 与 3 **不相等**，只能按各自定义解释，不得互相替代。

夏普比率频率一致性
------------------
用户输入的 rf_annual 是**年化**无风险利率；先转换为期间利率
rf_period = (1+rf_annual)^(1/K) − 1，再计算
Sharpe = mean(r − rf_period) / std(r, ddof=1) × √K。
rf_annual = 0 时明确标注为「零基准假设」。

对数收益率要求 r > −1（否则 log(1+r) 无定义），校验失败直接抛错。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25


# ---------------- 基础收益 ----------------
def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """r_t = P_t / P_{t−1} − 1；首行 NaN 保留由调用方决定。"""
    return prices.pct_change()


def log_returns(r: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """l_t = ln(1 + r_t)，要求 r_t > −1。"""
    arr = np.asarray(r, dtype=float)
    if np.any(arr <= -1):
        raise ValueError("存在 r ≤ −1 的收益率，log(1+r) 无定义（请检查价格数据）")
    out = np.log1p(r)
    return out if isinstance(out, (pd.Series, pd.DataFrame)) else pd.Series(out)


def arithmetic_mean(r) -> float:
    """μ = mean(r_t)（简单收益率的算术平均，期间口径）。"""
    return float(np.asarray(pd.Series(r).dropna(), dtype=float).mean())


def mean_log_return(r) -> float:
    """g = mean(ln(1+r_t))（平均对数收益率，精确值）。"""
    return float(np.log1p(pd.Series(r).dropna()).mean())


def geometric_mean_period(r) -> float:
    """G = exp(g) − 1（每期几何平均收益）。"""
    return float(np.expm1(mean_log_return(r)))


def annualized_from_period(period_return: float, periods_per_year: int = TRADING_DAYS) -> float:
    """把每期收益按频率复利年化：(1+period_return)^K − 1。

    当 period_return = G = exp(g)−1（每期几何收益）时，本式恒等于 exp(K·g)−1。
    为避免"把平均对数收益率 g 当简单率复利"的口径错误，
    年化几何收益请统一调用 annualized_geometric_from_log(g, K)。
    """
    return float((1.0 + period_return) ** periods_per_year - 1.0)


def annualized_geometric_from_log(g: float, periods_per_year: int = TRADING_DAYS) -> float:
    """平均对数收益率 g（每期）→ 按频率年化几何收益：G_annual = exp(K·g) − 1。

    这是 g 的唯一严格年化复利转换（g 是对数率，不是简单收益率）。
    使用 np.expm1 以降低小数值的浮点误差。
    """
    return float(np.expm1(periods_per_year * float(g)))


def cagr_from_nav(nav_end: float, nav_start: float, start, end) -> float:
    """CAGR = (NAV_end/NAV_start)^(1/Y) − 1，Y 为**实际经过年数**（按 365.25 天）。"""
    y = (pd.Timestamp(end) - pd.Timestamp(start)).days / DAYS_PER_YEAR
    if y <= 0:
        raise ValueError("CAGR 要求 end > start（实际经过年数必须为正）")
    if nav_start <= 0 or nav_end <= 0:
        raise ValueError("NAV 必须为正")
    return float((nav_end / nav_start) ** (1.0 / y) - 1.0)


def annualized_vol(r, periods_per_year: int = TRADING_DAYS) -> float:
    """σ_annual = σ_period × √K；σ_period 为样本标准差(ddof=1)。"""
    return float(pd.Series(r).dropna().std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(r, rf_annual: float = 0.0, periods_per_year: int = TRADING_DAYS) -> dict:
    """夏普比率（频率一致版本），返回比率与中间量。"""
    x = pd.Series(r).dropna()
    if len(x) < 2:
        return {"sharpe": float("nan"), "rf_period": float("nan"),
                "excess_mean_period": float("nan"), "vol_annual": float("nan"),
                "rf_annual": rf_annual, "zero_rf": rf_annual == 0.0}
    rf_period = float((1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0)
    excess = x - rf_period
    vol_p = float(excess.std(ddof=1))
    vol_a = vol_p * np.sqrt(periods_per_year)
    sharpe = float(excess.mean() / vol_p * np.sqrt(periods_per_year)) if vol_p > 0 else float("nan")
    return {
        "sharpe": sharpe,
        "rf_period": rf_period,
        "rf_annual": rf_annual,
        "excess_mean_period": float(excess.mean()),
        "vol_annual": vol_a,
        "zero_rf": rf_annual == 0.0,
        "formula": "Sharpe = mean(r−rf_period)/std(r−rf_period)×√K；rf_period=(1+rf_annual)^(1/K)−1",
    }


def drawdown_series(nav: pd.Series) -> pd.Series:
    """DD_t = NAV_t / max(NAV_0..NAV_t) − 1。"""
    nav = pd.Series(nav).astype(float)
    return nav / nav.cummax() - 1.0


def max_drawdown(nav: pd.Series) -> float:
    """MDD = min(DD_t)（返回负数或 0）。"""
    dd = drawdown_series(nav)
    return float(dd.min()) if len(dd) else float("nan")


def nav_from_returns(r: pd.Series, start_value: float = 1.0, start_date=None) -> pd.Series:
    """由期间收益构造净值序列（首日 = start_value）。"""
    x = pd.Series(r).dropna()
    nav = start_value * (1.0 + x).cumprod()
    if start_date is not None:
        nav = pd.concat([pd.Series([start_value], index=[pd.Timestamp(start_date)]), nav])
    return nav


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def sample_covariance(returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """样本协方差（ddof=1）× K 年化；索引/列均为 asset_id。"""
    return pd.DataFrame(
        returns.cov(ddof=1).to_numpy() * periods_per_year,
        index=returns.columns, columns=returns.columns,
    )


def rolling_metrics(r: pd.Series, window: int, periods_per_year: int = TRADING_DAYS,
                    rf_annual: float = 0.0) -> pd.DataFrame:
    """滚动累计收益、滚动年化波动、滚动年化几何收益、滚动夏普（rf 频率一致）。"""
    x = pd.Series(r).dropna()
    K = periods_per_year
    rf_period = float((1.0 + rf_annual) ** (1.0 / K) - 1.0)
    roll_cum = (1 + x).rolling(window).apply(np.prod, raw=True) - 1
    roll_std = x.rolling(window).std(ddof=1)
    roll_vol = roll_std * np.sqrt(K)
    roll_geo = np.exp(np.log1p(x).rolling(window).mean() * K) - 1
    roll_sharpe = (x.rolling(window).mean() - rf_period) / roll_std * np.sqrt(K)
    return pd.DataFrame({
        "滚动累计收益": roll_cum,
        "滚动年化波动率": roll_vol,
        "滚动年化几何收益": roll_geo,
        "滚动夏普比率": roll_sharpe,
    })


# ---------------- 绩效汇总（页面/回测共用） ----------------
def performance_summary(
    r: pd.Series,
    price_index: pd.DatetimeIndex | None = None,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    start_value: float = 1.0,
) -> dict:
    """统一绩效指标字典（所有页面、所有模型、敏感性分析共用这一实现）。

    CAGR 年数口径（V2.3 统一）
    --------------------------
    每条收益 x[t] 表示「上一交易日收盘 → 本交易日收盘」的持有收益，
    所以净值 1.0 位于**首个收益日的前一交易日**，期末净值位于最后收益日。
    旧实现直接取 `price_index[0]` 与 `price_index[-1]`：当 price_index 是
    全样本价格日历、而收益序列只从建仓日开始（`core/backtest.py`）时，
    d0 会退到数据第一天，实际年数被全样本跨度稀释，CAGR 被明显低估。
    现统一为：`d1 = x.index[-1]`；`d0` 取 price_index 中严格早于
    `x.index[0]` 的最后一个交易日；若调用方未提供更早日期，则退化为
    `d0 = x.index[0]`，保持静态分析与既有独立验算的原有口径。
    不重复首日、不人为增加虚假交易日。
    """
    x = pd.Series(r).dropna()
    if len(x) < 2:
        raise ValueError("绩效计算至少需要 2 条收益")
    d1 = pd.Timestamp(x.index[-1])
    if price_index is not None and len(price_index) >= 2:
        cal = pd.DatetimeIndex(price_index)
        prior = cal[cal < pd.Timestamp(x.index[0])]
        d0 = pd.Timestamp(prior[-1]) if len(prior) else pd.Timestamp(x.index[0])
    else:
        d0 = pd.Timestamp(x.index[0])
    y = (d1 - d0).days / DAYS_PER_YEAR
    nav = nav_from_returns(x, start_value=start_value)
    nav_start, nav_end = float(start_value), float(nav.iloc[-1])
    # 最大回撤的净值序列必须包含起点 1.0，否则首个收益期的下跌会被漏掉；
    # 与 core/diagnosis.py::performance_on_window 保持同一口径。
    nav_for_mdd = pd.concat([pd.Series([start_value], index=[d0]), nav])
    mu = arithmetic_mean(x)
    g = mean_log_return(x)
    G = geometric_mean_period(x)
    sr = sharpe_ratio(x, rf_annual, periods_per_year)
    return {
        "样本数": int(len(x)),
        "区间": f"{d0.date()} ~ {d1.date()}",
        "实际年数Y": float(y),
        "累计收益": nav_end / nav_start - 1.0,
        "算术平均收益(每期)": mu,
        "年化算术平均收益": mu * periods_per_year,
        "平均对数收益(每期)": g,
        "年化对数增长率": g * periods_per_year,
        "几何平均收益(每期)": G,
        "年化几何收益(按频率)": annualized_geometric_from_log(g, periods_per_year),
        "实际CAGR(按实际年数)": cagr_from_nav(nav_end, nav_start, d0, d1) if y > 0 else float("nan"),
        "年化波动率": annualized_vol(x, periods_per_year),
        "夏普比率": sr["sharpe"],
        "最大回撤": max_drawdown(nav_for_mdd),
        "期末净值": nav_end,
        "无风险利率(年化)": rf_annual,
        "零基准假设": sr["zero_rf"],
        "年化频率K": periods_per_year,
    }


def summary_frame(summaries: dict[str, dict]) -> pd.DataFrame:
    """多个组合的绩效对比表（行=组合，列=指标），数值列保持 float 便于格式化。"""
    return pd.DataFrame(summaries).T


# ---------------- 算术/几何近似（老师重点研究） ----------------
def arithmetic_geometric_check(
    r: pd.Series, periods_per_year: int = TRADING_DAYS, ddof: int = 1
) -> dict:
    """算术/几何/对数收益关系的验证（V2.1 修正版；**均为近似，不是恒等式**）。

    数学定义（全部为**每期**口径，r_t > −1 才有 ln(1+r)）：
        μ = mean(r_t)                       算术平均简单收益率
        g_exact = mean(ln(1+r_t))           平均对数收益率（精确）
        G = exp(g_exact) − 1                每期几何收益率
        G_annual = exp(K·g_exact) − 1       按频率年化的几何收益率（K 期/年）
        CAGR = (NAV_T/NAV_0)^(1/Y) − 1     按**实际经过年数** Y，与 G_annual 可不同

    两种二阶近似（都只是近似）：
        A（常用金融近似）   g_A = μ − 0.5·σ²,   σ² = Var(r)（默认 ddof=1 样本方差）
        B（展开细节/验证）  g_B = μ − 0.5·mean(r²),  mean(r²) = Var(r) + μ²（二阶原点矩）
    ln(1+r) = r − r²/2 + r³/3 − …，故二阶 Taylor 严格对应 −0.5·E[r²]（形式 B）；
    形式 A 用 Var 代替 E[r²]，在 μ≠0 时与 B 相差 0.5μ²。
    **采用 ddof=1 的样本方差时，A 也不是严格的二阶 Taylor 恒等式**，
    两者均须保留三阶及以上项与偏度误差，误差符号可正可负。

    三项误差（A 为主展示口径；B 同时给出）：
        单期对数误差        error_period = g_A − g_exact
        线性年化对数误差    error_annual_linear = error_period × K
        年化复利收益率误差  error_annual_compound = exp(K·g_A) − exp(K·g_exact)
                            = expm1(K·g_A) − expm1(K·g_exact)
    最后一项是**收益率之差**（不是对数率之差），用 np.expm1 计算以降低浮点误差。

    bp 定义：收益率（或对数收益率）差 × 10000，1bp = 0.01 个百分点（pp）。
    另返回误差符号与绝对误差，便于检查"近似必然低估"之类的错误论断。
    """
    x = pd.Series(r).dropna()
    if len(x) < 2:
        raise ValueError("近似验证至少需要 2 条收益")
    if np.any(x.to_numpy(dtype=float) <= -1):
        raise ValueError("存在 r ≤ −1，ln(1+r) 无定义，无法进行算术/几何验证")
    if ddof not in (0, 1):
        raise ValueError("ddof 仅支持 0 或 1")
    K = periods_per_year
    mu = float(x.mean())
    var_used = float(x.var(ddof=ddof))
    var_ddof1 = float(x.var(ddof=1))
    var_ddof0 = float(x.var(ddof=0))
    m2 = float(np.mean(x.to_numpy(dtype=float) ** 2))   # 二阶原点矩 E[r²] 的样本均值
    g_exact = float(np.log1p(x).mean())

    g_A = mu - 0.5 * var_used                            # 常用波动拖累近似
    g_B = mu - 0.5 * m2                                  # 更贴近样本二阶 Taylor 展开
    err_A = g_A - g_exact
    err_B = g_B - g_exact
    lin_A, lin_B = err_A * K, err_B * K
    cmp_A = float(np.expm1(K * g_A) - np.expm1(K * g_exact))
    cmp_B = float(np.expm1(K * g_B) - np.expm1(K * g_exact))

    bp = 10000.0
    return {
        # ---- 基础量 ----
        "mu_arith_period": mu,
        "var_simple_period": var_used,
        "var_simple_ddof1": var_ddof1,
        "var_simple_ddof0": var_ddof0,
        "mean_square_period": m2,
        "ddof": ddof,
        "sigma_annual": annualized_vol(x, K),
        "g_exact_period": g_exact,
        "geom_mean_period": float(np.expm1(g_exact)),
        # ---- 两种近似（A 为主口径，旧键 g_approx_period 保持兼容）----
        "g_approx_period": g_A,
        "g_approx_A_period": g_A,
        "g_approx_B_period": g_B,
        "approx_A_minus_B": g_A - g_B,
        "ddof_effect_on_A": (mu - 0.5 * var_ddof1) - (mu - 0.5 * var_ddof0),
        # ---- 按频率年化的几何收益（严格 exp(Kg)−1）----
        "annualized_geom_exact": annualized_geometric_from_log(g_exact, K),
        "annualized_geom_A": annualized_geometric_from_log(g_A, K),
        "annualized_geom_B": annualized_geometric_from_log(g_B, K),
        # ---- 三项误差（A）及 B 的对应值 ----
        "error_period": err_A,
        "error_A_period": err_A,
        "error_B_period": err_B,
        "error_annual_linear": lin_A,
        "error_A_annual_linear": lin_A,
        "error_B_annual_linear": lin_B,
        "error_annual_compound": cmp_A,
        "error_A_annual_compound": cmp_A,
        "error_B_annual_compound": cmp_B,
        # ---- bp（×10000）、符号、绝对误差 ----
        "error_period_bp": err_A * bp,
        "error_annual_linear_bp": lin_A * bp,
        "error_annual_compound_bp": cmp_A * bp,
        "error_B_period_bp": err_B * bp,
        "error_B_annual_linear_bp": lin_B * bp,
        "error_B_annual_compound_bp": cmp_B * bp,
        "error_sign": int(np.sign(err_A)) if err_A != 0 else 0,
        "error_annual_compound_sign": int(np.sign(cmp_A)) if cmp_A != 0 else 0,
        "abs_error_period": abs(err_A),
        "abs_error_annual_linear": abs(lin_A),
        "abs_error_annual_compound": abs(cmp_A),
        "abs_error_B_period": abs(err_B),
        # ---- 波动拖累 ----
        "vol_drag_exact": mu - g_exact,
        "vol_drag_approx": 0.5 * var_used,           # A（兼容旧键）
        "vol_drag_approx_A": 0.5 * var_used,
        "vol_drag_approx_B": 0.5 * m2,
        "max_abs_return": float(x.abs().max()),
        "n": int(len(x)),
        "K": K,
        "formula": "g ≈ μ − 0.5σ²（A，常用近似）｜g ≈ μ − 0.5·mean(r²)（B，二阶 Taylor 展开细节）；两者均为近似",
        "caveat": (
            "A、B 均为**近似**，**并非恒等式**：忽略 ln(1+r) 展开的三阶及以上项与偏度；"
            "ddof=1 的样本方差使 A 不是严格的二阶 Taylor 恒等式。"
            "误差符号不固定。按频率年化几何收益 exp(Kg)−1 与按实际经过年数计算的 CAGR "
            "是两种口径，样本跨度不等于 N/K 时数值不同，不得混用。"
            "bp = 收益率（对数收益率）差 × 10000，1bp = 0.01pp。"
        ),
        "definitions": {
            "mu": "mean(r_t)，简单收益率算术平均",
            "g_exact": "mean(ln(1+r_t))，平均对数收益率（精确）",
            "G": "exp(g)−1，每期几何收益率",
            "G_annual": "exp(K·g)−1，按频率年化的几何收益率",
            "CAGR": "(NAV_T/NAV_0)^(1/Y)−1，Y 为实际经过年数，与 G_annual 不同口径",
            "A": "μ − 0.5Var(r)，常用波动拖累近似（金融惯用）",
            "B": "μ − 0.5mean(r²)，更贴近样本二阶 Taylor 展开",
            "bp": "收益率差 × 10000",
        },
    }
