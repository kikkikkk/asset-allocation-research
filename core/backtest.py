# -*- coding: utf-8 -*-
"""P4 统一回测引擎（V2.1；所有模型共用；无前视偏差强制约束）。

唯一时间规则（只有日收盘数据）
------------------------------
对第 j 个交易事件（j 为收益矩阵中的位置索引）：

    t 日收盘      已获得 t 日及之前全部历史数据
    t 日收盘后    估计协方差并生成目标权重 → signal_date = info_cutoff = R.index[j−1]
    t+1 日收盘    按收盘价近似成交         → execution_date = R.index[j]
    t+2 日起      新权重开始产生收益       → holding_start = R.index[j+1]

    恒有 est_end ≤ info_cutoff < execution_date < holding_start（每次事件单独校验）。

- 权重只用 `est_returns = R.iloc[j−W : j]`（严格早于执行日），**无未来信息**；
- 执行日当日收益由**执行前**的持仓实现（收盘才成交）；新权重从下一期起生效；
- 不虚构次日开盘价：以执行日收盘价作为可执行价格近似，页面明确标注；
- 相邻交易的持有期首尾相接、无重叠、无空窗。

首次建仓（V2.1 修复）
---------------------
- 回测开始时资金**全为现金**，初始净值 = 1，**零收益现金假设**（现金收益率为 0，
  仅代表"不计利息的记账现金"，不代表真实无风险利率）；
- 首个 execution_date 收盘建仓，建仓前不享有任何目标资产收益（当日 r_port = 0）；
- 建仓后的下一个收益区间才开始按目标权重计算持有收益；
- 从现金买入资产按资金权重分配，**计提初始交易成本**，成本只在实际执行时点扣一次。

权重漂移（两次交易之间，无交易、无外部现金流）
---------------------------------------------
    r_p,t   = Σ_i w_i,t-1 · r_i,t
    w_i,t   = w_i,t-1 · (1 + r_i,t) / (1 + r_p,t)
默认**不**每天恢复目标权重；只有交易事件才把权重设回目标。
若 1 + r_p,t ≈ 0（组合价值归零、权重无定义）直接抛错。

换手与交易成本（主口径唯一，不混用两种换手定义）
------------------------------------------------
    主指标 turnover_abs_t = Σ_i |w_target,i − w_pre,i|  （**双边绝对权重变化**）
    cost_t = turnover_abs_t × cost_bps / 10000           （买入与卖出使用同一成本率 c）
    r_net,t = (1 + r_port,t)(1 − cost_t) − 1
参考口径（不参与成本计算）：
    one_side_turnover = 0.5 × turnover_abs（仅再平衡事件的"单边"口径）
    traded_notional   = 建仓事件 1.0（现金全额买入）；再平衡事件 0.5 × turnover_abs
首次建仓时 w_pre = 0（现金），故 turnover_abs = Σw = 1，初始成本 = c。

事件区分：is_initial_build（仅 1 次）/ is_rebalance（后续调仓）/ is_trade = 前两者之和。
三种计数分别记录，避免同一页面出现"调仓次数相差 1"。

样本外区间为空时直接抛错（不显示虚假结果）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import returns as R
from core.covariance import estimate_covariance
from core.models import Constraints, ModelResult, solve_model

TRADING_DAYS = 252

WINDOW_PRESETS: dict[str, int] = {
    "3个月（63交易日）": 63,
    "6个月（126交易日）": 126,
    "12个月（252交易日）": 252,
    "24个月（504交易日）": 504,
    "36个月（756交易日）": 756,
}
REBALANCE_PRESETS: dict[str, int] = {
    "不再平衡（买入持有）": 0,
    "月度（21交易日）": 21,
    "季度（63交易日）": 63,
    "半年度（126交易日）": 126,
    "年度（252交易日）": 252,
}


@dataclass
class BacktestConfig:
    model: str = "最小方差"
    cov_method: str = "sample"
    ewma_lambda: float = 0.94
    window: int = 252                 # 估计窗口（**固定交易日数量**）
    window_label: str = "12个月（252交易日）"
    rebalance_every: int = 21         # 0 = 不再平衡
    rebalance_label: str = "月度（21交易日）"
    cost_bps: float = 0.0
    rf_annual: float = 0.0
    periods_per_year: int = TRADING_DAYS
    constraints: Constraints = field(default_factory=Constraints)
    custom_weights: dict | None = None
    solver: str = "auto"
    min_obs: int = 60
    expanding: bool = False           # True = 用 j 之前全部历史（仍需 ≥ min_obs）

    def validate(self, n_obs: int) -> None:
        if not self.expanding and self.window < self.min_obs:
            raise ValueError(f"估计窗口 {self.window} 条 < 最少 {self.min_obs} 条")
        need = self.min_obs if self.expanding else self.window
        if n_obs <= need + 2:
            raise ValueError(
                f"总收益样本 {n_obs} 条，估计窗口需 {need} 条，"
                "剩余样本不足 3 条，**没有样本外数据可回测**（不会偷偷缩短窗口）。"
            )
        if self.rebalance_every < 0:
            raise ValueError("调仓间隔不得为负")

    def to_dict(self) -> dict:
        return {
            "模型": self.model, "协方差方法": self.cov_method,
            "EWMA λ": self.ewma_lambda if self.cov_method == "ewma" else "—",
            "估计窗口": self.window_label + ("（expanding）" if self.expanding else ""),
            "估计交易日数": self.window if not self.expanding else "expanding",
            "再平衡频率": self.rebalance_label,
            "调仓间隔(交易日)": self.rebalance_every,
            "交易成本(bp)": self.cost_bps,
            "无风险利率(年化)": self.rf_annual,
            "最小观测数": self.min_obs,
            "执行假设": "t日收盘后信号→t+1日按收盘价近似成交→t+2日起新权重产生收益；首日现金建仓并计初始成本（零收益现金假设）",
            "成本口径": "turnover=Σ|目标权重−交易前权重|（双边绝对权重变化），cost=turnover×bp/10000，买卖同费率",
        }


@dataclass
class BacktestResult:
    ok: bool
    config: BacktestConfig
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    rebalance_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance: dict = field(default_factory=dict)
    oos_start: pd.Timestamp | None = None
    oos_end: pd.Timestamp | None = None
    n_oos: int = 0
    n_rebalance: int = 0                  # 后续再平衡次数（不含首次建仓）
    n_initial_build: int = 0              # 首次建仓次数（0 或 1）
    n_trade_events: int = 0               # 总交易事件 = 建仓 + 再平衡
    n_planned_trades: int = 0             # 计划交易事件数（成功 + 失败）
    n_failed_trades: int = 0              # 失败交易事件总数（建仓失败 + 调仓失败）
    n_failed_build: int = 0               # 首次建仓失败次数
    n_failed_rebalance: int = 0           # 后续调仓失败次数（失败时继续持有漂移权重并记录）
    turnover_total: float = 0.0
    cost_total: float = 0.0
    forecast_eval: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def weights_target(self) -> pd.DataFrame:
        cols = [c for c in self.daily.columns if c.startswith("target_")]
        out = self.daily[cols].copy()
        out.columns = [c[7:] for c in cols]
        return out

    def weights_actual(self) -> pd.DataFrame:
        cols = [c for c in self.daily.columns if c.startswith("actual_")]
        out = self.daily[cols].copy()
        out.columns = [c[7:] for c in cols]
        return out

    def weights_pre_trade(self) -> pd.DataFrame:
        cols = [c for c in self.daily.columns if c.startswith("pre_trade_")]
        out = self.daily[cols].copy()
        out.columns = [c[10:] for c in cols]
        return out

    def weights_post_trade(self) -> pd.DataFrame:
        cols = [c for c in self.daily.columns if c.startswith("post_trade_")]
        out = self.daily[cols].copy()
        out.columns = [c[11:] for c in cols]
        return out


def compute_target_weights(
    est_returns: pd.DataFrame,
    config: BacktestConfig,
    as_of: pd.Timestamp,
) -> tuple[ModelResult, dict]:
    """统一的"估计协方差 → 求模型权重"入口；回测与静态分析、敏感性分析共用。"""
    cov = estimate_covariance(
        est_returns, method=config.cov_method, ewma_lambda=config.ewma_lambda,
        periods_per_year=config.periods_per_year, min_obs=config.min_obs,
    )
    params = {
        "solver": config.solver,
        "weights": config.custom_weights,
        # 兼容旧调用；V2.1 起类别映射优先来自 config.constraints.asset_class_map
        "asset_class_map": dict(config.constraints.asset_class_map),
    }
    res = solve_model(config.model, est_returns, cov, config.constraints, **params)
    return res, cov.to_dict()


def simulate_portfolio(
    returns_df: pd.DataFrame,
    target_schedule: list[tuple[int, np.ndarray, dict]],
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """底层逐日模拟器（V2.1）：按 (执行日位置 j, 目标权重, 事件信息) 调度交易。

    - 起始资金为**现金**（权重全 0、净值 1、现金收益率 0 = 零收益现金假设）；
    - 首个执行日 = 首次建仓：当日 r_port = 0（不享有目标资产收益），收盘按目标权重
      全额买入并**计提初始交易成本**（turnover_abs = Σw = 1）；
    - 其后每个执行日：当日收益由执行前的漂移权重实现，收盘换仓并计提成本；
    - 新权重一律从执行日的**下一期**起产生收益；
    - 非交易日权重按 w_i,t = w_i,t−1(1+r_i,t)/(1+r_p,t) 自然漂移，不恢复目标权重。

    成本主口径：turnover = Σ|w_target − w_pre|（双边绝对权重变化），
    cost = turnover × cost_bps / 10000，买卖使用同一成本率。
    one_side_turnover = 0.5×turnover 与 traded_notional 仅为参考口径，不参与成本计算。
    成交金额口径（每事件同时记录，互不替代）：
        buy_amount  = Σ max(w_target − w_pre, 0)   买入成交额（占净值比例）
        sell_amount = Σ max(w_pre − w_target, 0)   卖出成交额（占净值比例）
        建仓：buy=1.0、sell=0.0（从现金全额买入）；再平衡（完全投资）：buy = sell。
    自融资（self-financing）简化假设：交易成本按比例从净值中扣除
    （r_net = (1+r_port)(1−cost) − 1，每执行时点只扣一次），**成交后实际权重仍精确等于
    目标权重**——即假设成本从组合外部现金支付、不消耗投资本金。这是简化模型：
    真实世界里成本会减少可投资金、使成交后权重略低于目标。
    """
    assets = list(returns_df.columns)
    Rm = returns_df.to_numpy(dtype=float)
    n, p = Rm.shape
    sched = {j: (w, info) for j, w, info in target_schedule}
    if not sched:
        raise ValueError("无调仓计划（交易计划为空），无法开始回测")
    if cost_bps < 0:
        raise ValueError("交易成本不得为负")

    j0 = min(sched)
    zero = np.zeros(p, dtype=float)
    w_cur = zero.copy()                       # 现金：无风险资产持仓权重为 0
    target = np.asarray(sched[j0][0], dtype=float).copy()
    cum_cost = cum_turnover = 0.0
    rows = []

    for j in range(j0, n):
        r_t = Rm[j]
        is_build = j == j0
        is_rebal = (j in sched) and not is_build
        is_trade = is_build or is_rebal
        if is_rebal:
            target = np.asarray(sched[j][0], dtype=float).copy()

        # 1) 当日收益：建仓日前/当日仍是现金 → 0；其余按执行前漂移权重实现
        if is_build:
            r_p = 0.0
            w_pre_trade = zero.copy()         # 交易前为现金
            w_after_return = zero.copy()
        else:
            r_p = float(w_cur @ r_t)
            denom = 1.0 + r_p
            if abs(denom) < 1e-12:
                raise ValueError(
                    f"{returns_df.index[j].date()} 组合收益 r_p={r_p:.6f} 使净值归零，"
                    "权重漂移 w_i,t=w_i,t−1(1+r_i,t)/(1+r_p,t) 无定义，回测中止"
                )
            w_after_return = w_cur * (1.0 + r_t) / denom
            w_pre_trade = w_after_return.copy()

        # 2) 收盘执行：换手、成本（只扣一次）、成交后权重
        turn = cost = one_side = traded = 0.0
        buy_amt = sell_amt = 0.0
        if is_trade:
            turn = float(np.abs(target - w_pre_trade).sum())   # 主口径：双边绝对权重变化
            one_side = 0.5 * turn
            # 建仓：从现金全额买入，单边成交金额 = 1；再平衡：0.5×Σ|Δw|
            traded = 1.0 if is_build else one_side
            buy_amt = float(np.clip(target - w_pre_trade, 0.0, None).sum())   # 买入成交额
            sell_amt = float(np.clip(w_pre_trade - target, 0.0, None).sum())  # 卖出成交额
            cost = turn * cost_bps / 10000.0
            w_post_trade = target.copy()
        else:
            w_post_trade = w_after_return.copy()

        r_net = (1.0 + r_p) * (1.0 - cost) - 1.0
        cum_cost += cost
        cum_turnover += turn
        w_cur = w_post_trade.copy()          # 下一期起按成交后权重产生收益

        rows.append({
            "r_port": r_p, "r_net": r_net, "cost": cost,
            "turnover": turn, "one_side_turnover": one_side,
            "traded_notional": traded, "buy_amount": buy_amt, "sell_amount": sell_amt,
            "cumulative_cost": cum_cost,
            "cumulative_turnover": cum_turnover,
            "is_initial_build": is_build, "is_rebalance": is_rebal, "is_trade": is_trade,
            **{f"target_{a}": target[i] for i, a in enumerate(assets)},
            **{f"actual_{a}": w_post_trade[i] for i, a in enumerate(assets)},
            **{f"pre_trade_{a}": w_pre_trade[i] for i, a in enumerate(assets)},
            **{f"post_trade_{a}": w_post_trade[i] for i, a in enumerate(assets)},
        })

    out = pd.DataFrame(rows, index=returns_df.index[j0:])
    out["nav"] = (1 + out["r_net"]).cumprod()
    out["drawdown"] = R.drawdown_series(out["nav"])
    return out


def run_backtest(returns_df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Walk-forward 主引擎：滚动估窗 → 求权重 → 执行 → 持有 → 评价。"""
    n = len(returns_df)
    try:
        config.validate(n)
    except ValueError as exc:
        return BacktestResult(ok=False, config=config, error=str(exc))

    first = config.min_obs if config.expanding else config.window
    step = config.rebalance_every if config.rebalance_every > 0 else max(n - first, 1)
    decisions = list(range(first, n - 1, step))   # 留出 j+1 作为持有收益起点
    if not decisions:
        return BacktestResult(
            ok=False, config=config,
            error=f"按窗口 {first} 与调仓间隔 {config.rebalance_every} 无法形成任何样本外持有期。",
        )

    schedule: list[tuple[int, np.ndarray, dict]] = []
    log_rows: list[dict] = []
    failed: list[str] = []          # 人类可读失败原因（告警用）
    failed_meta: list[dict] = []    # 结构化失败记录（日志/导出用）

    def _record_failure(k: int, j: int, reason: str) -> None:
        """记录一次失败交易事件：区分首次建仓失败与后续调仓失败，策略继续持有原权重。"""
        kind = "首次建仓失败" if k == 0 else "调仓失败"
        failed.append(f"{returns_df.index[j].date()}（{kind}）：{reason}")
        failed_meta.append({
            "execution_date": returns_df.index[j],
            "event_kind": kind,
            "reason": reason,
        })

    n_planned = len(decisions)
    for k, j in enumerate(decisions):
        lo = 0 if config.expanding else j - config.window
        est = returns_df.iloc[lo:j]
        if len(est) < config.min_obs:
            _record_failure(k, j, f"估计样本 {len(est)} < 最少 {config.min_obs}，跳过该时点")
            continue
        try:
            res, cov_meta = compute_target_weights(est, config, returns_df.index[j - 1])
        except Exception as exc:  # noqa: BLE001 —— 单期失败记录，不丢失其他期
            _record_failure(k, j, f"{type(exc).__name__}: {exc}")
            continue
        if res.status != "success" or res.weights.isna().any():
            _record_failure(k, j, f"模型求解失败（{res.message}）")
            continue
        w = res.weights.to_numpy(dtype=float)
        schedule.append((j, w, {"res": res}))
        info_cutoff = returns_df.index[j - 1]
        execution_date = returns_df.index[j]
        holding_start = returns_df.index[min(j + 1, n - 1)]
        # 每次事件单独校验严格时序
        if not (est.index[-1] <= info_cutoff < execution_date < holding_start):
            failed.append(f"{execution_date.date()}：时间顺序不满足 est_end≤info_cutoff<execution<holding_start，跳过")
            schedule.pop()
            continue
        log_rows.append({
            "est_start": est.index[0],
            "est_end": est.index[-1],
            "info_cutoff": info_cutoff,
            "signal_date": info_cutoff,
            "exec_date": execution_date,
            "execution_date": execution_date,
            "hold_return_start": holding_start,
            "holding_start": holding_start,
            "n_est": int(len(est)),
            "model": config.model,
            "cov_method": config.cov_method,
            "status": res.status,
            "solver": res.diagnostics.get("求解器", "—"),
            "predicted_vol": res.predicted_vol,
            "constraint_check": str(res.diagnostics.get("约束检查", "")),
            "constraints_satisfied": bool(
                res.diagnostics.get("约束检查", {}).get("全部约束满足", False)
            ),
            "failure_category": res.failure_category,
            **{f"w_{a}": float(res.weights[a]) for a in returns_df.columns},
        })

    if not schedule:
        return BacktestResult(
            ok=False, config=config,
            error="所有交易期的权重求解均失败，未生成任何回测结果（不返回等权兜底）：" + "；".join(failed[:5]),
        )

    # ---- 持有期终点统一重算（独立审查修复）----
    # 每个成功事件的持有收益区间 = [holding_start, 下一个**成功执行日**]，
    # 其中「下一个执行日」的收益仍由旧（漂移）权重实现，新权重从再下一期才生效，
    # 因此该日正确归属于**上一持有期**，相邻区间首尾相接、不重不漏。
    # 关键：失败调仓时策略继续持有上一成功事件的漂移权重，直到下一个成功执行日；
    # 若沿用「下一个计划决策日」作为 hold_end，失败决策之后的收益会落入无人评价的断档。
    success_j = [j for j, _w, _info in schedule]   # 与 log_rows 一一对应、按决策顺序递增
    for k, row in enumerate(log_rows):
        next_success = success_j[k + 1] if k + 1 < len(success_j) else n - 1
        row["hold_end"] = returns_df.index[next_success]

    try:
        daily = simulate_portfolio(returns_df, schedule, cost_bps=config.cost_bps)
    except Exception as exc:  # noqa: BLE001
        return BacktestResult(ok=False, config=config, error=f"组合模拟失败：{exc}")

    # ---- 用模拟结果回填交易事件字段（三类事件明确区分，避免计数差 1）----
    import json
    assets = list(returns_df.columns)
    trade_rows = daily[daily["is_trade"]].to_dict("index")

    def _wjson(values) -> str:
        return json.dumps({a: round(float(v), 8) for a, v in zip(assets, values)},
                          ensure_ascii=False)

    log = pd.DataFrame(log_rows)
    num_cols = ["turnover", "transaction_cost", "traded_notional", "buy_amount", "sell_amount",
                "realized_vol_fixed_weight", "realized_vol_gross_drift", "realized_vol_net",
                "forecast_error", "eval_n_obs"]
    bool_cols = ["is_initial_build", "is_rebalance", "is_trade"]
    str_cols = ["pre_trade_weights", "post_trade_weights", "target_weights"]
    for c in num_cols:
        log[c] = np.nan
    for c in bool_cols:
        log[c] = False
    for c in str_cols:
        log[c] = ""
    log["event_kind"] = "成功交易"
    log["失败原因"] = ""
    K = config.periods_per_year
    SHORT_HOLD_MIN_OBS = 21   # 短持有期警告阈值：有效观测 < 21 时实现风险估计样本不足
    short_hold_events = 0
    for i, row in log.iterrows():
        if pd.isna(row["execution_date"]) or pd.isna(row["holding_start"]):
            continue   # 失败事件行：无成交、无持有期，字段保持缺失
        d = trade_rows.get(row["execution_date"])
        if d is not None:
            pre = [d[f"pre_trade_{a}"] for a in assets]
            post = [d[f"post_trade_{a}"] for a in assets]
            log.at[i, "turnover"] = d["turnover"]
            log.at[i, "transaction_cost"] = d["cost"]
            log.at[i, "traded_notional"] = d["traded_notional"]
            log.at[i, "buy_amount"] = d["buy_amount"]
            log.at[i, "sell_amount"] = d["sell_amount"]
            log.at[i, "is_initial_build"] = bool(d["is_initial_build"])
            log.at[i, "is_rebalance"] = bool(d["is_rebalance"])
            log.at[i, "is_trade"] = bool(d["is_trade"])
            log.at[i, "pre_trade_weights"] = _wjson(pre)
            log.at[i, "post_trade_weights"] = _wjson(post)
            log.at[i, "target_weights"] = _wjson(post)
        # ---- 实现风险（**事后**计算，绝不参与该时点的权重估计）----
        # 三种口径，逐层区分权重漂移 / 再平衡 / 交易成本的影响：
        #   realized_vol_fixed_weight：持有期内**固定目标权重**组合收益的年化波动率
        #                              ——协方差预测误差的**主要评价对象**（与预测口径一致）；
        #   realized_vol_gross_drift ：持有期内**实际漂移权重**毛收益(r_port)的年化波动率
        #                              ——含权重漂移与再平衡时点影响、不含成本；
        #   realized_vol_net         ：持有期内**实际净收益**(r_net)的年化波动率
        #                              ——策略实际表现指标（额外含交易成本影响）。
        seg = returns_df.loc[row["holding_start"]:row["hold_end"]]
        n_obs = int(len(seg))
        log.at[i, "eval_n_obs"] = n_obs
        if n_obs >= 3:
            w_t = np.array([float(row[f"w_{a}"]) for a in assets], dtype=float)
            fw = seg.to_numpy(dtype=float) @ w_t                       # 固定目标权重收益
            log.at[i, "realized_vol_fixed_weight"] = float(np.std(fw, ddof=1) * np.sqrt(K))
            log.at[i, "realized_vol_gross_drift"] = float(
                daily.loc[seg.index, "r_port"].std(ddof=1) * np.sqrt(K))
            log.at[i, "realized_vol_net"] = float(
                daily.loc[seg.index, "r_net"].std(ddof=1) * np.sqrt(K))
        if 0 < n_obs < SHORT_HOLD_MIN_OBS:
            short_hold_events += 1
    # 与预测口径一致的预测误差：固定权重实现波动率 − 预测波动率（主口径）
    log["forecast_error"] = log["realized_vol_fixed_weight"] - log["predicted_vol"]
    log["forecast_error_net"] = log["realized_vol_net"] - log["predicted_vol"]  # 参考（含成本）

    # ---- 失败交易事件并入日志（策略继续持有漂移权重，但必须明确记录）----
    if failed_meta:
        frows = []
        for fm in failed_meta:
            frows.append({
                "execution_date": fm["execution_date"], "exec_date": fm["execution_date"],
                "signal_date": pd.NaT, "info_cutoff": pd.NaT,
                "est_start": pd.NaT, "est_end": pd.NaT,
                "hold_return_start": pd.NaT, "holding_start": pd.NaT, "hold_end": pd.NaT,
                "n_est": 0, "model": config.model, "cov_method": config.cov_method,
                "status": "failed", "solver": "—", "predicted_vol": np.nan,
                "failure_category": "数值求解失败/输入校验失败",
                "event_kind": fm["event_kind"], "失败原因": fm["reason"],
                "is_initial_build": False, "is_rebalance": False, "is_trade": False,
            })
        log = pd.concat([log, pd.DataFrame(frows)], ignore_index=True)
        log = log.sort_values("execution_date").reset_index(drop=True)

    ok_log = log[log["status"] == "success"]
    err = ok_log["forecast_error"].dropna().to_numpy(dtype=float)
    err_net = ok_log["forecast_error_net"].dropna().to_numpy(dtype=float)
    # 稳健性检验：**排除**有效观测 < 21 的短持有期事件后重算 MAE/RMSE。
    # 原 MAE/RMSE（全有效样本）保持不变，两者并列并明确标注，便于判断短样本是否主导结论。
    robust = ok_log[(ok_log["eval_n_obs"] >= SHORT_HOLD_MIN_OBS)]
    err_robust = robust["forecast_error"].dropna().to_numpy(dtype=float)
    n_obs_arr = ok_log["eval_n_obs"].dropna().to_numpy(dtype=float)
    n_too_short_for_vol = int(((n_obs_arr > 0) & (n_obs_arr < 3)).sum())  # 无有效实现波动率
    n_failed = len(failed_meta)
    n_failed_build = sum(1 for fm in failed_meta if fm["event_kind"] == "首次建仓失败")
    forecast_summary = {
        "事件数(成功)": int((log["status"] == "success").sum()),
        "失败交易事件": n_failed,
        "有效评价数": int(len(err)),
        "MAE": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "RMSE": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
        "平均偏差(实现−预测)": float(np.mean(err)) if len(err) else float("nan"),
        "误差标准差": float(np.std(err, ddof=1)) if len(err) > 1 else float("nan"),
        "MAE(净收益口径,参考)": float(np.mean(np.abs(err_net))) if len(err_net) else float("nan"),
        # ---- 稳健性：排除短样本（<21 个有效观测）后重算，原指标不变 ----
        "MAE(排除短样本,稳健性)": float(np.mean(np.abs(err_robust))) if len(err_robust) else float("nan"),
        "RMSE(排除短样本,稳健性)": float(np.sqrt(np.mean(err_robust ** 2))) if len(err_robust) else float("nan"),
        "有效评价数(排除短样本)": int(len(err_robust)),
        "排除的短样本事件数(3≤观测<21)": int(len(err) - len(err_robust)),
        "未达最小估计样本事件数(观测<3)": n_too_short_for_vol,
        "评价期限观测数(最小/中位/最大)": (
            f"{int(n_obs_arr.min())} / {float(np.median(n_obs_arr)):.0f} / {int(n_obs_arr.max())}"
            if len(n_obs_arr) else "—"),
        "口径": (
            "预测波动率 = 估计期协方差 Σ 与目标权重 w 的年化 wᵀΣw 开方（预测时点 = 信号日收盘后）；"
            "实现波动率（主评价口径）= 持有期**固定目标权重**假设收益标准差×√K（与预测口径一致，"
            "区别于实际漂移权重的策略收益 gross_drift / net）；"
            "评价起点 = holding_start、终点 = 下一成功执行日（该日收益归上一持有期），样本数 = eval_n_obs，"
            "相邻区间首尾相接不重不漏；MAE/RMSE 均基于该一致口径。"
            "「MAE/RMSE(排除短样本,稳健性)」仅剔除 3≤eval_n_obs<21 的事件后重算，"
            "**原 MAE/RMSE 保留不变**，两者并列供对照；观测<3 的事件无有效实现波动率、"
            "不参与任何 MAE/RMSE；若排除后无剩余事件则稳健性指标记为 NaN（不伪造、不以 0 代替）；"
            "realized_vol_net（含漂移/成本）仅作策略表现参考；"
            "实现风险只在成交后用于事后评价，不参与该时点权重估计"
        ),
        "可比性限制": (
            "评价期限 = 单次持有期长度，由再平衡频率决定（月度≈21、季度≈63、年度≈252 个交易日）；"
            "不同频率/不同窗口下每事件观测数不同，实现波动率的抽样误差与期限依赖不同，"
            "**MAE/RMSE 不可直接跨再平衡频率比较**；仅可在相同频率、相近评价期限内比较，"
            "或使用排除短样本后的稳健性指标与按频率分组的对比"
        ),
    }

    perf = R.performance_summary(
        daily["r_net"], price_index=returns_df.index,
        rf_annual=config.rf_annual, periods_per_year=config.periods_per_year,
    )
    warns = list(failed)
    if n_failed > 0:
        warns.append(
            f"⚠️ **本次回测非完全成功**：计划 {n_planned} 次交易事件，成功 "
            f"{n_planned - n_failed} 次、失败 {n_failed} 次（首次建仓失败 "
            f"{n_failed_build} 次、后续调仓失败 {n_failed - n_failed_build} 次）。"
            "失败时点的策略**继续持有漂移权重**直至下一成功调仓；"
            "与其他策略比较时请注意各策略失败次数是否相近，否则比较公平性受影响。"
        )
    if short_hold_events > 0:
        warns.append(
            f"{short_hold_events} 个持有期的有效观测数 < {SHORT_HOLD_MIN_OBS}："
            "短持有期的实现风险（年化波动率）估计**样本不足**、误差大，"
            f"对应预测误差评价仅供参考。已单独提供「MAE/RMSE(排除短样本,稳健性)」"
            f"（从主样本中排除 {int(len(err) - len(err_robust))} 个 3≤观测<21 的事件；"
            f"另有 {n_too_short_for_vol} 个观测<3 的事件本就无有效实现波动率）"
            "供对照，原全样本指标保留不变。"
        )
    warns.append(
        "⚠️ **预测误差可比性**：评价期限 = 单次持有期长度，由再平衡频率决定"
        "（月度≈21、季度≈63、年度≈252 个交易日）。不同频率/窗口下每事件观测数不同，"
        "实现波动率抽样误差与期限依赖不同，**MAE/RMSE 不可直接跨再平衡频率比较**；"
        "请在相同频率与相近评价期限内比较，或参考排除短样本后的稳健性指标。"
    )
    if config.cost_bps == 0:
        warns.append("当前交易成本设为 0，回测未计入任何交易费用")
    warns.extend([
        "执行假设：t 日收盘后生成信号 → t+1 日按收盘价近似成交 → t+2 日起新权重产生收益；"
        "只有日收盘数据，不虚构次日开盘价。",
        "首次建仓：起始资金为现金（**零收益现金假设**，现金收益率 0，不代表真实无风险利率），"
        "建仓日不享有目标资产收益，收盘建仓并计提初始交易成本。",
        "成本主口径 turnover = Σ|目标权重 − 交易前权重|（双边绝对权重变化），"
        "cost = turnover × 成本率(bp)/10000，买卖同费率，每个执行时点只扣一次；"
        "one_side_turnover / traded_notional / buy_amount / sell_amount 仅为参考口径，"
        "不参与成本计算。自融资简化假设：成本按比例从净值扣除，成交后权重仍精确等于目标权重。",
        "实现风险三种口径：固定目标权重（协方差预测误差的主评价对象）、"
        "漂移权重毛收益（含漂移/再平衡影响）、净收益（再含成本影响）。",
    ])
    return BacktestResult(
        ok=True, config=config, daily=daily, rebalance_log=log, performance=perf,
        oos_start=daily.index[0], oos_end=daily.index[-1], n_oos=len(daily),
        n_initial_build=int(daily["is_initial_build"].sum()),
        n_rebalance=int(daily["is_rebalance"].sum()),
        n_trade_events=int(daily["is_trade"].sum()),
        n_planned_trades=n_planned,
        n_failed_trades=n_failed,
        n_failed_build=n_failed_build,
        n_failed_rebalance=n_failed - n_failed_build,
        turnover_total=float(daily["turnover"].sum()),
        cost_total=float(daily["cost"].sum()),
        forecast_eval=log.loc[log["status"] == "success", [
            "execution_date", "signal_date", "predicted_vol",
            "realized_vol_fixed_weight", "realized_vol_gross_drift", "realized_vol_net",
            "forecast_error", "eval_n_obs"]].copy(),
        forecast_summary=forecast_summary,
        warnings=warns,
    )


def buy_and_hold(
    returns_df: pd.DataFrame,
    weights: dict | pd.Series,
    initial_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """买入持有（不做任何再平衡），与 `simulate_portfolio` 的"仅建仓"路径完全一致。

    首日为现金（零收益现金假设，r_port=0），收盘按目标权重全额建仓并计提初始成本；
    次日起按权重漂移公式逐日推进，不再交易。
    """
    w0 = np.array([float(weights[a]) for a in returns_df.columns], dtype=float)
    w0 = w0 / w0.sum()
    Rm = returns_df.to_numpy(dtype=float)
    p = len(w0)
    w = np.zeros(p, dtype=float)              # 现金
    navs, ws, rps, costs = [], [], [], []
    nav = 1.0
    for t in range(len(Rm)):
        if t == 0:
            r_p = 0.0
            cost = float(np.abs(w0 - w).sum()) * initial_cost_bps / 10000.0
            w = w0.copy()                     # 收盘建仓，下一期起生效
        else:
            r_p = float(w @ Rm[t])
            denom = 1.0 + r_p
            if abs(denom) < 1e-12:
                raise ValueError(f"组合收益 r_p={r_p} 导致净值归零，权重漂移无定义")
            w = w * (1 + Rm[t]) / denom
            cost = 0.0
        r_net = (1 + r_p) * (1 - cost) - 1
        nav *= 1 + r_net
        rps.append(r_p)
        costs.append(cost)
        navs.append(nav)
        ws.append(w.copy())
    idx = returns_df.index
    out = pd.DataFrame({"r_port": rps, "r_net": (np.array(rps) + 0.0), "nav": navs}, index=idx)
    out["r_net"] = [((1 + a) * (1 - c) - 1) for a, c in zip(rps, costs)]
    out["drawdown"] = R.drawdown_series(out["nav"])
    for i, a in enumerate(returns_df.columns):
        out[f"actual_{a}"] = [x[i] for x in ws]
        out[f"target_{a}"] = float(w0[i])
    out["cost"] = costs
    out["turnover"] = [float(np.abs(w0).sum()) if t == 0 else 0.0 for t in range(len(Rm))]
    out["is_initial_build"] = [t == 0 for t in range(len(Rm))]
    out["is_rebalance"] = False
    out["is_trade"] = out["is_initial_build"]
    return out


def compare_rebalance_frequencies(
    returns_df: pd.DataFrame,
    weights: dict | pd.Series,
    frequencies: dict[str, int] | None = None,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """固定目标权重下不同再平衡频率的表现（DR 模块 / 敏感性分析共用）。

    所有频率都从**同一个首日现金建仓**开始，走同一模拟器，
    保证只有"后续再平衡频率"不同、区间与成本口径一致。
    """
    freqs = frequencies or {
        "不再平衡（买入持有）": 0, "月度（21交易日）": 21,
        "季度（63交易日）": 63, "半年度（126交易日）": 126,
        "年度（252交易日）": 252,
    }
    w = np.array([float(weights[a]) for a in returns_df.columns], dtype=float)
    w = w / w.sum()
    rows = {}
    for label, every in freqs.items():
        try:
            if every == 0:
                d = buy_and_hold(returns_df, weights, initial_cost_bps=cost_bps)
            else:
                sched = [(j, w.copy(), {}) for j in range(0, len(returns_df) - 1, every)]
                d = simulate_portfolio(returns_df, sched, cost_bps=cost_bps)
            perf = R.performance_summary(d["r_net"], price_index=returns_df.index)
            rows[label] = {
                "年化收益(实际CAGR)": perf["实际CAGR(按实际年数)"],
                "年化波动率": perf["年化波动率"],
                "夏普比率": perf["夏普比率"],
                "最大回撤": perf["最大回撤"],
                "累计换手": float(d["turnover"].sum()),
                "建仓次数": int(d["is_initial_build"].sum()),
                "后续再平衡次数": int(d["is_rebalance"].sum()),
                "总交易事件次数": int(d["is_trade"].sum()),
                "累计成本": float(d["cost"].sum()),
                "调仓次数": int(d["is_trade"].sum()),
                "期末净值": perf["期末净值"],
                "样本外天数": len(d),
            }
        except Exception as exc:  # noqa: BLE001 —— 单个频率失败不影响其他频率
            rows[label] = {"错误": str(exc)}
    return pd.DataFrame(rows).T
