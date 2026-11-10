# -*- coding: utf-8 -*-
"""P3 协方差估计统一接口（N 资产）。

三种方法（V2.1 审查后保持算法不变）
------------------------------------
1. **sample**：样本协方差，ddof=1，×K 年化。
   - 需明确：历史窗口、有效观测数、数据频率、自由度(ddof=1)。
2. **ewma**：指数加权协方差，衰减系数 λ 用户可选（日频 0.94 仅为示例默认，非最优）。
   - 去均值：用估计窗口内样本均值（不是 0，也不是全样本均值）；
   - 加权：w_t ∝ λ^(T−1−t)（最近一期权重最大），归一化后加权外积；
   - 等价于递推 S_t = λS_{t−1} + (1−λ)(r_t−μ)(r_t−μ)' 的稳态解，
     **无需主观初始化矩阵**（避免初始化偏差；诊断里给出有效样本量）。
3. **ledoit_wolf**（V2 可复现基准，V2.1 **不替换算法**）：Ledoit–Wolf 型单参数收缩，
   目标 **μI（scaled identity）**，μ = trace(S)/N，S 为 MLE 协方差(1/n、按样本均值中心化)，
   Σ = δ·μI + (1−δ)·S，δ = clip(b²/d², 0, 1) 为**简化**收缩强度估计。

   **与可信标准实现的关系（V2.1 核实）**：
   - scikit-learn `sklearn.covariance.LedoitWolf` 的目标**同样是 μI**
     （源码 `_ledoit_wolf`: `(1−δ)S` 后对角加 `δ·μ`），不是 constant-correlation 目标；
   - 差别在于 δ：sklearn 用 Ledoit–Wolf (2004) 标准 π−ρ/γ 公式 `δ=clip(κ/n,0,1)`，
     本模块用简化的 `b²/d²`，两者数值不同（见 `tests/test_covariance_v21.py` 数值对照）；
   - 因此本方法应称为**简化的 μI 收缩基准**，不能称为"适用于所有资产的标准/最优 LW"；
   - 异方差资产池（股票方差 ~3% vs 债券 ~0.008%）下 μI 目标会明显拉平方差，
     **不保证优于**样本协方差，也不能因预测波动率更高就判定方法更差；
   - 如需标准 δ 公式或常数相关目标，应作为**独立命名**的新方法加入，
     不得在 `ledoit_wolf` 名称下偷偷替换算法。

所有方法统一返回 CovarianceResult（矩阵、估计窗口、方法、有效观测数、估计日期、诊断信息），
并校验对称性 / 有限值 / 正半定性；校验失败抛错，绝不静默返回单位矩阵或等权。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

COV_METHODS = ["sample", "ewma", "ledoit_wolf"]
COV_METHOD_LABELS = {
    "sample": "样本协方差（ddof=1）",
    "ewma": "EWMA 指数加权（λ 可调）",
    "ledoit_wolf": "Ledoit–Wolf μI 目标（简化 δ，可复现基准）",
}


@dataclass
class CovarianceResult:
    matrix: pd.DataFrame                 # 年化协方差，index/columns = asset_id
    method: str
    n_obs: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    as_of: pd.Timestamp                  # 信息截止日 = window_end（收盘后）
    periods_per_year: int = TRADING_DAYS
    diagnostics: dict = field(default_factory=dict)
    ok: bool = True
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "方法": self.method, "有效观测数": self.n_obs,
            "估计窗口": f"{self.window_start.date()} ~ {self.window_end.date()}",
            "信息截止日": str(self.as_of.date()), "年化频率K": self.periods_per_year,
            "状态": "成功" if self.ok else "失败", "说明": self.message,
            **{f"诊断_{k}": v for k, v in self.diagnostics.items()},
        }


def _validate(returns: pd.DataFrame, min_obs: int = 20) -> np.ndarray:
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("协方差估计需要至少 2 个资产的二维收益矩阵")
    if X.shape[0] < min_obs:
        raise ValueError(f"有效观测仅 {X.shape[0]} 条（要求 ≥ {min_obs}），不得偷偷缩短窗口")
    if not np.isfinite(X).all():
        raise ValueError("收益矩阵含 NaN/Inf，请先做数据质量校验")
    return X


def _validate_matrix(cov: np.ndarray, labels) -> pd.DataFrame:
    if not np.isfinite(cov).all():
        raise ValueError("协方差矩阵含非有限值")
    if not np.allclose(cov, cov.T, atol=1e-14):
        raise ValueError("协方差矩阵不对称")
    eig = np.linalg.eigvalsh(cov)
    out = pd.DataFrame(cov, index=labels, columns=labels)
    if eig.min() < -1e-12:
        raise ValueError(f"协方差矩阵非正半定（最小特征值 {eig.min():.3e}）")
    return out


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "sample",
    ewma_lambda: float = 0.94,
    periods_per_year: int = TRADING_DAYS,
    min_obs: int = 20,
) -> CovarianceResult:
    """估计年化协方差矩阵；returns 为**已按窗口切片**的期间简单收益矩阵。"""
    X = _validate(returns, min_obs)
    n, p = X.shape
    labels = list(returns.columns)
    start, end = pd.Timestamp(returns.index[0]), pd.Timestamp(returns.index[-1])
    diag: dict = {}

    if method == "sample":
        cov_d = np.cov(X, rowvar=False, ddof=1)
        diag.update({
            "自由度": "ddof=1（无偏样本协方差）",
            "频率": f"{periods_per_year} 期/年",
            "收缩强度": "—", "衰减系数": "—",
        })
        msg = "样本协方差（无偏，ddof=1）"
    elif method == "ewma":
        if not (0 < ewma_lambda < 1):
            raise ValueError(f"EWMA 衰减系数 λ 必须在 (0,1)，当前 {ewma_lambda}")
        ages = np.arange(n - 1, -1, -1)                  # 最近一期 age=0
        w = ewma_lambda ** ages
        w /= w.sum()
        mu = X.mean(axis=0)                              # 窗口内样本均值（去均值处理）
        Xc = X - mu
        cov_d = (Xc * w[:, None]).T @ Xc                 # 归一化加权外积
        eff_n = 1.0 / float((w ** 2).sum())              # 有效样本量
        diag.update({
            "衰减系数λ": ewma_lambda,
            "有效样本量": round(eff_n, 1),
            "均值处理": "估计窗口内样本均值去均值",
            "加权方式": "w_t ∝ λ^(T−1−t)，归一化；等价于递推稳态解，无主观初始化",
            "自由度": "加权（非 ddof）",
            "频率": f"{periods_per_year} 期/年",
            "收缩强度": "—",
            "窗口提示": "λ 决定记忆长度；当有效样本量远小于窗口长度时，延长窗口基本不改变估计",
        })
        msg = f"EWMA（λ={ewma_lambda}，有效样本量≈{eff_n:.0f}）"
    elif method == "ledoit_wolf":
        Xc = X - X.mean(axis=0, keepdims=True)
        S = Xc.T @ Xc / n                                # MLE 协方差
        mu = float(np.trace(S) / p)
        target = mu * np.eye(p)
        d2 = float(np.sum((S - target) ** 2))
        outer = np.einsum("ti,tj->tij", Xc, Xc)          # (n,p,p)
        diff = outer - S
        b2_bar = float(np.sum(diff ** 2) / (p * n ** 2))
        delta = 0.0 if d2 <= 1e-18 else float(min(max(b2_bar / d2, 0.0), 1.0))
        cov_d = delta * target + (1 - delta) * S
        diag.update({
            "收缩目标": "μI（scaled identity，μ=trace(S)/N）",
            "收缩强度δ": round(delta, 6),
            "d²": round(d2, 12), "b²_bar": round(b2_bar, 12),
            "自由度": "MLE（1/n）",
            "频率": f"{periods_per_year} 期/年",
            "衰减系数": "—",
            "局限": "股债波动率差异大时 μI 目标会拉平方差；不保证优于样本协方差；δ 为简化估计",
            "标准实现关系": "sklearn LedoitWolf 同为 μI 目标但用标准 π−ρ/γ 的 δ；本方法仅作可复现基准",
            "中心化": "按估计窗口内样本均值中心化；S 为 MLE(1/n)",
        })
        msg = f"Ledoit–Wolf 收缩（δ={delta:.4f}，目标 μI，简化 δ 基准）"
    else:
        raise ValueError(f"未知协方差估计方法：{method}")

    cov_ann = cov_d * periods_per_year
    frame = _validate_matrix(cov_ann, labels)
    eig = np.linalg.eigvalsh(frame.to_numpy())
    diag["最小特征值(年化)"] = float(eig.min())
    diag["条件数"] = float(eig.max() / eig.min()) if eig.min() > 1e-14 else float("inf")
    return CovarianceResult(
        matrix=frame, method=method, n_obs=int(n), window_start=start,
        window_end=end, as_of=end, periods_per_year=periods_per_year,
        diagnostics=diag, ok=True, message=msg,
    )


def covariance_difference(a: CovarianceResult, b: CovarianceResult) -> dict:
    """两种估计的矩阵差异诊断（Frobenius 范数、最大绝对差、权重影响在模型层计算）。"""
    diff = (a.matrix - b.matrix).to_numpy()
    return {
        "methods": f"{a.method} vs {b.method}",
        "frobenius_norm": float(np.linalg.norm(diff)),
        "max_abs_diff": float(np.abs(diff).max()),
        "diag_diff": dict(zip(a.matrix.index, np.diag(diff).tolist())),
    }
