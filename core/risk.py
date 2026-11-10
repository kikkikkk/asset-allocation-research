# -*- coding: utf-8 -*-
"""V2.1 P4：最小方差模型的权重集中度、风险贡献与权重上限情景。

**只做诊断，不做风险平价优化**（风险平价等仍是预留接口）。

风险贡献（Σ 为年化协方差矩阵）
    σ_p = sqrt(wᵀΣw)
    MRC_i = (Σw)_i / σ_p              边际风险贡献
    RC_i  = w_i · MRC_i               总风险贡献，Σ_i RC_i = σ_p
    VC_i  = w_i · (Σw)_i             方差贡献，  Σ_i VC_i = σ_p²
资金权重 w_i 与风险贡献 RC_i 是不同概念：低波动资产常有高资金权重、
低风险贡献；高波动资产可能资金权重小但风险贡献大。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.models import Constraints, ModelResult, solve_model
from core.covariance import CovarianceResult


def concentration_metrics(weights: pd.Series | dict) -> dict:
    """权重集中度：HHI = Σw²；有效资产数 = 1/HHI；最大单资产权重。"""
    w = pd.Series(weights, dtype=float)
    if w.isna().any():
        return {"error": "权重含 NaN，无法计算集中度"}
    w = w.clip(lower=0)
    s = float(w.sum())
    if s <= 0:
        return {"error": "权重和 ≤ 0，无法计算集中度"}
    wn = w / s
    hhi = float((wn ** 2).sum())
    return {
        "最大单资产权重": float(wn.max()),
        "HHI": hhi,
        "有效资产数": float(1.0 / hhi) if hhi > 0 else float("nan"),
        "非零权重资产数": int((wn > 1e-8).sum()),
        "资产数": int(len(wn)),
        "口径": "HHI=Σw²（∈[1/N,1]）；有效资产数=1/HHI；与风险贡献不同",
    }


def risk_contributions(weights: pd.Series | dict, cov_ann: pd.DataFrame) -> dict:
    """计算 σ_p、MRC、RC、VC 并验证两条恒等式。"""
    ids = list(cov_ann.columns)
    w = pd.Series({a: float(pd.Series(weights).get(a, 0.0)) for a in ids}, index=ids)
    C = cov_ann.to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    var_p = float(wv @ C @ wv)
    sigma_p = float(np.sqrt(max(var_p, 0.0)))
    sw = C @ wv
    if sigma_p < 1e-14:
        return {
            "sigma_p": sigma_p, "var_p": var_p, "frame": pd.DataFrame(index=ids),
            "sum_RC": float("nan"), "sum_VC": float(var_p),
            "identity_RC": False, "identity_VC": bool(abs(float(wv @ sw) - var_p) < 1e-12),
            "warning": "组合波动率≈0（零波动/完全对冲），MRC 与风险贡献占比无定义，未强行输出",
        }
    mrc = sw / sigma_p
    rc = wv * mrc
    vc = wv * sw
    frame = pd.DataFrame({
        "资金权重": wv,
        "边际风险贡献MRC": mrc,
        "风险贡献RC": rc,
        "风险贡献占比": rc / sigma_p,
        "方差贡献VC": vc,
        "方差贡献占比": vc / var_p if var_p > 0 else np.nan,
    }, index=ids)
    return {
        "sigma_p": sigma_p, "var_p": var_p, "frame": frame,
        "sum_RC": float(rc.sum()), "sum_VC": float(vc.sum()),
        "identity_RC": bool(abs(float(rc.sum()) - sigma_p) < 1e-9),
        "identity_VC": bool(abs(float(vc.sum()) - var_p) < 1e-12),
        "warning": "",
        "note": "ΣRC=σ_p、ΣVC=σ_p² 已逐项校验；资金权重 ≠ 风险贡献",
    }


def weight_cap_curve(
    returns: pd.DataFrame,
    cov: CovarianceResult,
    constraints: Constraints,
    caps=(0.5, 0.7, 0.9, 1.0),
) -> pd.DataFrame:
    """不同单资产权重上限下的最小方差解（含不可行情景的明确标记）。"""
    rows = []
    n = len(returns.columns)
    for cap in caps:
        c = Constraints(
            min_weight=min(constraints.min_weight, cap),
            max_weight=cap,
            class_caps=dict(constraints.class_caps),
            asset_class_map=dict(constraints.asset_class_map),
        )
        if cap * n < 1 - 1e-12:
            rows.append({
                "单资产上限": cap, "状态": "不可行",
                "原因": f"{n} 个资产 × 上限 {cap:.0%} < 100%，无法满足 Σw=1",
                "预测年化波动率": np.nan, "最小方差目标": np.nan,
                "最大单资产权重": np.nan, "有效资产数": np.nan,
            })
            continue
        res = solve_model("最小方差", returns, cov, c)
        if res.status != "success":
            rows.append({
                "单资产上限": cap, "状态": "失败",
                "原因": f"{res.failure_category}：{res.message}",
                "预测年化波动率": np.nan, "最小方差目标": np.nan,
                "最大单资产权重": np.nan, "有效资产数": np.nan,
            })
            continue
        cm = concentration_metrics(res.weights)
        rows.append({
            "单资产上限": cap, "状态": "成功",
            "原因": res.diagnostics.get("求解器", ""),
            "预测年化波动率": res.predicted_vol,
            "最小方差目标": res.predicted_variance,
            "最大单资产权重": cm["最大单资产权重"],
            "有效资产数": cm["有效资产数"],
            **{f"w_{a}": float(res.weights[a]) for a in returns.columns},
        })
    return pd.DataFrame(rows)


def model_diagnostics(
    res: ModelResult, cov: CovarianceResult, constraints: Constraints
) -> dict:
    """最小方差模型的一次性完整诊断（页面3/6/导出共用）。"""
    if res.status != "success":
        return {
            "状态": "failed", "失败类别": res.failure_category,
            "原因": res.message, "约束定义": constraints.describe(),
        }
    conc = concentration_metrics(res.weights)
    rc = risk_contributions(res.weights, cov.matrix)
    eig = np.linalg.eigvalsh(cov.matrix.to_numpy(dtype=float))
    return {
        "状态": "success",
        "模型": res.model_name,
        "求解器": res.diagnostics.get("求解器", ""),
        "权重": res.weights.to_dict(),
        "权重和": float(res.weights.sum()),
        "预测年化方差": res.predicted_variance,
        "预测年化波动率": res.predicted_vol,
        "集中度": conc,
        "风险贡献": rc,
        "约束检查": res.diagnostics.get("约束检查", {}),
        "约束定义": constraints.describe(),
        "协方差最小特征值": float(eig.min()),
        "协方差条件数": float(eig.max() / eig.min()) if eig.min() > 1e-14 else float("inf"),
        "与精确解方差差": res.diagnostics.get("与精确解方差差", "—"),
        "估计窗口": f"{res.est_start.date()} ~ {res.est_end.date()}" if res.est_start is not None else "",
    }
