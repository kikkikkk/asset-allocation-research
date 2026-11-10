# -*- coding: utf-8 -*-
"""P3 统一配置模型接口 + 等权 / 自定义 / 最小方差（N 资产）。

所有模型接受：资产收益矩阵、协方差估计结果、约束、参数；
统一输出 ModelResult（目标权重、模型名、参数、估计区间、求解状态、预测风险、诊断信息）。

**最小方差求解的可靠性**
- 主求解器：scipy.optimize.minimize(SLSQP)（若 scipy 可用）；
- 精确基准/无依赖回退：**主动集枚举精确解**（枚举所有支撑集，解等式约束 QP 并校验 KKT），
  N ≤ 10 时可用；同时作为 SLSQP 结果的交叉验证基准；
- 两者均失败 → status="failed"、weights 为 NaN、明确失败原因，
  **绝不静默返回等权**。

预留接口（本轮仅占位，调用会抛 NotImplementedError，UI 明确标注"未实现"）：
风险平价、最大夏普、最大分散度、目标波动率。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from core.covariance import CovarianceResult

try:  # scipy 可选：缺失时自动使用精确主动集求解器
    from scipy.optimize import minimize as _scipy_minimize
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


@dataclass
class Constraints:
    """组合约束（V2.1：**所有求解路径共用同一对象**）。

    默认：完全投资、不做空、不使用杠杆。
    `asset_class_map` 属于约束定义的一部分，不再由某个页面临时注入，
    以保证静态模型页、滚动回测、敏感性分析、导出结果得到完全相同的约束。
    """

    min_weight: float = 0.0
    max_weight: float = 1.0
    class_caps: dict[str, tuple[float, float]] = field(default_factory=dict)
    asset_class_map: dict[str, str] = field(default_factory=dict)
    allow_short: bool = False
    leverage: bool = False

    def validate(self, n_assets: int, asset_ids=None) -> None:
        """求解前的可行性检查；不可行直接抛错，不静默放松约束。"""
        if self.allow_short or self.leverage:
            raise ValueError("本版本不支持做空与杠杆（约束固定为 long-only、权重和=1）")
        if self.min_weight < 0:
            raise ValueError("min_weight 不得为负（不做空约束）")
        if not (0 <= self.min_weight <= self.max_weight <= 1):
            raise ValueError("权重上下限需满足 0 ≤ min ≤ max ≤ 1")
        if self.min_weight * n_assets > 1 + 1e-12:
            raise ValueError(
                f"单资产下限 {self.min_weight:.2f} × {n_assets} 个资产 > 1，约束不可行")
        if self.max_weight * n_assets < 1 - 1e-12:
            raise ValueError(
                f"单资产上限 {self.max_weight:.2f} × {n_assets} 个资产 < 1，约束不可行"
                "（例如 3 个资产各 20% 上限无法达到权重和 100%）")
        if not self.class_caps:
            return
        if not self.asset_class_map:
            raise ValueError(
                "已启用资产类别约束，但缺少资产类别映射 asset_class_map；"
                "**不得静默跳过类别约束**，请补齐映射或关闭类别约束。")
        ids = list(asset_ids) if asset_ids is not None else list(self.asset_class_map)
        missing = [a for a in ids if a not in self.asset_class_map]
        if missing:
            raise ValueError(f"资产类别映射缺少以下资产：{missing}")
        present = {self.asset_class_map.get(a) for a in ids}
        for cls, (lo, hi) in self.class_caps.items():
            if not (0 <= lo <= hi <= 1):
                raise ValueError(f"资产类别 {cls} 的上下限不合法")
            if cls not in present:
                raise ValueError(f"类别约束 {cls} 在当前资产池中没有任何资产，约束不成立")
            k = sum(1 for a in ids if self.asset_class_map.get(a) == cls)
            if k * self.max_weight < lo - 1e-12:
                raise ValueError(
                    f"类别 {cls} 下限 {lo:.2f} 不可行：{k} 个资产、单资产上限 {self.max_weight:.2f}")
            if k * self.min_weight > hi + 1e-12:
                raise ValueError(
                    f"类别 {cls} 上限 {hi:.2f} 与单资产下限 {self.min_weight:.2f} 冲突（{k} 个资产）")
        capped = set(self.class_caps)
        capacity = sum(
            min(hi, sum(1 for a in ids if self.asset_class_map.get(a) == cls) * self.max_weight)
            for cls, (_lo, hi) in self.class_caps.items()
        )
        capacity += sum(1 for a in ids if self.asset_class_map.get(a) not in capped) * self.max_weight
        lo_total = sum(lo for lo, _hi in self.class_caps.values())
        if lo_total > 1 + 1e-12:
            raise ValueError(f"各类别权重下限之和 {lo_total:.2f} > 1，约束不可行")
        if capacity < 1 - 1e-12:
            raise ValueError(
                f"类别上限与单资产上限组合后最大可投资权重仅 {capacity:.2f} < 1，约束不可行")

    def describe(self) -> dict:
        return {
            "完全投资": "Σw = 1", "不做空": not self.allow_short,
            "无杠杆": not self.leverage,
            "单资产下限": self.min_weight, "单资产上限": self.max_weight,
            "类别约束": {k: {"下限": v[0], "上限": v[1]} for k, v in self.class_caps.items()},
            "资产类别映射": dict(self.asset_class_map),
        }


@dataclass
class ModelResult:
    model_name: str
    weights: pd.Series                 # index = asset_id
    status: str                        # "success" | "failed"
    message: str
    params: dict = field(default_factory=dict)
    est_start: pd.Timestamp | None = None
    est_end: pd.Timestamp | None = None
    as_of: pd.Timestamp | None = None  # 信息截止（权重计算所用最后一条数据日期）
    predicted_variance: float = float("nan")
    predicted_vol: float = float("nan")
    failure_category: str = ""         # 输入数据不足/协方差矩阵异常/理论不可行/数值求解失败
    diagnostics: dict = field(default_factory=dict)

    def constraint_check(self, c: Constraints) -> dict:
        w = self.weights
        base = {
            "权重和": float(w.sum()),
            "权重和=1": bool(abs(float(w.sum()) - 1) < 1e-7),
            "最小权重": float(w.min()),
            "最大权重": float(w.max()),
            "非负": bool((w >= -1e-10).all()),
            "满足上限": bool((w <= c.max_weight + 1e-8).all()),
            "满足下限": bool((w >= c.min_weight - 1e-8).all()),
            "无NaN": bool(w.notna().all()),
        }
        classes = {}
        for cls, (lo, hi) in c.class_caps.items():
            ids = [a for a in w.index if c.asset_class_map.get(a) == cls]
            val = float(w.reindex(ids).sum()) if ids else 0.0
            classes[cls] = {
                "资产数": len(ids), "权重": val, "下限": lo, "上限": hi,
                "满足": bool(lo - 1e-8 <= val <= hi + 1e-8),
            }
        base["类别约束"] = classes
        base["类别约束全部满足"] = all(v["满足"] for v in classes.values()) if classes else True
        base["全部约束满足"] = bool(
            base["权重和=1"] and base["非负"] and base["满足上限"]
            and base["满足下限"] and base["无NaN"] and base["类别约束全部满足"]
        )
        return base


class AllocationModel(ABC):
    """统一模型接口：新增模型只需实现 solve，不需要改动回测/图表/导出。"""

    name: str = "基类"
    implemented: bool = True

    @abstractmethod
    def solve(
        self,
        returns: pd.DataFrame,
        cov: CovarianceResult,
        constraints: Constraints,
        **params,
    ) -> ModelResult:
        ...


def _envelope(cov: CovarianceResult, constraints: Constraints, params: dict) -> dict:
    return {
        "est_start": cov.window_start,
        "est_end": cov.window_end,
        "as_of": cov.as_of,
        "params": {
            "cov_method": cov.method,
            "n_obs": cov.n_obs,
            "min_weight": constraints.min_weight,
            "max_weight": constraints.max_weight,
            "class_caps": constraints.class_caps,
            "asset_class_map": dict(constraints.asset_class_map),
            "constraint_describe": constraints.describe(),
            **params,
        },
    }


class EqualWeightModel(AllocationModel):
    name = "等权（1/N）"

    def solve(self, returns, cov, constraints, **params) -> ModelResult:
        ids = list(returns.columns)
        n = len(ids)
        w = pd.Series(1.0 / n, index=ids)
        if (w < constraints.min_weight - 1e-12).any() or (w > constraints.max_weight + 1e-12).any():
            return ModelResult(
                self.name, pd.Series(np.nan, index=ids), "failed",
                "等权权重不满足单资产上下限约束", **_envelope(cov, constraints, params),
            )
        var = float(w.to_numpy() @ cov.matrix.to_numpy() @ w.to_numpy())
        return ModelResult(
            self.name, w, "success", "等权组合，无需优化求解",
            predicted_variance=var, predicted_vol=float(np.sqrt(max(var, 0))),
            diagnostics={"求解器": "解析（1/N）"}, **_envelope(cov, constraints, params),
        )


class CustomWeightModel(AllocationModel):
    name = "自定义目标权重"

    def solve(self, returns, cov, constraints, weights: dict | None = None, **params) -> ModelResult:
        ids = list(returns.columns)
        if not weights:
            return ModelResult(self.name, pd.Series(np.nan, index=ids), "failed",
                               "未提供自定义权重", **_envelope(cov, constraints, params))
        w = pd.Series({i: float(weights.get(i, 0.0)) for i in ids}, index=ids)
        if abs(float(w.sum()) - 1) > 1e-6:
            return ModelResult(self.name, pd.Series(np.nan, index=ids), "failed",
                               f"自定义权重之和为 {w.sum():.6f}，必须等于 1",
                               **_envelope(cov, constraints, params))
        if (w < -1e-12).any() or (w < constraints.min_weight - 1e-12).any() or (w > constraints.max_weight + 1e-12).any():
            return ModelResult(self.name, pd.Series(np.nan, index=ids), "failed",
                               "自定义权重存在负值或超出上下限约束",
                               **_envelope(cov, constraints, params))
        var = float(w.to_numpy() @ cov.matrix.to_numpy() @ w.to_numpy())
        return ModelResult(
            self.name, w, "success", "用户指定权重，不参与优化求解（无估计窗口依赖）",
            predicted_variance=var, predicted_vol=float(np.sqrt(max(var, 0))),
            diagnostics={"求解器": "用户指定"}, **_envelope(cov, constraints, params),
        )


# ---------------- 最小方差求解器 ----------------
def exact_active_set_min_variance(cov: np.ndarray, max_n: int = 10) -> tuple[np.ndarray, float, dict]:
    """枚举支撑集的**精确** long-only 最小方差解（无外部优化依赖，可作基准/回退）。

    对每个非空支撑集 S：解 w_S = Σ_S^{-1}1 / (1'Σ_S^{-1}1)，
    检查 w_S ≥ 0 与 KKT 条件 (Σw)_i ≥ w'Σw（对所有 i∉S），取目标最小者。
    """
    n = cov.shape[0]
    if n > max_n:
        raise ValueError(f"主动集精确解最多支持 {max_n} 个资产（当前 {n}），请改用 SLSQP")
    best_w, best_var, best_k = None, np.inf, 0
    for k in range(1, n + 1):
        for S in combinations(range(n), k):
            sub = cov[np.ix_(S, S)]
            try:
                inv1 = np.linalg.pinv(sub) @ np.ones(k)
                denom = float(np.ones(k) @ inv1)
                if denom <= 1e-16:
                    continue
                ws = inv1 / denom
                if (ws < -1e-10).any():
                    continue
                w = np.zeros(n)
                w[list(S)] = ws
                var = float(w @ cov @ w)
                grad = cov @ w
                if (grad[np.setdiff1d(np.arange(n), list(S))] < var - 1e-10).any():
                    continue  # KKT 不满足：集合外资产加入可降低方差
                if var < best_var - 1e-18:
                    best_w, best_var, best_k = w, var, k
            except np.linalg.LinAlgError:
                continue
    if best_w is None:
        raise ValueError("主动集枚举未找到可行解（协方差矩阵可能病态）")
    return best_w, best_var, {"active_set_size": best_k, "subsets_checked": 2**n - 1}


class MinVarianceModel(AllocationModel):
    """min wᵀΣw, s.t. Σw = 1, w ≥ 0（默认完全投资、不做空、无杠杆）。"""

    name = "最小方差"

    def solve(self, returns, cov, constraints, solver: str = "auto", **params) -> ModelResult:
        ids = list(returns.columns)
        n = len(ids)
        env = _envelope(cov, constraints, {**params, "solver": solver})
        # V2.1：类别映射优先取自统一约束对象；params 仅为向后兼容的 fallback
        amap = constraints.asset_class_map or params.get("asset_class_map") or {}
        if constraints.class_caps and amap and not constraints.asset_class_map:
            constraints = Constraints(**{**constraints.__dict__, "asset_class_map": dict(amap)})
        # 分类 1：理论不可行 / 约束定义错误（求解前明确报错，不静默放松）
        try:
            constraints.validate(n, asset_ids=ids)
        except ValueError as exc:
            return ModelResult(
                self.name, pd.Series(np.nan, index=ids), "failed",
                f"约束不可行或约束定义不完整：{exc}",
                failure_category="理论不可行", **env,
            )
        # 分类 2：输入数据不足（含单资产边界）
        if n < 2 or len(returns) < 3:
            return ModelResult(
                self.name, pd.Series(np.nan, index=ids), "failed",
                f"输入数据不足：{n} 个资产 / {len(returns)} 条收益样本（至少需 2 资产与 3 条观测）",
                failure_category="输入数据不足", **env,
            )
        # 分类 3：协方差矩阵异常
        try:
            C = cov.matrix.to_numpy(dtype=float)
            if C.shape != (n, n):
                raise ValueError("协方差矩阵维度与资产数不一致")
            if not np.isfinite(C).all():
                raise ValueError("协方差矩阵含非有限值")
            if not np.allclose(C, C.T, atol=1e-12):
                raise ValueError("协方差矩阵不对称")
            eig = np.linalg.eigvalsh(C)
            if eig.min() < -1e-10:
                raise ValueError(f"协方差矩阵非正半定（最小特征值 {eig.min():.3e}）")
            psd_projection = ""
            if eig.min() < 0:
                # 极小负特征值视为浮点误差，投影到 PSD 并记录，不静默改数
                vals, vecs = np.linalg.eigh((C + C.T) / 2)
                C = (vecs * np.maximum(vals, 0.0)) @ vecs.T
                C = (C + C.T) / 2
                psd_projection = f"最小特征值 {eig.min():.3e} 视为浮点误差，已投影至 PSD"
                eig = np.linalg.eigvalsh(C)
            cond = float(eig.max() / eig.min()) if eig.min() > 1e-14 else float("inf")
            diag_hint = (f"；条件数 {cond:.2e} 极大（近奇异/重复资产），解可能不唯一"
                         if cond > 1e12 else "")
        except (ValueError, np.linalg.LinAlgError) as exc:
            return ModelResult(
                self.name, pd.Series(np.nan, index=ids), "failed",
                f"协方差矩阵异常：{exc}", failure_category="协方差矩阵异常", **env,
            )

        bounds = [(constraints.min_weight, constraints.max_weight)] * n
        eq = [{"type": "eq", "fun": lambda w: float(w.sum() - 1),
               "jac": lambda w: np.ones_like(w)}]
        # 资产类别权重上下限：映射来自**统一约束对象**，静态/回测/敏感性路径完全一致
        class_cons = []
        if constraints.class_caps:
            for cls, (lo, hi) in constraints.class_caps.items():
                idx = [i for i, a in enumerate(ids) if amap.get(a) == cls]
                if not idx:
                    continue
                vec = np.zeros(n); vec[idx] = 1.0
                class_cons.append({"type": "ineq", "fun": (lambda w, v=vec, h=hi: h - float(v @ w)),
                                   "jac": lambda w, v=vec: -v})
                class_cons.append({"type": "ineq", "fun": (lambda w, v=vec, l=lo: float(v @ w) - l),
                                   "jac": lambda w, v=vec: v})

        diag: dict = {
            "约束": "Σw=1, w≥0, 不做空, 无杠杆",
            "约束定义": constraints.describe(),
            "协方差条件数": cond,
            "近奇异提示": diag_hint,
            "PSD投影": psd_projection or "无",
            "类别映射来源": "统一 Constraints.asset_class_map",
        }
        w, var, solver_used, failures = None, np.nan, "", []

        use_scipy = (solver in ("auto", "slsqp")) and _HAS_SCIPY
        if use_scipy:
            try:
                x0 = np.full(n, 1.0 / n)
                # 初值需满足上下限
                x0 = np.clip(x0, constraints.min_weight, constraints.max_weight)
                x0 = x0 / x0.sum()
                res = _scipy_minimize(
                    lambda w: float(w @ C @ w), x0, jac=lambda w: 2 * C @ w,
                    method="SLSQP", bounds=bounds,
                    constraints=eq + class_cons,
                    options={"ftol": 1e-14, "maxiter": 1000},
                )
                if res.success and np.isfinite(res.x).all() and abs(res.x.sum() - 1) < 1e-7:
                    w = np.asarray(res.x, dtype=float)
                    var = float(w @ C @ w)
                    solver_used = "scipy.optimize.minimize(SLSQP)"
                    diag["SLSQP"] = {"success": bool(res.success), "message": str(res.message),
                                     "iterations": int(getattr(res, "nit", -1))}
                else:
                    failures.append(f"SLSQP 未收敛或约束不满足：{res.message}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"SLSQP 异常：{type(exc).__name__}: {exc}")
        elif solver == "slsqp" and not _HAS_SCIPY:
            failures.append("请求 SLSQP 但 scipy 不可用")

        # 精确主动集解：作为无 scipy 的回退 **以及** SLSQP 的交叉验证基准
        exact_w = exact_var = None
        try:
            if not constraints.class_caps and constraints.min_weight == 0.0 and constraints.max_weight == 1.0:
                exact_w, exact_var, exact_diag = exact_active_set_min_variance(C)
                diag["主动集精确解"] = {"方差": exact_var, **exact_diag}
            else:
                diag["主动集精确解"] = "有额外上下限/类别约束，跳过（仅 SLSQP 适用）"
        except Exception as exc:  # noqa: BLE001
            diag["主动集精确解"] = f"不可用：{exc}"

        if w is None:
            if exact_w is not None:
                w, var, solver_used = exact_w, exact_var, "主动集枚举精确解"
            else:
                return ModelResult(
                    self.name, pd.Series(np.nan, index=ids), "failed",
                    "优化失败，且无可用精确回退；**未返回等权权重**。原因：" + "；".join(failures),
                    failure_category="数值求解失败", diagnostics=diag, **env,
                )
        else:
            if exact_w is not None:
                gap = float(w @ C @ w - exact_var)
                diag["与精确解方差差"] = gap
                diag["与精确解权重最大差"] = float(np.max(np.abs(w - exact_w)))
                if gap < -1e-10:
                    failures.append("SLSQP 目标值低于精确基准，疑似数值异常，已改用精确解")
                    w, var, solver_used = exact_w, exact_var, "主动集枚举精确解"

        w = np.clip(w, 0.0, 1.0)
        if w.sum() <= 0:
            return ModelResult(self.name, pd.Series(np.nan, index=ids), "failed",
                               "优化结果全为零权重", failure_category="数值求解失败",
                               diagnostics=diag, **env)
        w = w / w.sum()
        var = float(w @ C @ w)
        result = ModelResult(
            self.name, pd.Series(w, index=ids), "success",
            f"求解成功（{solver_used}）",
            predicted_variance=var, predicted_vol=float(np.sqrt(max(var, 0))),
            diagnostics={**diag, "求解器": solver_used, "告警": failures or "无"}, **env,
        )
        check = result.constraint_check(constraints)
        result.diagnostics["约束检查"] = check
        result.diagnostics["约束定义"] = constraints.describe()
        if not check["全部约束满足"]:
            result.status = "failed"
            result.message = (
                "优化结果未通过约束检查（含类别约束），不返回违反约束的权重；"
                f"失败项：{[k for k, v in check.items() if isinstance(v, bool) and not v]}"
            )
            result.weights = pd.Series(np.nan, index=ids)
            result.failure_category = "数值求解失败"
        return result


# ---------------- 预留接口（本轮未实现） ----------------
class _ReservedModel(AllocationModel):
    implemented = False

    def solve(self, returns, cov, constraints, **params) -> ModelResult:
        raise NotImplementedError(
            f"模型「{self.name}」为本轮预留接口，尚未实现；请勿在界面选择用于研究结论。"
        )


class RiskParityModel(_ReservedModel):
    name = "风险平价（预留·未实现）"


class MaxSharpeModel(_ReservedModel):
    name = "最大夏普（预留·未实现）"


class MaxDiversificationModel(_ReservedModel):
    name = "最大分散度（预留·未实现）"


MODEL_REGISTRY: dict[str, type[AllocationModel]] = {
    "等权（1/N）": EqualWeightModel,
    "自定义目标权重": CustomWeightModel,
    "最小方差": MinVarianceModel,
    "风险平价（预留·未实现）": RiskParityModel,
    "最大夏普（预留·未实现）": MaxSharpeModel,
    "最大分散度（预留·未实现）": MaxDiversificationModel,
}

IMPLEMENTED_MODELS = [k for k, v in MODEL_REGISTRY.items() if v.implemented]
RESERVED_MODELS = [k for k, v in MODEL_REGISTRY.items() if not v.implemented]


def solve_model(
    model_name: str,
    returns: pd.DataFrame,
    cov: CovarianceResult,
    constraints: Constraints,
    **params,
) -> ModelResult:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型：{model_name}")
    return MODEL_REGISTRY[model_name]().solve(returns, cov, constraints, **params)
