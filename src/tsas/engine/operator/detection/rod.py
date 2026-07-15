"""
ROD 异常检测算子

算法逻辑源自对应公开实现的 rod.ROD, 按 TSA-Suite 算子规范重写。
不依赖 第三方库, 不依赖 sklearn. 算法完全用 numpy 实现.

核心思想 (Almardeny et al., 2020):
    1. 对每个 3D 子空间, 以几何中位数为旋转中心计算"成本" =
       |x|^3 * cos(gamma) * sin^2(gamma) (Rodrigues 旋转平行六面体体积);
    2. MAD 修正 z 分数作为 3D 子空间异常分数;
    3. 对 >3D 数据, 所有 C(d, 3) 个 3D 子空间分别求分数, 经 sigmoid
       压缩后取平均; 对 <3D 数据自动补零到 3D.

包含:
    - RODScorer: 直接评分器, 输出异常分数
    - RODDetector: 端到端检测器, 组合 RODScorer + PercentileDecider
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
)

__all__ = [
    'RODScorerConfig',
    'RODScorer',
    'RODDetectorConfig',
    'RODDetector',
]


# ============================================================================
# 算法子函数 (纯 numpy 实现)
# ============================================================================


def _mad(costs: np.ndarray) -> tuple[np.ndarray, float]:
    """基于中位数的绝对偏差 (MAD), 度量旋转代价的离散程度.

    公式: 修正 z 分数 = 0.6745 * |c - median| / median(|c - median|)

    Args:
        costs: 旋转代价数组.

    Returns:
        tuple: (修正 z 分数数组, 中位数).
    """
    median = float(np.median(costs))
    diff = np.abs(costs - median)
    mad = float(np.median(diff))
    if mad == 0:
        # 避免除零; 当所有 cost 相同时, z 分数全部归一化为 0
        return np.zeros_like(costs), median
    return 0.6745 * diff / mad, median


def _geometric_median(x: np.ndarray, eps: float = 1e-5, max_iter: int = 1000) -> np.ndarray:
    """使用 Weiszfeld 算法 (Vardi-Zhang) 计算多元几何 L1 中位数.

    Args:
        x: 数据点矩阵, 形状 (n_samples, n_features).
        eps: 收敛阈值.
        max_iter: 最大迭代次数.

    Returns:
        np.ndarray: 几何中位数向量.
    """
    # 使用均值初始化 (Vardi-Zhang 风格)
    pts = np.asarray(x, dtype=float)
    if pts.shape[0] == 0:
        return pts
    gm = pts.mean(axis=0)
    for _ in range(max_iter):
        # 各样本到当前 gm 的欧氏距离
        diff = pts - gm
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        # 距离为 0: 此时 gm 就是某个数据点
        if np.any(dist == 0):
            return gm
        inv = 1.0 / dist
        w = inv / inv.sum()
        gm_new = np.sum(w[:, None] * pts, axis=0)
        # 检测收敛
        step = np.sqrt(np.sum((gm_new - gm) ** 2))
        gm = gm_new
        if step < eps:
            break
    return gm


def _scale_angles(gammas: np.ndarray) -> np.ndarray:
    """将弧度角按 [0, pi/2] / (pi/2, pi] 两段分别线性缩放到 [0.001, 0.955] / [pi/2 + 0.001, 2.186].

    纯 numpy 实现, 无 sklearn 依赖.

    Args:
        gammas: 角度数组 (弧度).

    Returns:
        np.ndarray: 缩放后的角度数组.
    """
    q1 = np.pi / 2.0
    first_mask = gammas <= q1
    second_mask = ~first_mask
    g1_min, g1_max = 0.001, 0.955
    g2_min, g2_max = q1 + 0.001, 2.186
    out = np.empty_like(gammas, dtype=float)
    if np.any(first_mask):
        v = gammas[first_mask]
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            out[first_mask] = (v - vmin) / (vmax - vmin) * (g1_max - g1_min) + g1_min
        else:
            out[first_mask] = g1_min
    if np.any(second_mask):
        v = gammas[second_mask]
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            out[second_mask] = (v - vmin) / (vmax - vmin) * (g2_max - g2_min) + g2_min
        else:
            out[second_mask] = g2_min
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _robust_scale(x: np.ndarray, center: np.ndarray | None = None, scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """纯 numpy 版 RobustScaler: 中位数 + IQR.

    等价于 sklearn.preprocessing.RobustScaler() 但不依赖 sklearn.
    """
    if center is None or scale is None:
        center = np.median(x, axis=0)
        q1 = np.percentile(x, 25, axis=0)
        q3 = np.percentile(x, 75, axis=0)
        scale = q3 - q1
        scale[scale == 0] = 1.0  # 避免除零
    return (x - center) / scale, center, scale


def _rod_3d(
    x: np.ndarray,
    gm: np.ndarray | None = None,
    median: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """计算 3D 数据的 ROD 异常分数.

    Args:
        x: 三维数据, 形状 (n_samples, 3).
        gm: 几何中位数, None 时自动计算.
        median: MAD 中位数, None 时自动计算.

    Returns:
        tuple: (异常分数, 几何中位数, MAD中位数).
    """
    if gm is None:
        gm = _geometric_median(x)
    norm_gm = np.linalg.norm(gm)
    _x = x - gm
    v_norm = np.linalg.norm(_x, axis=1)
    # 计算样本向量与几何中位数向量的夹角 (gamma)
    if norm_gm == 0:
        # 退化情形: 几何中位数为零向量, 无法用 acos 计算 gamma
        # 此时取 gamma = pi/2 (即所有样本都"侧向"), 后续 cos(g)=0, sin(g)=1
        gammas = np.full(x.shape[0], np.pi / 2.0)
    else:
        dots = np.dot(_x, gm)
        cos_vals = np.clip(dots / (v_norm * norm_gm), -1.0, 1.0)
        gammas = np.arccos(cos_vals)
    gammas = _scale_angles(gammas)
    # 旋转平行六面体体积: v^3 * cos(g) * sin^2(g)
    costs = np.power(v_norm, 3) * np.cos(gammas) * np.square(np.sin(gammas))
    decision_scores, median = _mad(costs)
    if median is not None:
        median = median  # 保留传入的中位数, 便于 fit/predict 语义一致
    return decision_scores, gm, median


def _process_subspace(subspace: np.ndarray) -> np.ndarray:
    """对单个 3D 子空间运行 ROD, 返回 sigmoid 压缩后的分数."""
    scores, _, _ = _rod_3d(subspace)
    return _sigmoid(np.nan_to_num(scores, nan=0.0))


def _robust_rod(X: np.ndarray) -> np.ndarray:
    """>3D 数据的 ROD 计算: RobustScaler + 所有 C(d, 3) 子空间平均.

    Args:
        X: 输入数据, 形状 (n_samples, n_features), d > 3.

    Returns:
        np.ndarray: 异常分数, 形状 (n_samples,).
    """
    X_s, _, _ = _robust_scale(X)
    d = X_s.shape[1]
    subspaces = [X_s[:, idx] for idx in combinations(range(d), 3)]
    sub_scores = np.column_stack([_process_subspace(s) for s in subspaces])
    return sub_scores.mean(axis=1)


# ============================================================================
# ROD Scorer
# ============================================================================


class RODScorerConfig(BaseModel):
    """ROD 评分器实例参数"""
    model_config = {"frozen": True}


class RODScorer(SingleScorerMixin[None],
                 UnsupervisedNumericOperatorMixin[None],
                 NumericOperator[None, RODScorerConfig, None]):
    """ROD 直接评分器 (Robust Outlyingness)"""
    _LEARNED_STATE_FILE = '_learned_state.npz'

    @classmethod
    def name(cls) -> str:
        return "rod_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """训练阶段无状态: 实际计算在 decision_function 内."""
        # ROD 在 第三方库 实现中仅在决策函数里使用几何中位数和 cost,
        # 不显式缓存训练集状态. 我们也保持一致: fit 不计算, 仅确认输入可用.
        if x.ndim != 2:
            raise ValueError("ROD expects 2-D input (n_samples, n_features)")

    def _can_run(self) -> None:
        pass

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """计算每个样本的 ROD 异常分数."""
        X = np.asarray(x, dtype=float)
        n_features = X.shape[1]
        # < 3D 补零到 3D
        if n_features < 3:
            X = np.hstack([X, np.zeros((X.shape[0], 3 - n_features))])
            n_features = 3
        if n_features == 3:
            scores, _, _ = _rod_3d(X)
            return scores
        # > 3D: RobustScaler + 所有 3D 子空间 sigmoid 平均
        return _robust_rod(X)

    # ROD 是无状态算子, 不需要 _save_fit_state / _load_fit_state 特殊处理;
    # 默认基类行为即满足需求.


# ============================================================================
# ROD Detector
# ============================================================================


class RODDetectorConfig(BaseModel):
    """ROD 检测器实例参数"""
    model_config = {"frozen": True}
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class RODDetector(UnsupervisedNumericOperatorMixin[None],
                  BaseDeciderMixin[None],
                  NumericOperator[None, RODDetectorConfig, None]):
    """ROD 检测器 = RODScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "rod_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = RODScorer(config=RODScorerConfig())
        self._decider = PercentileDecider(config=PercentileDeciderConfig(
            percentile=self.config.percentile,
        ))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._scorer.fit(x)
        scores = self._scorer.run(x)
        self._decider.fit(scores)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        scores = self._scorer.run(x)
        labels, _ = self._decider.run(scores)
        return labels

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        self._scorer.save(path / "_scorer")
        self._decider.save(path / "_decider")

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        self._scorer = RODScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
