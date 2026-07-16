"""
EIF 异常检测算子 (Extended Isolation Forest)

算法逻辑源自对应公开实现的 eif.EIF, 按 TSA-Suite 算子规范重写。
不依赖 第三方库, 也不依赖任何第三方扩展隔离森林包 (如 PyPI 上的 ``eif``).
扩展隔离树完全用 numpy 实现, 思路与原论文 Hariri et al. (TKDE 2019) 一致.

核心思想:
    标准 IForest 用轴对齐切割, 对相关特征产生 artifact;
    EIF 用随机超平面 (斜分叉) 切割, ``extension_level`` 控制自由度:
        - 0: 等价于标准 IForest (仅 1 维参与分叉, 轴对齐)
        - n_features - 1: 所有维度参与, 完全斜分叉 (默认)

异常分数 score(x) = 2^(-avg_path_length(x) / c(sample_size)),
其中 c(n) = 2 * (ln(n-1) + 0.5772) - 2*(n-1)/n 是 BST 平均不成功路径长度.

包含:
    - EIFScorer: 直接评分器, 输出异常分数
    - EIFDetector: 端到端检测器, 组合 EIFScorer + PercentileDecider
"""
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
    'EIFScorerConfig',
    'EIFScorer',
    'EIFDetectorConfig',
    'EIFDetector',
]


# ============================================================================
# 扩展隔离树节点
# ============================================================================


class _EIFNode:
    """扩展隔离树节点 (内部 / 外部).

    使用 __slots__ 节省内存, 因 EIF 通常构造上百棵树.
    """

    __slots__ = ("e", "size", "n_vec", "p_vec", "left", "right", "ntype")

    def __init__(
        self,
        e: int,
        size: int,
        n_vec: np.ndarray | None,
        p_vec: np.ndarray | None,
        left: "_EIFNode | None",
        right: "_EIFNode | None",
        ntype: str,
    ) -> None:
        self.e = e            # 当前深度
        self.size = size      # 该节点样本数
        self.n_vec = n_vec    # 分裂超平面法向量 (外部节点为 None)
        self.p_vec = p_vec    # 分裂超平面上参考点
        self.left = left
        self.right = right
        self.ntype = ntype    # 'inNode' (内部) 或 'exNode' (外部)


# ============================================================================
# 算法子函数
# ============================================================================


def _average_path_length(n: int) -> float:
    """BST 中 n 个点不成功搜索的平均路径长度 c(n).

    c(n) = 2 * (ln(n-1) + 0.5772) - 2*(n-1)/n
    """
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


def _build_tree(X: np.ndarray, e: int, max_depth: int, dim: int, exlevel: int, rng: np.random.Generator) -> _EIFNode:
    """递归构建扩展隔离树.

    使用随机超平面 (受 exlevel 控制自由度) 递归划分数据.
    """
    if e >= max_depth or X.shape[0] <= 1:
        return _EIFNode(e, X.shape[0], None, None, None, None, "exNode")
    # exlevel 控制非零分量的个数: 选择 (dim - exlevel - 1) 个置零维度
    n_zeros = dim - exlevel - 1
    if n_zeros < 0:
        n_zeros = 0
    if n_zeros >= dim:
        # 全部置零 -> 法向量为零 -> 退化为外部节点
        return _EIFNode(e, X.shape[0], None, None, None, None, "exNode")
    idxs = rng.choice(dim, n_zeros, replace=False)
    n_vec = rng.normal(0.0, 1.0, dim)
    n_vec[idxs] = 0.0
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    p_vec = rng.uniform(mins, maxs)
    w = (X - p_vec).dot(n_vec) < 0
    left = _build_tree(X[w], e + 1, max_depth, dim, exlevel, rng)
    right = _build_tree(X[~w], e + 1, max_depth, dim, exlevel, rng)
    return _EIFNode(e, X.shape[0], n_vec, p_vec, left, right, "inNode")


def _path_length(x: np.ndarray, root: _EIFNode) -> float:
    """计算单个样本在扩展隔离树中的路径长度."""
    node = root
    e = 0
    while node.ntype == "inNode":
        e += 1
        # 若法向量为零 (退化情形), 默认分配到右子节点
        if node.n_vec is None or np.allclose(node.n_vec, 0.0):
            node = node.right
        elif (x - node.p_vec).dot(node.n_vec) < 0:
            node = node.left
        else:
            node = node.right
    if node.size <= 1:
        return float(e)
    return e + _average_path_length(node.size)


# ============================================================================
# EIF Scorer
# ============================================================================


class EIFScorerConfig(BaseModel):
    """EIF 评分器实例参数"""
    model_config = {"frozen": True}
    n_estimators: int = Field(default=100, ge=10, le=500, description="树的数量")
    max_samples: int | None = Field(default=None, ge=1, description="每棵树采样数, None 则使用 min(256, n)")
    extension_level: int | None = Field(default=None, ge=0, description="超平面扩展级别, None 则使用 n_features - 1")
    random_state: int | None = Field(default=None, description="随机种子")


class EIFScorer(SingleScorerMixin[None],
                 UnsupervisedNumericOperatorMixin[None],
                 NumericOperator[None, EIFScorerConfig, None]):
    """EIF 直接评分器 (Extended Isolation Forest)"""
    _LEARNED_STATE_FILE = '_learned_state.npz'

    @classmethod
    def name(cls) -> str:
        return "eif_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._trees: list[_EIFNode] | None = None
        """已构建的扩展隔离树 (存根节点)"""
        self._dim: int = 0
        """训练数据维度"""
        self._sample_size: int = 0
        """每棵树的采样数"""
        self._c: float = 0.0
        """c(sample_size) — 用于归一化路径长度"""
        self._exlevel: int = 0
        """实际使用的扩展级别"""

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """构建扩展隔离森林."""
        X = np.asarray(x, dtype=float)
        if X.ndim != 2:
            raise ValueError("EIF expects 2-D input (n_samples, n_features)")
        n_samples, dim = X.shape

        # 解析 max_samples
        if self.config.max_samples is not None:
            sample_size = int(self.config.max_samples)
            sample_size = max(1, min(sample_size, n_samples))
        else:
            sample_size = min(256, n_samples)

        # 解析 extension_level: None → n_features - 1 (完全斜分叉)
        if self.config.extension_level is None:
            exlevel = dim - 1
        else:
            exlevel = int(self.config.extension_level)
        exlevel = max(0, min(exlevel, dim - 1))

        rng = np.random.default_rng(self.config.random_state)
        limit = int(np.ceil(np.log2(sample_size))) if sample_size > 1 else 1

        trees: list[_EIFNode] = []
        for _ in range(self.config.n_estimators):
            ix = rng.choice(n_samples, sample_size, replace=False)
            sample = X[ix]
            trees.append(_build_tree(sample, 0, limit, dim, exlevel, rng))

        self._trees = trees
        self._dim = dim
        self._sample_size = sample_size
        self._exlevel = exlevel
        self._c = _average_path_length(sample_size)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """计算每个样本的异常分数: score = 2^(-avg_path / c)."""
        if self._trees is None:
            raise RuntimeError("训练尚未完成, 请先调用 fit() 方法训练模型")
        X = np.asarray(x, dtype=float)
        if X.ndim != 2:
            raise ValueError("EIF expects 2-D input (n_samples, n_features)")
        if X.shape[1] != self._dim:
            raise ValueError(
                f"输入特征维度 ({X.shape[1]}) 与训练维度 ({self._dim}) 不一致"
            )
        n = X.shape[0]
        scores = np.zeros(n)
        for i in range(n):
            avg_path = np.mean([_path_length(X[i], t) for t in self._trees])
            scores[i] = 2.0 ** (-avg_path / self._c) if self._c > 0 else 0.0
        return scores

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        # 树结构用 pickle 序列化 (对象图包含递归 _EIFNode)
        import pickle
        with open(path / "_trees.pkl", "wb") as f:
            pickle.dump(self._trees, f)
        np.savez(
            path / self._LEARNED_STATE_FILE,
            dim=np.array([self._dim]),
            sample_size=np.array([self._sample_size]),
            c=np.array([self._c]),
            exlevel=np.array([self._exlevel]),
        )

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        import pickle
        with open(path / "_trees.pkl", "rb") as f:
            self._trees = pickle.load(f)
        data = np.load(path / self._LEARNED_STATE_FILE)
        self._dim = int(data['dim'][0])
        self._sample_size = int(data['sample_size'][0])
        self._c = float(data['c'][0])
        self._exlevel = int(data['exlevel'][0])
        self._fitted = True


# ============================================================================
# EIF Detector
# ============================================================================


class EIFDetectorConfig(BaseModel):
    """EIF 检测器实例参数"""
    model_config = {"frozen": True}
    n_estimators: int = Field(default=100, ge=10, le=500, description="树的数量")
    max_samples: int | None = Field(default=None, ge=1, description="每棵树采样数")
    extension_level: int | None = Field(default=None, ge=0, description="超平面扩展级别")
    random_state: int | None = Field(default=None, description="随机种子")
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class EIFDetector(UnsupervisedNumericOperatorMixin[None],
                  BaseDeciderMixin[None],
                  NumericOperator[None, EIFDetectorConfig, None]):
    """EIF 检测器 = EIFScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "eif_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = EIFScorer(config=EIFScorerConfig(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            extension_level=self.config.extension_level,
            random_state=self.config.random_state,
        ))
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
        self._scorer = EIFScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
