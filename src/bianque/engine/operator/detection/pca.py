# -*- coding: utf-8 -*-

"""
PCA 异常检测算子

基于主成分分析（PCA）重构误差的异常检测，属于 CompositeScorer 路径。
核心思想: 正常数据可以被低维主成分有效重构，异常数据的重构误差显著偏大。

包含:
    - PCAPredictor: PCA 预测器，学习主成分并输出重构值
    - PCAScorer: 组合评分器，PCAPredictor + ResidualComparator + IdentityMapper
    - PCADetector: 端到端检测器，PCAScorer + PercentileDecider

示例用法::

    # 端到端检测
    detector = PCADetector(n_components=2, percentile=95.0)
    detector.fit(train_data)
    scores, eo = detector.run(test_data)
"""

import numpy as np
from pydantic import BaseModel, Field

from bianque.engine.operator.base import UnsupervisedNumericOperatorMixin
from bianque.engine.operator.detection.base import BasePredictor

__all__ = [
    'PCAPredictorConfig',
    'PCAPredictorExtraOutput',
    'PCAPredictor',
]


class PCAPredictorConfig(BaseModel):
    """
    PCA 预测器实例参数

    Attributes:
        n_components: 保留的主成分数量，必须为正整数
    """
    n_components: int = Field(default=2, description="保留的主成分数量")


class PCAPredictorExtraOutput(BaseModel):
    """
    PCA 预测器附加输出

    Attributes:
        explained_variance_ratio: 各主成分的解释方差比
        n_components: 保留的主成分数量
    """
    explained_variance_ratio: list[float] = []
    """各主成分的解释方差比"""
    n_components: int = 2
    """保留的主成分数量"""


class PCAPredictor(UnsupervisedNumericOperatorMixin[None],
                   BasePredictor[PCAPredictorExtraOutput, PCAPredictorConfig, None]):
    """
    PCA 预测器 — 重构型预测器

    基于主成分分析的重构型预测器。训练阶段学习数据的主成分方向，
    推理阶段将输入投影到主成分空间再重构回原始空间。

    核心逻辑:
        - ``_fit``: 使用 SVD 分解学习主成分矩阵 ``_components`` 和均值 ``_mean``
        - ``_run``: 投影 → 重构，输出与输入同维度的重构值

    数学原理:
        - 投影: ``z = (x - mean) @ components.T``（降维）
        - 重构: ``x_hat = z @ components + mean``（升维）
        - 重构值即为预测值，后续与真实值的残差反映异常程度

    注意:
        当 n_components 大于特征数时，自动调整为特征数。

    泛型参数:
        - EO: PCAPredictorExtraOutput
        - C: PCAPredictorConfig
        - RP: None（无运行参数）
        - FP: None（无训练参数）
    """

    @classmethod
    def name(cls) -> str:
        return "pca_predictor"

    def __init__(self, *, oid: str | None = None, config: PCAPredictorConfig | None = None, **kwargs):
        """
        初始化 PCA 预测器

        Args:
            **kwargs: 透传给基类的参数，支持:
                - n_components (int): 保留的主成分数量，默认 2
                - oid (str): 算子标识
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._components: np.ndarray | None = None
        """主成分矩阵，形状 (n_components, n_features)"""
        self._mean: np.ndarray | None = None
        """训练数据的列均值向量"""
        self._explained_variance_ratio: np.ndarray | None = None
        """各主成分的解释方差比"""

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        学习主成分

        对去中心化的训练数据进行 SVD 分解，提取前 n_components 个主成分。
        当 n_components 大于特征数时，自动调整。

        Args:
            x(np.ndarray): 训练数据，形状 (n_samples, n_features)
            params(None): 无训练参数
        """
        # 去中心化
        self._mean = x.mean(axis=0)
        x_centered = x - self._mean

        # 自动调整 n_components（不超过特征数）
        effective_k = min(self.config.n_components, x.shape[1])

        # SVD 分解: Vh 的前 k 行为主成分方向
        # full_matrices=False 使 Vh 形状为 (min(n,p), p)，节省计算
        _, s, vh = np.linalg.svd(x_centered, full_matrices=False)

        # 提取前 effective_k 个主成分
        self._components = vh[:effective_k]

        # 计算解释方差比
        total_var = np.sum(s ** 2)
        if total_var > 0:
            self._explained_variance_ratio = (s[:effective_k] ** 2) / total_var
        else:
            # 退化情况: 所有特征为常数
            self._explained_variance_ratio = np.zeros(effective_k)

    def _run_data(self, x: np.ndarray, params: None) -> np.ndarray | tuple[np.ndarray, PCAPredictorExtraOutput]:

        """
        PCA 重构预测

        将输入投影到主成分空间再重构回原始空间，
        输出与输入同维度的重构值。

        Args:
            x(np.ndarray): 输入数据，形状 (n_samples, n_features)
            params(None): 无运行参数

        Returns:
            tuple[np.ndarray, PCAPredictorExtraOutput]:
                - 重构值 ndarray，形状 (n_samples, n_features)
                - 附加输出，包含解释方差比和主成分数
        """
        # 投影: (x - mean) @ components.T → (n_samples, n_components)
        z = (x - self._mean) @ self._components.T
        # 重构: z @ components + mean → (n_samples, n_features)
        pred = z @ self._components + self._mean

        eo = PCAPredictorExtraOutput(
            explained_variance_ratio=self._explained_variance_ratio.tolist(),
            n_components=self._components.shape[0],
        )
        return pred, eo
