# -*- coding: utf-8 -*-

"""
PCA 异常检测算子

基于主成分分析（PCA）重构误差的异常检测。
核心思想: 正常数据可以被低维主成分有效重构，异常数据的重构误差显著偏大。

包含:
    - PCAPredictor: PCA 预测器，学习主成分并输出重构值
    - PCAScorer: PCA 评分器，组合 PCAPredictor + ResidualScorer，输出异常分数
    - PCADetector: 端到端检测器，组合 PCAScorer + PercentileDecider，输出二分类标签

示例用法::

    # 直接评分
    scorer = PCAScorer(n_components=3)
    scorer.fit(train_data)
    scores, eo = scorer.run(test_data)

    # 端到端检测
    detector = PCADetector(n_components=3, percentile=95.0)
    detector.fit(train_data)
    labels, eo = detector.run(test_data)
"""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from bianque.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from bianque.engine.operator.detection.base import BaseDeciderMixin, BasePredictor, SingleScorerMixin
from bianque.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderExtraOutput,
)
from bianque.engine.operator.detection.residual_scorer import (
    ResidualScorer,
    ResidualScorerExtraOutput,
)

__all__ = [
    'PCAPredictorConfig',
    'PCAPredictorExtraOutput',
    'PCAPredictor',
    'PCAScorerConfig',
    'PCAScorerExtraOutput',
    'PCAScorer',
    'PCADetectorConfig',
    'PCADetectorExtraOutput',
    'PCADetector',
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

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray | tuple[
        np.ndarray, PCAPredictorExtraOutput]:

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


# ============================================================================
# PCA 评分器
# ============================================================================


class PCAScorerConfig(BaseModel):
    """
    PCA 评分器实例参数

    Attributes:
        n_components: PCA 保留的主成分数量
        metric: 残差计算方式，``"mse"`` 为均方误差，``"mae"`` 为平均绝对误差
    """
    n_components: int = Field(default=3, ge=1, description="PCA 保留的主成分数量")
    metric: Literal["mse", "mae"] = Field(default="mse", description="残差计算方式: 'mse' 或 'mae'")


class PCAScorerExtraOutput(BaseModel):
    """
    PCA 评分器附加输出

    聚合子组件 PCAPredictor 和 ResidualScorer 的附加输出。

    Attributes:
        pca_eo: PCAPredictor 的附加输出（解释方差比等），可能为 None
        residual_eo: ResidualScorer 的附加输出（逐变量分数），可能为 None
    """
    pca_eo: PCAPredictorExtraOutput | None = None
    """PCAPredictor 的附加输出"""
    residual_eo: ResidualScorerExtraOutput | None = None
    """ResidualScorer 的附加输出"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PCAScorer(SingleScorerMixin[None],
                UnsupervisedNumericOperatorMixin[None],
                NumericOperator[PCAScorerExtraOutput, PCAScorerConfig, None]):
    """
    PCA 评分器 — 组合 PCAPredictor + ResidualScorer

    基于 PCA 重构误差的异常评分器。

    内部数据流::

        输入 x → PCAPredictor.run(x) → (x_pred, pca_eo)
               → ResidualScorer.run((x, x_pred)) → (scores, residual_eo)
               → 输出: (1D 异常分数, PCAScorerExtraOutput)

    训练阶段仅训练 PCAPredictor（ResidualScorer 为 BiNumericOperator，无需训练）。

    泛型参数:
        - EO: PCAScorerExtraOutput
        - C: PCAScorerConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        return "pca_scorer"

    def __init__(self, *, oid: str | None = None, config: PCAScorerConfig | None = None, **kwargs):
        """
        初始化 PCA 评分器

        自动创建 PCAPredictor 和 ResidualScorer 子组件。

        Args:
            oid: 算子标识
            config: PCA 评分器配置
            **kwargs: 透传给基类的参数，支持:
                - n_components (int): PCA 主成分数，默认 3
                - metric (str): 残差计算方式，默认 "mse"
                - oid (str): 算子标识
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._predictor = PCAPredictor(n_components=self.config.n_components)
        self._scorer = ResidualScorer(metric=self.config.metric)

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练 PCA 评分器

        仅训练 PCAPredictor（学习主成分），ResidualScorer 无需训练。

        Args:
            x(np.ndarray): 训练数据，形状 (n_samples, n_features)
            params(None): 无训练参数
        """
        self._predictor.fit(x)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> tuple[
        np.ndarray, PCAScorerExtraOutput]:
        """
        计算 PCA 重构误差异常分数

        Args:
            x(np.ndarray): 输入数据，形状 (n_samples, n_features)
            params(None): 无运行参数
            idx(pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, PCAScorerExtraOutput]:
                - scores: 1D 异常分数，形状 (n_samples,)
                - eo: 聚合子组件的附加输出
        """
        # 步骤1: PCA 重构
        x_pred, pca_eo = self._predictor.run(x)

        # 步骤2: 残差评分
        scores, residual_eo = self._scorer.run((x, x_pred))

        return scores.ravel(), PCAScorerExtraOutput(pca_eo=pca_eo, residual_eo=residual_eo)


# ============================================================================
# PCA 检测器
# ============================================================================


class PCADetectorConfig(BaseModel):
    """
    PCA 检测器实例参数

    Attributes:
        n_components: PCA 保留的主成分数量
        metric: 残差计算方式，``"mse"`` 为均方误差，``"mae"`` 为平均绝对误差
        percentile: 百分位阈值
    """
    n_components: int = Field(default=3, ge=1, description="PCA 保留的主成分数量")
    metric: Literal["mse", "mae"] = Field(default="mse", description="残差计算方式: 'mse' 或 'mae'")
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class PCADetectorExtraOutput(BaseModel):
    """
    PCA 检测器附加输出

    聚合子组件 PCAScorer 和 PercentileDecider 的附加输出。

    Attributes:
        scorer_eo: PCAScorer 的附加输出，可能为 None
        decider_eo: PercentileDecider 的附加输出，可能为 None
    """
    scorer_eo: PCAScorerExtraOutput | None = None
    """PCAScorer 的附加输出"""
    decider_eo: PercentileDeciderExtraOutput | None = None
    """PercentileDecider 的附加输出"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PCADetector(UnsupervisedNumericOperatorMixin[None],
                  BaseDeciderMixin[None],
                  NumericOperator[PCADetectorExtraOutput, PCADetectorConfig, None]):
    """
    PCA 检测器 — 组合 PCAScorer + PercentileDecider

    端到端的 PCA 异常检测器:

    ::

        PCADetector
          ├── PCAScorer
          │     ├── PCAPredictor
          │     │     _fit: 学习主成分
          │     │     _run: 输出重构值
          │     └── ResidualScorer
          │           _run: 计算重构残差分数
          └── PercentileDecider
                _fit: 学习训练分数的百分位阈值
                _run: scores > threshold → labels

    使用示例::

        detector = PCADetector(n_components=3, percentile=95.0)
        detector.fit(train_data)
        labels, eo = detector.run(test_data)

    泛型参数:
        - EO: PCADetectorExtraOutput
        - C: PCADetectorConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        return "pca_detector"

    def __init__(self, *, oid: str | None = None, config: PCADetectorConfig | None = None, **kwargs):
        """
        初始化 PCA 检测器

        自动创建 PCAScorer 和 PercentileDecider 子组件。

        Args:
            oid: 算子标识
            config: PCA 检测器配置
            **kwargs: 透传给基类的参数，支持:
                - n_components (int): PCA 主成分数，默认 3
                - metric (str): 残差计算方式，默认 "mse"
                - percentile (float): 百分位阈值，默认 95.0
                - oid (str): 算子标识
        """
        super().__init__(oid=oid, config=config, **kwargs)
        scorer_config = PCAScorerConfig(
            n_components=self.config.n_components,
            metric=self.config.metric,
        )
        self._scorer = PCAScorer(config=scorer_config)
        self._decider = PercentileDecider(percentile=self.config.percentile)

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练 PCA 检测器

        1. 训练 PCAScorer（学习 PCA 主成分）
        2. 用训练数据计算训练分数
        3. 用训练分数训练 PercentileDecider

        Args:
            x(np.ndarray): 训练数据，形状 (n_samples, n_features)
            params(None): 无训练参数
        """
        # 步骤1: 训练评分器
        self._scorer.fit(x)
        # 步骤2: 计算训练分数
        scores, _ = self._scorer.run(x)
        # 步骤3: 训练决策器（学习百分位阈值）
        self._decider.fit(scores)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> tuple[
        np.ndarray, PCADetectorExtraOutput]:
        """
        检测推理: PCAScorer → PercentileDecider

        Args:
            x(np.ndarray): 输入数据，形状 (n_samples, n_features)
            params(None): 无运行参数
            idx(pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, PCADetectorExtraOutput]:
                - labels: 二分类标签，1=异常/0=正常
                - eo: 聚合子组件的附加输出
        """
        # 步骤1: 评分器推理
        scores, scorer_eo = self._scorer.run(x)
        # 步骤2: 决策器推理
        labels, decider_eo = self._decider.run(scores)

        return labels, PCADetectorExtraOutput(scorer_eo=scorer_eo, decider_eo=decider_eo)
