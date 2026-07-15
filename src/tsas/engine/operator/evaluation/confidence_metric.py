# -*- coding: utf-8 -*-

"""
Confidence 评价指标算子（PyOD 移植）

基于 Beta-Binomial 后验估计"模型在略不同训练集上做出相同预测"的概率：
1. 对每个 test_score，统计训练集中分数不超过它的样本数 n_instances；
2. Laplace 平滑得后验 posterior = (1 + n_instances) / (2 + n)；
3. 用二项 CDF 算 confidence = 1 - binom.cdf(n - ⌊n·contamination⌋, n, posterior)；
4. 对预测为异常的样本（test_score > threshold）翻转 confidence。

Reference:
    Perini, Vercellis (2023). *Probabilistic Confidence of Anomaly Detectors*. SDM.

核心组件:
    - ConfidenceResult: 置信度结果（Pydantic BaseModel）
    - ConfidenceConfig: 配置类（继承 BaseMetricConfig）
    - ConfidenceMetric: 置信度指标算子

使用示例::

    from tsas.engine.operator.evaluation import ConfidenceMetric

    op = ConfidenceMetric(threshold=0.5)
    result = op.run((train_scores, test_scores))
    print(result.confidence)
"""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import binom

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'ConfidenceResult',
    'ConfidenceConfig',
    'ConfidenceMetric',
]


class ConfidenceResult(BaseModel):
    """置信度结果

    Attributes:
        confidence (np.ndarray): 与 test_scores 等长的置信度数组，取值 [0, 1]
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    confidence: np.ndarray = Field(description="与 test_scores 等长的置信度数组，取值 [0, 1]")


class ConfidenceConfig(BaseMetricConfig):
    """置信度评价指标配置

    Attributes:
        threshold (float): 决策阈值；``test_score > threshold`` 判为异常
        contamination (float): 污染比例，默认 0.1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"confidence": "confidence"}``
    """
    model_config = ConfigDict(frozen=True)

    threshold: float = Field(description="决策阈值")
    contamination: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="污染比例（默认 0.1）",
    )
    main_scores: dict[str, str] | None = Field(
        default=None,
        description="主评分路径映射；confidence 字段为 ndarray，无单一主评分，默认 None",
    )


class ConfidenceMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        ConfidenceResult,
        ConfidenceConfig,
        None,
    ],
):
    """Confidence 评价指标算子

    Input:
        train_scores: 训练集异常分数，shape (n_train,)
        test_scores: 待评估分数，shape (n_test,)

    Output:
        ConfidenceResult: 与 test_scores 等长的置信度数组

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (train_scores, test_scores)
        MR: ConfidenceResult — 置信度结果
        MC: ConfidenceConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "confidence_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> ConfidenceResult:
        train_scores, test_scores = x
        train = np.asarray(train_scores, dtype=float).ravel()
        test = np.asarray(test_scores, dtype=float).ravel()
        n = train.shape[0]
        if n == 0:
            raise ValueError("train_scores must be non-empty")

        config = self.config
        if config is None:
            raise ValueError("ConfidenceMetric 需要显式提供 threshold 配置")
        threshold = config.threshold
        contamination = config.contamination

        sorted_train = np.sort(train)
        n_instances = np.searchsorted(sorted_train, test, side="right")
        posterior_prob = (1.0 + n_instances) / (2.0 + n)
        k = n - int(np.floor(n * contamination))
        confidence = 1.0 - binom.cdf(k, n, posterior_prob)

        prediction = (test > threshold).astype(int)
        if prediction.any():
            np.place(
                confidence,
                prediction == 1,
                1.0 - confidence[prediction == 1],
            )

        return ConfidenceResult(confidence=np.asarray(confidence, dtype=float))