# -*- coding: utf-8 -*-

"""
Hamming Loss 评价指标算子（多标签）

``mean(y_true != y_pred)``：所有样本 × 所有标签位置上的错误比例。
越小越好。

核心组件:
    - HammingLossConfig: 配置类（继承 BaseMetricConfig）
    - HammingLossMetric: Hamming Loss 指标算子

使用示例::

    from tsas.engine.operator.evaluation import HammingLossMetric

    op = HammingLossMetric()
    result = op.run((y_true, y_pred))
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'HammingLossConfig',
    'HammingLossMetric',
]


class HammingLossConfig(BaseMetricConfig):
    """Hamming Loss 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"hamming_loss": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"hamming_loss": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class HammingLossMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        HammingLossConfig,
        None,
    ],
):
    """Hamming Loss 评价指标算子

    多标签场景下，统计 ``y_true`` 与 ``y_pred`` 在所有 ``(sample, label)`` 位置
    上的错误比例。形状必须相同，shape=(n_samples, n_labels)。

    Input:
        y_truth: 真实标签矩阵，shape=(n_samples, n_labels)
        y_pred: 预测标签矩阵，shape 与 y_truth 相同

    Output:
        float: Hamming Loss 值，范围 ``[0, 1]``，越小越好

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 二维矩阵对
        MR: float — 标量 Hamming Loss
        MC: HammingLossConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "hamming_loss_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> float:
        y_true, y_pred = x
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true 和 y_pred 形状不一致: {y_true.shape} vs {y_pred.shape}")

        n_samples, n_labels = y_true.shape
        total_errors = int(np.sum(y_true != y_pred))
        return float(total_errors / (n_samples * n_labels))