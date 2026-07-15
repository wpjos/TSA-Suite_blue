# -*- coding: utf-8 -*-

"""
Multi-Label Accuracy 评价指标算子（Jaccard Index 均值）

每个样本的标签交集与并集之比的均值。

核心组件:
    - MultiLabelAccuracyConfig: 配置类（继承 BaseMetricConfig）
    - MultiLabelAccuracyMetric: Multi-Label Accuracy 指标算子

使用示例::

    from tsas.engine.operator.evaluation import MultiLabelAccuracyMetric

    op = MultiLabelAccuracyMetric()
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
    'MultiLabelAccuracyConfig',
    'MultiLabelAccuracyMetric',
]


class MultiLabelAccuracyConfig(BaseMetricConfig):
    """Multi-Label Accuracy 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"ml_accuracy": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"ml_accuracy": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class MultiLabelAccuracyMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        MultiLabelAccuracyConfig,
        None,
    ],
):
    """Multi-Label Accuracy 评价指标算子

    对每个样本计算 ``|y_true ∩ y_pred| / |y_true ∪ y_pred|``（Jaccard Index），
    再取所有样本的均值。当并集为 0 时，若交集也为 0 返回 1.0，否则返回 0.0。

    Input:
        y_truth: 真实标签矩阵，shape=(n_samples, n_labels)
        y_pred: 预测标签矩阵，shape 与 y_truth 相同

    Output:
        float: Multi-Label Accuracy 值，范围 ``[0, 1]``

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 二维矩阵对
        MR: float — 标量 Multi-Label Accuracy
        MC: MultiLabelAccuracyConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "multi_label_accuracy_metric"

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

        n_samples = y_true.shape[0]
        accuracies: list[float] = []
        for i in range(n_samples):
            intersection = int(np.sum((y_true[i] == 1) & (y_pred[i] == 1)))
            union = int(np.sum((y_true[i] == 1) | (y_pred[i] == 1)))
            if union == 0:
                acc = 1.0 if intersection == 0 else 0.0
            else:
                acc = intersection / union
            accuracies.append(acc)

        return float(np.mean(accuracies))