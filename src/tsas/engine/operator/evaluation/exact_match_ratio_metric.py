# -*- coding: utf-8 -*-

"""
Exact Match Ratio 评价指标算子（多标签）

所有标签都预测正确的样本比例。越大越好。

核心组件:
    - ExactMatchRatioConfig: 配置类（继承 BaseMetricConfig）
    - ExactMatchRatioMetric: Exact Match Ratio 指标算子

使用示例::

    from tsas.engine.operator.evaluation import ExactMatchRatioMetric

    op = ExactMatchRatioMetric()
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
    'ExactMatchRatioConfig',
    'ExactMatchRatioMetric',
]


class ExactMatchRatioConfig(BaseMetricConfig):
    """Exact Match Ratio 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"emr": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"emr": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class ExactMatchRatioMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        ExactMatchRatioConfig,
        None,
    ],
):
    """Exact Match Ratio 评价指标算子

    多标签场景下，所有 ``n_labels`` 个标签都预测正确的样本占总样本的比例。

    Input:
        y_truth: 真实标签矩阵，shape=(n_samples, n_labels)
        y_pred: 预测标签矩阵，shape 与 y_truth 相同

    Output:
        float: Exact Match Ratio 值，范围 ``[0, 1]``，越大越好

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 二维矩阵对
        MR: float — 标量 Exact Match Ratio
        MC: ExactMatchRatioConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "exact_match_ratio_metric"

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

        exact_matches = np.all(y_true == y_pred, axis=1)
        return float(np.mean(exact_matches))