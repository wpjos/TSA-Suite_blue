# -*- coding: utf-8 -*-

"""
Weighted-F1 评价指标算子

按各类别样本数加权平均的 F1 分数（权重 = 各类别真实样本数）。
受类别分布影响。

核心组件:
    - WeightedF1Config: 配置类（继承 BaseMetricConfig）
    - WeightedF1Metric: Weighted-F1 指标算子

使用示例::

    from tsas.engine.operator.evaluation import WeightedF1Metric

    op = WeightedF1Metric()
    result = op.run((y_truth, y_pred))
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'WeightedF1Config',
    'WeightedF1Metric',
]


class WeightedF1Config(BaseMetricConfig):
    """Weighted-F1 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"weighted_f1": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"weighted_f1": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class WeightedF1Metric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        WeightedF1Config,
        None,
    ],
):
    """Weighted-F1 评价指标算子

    按各类别真实样本数加权的 F1 平均值。类别越多样本权重越大。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        float: Weighted-F1 值，范围 ``[0, 1]``

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: float — 标量 Weighted-F1
        MC: WeightedF1Config — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "weighted_f1_metric"

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
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true 和 y_pred 长度不一致: {len(y_true)} vs {len(y_pred)}"
            )

        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s: list[float] = []
        weights: list[float] = []

        for label in labels:
            tp = int(np.sum((y_true == label) & (y_pred == label)))
            fp = int(np.sum((y_true != label) & (y_pred == label)))
            fn = int(np.sum((y_true == label) & (y_pred != label)))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
            weights.append(float(np.sum(y_true == label)))

        return float(np.average(f1s, weights=weights))