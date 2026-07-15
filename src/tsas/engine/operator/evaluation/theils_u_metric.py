# -*- coding: utf-8 -*-

"""
Theil's U 评价指标算子

``sqrt(mean((y_true - y_pred)²)) / sqrt(mean((y_true - y_naive)²))``，
衡量相对于朴素预测的预测误差；< 1 好于朴素预测，= 1 与朴素预测相当，> 1 差于朴素预测。

核心组件:
    - TheilsUMetricConfig: 配置类（继承 BaseMetricConfig）
    - TheilsUMetric: Theil's U 指标算子

使用示例::

    from tsas.engine.operator.evaluation import TheilsUMetric

    op = TheilsUMetric()
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
    'TheilsUMetricConfig',
    'TheilsUMetric',
]


class TheilsUMetricConfig(BaseMetricConfig):
    """Theil's U 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"theils_u": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"theils_u": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class TheilsUMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        TheilsUMetricConfig,
        None,
    ],
):
    """Theil's U 评价指标算子

    计算 ``sqrt(mean((y_true - y_pred)²)) / sqrt(mean((y_true - y_naive)²))``。
    ``y_naive`` 是 ``y_true`` 的 roll-1 朴素预测（首点回填）。
    ``len(y_true) <= 1`` 时返回 1.0；分母为 0 且分子为 0 时返回 1.0，否则返回 inf。

    Input:
        y_truth: 真实值数组
        y_pred: 预测值数组，与 ``y_truth`` 等长

    Output:
        float: Theil's U 值；< 1 好于朴素预测，= 1 与朴素预测相当，> 1 差于朴素预测

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 实测-预测对
        MR: float — 标量 Theil's U
        MC: TheilsUMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "theils_u_metric"

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

        if len(y_true) <= 1:
            return 1.0

        y_naive = np.roll(y_true, 1)
        y_naive[0] = y_true[0]

        numerator = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        denominator = float(np.sqrt(np.mean((y_true - y_naive) ** 2)))

        if denominator == 0:
            return 1.0 if numerator == 0 else float("inf")

        return float(numerator / denominator)