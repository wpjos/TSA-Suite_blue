# -*- coding: utf-8 -*-

"""
RMSE 评价指标算子（均方根误差）

``sqrt(MSE)``，与原值量纲一致，常作为回归评估的首选指标。
无状态纯函数，输出标量 float。

核心组件:
    - RMSEMetricConfig: 配置类（继承 BaseMetricConfig）
    - RMSEMetric: RMSE 指标算子

使用示例::

    from tsas.engine.operator.evaluation import RMSEMetric

    op = RMSEMetric()
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
    'RMSEMetricConfig',
    'RMSEMetric',
]


class RMSEMetricConfig(BaseMetricConfig):
    """RMSE 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"rmse": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"rmse": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class RMSEMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        RMSEMetricConfig,
        None,
    ],
):
    """RMSE 评价指标算子

    计算 ``sqrt(mean((y_true - y_pred) ** 2))``，输出非负浮点数。

    Input:
        y_truth: 真实值数组
        y_pred: 预测值数组，与 ``y_truth`` 等长

    Output:
        float: RMSE 值，非负

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 实测-预测对
        MR: float — 标量 RMSE
        MC: RMSEMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "rmse_metric"

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
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))