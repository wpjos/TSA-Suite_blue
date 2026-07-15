# -*- coding: utf-8 -*-

"""
MAE 评价指标算子（平均绝对误差）

计算 ``y_true`` 与 ``y_pred`` 之差的绝对值的均值，衡量回归预测的整体误差水平。
无状态纯函数，输出标量 float。

核心组件:
    - MAEMetricConfig: 配置类（继承 BaseMetricConfig）
    - MAEMetric: MAE 指标算子

使用示例::

    from tsas.engine.operator.evaluation import MAEMetric

    op = MAEMetric()
    result = op.run((y_true, y_pred))
    print(result)

    # HPO 集成
    op = MAEMetric(main_scores={"mae": "_"})
    scores = op.scores((y_true, y_pred))  # -> {"mae": 0.123}
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'MAEMetricConfig',
    'MAEMetric',
]


class MAEMetricConfig(BaseMetricConfig):
    """MAE 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"mae": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"mae": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class MAEMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        MAEMetricConfig,
        None,
    ],
):
    """MAE 评价指标算子

    计算 ``mean(|y_true - y_pred|)``，输出非负浮点数。

    Input:
        y_truth: 真实值数组
        y_pred: 预测值数组，与 ``y_truth`` 等长

    Output:
        float: MAE 值，非负

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 实测-预测对
        MR: float — 标量 MAE
        MC: MAEMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "mae_metric"

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
        return float(np.mean(np.abs(y_true - y_pred)))