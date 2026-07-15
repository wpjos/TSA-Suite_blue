# -*- coding: utf-8 -*-

"""
SMAPE 评价指标算子（对称平均绝对百分比误差）

``mean(|y_pred - y_true| / ((|y_true| + |y_pred|) / 2)) * 100``，
解决 MAPE 在真实值接近零时不稳定的问题，分母两侧都有值故更稳健。

核心组件:
    - SMAPEMetricConfig: 配置类（继承 BaseMetricConfig）
    - SMAPEMetric: SMAPE 指标算子

使用示例::

    from tsas.engine.operator.evaluation import SMAPEMetric

    op = SMAPEMetric()
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
    'SMAPEMetricConfig',
    'SMAPEMetric',
]


class SMAPEMetricConfig(BaseMetricConfig):
    """SMAPE 评价指标配置

    Attributes:
        epsilon (float): 防除零平滑因子，默认 ``1e-10``
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"smape": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    epsilon: float = Field(default=1e-10, description="防除零平滑因子，加到分母 (|y_true|+|y_pred|)/2 上")
    main_scores: dict[str, str] | None = Field(
        default={"smape": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class SMAPEMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        SMAPEMetricConfig,
        None,
    ],
):
    """SMAPE 评价指标算子

    计算 ``mean(|y_pred - y_true| / ((|y_true| + |y_pred|) / 2 + epsilon)) * 100``。
    与 MAPE 相比，分母两侧都有数值，对真实值接近零更稳健。

    Input:
        y_truth: 真实值数组
        y_pred: 预测值数组，与 ``y_truth`` 等长

    Output:
        float: SMAPE 值（百分比），非负

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 实测-预测对
        MR: float — 标量 SMAPE（百分比）
        MC: SMAPEMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "smape_metric"

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
        config = self.config
        epsilon = 1e-10 if config is None else config.epsilon
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return float(np.mean(numerator / denominator) * 100)