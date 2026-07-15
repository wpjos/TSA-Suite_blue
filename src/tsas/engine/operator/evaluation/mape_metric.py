# -*- coding: utf-8 -*-

"""
MAPE 评价指标算子（平均绝对百分比误差）

``mean(|y_true - y_pred| / |y_true|) * 100``，以百分比形式衡量预测偏差。
真实值越接近 0 越不稳定；通过 ``epsilon`` 平滑防除零。

核心组件:
    - MAPEMetricConfig: 配置类（继承 BaseMetricConfig）
    - MAPEMetric: MAPE 指标算子

使用示例::

    from tsas.engine.operator.evaluation import MAPEMetric

    op = MAPEMetric()
    result = op.run((y_true, y_pred))

    op = MAPEMetric(epsilon=1e-8)
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
    'MAPEMetricConfig',
    'MAPEMetric',
]


class MAPEMetricConfig(BaseMetricConfig):
    """MAPE 评价指标配置

    Attributes:
        epsilon (float): 防除零平滑因子，默认 ``1e-10``
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"mape": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    epsilon: float = Field(default=1e-10, description="防除零平滑因子，加到 |y_true| 分母")
    main_scores: dict[str, str] | None = Field(
        default={"mape": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class MAPEMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        MAPEMetricConfig,
        None,
    ],
):
    """MAPE 评价指标算子

    计算 ``mean(|y_true - y_pred| / (|y_true| + epsilon)) * 100``。
    当 ``y_true`` 含 0 时分母加上 ``epsilon`` 防止除零。

    Input:
        y_truth: 真实值数组
        y_pred: 预测值数组，与 ``y_truth`` 等长

    Output:
        float: MAPE 值（百分比），非负

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 实测-预测对
        MR: float — 标量 MAPE（百分比）
        MC: MAPEMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "mape_metric"

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
        return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)