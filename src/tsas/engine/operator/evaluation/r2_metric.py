# -*- coding: utf-8 -*-

"""
R² 评价指标算子（决定系数）

``1 - SS_res / SS_tot``，衡量模型解释方差的比例。
``SS_tot = 0`` 时返回 0.0（与 bqlib 一致）。

核心组件:
    - R2MetricConfig: 配置类（继承 BaseMetricConfig）
    - R2Metric: R² 指标算子

使用示例::

    from tsas.engine.operator.evaluation import R2Metric

    op = R2Metric()
    result = op.run((y_true, y_pred))
    print(result)
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'R2MetricConfig',
    'R2Metric',
]


class R2MetricConfig(BaseMetricConfig):
    """R² 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"r2": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"r2": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class R2Metric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        R2MetricConfig,
        None,
    ],
):
    """R² 评价指标算子

    计算 ``1 - SS_res / SS_tot``，取值范围 ``(-∞, 1]``。
    ``SS_tot = 0`` 时返回 0.0（bqlib 兼容行为）。

    Input:
        y_truth: 真实值数组
        y_pred: 预测值数组，与 ``y_truth`` 等长

    Output:
        float: R² 值；1 表示完美拟合，0 表示与均值预测相当，可为负

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 实测-预测对
        MR: float — 标量 R²
        MC: R2MetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "r2_metric"

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
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)