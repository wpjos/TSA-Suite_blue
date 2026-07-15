# -*- coding: utf-8 -*-

"""
MASE 评价指标算子（平均绝对标度误差）

``mean(|y_true - y_pred|) / naive_error``，naive_error 由 ``y_train`` 的季节性差分计算。
MASE < 1 表示优于朴素预测。

核心组件:
    - MASEMetricConfig: 配置类（继承 BaseMetricConfig）
    - MASEMetric: MASE 指标算子

使用示例::

    from tsas.engine.operator.evaluation import MASEMetric

    op = MASEMetric()
    result = op.run((y_true, y_pred))

    # 提供训练数据
    op = MASEMetric(y_train=train_array, seasonality=24)
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
    'MASEMetricConfig',
    'MASEMetric',
]


class MASEMetricConfig(BaseMetricConfig):
    """MASE 评价指标配置

    Attributes:
        y_train (list[float] | None): 训练数据；``None`` 时使用 ``y_true`` 做朴素误差
        seasonality (int): 季节周期，``1`` 表示非季节性，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"mase": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    y_train: list[float] | None = Field(
        default=None,
        description="训练数据，用于计算朴素预测误差；None 时使用 y_true 自身",
    )
    seasonality: int = Field(default=1, ge=1, description="季节周期；1 表示非季节性")
    main_scores: dict[str, str] | None = Field(
        default={"mase": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class MASEMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        MASEMetricConfig,
        None,
    ],
):
    """MASE 评价指标算子

    计算 ``mean(|y_true - y_pred|) / naive_error``。
    naive_error = mean(|y_train[seasonality:] - y_train[:-seasonality]|)；
    当 ``y_train <= seasonality`` 或 ``y_train`` 较小时退化使用 ``mean(|y_true|)``。

    Input:
        y_truth: 真实值数组
        y_pred: 预测值数组，与 ``y_truth`` 等长

    Output:
        float: MASE 值；< 1 表示优于朴素预测，= 1 与朴素预测相当，> 1 差于朴素预测

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred) 实测-预测对
        MR: float — 标量 MASE
        MC: MASEMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "mase_metric"

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
        y_train = np.asarray(y_true) if config is None or config.y_train is None else np.asarray(config.y_train)
        seasonality = 1 if config is None else config.seasonality

        if len(y_train) <= seasonality:
            naive_error = float(np.mean(np.abs(y_true)))
        else:
            naive_error = float(np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])))

        if naive_error == 0:
            return float(np.mean(np.abs(y_true - y_pred)))

        return float(np.mean(np.abs(y_true - y_pred)) / naive_error)