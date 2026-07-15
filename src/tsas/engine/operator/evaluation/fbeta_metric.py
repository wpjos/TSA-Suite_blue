# -*- coding: utf-8 -*-

"""
F-β 评价指标算子

``(1 + β²) * (Precision * Recall) / (β² * Precision + Recall)``，
通过 β 参数在精确率和召回率之间进行加权平衡。

核心组件:
    - FBetaMetricConfig: 配置类（继承 BaseMetricConfig）
    - FBetaMetric: F-β 指标算子

使用示例::

    from tsas.engine.operator.evaluation import FBetaMetric

    op = FBetaMetric()
    result = op.run((y_truth, y_pred))  # β=1 等同 F1

    op = FBetaMetric(beta=2.0)
    result = op.run((y_truth, y_pred))  # 偏重 Recall
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'FBetaMetricConfig',
    'FBetaMetric',
]


class FBetaMetricConfig(BaseMetricConfig):
    """F-β 评价指标配置

    Attributes:
        beta (float): 权重系数，>1 偏重 Recall，<1 偏重 Precision，默认 1（F1）
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"fbeta": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    beta: float = Field(default=1.0, gt=0, description="权重系数，>1 偏重 Recall，<1 偏重 Precision")
    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={"fbeta": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class FBetaMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        FBetaMetricConfig,
        None,
    ],
):
    """F-β 评价指标算子

    计算 ``(1 + β²) * (P * R) / (β² * P + R)``；当 ``P + R == 0`` 时返回 0.0。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        float: F-β 值，范围 ``[0, 1]``

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: float — 标量 F-β
        MC: FBetaMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "fbeta_metric"

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
        beta = 1.0 if config is None else config.beta
        pos_label = 1 if config is None else config.pos_label

        tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
        fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
        fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        beta_sq = beta ** 2
        return float((1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall))