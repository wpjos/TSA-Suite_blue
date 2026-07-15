# -*- coding: utf-8 -*-

"""
AP 评价指标算子（Average Precision，平均精度）

按 score 降序遍历，对每个正例样本计算截至当前的 precision 并取均值。
无正例样本时返回 0.0。

核心组件:
    - AveragePrecisionMetricConfig: 配置类（继承 BaseMetricConfig）
    - AveragePrecisionMetric: AP 指标算子

使用示例::

    from tsas.engine.operator.evaluation import AveragePrecisionMetric

    op = AveragePrecisionMetric()
    result = op.run((y_truth, y_score))
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'AveragePrecisionMetricConfig',
    'AveragePrecisionMetric',
]


class AveragePrecisionMetricConfig(BaseMetricConfig):
    """AP 评价指标配置

    Attributes:
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"ap": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={"ap": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class AveragePrecisionMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        AveragePrecisionMetricConfig,
        None,
    ],
):
    """AP（Average Precision）评价指标算子

    按 ``y_score`` 降序遍历，对每个等于 ``pos_label`` 的样本记录截至当前位置
    的 ``precision = num_pos / (i + 1)``，最终对所有记录的 precision 取均值。
    无正例时返回 0.0。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_score: 预测得分数组（越高越可能是正例），与 y_truth 等长

    Output:
        float: AP 值，范围 ``[0, 1]``

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: float — 标量 AP
        MC: AveragePrecisionMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "average_precision_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> float:
        y_true, y_score = x
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        if len(y_true) != len(y_score):
            raise ValueError(
                f"y_true 和 y_score 长度不一致: {len(y_true)} vs {len(y_score)}"
            )
        config = self.config
        pos_label = 1 if config is None else config.pos_label

        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]

        precisions: list[float] = []
        num_pos = 0
        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == pos_label:
                num_pos += 1
                precisions.append(num_pos / (i + 1))

        if num_pos == 0:
            return 0.0
        return float(np.mean(precisions))