# -*- coding: utf-8 -*-

"""
Precision@K 评价指标算子

异常得分最高的前 K 个样本中的精确率。常用于排序类异常检测评估。

核心组件:
    - PrecisionAtKConfig: 配置类（继承 BaseMetricConfig）
    - PrecisionAtKMetric: Precision@K 指标算子

使用示例::

    from tsas.engine.operator.evaluation import PrecisionAtKMetric

    op = PrecisionAtKMetric()
    result = op.run((y_truth, y_score))

    op = PrecisionAtKMetric(k=50)
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
    'PrecisionAtKConfig',
    'PrecisionAtKMetric',
]


class PrecisionAtKConfig(BaseMetricConfig):
    """Precision@K 评价指标配置

    Attributes:
        k (int): 取 top-k 个样本，默认 10
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"patk": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    k: int = Field(default=10, ge=1, description="取前 k 个最高分样本计算 precision")
    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={"patk": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class PrecisionAtKMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        PrecisionAtKConfig,
        None,
    ],
):
    """Precision@K 评价指标算子

    取 ``y_score`` 最高的前 ``k`` 个样本，统计其中等于 ``pos_label`` 的比例。
    当 ``k <= 0`` 或 ``k > len(y_truth)`` 时抛出 ValueError。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_score: 预测得分数组（越高越可能是正例），与 y_truth 等长

    Output:
        float: Precision@K 值，范围 ``[0, 1]``

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: float — 标量 Precision@K
        MC: PrecisionAtKConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "precision_at_k_metric"

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
        k = 10 if config is None else config.k
        pos_label = 1 if config is None else config.pos_label

        if k <= 0 or k > len(y_true):
            raise ValueError(f"k must be between 1 and {len(y_true)}")

        top_indices = np.argsort(y_score)[::-1][:k]
        top_true = y_true[top_indices]
        return float(np.sum(top_true == pos_label) / k)