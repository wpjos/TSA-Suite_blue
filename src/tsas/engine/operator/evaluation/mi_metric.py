# -*- coding: utf-8 -*-

"""
Mutual Information（MI）评价指标算子（聚类）

两个标签分配之间的互信息 ``Σ p_ij * log(p_ij / (p_i * p_j))``。

核心组件:
    - MutualInfoConfig: 配置类（继承 BaseMetricConfig）
    - MutualInfoMetric: MI 指标算子

使用示例::

    from tsas.engine.operator.evaluation import MutualInfoMetric

    op = MutualInfoMetric()
    result = op.run((y_true, y_pred))
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.ari_metric import _contingency_matrix
from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'MutualInfoConfig',
    'MutualInfoMetric',
    'mutual_info_score',
]


def mutual_info_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算互信息（MI），衡量两个标签分配之间的信息共享量。"""
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    contingency = _contingency_matrix(y_true, y_pred)

    p_i = np.sum(contingency, axis=1) / n_samples
    p_j = np.sum(contingency, axis=0) / n_samples
    p_ij = contingency / n_samples

    mi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if p_ij[i, j] > 0 and p_i[i] > 0 and p_j[j] > 0:
                mi += p_ij[i, j] * np.log(p_ij[i, j] / (p_i[i] * p_j[j]))

    return float(mi)


class MutualInfoConfig(BaseMetricConfig):
    """Mutual Information 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"mi": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"mi": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class MutualInfoMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        MutualInfoConfig,
        None,
    ],
):
    """Mutual Information 评价指标算子

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        float: 互信息值，非负；值越大表示一致性越高

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: float — 标量 MI
        MC: MutualInfoConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "mi_metric"

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
        return mutual_info_score(y_true, y_pred)