# -*- coding: utf-8 -*-

"""
Entropy（信息熵）评价指标算子（聚类 / 自评估）

标签分布的香农熵 ``H = -Σ p_i * log(p_i)``，衡量标签分配的不确定性。

核心组件:
    - EntropyConfig: 配置类（继承 BaseMetricConfig）
    - EntropyMetric: 信息熵指标算子

使用示例::

    from tsas.engine.operator.evaluation import EntropyMetric

    op = EntropyMetric()
    result = op.run(labels)
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'EntropyConfig',
    'EntropyMetric',
    'entropy',
]


def entropy(labels: np.ndarray) -> float:
    """计算标签分布的信息熵，衡量标签分配的不确定性。

    Args:
        labels: 标签数组

    Returns:
        熵值，非负
    """
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n_samples

    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * np.log(p)

    return float(ent)


class EntropyConfig(BaseMetricConfig):
    """信息熵评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"entropy": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"entropy": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class EntropyMetric(
    BaseMetricOperator[
        np.ndarray,
        float,
        EntropyConfig,
        None,
    ],
):
    """信息熵评价指标算子

    Input:
        labels: 标签数组（一维离散）

    Output:
        float: 熵值，非负；越大表示分布越均匀

    泛型参数:
        I: np.ndarray — 一维标签数组
        MR: float — 标量熵
        MC: EntropyConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "entropy_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: np.ndarray,
        *,
        params: None,
    ) -> float:
        labels = np.asarray(x).ravel()
        return entropy(labels)