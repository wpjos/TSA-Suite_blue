# -*- coding: utf-8 -*-

"""
Adjusted Rand Index（ARI）评价指标算子（聚类）

衡量聚类结果与真实标签的一致性，并消除随机偶然因素。
``ARI = (RI - E[RI]) / (max(RI) - E[RI])``，其中 RI 由列联表
``C(n_ij, 2)`` 求和得到。

核心组件:
    - ARIConfig: 配置类（继承 BaseMetricConfig）
    - ARIMetric: ARI 指标算子

使用示例::

    from tsas.engine.operator.evaluation import ARIMetric

    op = ARIMetric()
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
    'ARIConfig',
    'ARIMetric',
]


class ARIConfig(BaseMetricConfig):
    """Adjusted Rand Index 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"ari": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"ari": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class ARIMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        ARIConfig,
        None,
    ],
):
    """Adjusted Rand Index 评价指标算子

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        float: ARI 值，范围 ``[-1, 1]``；1 表示完全一致，0 表示与随机聚类相当

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: float — 标量 ARI
        MC: ARIConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "ari_metric"

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

        n_samples = len(y_true)
        contingency = _contingency_matrix(y_true, y_pred)
        a = np.sum(contingency, axis=1)
        b = np.sum(contingency, axis=0)

        def n_choose_2(n: int) -> int:
            return n * (n - 1) // 2

        sum_nij_c2 = sum(n_choose_2(int(n_ij)) for n_ij in contingency.flatten())
        sum_ai_c2 = sum(n_choose_2(int(ai)) for ai in a)
        sum_bj_c2 = sum(n_choose_2(int(bj)) for bj in b)
        total_c2 = n_choose_2(n_samples)

        expected_ri = (sum_ai_c2 * sum_bj_c2) / total_c2 if total_c2 > 0 else 0
        max_ri = (sum_ai_c2 + sum_bj_c2) / 2

        if max_ri == expected_ri:
            return 1.0

        return float((sum_nij_c2 - expected_ri) / (max_ri - expected_ri))


def _contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """构造真实标签与预测标签的列联表（行=真实类别，列=预测类别）。"""
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    n_true = len(true_labels)
    n_pred = len(pred_labels)

    true_to_idx = {label: i for i, label in enumerate(true_labels)}
    pred_to_idx = {label: i for i, label in enumerate(pred_labels)}

    contingency = np.zeros((n_true, n_pred), dtype=int)
    for t, p in zip(y_true, y_pred):
        contingency[true_to_idx[t], pred_to_idx[p]] += 1

    return contingency