# -*- coding: utf-8 -*-

"""
Normalized Mutual Information（NMI）评价指标算子（聚类）

按 ``MI / sqrt(H(y_true) * H(y_pred))`` 归一化的互信息。

核心组件:
    - NMIConfig: 配置类（继承 BaseMetricConfig）
    - NMIMetric: NMI 指标算子

使用示例::

    from tsas.engine.operator.evaluation import NMIMetric

    op = NMIMetric()
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
    'NMIConfig',
    'NMIMetric',
]


class NMIConfig(BaseMetricConfig):
    """Normalized Mutual Information 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"nmi": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"nmi": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class NMIMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        NMIConfig,
        None,
    ],
):
    """Normalized Mutual Information 评价指标算子

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        float: NMI 值，范围 ``[0, 1]``；1 表示完全一致

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: float — 标量 NMI
        MC: NMIConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "nmi_metric"

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

        contingency = _contingency_matrix(y_true, y_pred)
        n_samples = len(y_true)
        if n_samples == 0:
            return 1.0

        p_i = np.sum(contingency, axis=1) / n_samples
        p_j = np.sum(contingency, axis=0) / n_samples
        p_ij = contingency / n_samples

        mi = 0.0
        for i in range(contingency.shape[0]):
            for j in range(contingency.shape[1]):
                if p_ij[i, j] > 0 and p_i[i] > 0 and p_j[j] > 0:
                    mi += p_ij[i, j] * np.log(p_ij[i, j] / (p_i[i] * p_j[j]))

        h_true = -float(np.sum(p_i[p_i > 0] * np.log(p_i[p_i > 0])))
        h_pred = -float(np.sum(p_j[p_j > 0] * np.log(p_j[p_j > 0])))

        if h_true == 0 or h_pred == 0:
            return 1.0 if h_true == h_pred else 0.0

        return float(mi / np.sqrt(h_true * h_pred))