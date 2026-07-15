# -*- coding: utf-8 -*-

"""
Cohen's Kappa 评价指标算子

衡量分类一致性并消除随机因素的影响；``kappa = (p_o - p_e) / (1 - p_e)``。
``p_e == 1`` 时返回 1.0。

核心组件:
    - CohenKappaConfig: 配置类（继承 BaseMetricConfig）
    - CohenKappaMetric: Cohen's Kappa 指标算子

使用示例::

    from tsas.engine.operator.evaluation import CohenKappaMetric

    op = CohenKappaMetric()
    result = op.run((y_truth, y_pred))
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'CohenKappaConfig',
    'CohenKappaMetric',
]


class CohenKappaConfig(BaseMetricConfig):
    """Cohen's Kappa 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"cohen_kappa": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"cohen_kappa": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class CohenKappaMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        CohenKappaConfig,
        None,
    ],
):
    """Cohen's Kappa 评价指标算子

    ``kappa = (p_o - p_e) / (1 - p_e)``，取值范围 ``[-1, 1]``。
    ``p_e == 1`` 时返回 1.0（与 bqlib 一致）。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        float: Kappa 值；1 表示完全一致，0 表示与随机相当，-1 表示完全相反

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: float — 标量 Kappa
        MC: CohenKappaConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "cohen_kappa_metric"

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

        labels = np.unique(np.concatenate([y_true, y_pred]))
        n = len(y_true)
        n_labels = len(labels)
        label_to_idx = {label: i for i, label in enumerate(labels)}

        conf_mat = np.zeros((n_labels, n_labels), dtype=int)
        for t, p in zip(y_true, y_pred):
            conf_mat[label_to_idx[t], label_to_idx[p]] += 1

        p_o = float(np.trace(conf_mat)) / n
        row_totals = conf_mat.sum(axis=1)
        col_totals = conf_mat.sum(axis=0)
        p_e = float(np.sum(row_totals * col_totals)) / (n * n)

        if p_e == 1:
            return 1.0
        return float((p_o - p_e) / (1 - p_e))