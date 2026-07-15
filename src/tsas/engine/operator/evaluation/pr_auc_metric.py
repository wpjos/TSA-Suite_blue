# -*- coding: utf-8 -*-

"""
PR-AUC 评价指标算子

``PR-AUC = ∫ Precision d(Recall)``，使用梯形法则计算 PR 曲线下面积，
对正例类别的排序质量更敏感（不平衡场景下比 ROC-AUC 更敏感）。

核心组件:
    - PrAucMetricConfig: 配置类（继承 BaseMetricConfig）
    - PrAucMetric: PR-AUC 指标算子

使用示例::

    from tsas.engine.operator.evaluation import PrAucMetric

    op = PrAucMetric()
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
    'PrAucMetricConfig',
    'PrAucMetric',
]


class PrAucMetricConfig(BaseMetricConfig):
    """PR-AUC 评价指标配置

    Attributes:
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"pr_auc": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={"pr_auc": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class PrAucMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        PrAucMetricConfig,
        None,
    ],
):
    """PR-AUC 评价指标算子

    计算 PR 曲线下面积，使用梯形法则积分（Recall 横轴、Precision 纵轴）。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_score: 预测得分数组（越高越可能是正例），与 y_truth 等长

    Output:
        float: PR-AUC 值，范围 ``[0, 1]``

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: float — 标量 PR-AUC
        MC: PrAucMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "pr_auc_metric"

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

        ps, rs = self._precision_recall_curve(y_true, y_score, pos_label)
        if hasattr(np, "trapz"):
            return float(-np.trapz(ps, rs))
        return float(-np.trapezoid(ps, rs))

    @staticmethod
    def _precision_recall_curve(
        y_true: np.ndarray,
        y_score: np.ndarray,
        pos_label: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算 PR 曲线的 (Precision, Recall) 数据点。

        Args:
            y_true: 真实标签数组
            y_score: 预测得分数组
            pos_label: 正例标签值

        Returns:
            (ps, rs): Precision 和 Recall 数组
        """
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        tps = np.cumsum(y_true == pos_label)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

        ps = tps / (tps + fps)
        rs = tps / np.sum(y_true == pos_label)

        ps = np.r_[ps[::-1], 1]
        rs = np.r_[rs[::-1], 0]
        return ps, rs