# -*- coding: utf-8 -*-

"""
ROC-AUC 评价指标算子

``AUC = ∫ TPR d(FPR)``，使用梯形法则计算 ROC 曲线下面积，
衡量分类器整体排序能力；值越大表示排序能力越强。

核心组件:
    - RocAucMetricConfig: 配置类（继承 BaseMetricConfig）
    - RocAucMetric: ROC-AUC 指标算子

使用示例::

    from tsas.engine.operator.evaluation import RocAucMetric

    op = RocAucMetric()
    result = op.run((y_truth, y_score))
    print(result)  # float

    # HPO 集成
    op = RocAucMetric(main_scores={"auc": "_"})
    scores = op.scores((y_truth, y_score))
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'RocAucMetricConfig',
    'RocAucMetric',
]


class RocAucMetricConfig(BaseMetricConfig):
    """ROC-AUC 评价指标配置

    Attributes:
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"roc_auc": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={"roc_auc": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class RocAucMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        RocAucMetricConfig,
        None,
    ],
):
    """ROC-AUC 评价指标算子

    计算 ROC 曲线下面积，使用梯形法则积分（FPR 横轴、TPR 纵轴）。
    与 sklearn 兼容；同时支持 numpy 的 ``trapezoid``（>=2.0）和 ``trapz``。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_score: 预测得分数组（越高越可能是正例），与 y_truth 等长

    Output:
        float: ROC-AUC 值，范围 ``[0, 1]``；1 表示完美排序

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: float — 标量 AUC
        MC: RocAucMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "roc_auc_metric"

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

        fpr_arr, tpr_arr = self._roc_curve(y_true, y_score, pos_label)
        if hasattr(np, "trapz"):
            return float(np.trapz(tpr_arr, fpr_arr))
        return float(np.trapezoid(tpr_arr, fpr_arr))

    @staticmethod
    def _roc_curve(
        y_true: np.ndarray,
        y_score: np.ndarray,
        pos_label: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算 ROC 曲线的 (FPR, TPR) 数据点。

        采用与 sklearn 兼容的合并排序 + 阈值扫描算法：
        1. 按 ``y_score`` 降序（mergesort 稳定排序）排列
        2. 对每个 distinct value 边界累加 TP/FP
        3. 累加序列前置 0 作为起点
        4. 归一化得到 FPR/TPR

        Args:
            y_true: 真实标签数组
            y_score: 预测得分数组
            pos_label: 正例标签值

        Returns:
            (fpr_arr, tpr_arr): 递增排序的 FPR 和 TPR 数组
        """
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        tps = np.cumsum(y_true == pos_label)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        n_pos = np.sum(y_true == pos_label)
        n_neg = len(y_true) - n_pos

        fpr_arr = fps / n_neg if n_neg > 0 else fps
        tpr_arr = tps / n_pos if n_pos > 0 else tps
        return fpr_arr, tpr_arr