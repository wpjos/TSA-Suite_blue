# -*- coding: utf-8 -*-

"""
Soft Event-Based F1 评价指标算子

基于 IoU 的软事件级 F1：每个真实/预测事件都找对侧 IoU 最大的匹配，
Precision/Recall = 所有事件匹配度的均值，部分重叠给部分分。

核心组件:
    - SoftEventBasedF1Result: 软事件级 F1 结果（Pydantic BaseModel）
    - SoftEventBasedF1Config: 配置类（继承 BaseMetricConfig）
    - SoftEventBasedF1Metric: 软事件级 F1 指标算子

使用示例::

    from tsas.engine.operator.evaluation import SoftEventBasedF1Metric

    op = SoftEventBasedF1Metric()
    result = op.run((y_truth, y_pred))
    print(result.f1)
"""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.evaluation._vus_utils import events_inclusive
from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'SoftEventBasedF1Result',
    'SoftEventBasedF1Config',
    'SoftEventBasedF1Metric',
]


class SoftEventBasedF1Result(BaseModel):
    """软事件级 F1 评价指标结果

    Attributes:
        n_true_events (int): 真实事件数
        n_pred_events (int): 预测事件数
        precision (float): 软事件级 Precision（IoU 均值）
        recall (float): 软事件级 Recall（IoU 均值）
        f1 (float): 软事件级 F1
    """
    model_config = ConfigDict(frozen=True)

    n_true_events: int = Field(description="真实事件数")
    n_pred_events: int = Field(description="预测事件数")
    precision: float = Field(description="软事件级 Precision（IoU 均值）")
    recall: float = Field(description="软事件级 Recall（IoU 均值）")
    f1: float = Field(description="软事件级 F1")


class SoftEventBasedF1Config(BaseMetricConfig):
    """软事件级 F1 评价指标配置

    Attributes:
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"soft_event_f1": "f1"}``
    """
    model_config = ConfigDict(frozen=True)

    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={"soft_event_f1": "f1"},
        description="主评分路径映射；float 字段使用对应属性名",
    )


class SoftEventBasedF1Metric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        SoftEventBasedF1Result,
        SoftEventBasedF1Config,
        None,
    ],
):
    """软事件级 F1 评价指标算子

    把"事件是否重叠"的二值判定换成 IoU 连续得分：每个真实事件/预测事件
    找到与之 IoU 最大的对侧事件，对应 IoU 作为该事件的"匹配度"。
    Precision/Recall = 所有事件匹配度的均值，完全对齐得 1.0。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        SoftEventBasedF1Result: 包含软事件级 Precision/Recall/F1

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: SoftEventBasedF1Result — 软事件级指标结果
        MC: SoftEventBasedF1Config — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "soft_event_based_f1_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> SoftEventBasedF1Result:
        y_true, y_pred = x
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true 和 y_pred 长度不一致: {len(y_true)} vs {len(y_pred)}"
            )
        config = self.config
        pos_label = 1 if config is None else config.pos_label

        true_events = events_inclusive(y_true, pos_label)
        pred_events = events_inclusive(y_pred, pos_label)

        if not true_events and not pred_events:
            return SoftEventBasedF1Result(
                n_true_events=0,
                n_pred_events=0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
            )
        if not true_events or not pred_events:
            return SoftEventBasedF1Result(
                n_true_events=len(true_events),
                n_pred_events=len(pred_events),
                precision=0.0,
                recall=0.0,
                f1=0.0,
            )

        recall_contrib = [max(_iou(p, t) for p in pred_events) for t in true_events]
        precision_contrib = [max(_iou(p, t) for t in true_events) for p in pred_events]

        rec = float(np.mean(recall_contrib))
        prec = float(np.mean(precision_contrib))
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

        return SoftEventBasedF1Result(
            n_true_events=len(true_events),
            n_pred_events=len(pred_events),
            precision=prec,
            recall=rec,
            f1=f1,
        )


def _iou(pred_event: tuple[int, int], true_event: tuple[int, int]) -> float:
    """计算两个事件的 Intersection over Union。

    Args:
        pred_event: 预测事件 ``(start, end)``（闭区间）
        true_event: 真实事件 ``(start, end)``（闭区间）

    Returns:
        IoU 值，范围 ``[0, 1]``。
    """
    p_start, p_end = pred_event
    t_start, t_end = true_event
    inter_start = max(p_start, t_start)
    inter_end = min(p_end, t_end)
    inter = max(0, inter_end - inter_start + 1)
    if inter == 0:
        return 0.0
    union = (p_end - p_start + 1) + (t_end - t_start + 1) - inter
    return inter / union if union > 0 else 0.0