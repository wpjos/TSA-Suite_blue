# -*- coding: utf-8 -*-

"""
Event-Based F1 评价指标算子

将连续异常点视为事件，计算事件级 F1：按"预测段与真实段至少重叠
``overlap_threshold`` 个点"判定匹配，再求 Precision/Recall/F1。

核心组件:
    - EventBasedF1Result: 事件级 F1 结果（Pydantic BaseModel）
    - EventBasedF1Config: 配置类（继承 BaseMetricConfig）
    - EventBasedF1Metric: 事件级 F1 指标算子

使用示例::

    from tsas.engine.operator.evaluation import EventBasedF1Metric

    op = EventBasedF1Metric()
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
    'EventBasedF1Result',
    'EventBasedF1Config',
    'EventBasedF1Metric',
]


class EventBasedF1Result(BaseModel):
    """事件级 F1 评价指标结果

    Attributes:
        n_true_events (int): 真实事件数
        n_pred_events (int): 预测事件数
        tp (int): 匹配的真实事件数
        fp (int): 未匹配的预测事件数
        fn (int): 未匹配的真实事件数
        precision (float): 事件级 Precision
        recall (float): 事件级 Recall
        f1 (float): 事件级 F1
    """
    model_config = ConfigDict(frozen=True)

    n_true_events: int = Field(description="真实事件数")
    n_pred_events: int = Field(description="预测事件数")
    tp: int = Field(description="匹配的真实事件数")
    fp: int = Field(description="未匹配的预测事件数")
    fn: int = Field(description="未匹配的真实事件数")
    precision: float = Field(description="事件级 Precision")
    recall: float = Field(description="事件级 Recall")
    f1: float = Field(description="事件级 F1")


class EventBasedF1Config(BaseMetricConfig):
    """事件级 F1 评价指标配置

    Attributes:
        pos_label (int): 正例标签值，默认 1
        overlap_threshold (int): 预测段与真实段最少重叠点数，默认 0
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"event_f1": "f1"}``
    """
    model_config = ConfigDict(frozen=True)

    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    overlap_threshold: int = Field(
        default=0,
        ge=0,
        description="预测事件与真实事件需要重叠的最少点数",
    )
    main_scores: dict[str, str] | None = Field(
        default={"event_f1": "f1"},
        description="主评分路径映射；float 字段使用对应属性名",
    )


class EventBasedF1Metric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        EventBasedF1Result,
        EventBasedF1Config,
        None,
    ],
):
    """事件级 F1 评价指标算子

    将 ``y_truth`` / ``y_pred`` 中的 ``pos_label`` 连续段视为事件，
    通过贪心匹配"预测事件与真实事件重叠点数 ≥ ``overlap_threshold``"
    的对来计算 TP/FP/FN 与 Precision/Recall/F1。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        EventBasedF1Result: 包含事件级 Precision/Recall/F1 与 TP/FP/FN 计数

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: EventBasedF1Result — 事件级指标结果
        MC: EventBasedF1Config — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "event_based_f1_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> EventBasedF1Result:
        y_true, y_pred = x
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true 和 y_pred 长度不一致: {len(y_true)} vs {len(y_pred)}"
            )
        config = self.config
        pos_label = 1 if config is None else config.pos_label
        overlap_threshold = 0 if config is None else config.overlap_threshold

        true_events = events_inclusive(y_true, pos_label)
        pred_events = events_inclusive(y_pred, pos_label)

        matched_true: set[int] = set()
        matched_pred: set[int] = set()
        for i, t_event in enumerate(true_events):
            for j, p_event in enumerate(pred_events):
                if j in matched_pred:
                    continue
                # 闭区间重叠判定
                if not (p_event[1] < t_event[0] or p_event[0] > t_event[1]):
                    overlap_start = max(t_event[0], p_event[0])
                    overlap_end = min(t_event[1], p_event[1])
                    overlap = overlap_end - overlap_start + 1
                    if overlap >= overlap_threshold:
                        matched_true.add(i)
                        matched_pred.add(j)
                        break

        tp = len(matched_true)
        fp = len(pred_events) - len(matched_pred)
        fn = len(true_events) - len(matched_true)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return EventBasedF1Result(
            n_true_events=len(true_events),
            n_pred_events=len(pred_events),
            tp=tp,
            fp=fp,
            fn=fn,
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
        )