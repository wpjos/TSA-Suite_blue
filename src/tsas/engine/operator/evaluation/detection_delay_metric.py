# -*- coding: utf-8 -*-

"""
Detection Delay 评价指标算子

计算从每个真实异常事件开始到首次被检测到的延迟（样本点数）。
未检出时以事件长度作为延迟。

核心组件:
    - DetectionDelayConfig: 配置类（继承 BaseMetricConfig）
    - DetectionDelayMetric: 检测延迟指标算子

使用示例::

    from tsas.engine.operator.evaluation import DetectionDelayMetric

    op = DetectionDelayMetric(threshold=0.5)
    result = op.run((y_truth, y_score))
    print(result)  # 平均延迟

    op = DetectionDelayMetric(threshold=0.5, return_all=True)
    result = op.run((y_truth, y_score))  # 各事件延迟列表
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation._vus_utils import events_inclusive
from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'DetectionDelayConfig',
    'DetectionDelayMetric',
]


class DetectionDelayConfig(BaseMetricConfig):
    """检测延迟评价指标配置

    Attributes:
        threshold (float): 判定异常的阈值，``y_score >= threshold`` 视为检出
        pos_label (int): 正例标签值，默认 1
        return_all (bool): ``True`` 返回每事件延迟列表，``False`` 返回平均延迟
        check_before_event (bool): ``True`` 允许在事件开始前检测到（提前预警）
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"delay": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    threshold: float = Field(description="判定异常的阈值，y_score >= threshold 视为检出")
    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    return_all: bool = Field(
        default=False,
        description="True 返回每事件延迟列表；False 返回平均延迟",
    )
    check_before_event: bool = Field(
        default=False,
        description="True 允许在事件开始前检测到（提前预警）；False 仅在事件开始后检测",
    )
    main_scores: dict[str, str] | None = Field(
        default={"delay": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class DetectionDelayMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float | list[float],
        DetectionDelayConfig,
        None,
    ],
):
    """检测延迟评价指标算子

    对每个真实异常事件，搜索 ``y_score >= threshold`` 的最早位置，
    延迟 = ``max(0, first_idx - start)``。未检出时以事件长度作为延迟。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_score: 异常得分数组（越高越异常），与 y_truth 等长

    Output:
        - ``config.return_all=False``：平均延迟（float）
        - ``config.return_all=True``：各事件延迟列表（list[float]），单位为样本点数

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: float | list[float] — 平均延迟或各事件延迟列表
        MC: DetectionDelayConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "detection_delay_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> float | list[float]:
        y_true, y_score = x
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        if len(y_true) != len(y_score):
            raise ValueError(
                f"y_true 和 y_score 长度不一致: {len(y_true)} vs {len(y_score)}"
            )
        config = self.config
        if config is None:
            raise ValueError("DetectionDelayMetric 需要显式提供 threshold 配置")
        threshold = config.threshold
        pos_label = config.pos_label
        return_all = config.return_all
        check_before_event = config.check_before_event

        true_events = events_inclusive(y_true, pos_label)
        if len(true_events) == 0:
            return [] if return_all else 0.0

        delays: list[float] = []
        for start, end in true_events:
            search_start = 0 if check_before_event else start
            delay = float(end - start + 1)
            for i in range(search_start, end + 1):
                if y_score[i] >= threshold:
                    delay = float(max(0, i - start))
                    break
            delays.append(delay)

        if return_all:
            return delays
        return float(np.mean(delays))