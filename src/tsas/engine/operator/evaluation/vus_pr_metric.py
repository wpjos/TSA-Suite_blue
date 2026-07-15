# -*- coding: utf-8 -*-

"""
VUS-PR 评价指标算子（Volume Under the Surface - PR）

同 :class:`VusRocMetric`，但积分 PR 曲线（Recall 横轴、Precision 纵轴）。
不平衡场景下比 VUS-ROC 更敏感。

核心组件:
    - VusPrMetricConfig: 配置类（继承 BaseMetricConfig）
    - VusPrMetric: VUS-PR 指标算子

使用示例::

    from tsas.engine.operator.evaluation import VusPrMetric

    op = VusPrMetric()
    result = op.run((y_truth, y_score))

    op = VusPrMetric(sliding_window=10, num_thresholds=100)
    result = op.run((y_truth, y_score))
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation._vus_utils import _curve_at_window
from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'VusPrMetricConfig',
    'VusPrMetric',
]


class VusPrMetricConfig(BaseMetricConfig):
    """VUS-PR 评价指标配置

    Attributes:
        sliding_window (int): 最大扩展窗口大小（采样点），默认 100
        num_thresholds (int): 每个窗口的阈值采样点数，默认 250
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"vus_pr": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    sliding_window: int = Field(default=100, ge=0, description="最大扩展窗口大小（采样点）")
    num_thresholds: int = Field(default=250, ge=2, description="每个窗口的阈值采样点数")
    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={"vus_pr": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class VusPrMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        VusPrMetricConfig,
        None,
    ],
):
    """VUS-PR 评价指标算子

    对 ``window = 0..sliding_window`` 内的 range-based PR-AUC 取均值。

    Input:
        y_truth: 真实标签数组（0/1 硬标签），shape (N,)
        y_score: 连续异常得分（越高越异常），shape (N,)

    Output:
        float: VUS-PR 值，范围通常 ``[0, 1]``；1 表示完美

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: float — 标量 VUS-PR
        MC: VusPrMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "vus_pr_metric"

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
        y_score = np.asarray(y_score, dtype=float).ravel()
        if y_true.shape != y_score.shape:
            raise ValueError(
                f"y_true 和 y_score 形状不一致: {y_true.shape} vs {y_score.shape}"
            )
        config = self.config
        sliding_window = 100 if config is None else config.sliding_window
        num_thresholds = 250 if config is None else config.num_thresholds
        pos_label = 1 if config is None else config.pos_label

        if sliding_window < 0:
            raise ValueError("sliding_window must be >= 0")

        y_true_bin = (y_true == pos_label)
        if int(sliding_window) == 0:
            _, auc = _curve_at_window(
                y_true_bin, y_score, 0, int(num_thresholds)
            )
            return float(auc)
        auc_sum = 0.0
        n_w = 0
        for w in range(int(sliding_window) + 1):
            _, a = _curve_at_window(y_true_bin, y_score, w, int(num_thresholds))
            auc_sum += a
            n_w += 1
        return auc_sum / n_w