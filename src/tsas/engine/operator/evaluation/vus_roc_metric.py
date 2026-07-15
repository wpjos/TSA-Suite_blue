# -*- coding: utf-8 -*-

"""
VUS-ROC 评价指标算子（Volume Under the Surface - ROC）

把每个真实异常段沿时间轴两侧各扩展 ``sliding_window // 2`` 个采样点，
扩展带内使用 ``sqrt(1 - d / sliding_window)`` 衰减权重。
扫描全部阈值后对 range-aware TPR-FPR 曲线做梯形积分，并对
``window = 0..sliding_window`` 取均值——即为 VUS-ROC。

参考：
    Paparrizos, Boniol, Palpanas, Tsay, Elmore & Franklin,
    "Volume Under the Surface: A New Accuracy Evaluation Measure for
    Time-Series Anomaly Detection", PVLDB 2022.

核心组件:
    - VusRocMetricConfig: 配置类（继承 BaseMetricConfig）
    - VusRocMetric: VUS-ROC 指标算子

使用示例::

    from tsas.engine.operator.evaluation import VusRocMetric

    op = VusRocMetric()
    result = op.run((y_truth, y_score))

    op = VusRocMetric(sliding_window=10, num_thresholds=100)
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
    'VusRocMetricConfig',
    'VusRocMetric',
]


class VusRocMetricConfig(BaseMetricConfig):
    """VUS-ROC 评价指标配置

    Attributes:
        sliding_window (int): 最大扩展窗口大小（采样点），默认 100
        num_thresholds (int): 每个窗口的阈值采样点数，默认 250
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"vus_roc": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    sliding_window: int = Field(default=100, ge=0, description="最大扩展窗口大小（采样点）")
    num_thresholds: int = Field(default=250, ge=2, description="每个窗口的阈值采样点数")
    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={"vus_roc": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class VusRocMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        VusRocMetricConfig,
        None,
    ],
):
    """VUS-ROC 评价指标算子

    对 ``window = 0..sliding_window`` 内的 range-based ROC-AUC 取均值。
    TPR 同时乘以 ``existence_ratio``（每段至少被命中一次的比例），
    缓解"单点错配碰巧命中"。

    Input:
        y_truth: 真实标签数组（0/1 硬标签），shape (N,)
        y_score: 连续异常得分（越高越异常），shape (N,)

    Output:
        float: VUS-ROC 值，范围通常 ``[0, 1]``；1 表示完美

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: float — 标量 VUS-ROC
        MC: VusRocMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "vus_roc_metric"

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
            auc, _ = _curve_at_window(
                y_true_bin, y_score, 0, int(num_thresholds)
            )
            return float(auc)
        auc_sum = 0.0
        n_w = 0
        for w in range(int(sliding_window) + 1):
            a, _ = _curve_at_window(y_true_bin, y_score, w, int(num_thresholds))
            auc_sum += a
            n_w += 1
        return auc_sum / n_w