# -*- coding: utf-8 -*-

"""
Range-AUC-PR 评价指标算子（单窗口）

把每个真实异常段沿时间轴两侧各扩展 ``sliding_window // 2`` 个采样点，
扩展带内使用 ``sqrt(1 - d / sliding_window)`` 的衰减权重；
扫描全部阈值后对 range-aware Recall-Precision 曲线做梯形积分。

与 VUS-PR 的区别：本指标只取固定窗口 ``sliding_window`` 单次计算，
不做 ``0..sliding_window`` 的均值。不平衡场景下比 Range-AUC-ROC 更敏感。

参考：
    Paparrizos, Boniol, Palpanas, Tsay, Elmore & Franklin,
    "Volume Under the Surface: A New Accuracy Evaluation Measure for
    Time-Series Anomaly Detection", PVLDB 2022.

核心组件:
    - RangeAucPrConfig: 配置类（继承 BaseMetricConfig）
    - RangeAucPrMetric: Range-AUC-PR 指标算子

使用示例::

    from tsas.engine.operator.evaluation import RangeAucPrMetric

    op = RangeAucPrMetric(sliding_window=10)
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
    'RangeAucPrConfig',
    'RangeAucPrMetric',
]


class RangeAucPrConfig(BaseMetricConfig):
    """Range-AUC-PR 评价指标配置

    Attributes:
        sliding_window (int): 扩展窗口大小（采样点），必须显式提供
        num_thresholds (int): 阈值采样点数，默认 250
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"range_auc_pr": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    sliding_window: int = Field(description="扩展窗口大小（采样点）")
    num_thresholds: int = Field(default=250, ge=2, description="阈值采样点数")
    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={"range_auc_pr": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class RangeAucPrMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        RangeAucPrConfig,
        None,
    ],
):
    """Range-AUC-PR 评价指标算子

    对固定 ``sliding_window`` 下的 range-aware Recall-Precision 曲线做
    梯形积分。不平衡场景下比 Range-AUC-ROC 更敏感。

    Input:
        y_truth: 真实标签数组（0/1 硬标签），shape (N,)
        y_score: 连续异常得分（越高越异常），shape (N,)

    Output:
        float: Range-AUC-PR 值，范围 ``[0, 1]``

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: float — 标量 Range-AUC-PR
        MC: RangeAucPrConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "range_auc_pr_metric"

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
        if config is None:
            raise ValueError("RangeAucPrMetric 需要显式提供 sliding_window 配置")
        sliding_window = int(config.sliding_window)
        num_thresholds = int(config.num_thresholds)
        pos_label = config.pos_label

        if sliding_window < 0:
            raise ValueError("sliding_window must be >= 0")
        if num_thresholds < 2:
            raise ValueError("num_thresholds must be >= 2")

        y_true_bin = (y_true == pos_label)
        _, auc = _curve_at_window(y_true_bin, y_score, sliding_window, num_thresholds)
        return float(auc)