# -*- coding: utf-8 -*-

"""
Micro-F1 评价指标算子

汇总全局 TP/FP/FN 后计算的 F1 分数。在多分类中 Micro-F1 等同于 Accuracy。

核心组件:
    - MicroF1Config: 配置类（继承 BaseMetricConfig）
    - MicroF1Metric: Micro-F1 指标算子

使用示例::

    from tsas.engine.operator.evaluation import MicroF1Metric

    op = MicroF1Metric()
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
    'MicroF1Config',
    'MicroF1Metric',
]


class MicroF1Config(BaseMetricConfig):
    """Micro-F1 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"micro_f1": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"micro_f1": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class MicroF1Metric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        MicroF1Config,
        None,
    ],
):
    """Micro-F1 评价指标算子

    对每个类别累加 TP/FP/FN 得到全局总计，再按二分类公式计算 F1。
    Micro-F1 在多分类场景中等同于 Accuracy。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        float: Micro-F1 值，范围 ``[0, 1]``

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: float — 标量 Micro-F1
        MC: MicroF1Config — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "micro_f1_metric"

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
        tp_total = 0
        fp_total = 0
        fn_total = 0
        for label in labels:
            tp_total += int(np.sum((y_true == label) & (y_pred == label)))
            fp_total += int(np.sum((y_true != label) & (y_pred == label)))
            fn_total += int(np.sum((y_true == label) & (y_pred != label)))

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return float(f1)