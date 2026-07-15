# -*- coding: utf-8 -*-

"""
Macro-F1 评价指标算子

各类别 F1 分数的算术平均值。对所有类别赋相同权重，适用于类别不均衡场景。

核心组件:
    - MacroF1Config: 配置类（继承 BaseMetricConfig）
    - MacroF1Metric: Macro-F1 指标算子

使用示例::

    from tsas.engine.operator.evaluation import MacroF1Metric

    op = MacroF1Metric()
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
    'MacroF1Config',
    'MacroF1Metric',
]


class MacroF1Config(BaseMetricConfig):
    """Macro-F1 评价指标配置

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"macro_f1": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"macro_f1": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class MacroF1Metric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        MacroF1Config,
        None,
    ],
):
    """Macro-F1 评价指标算子

    计算各类别 F1 分数的算术平均值。每个类别单独计 TP/FP/FN，F1=2PR/(P+R)，
    最后跨类别取均值。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        float: Macro-F1 值，范围 ``[0, 1]``

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: float — 标量 Macro-F1
        MC: MacroF1Config — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "macro_f1_metric"

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
        f1s: list[float] = []
        for label in labels:
            tp = int(np.sum((y_true == label) & (y_pred == label)))
            fp = int(np.sum((y_true != label) & (y_pred == label)))
            fn = int(np.sum((y_true == label) & (y_pred != label)))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)

        return float(np.mean(f1s))