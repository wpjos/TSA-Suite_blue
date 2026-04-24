# -*- coding: utf-8 -*-

"""
评价指标算子包

提供基于 BaseMetricOperator 的评价指标算子实现，支持 HPO 单目标/多目标优化。

模块组成:
    - base_metric: 基础类型（BaseMetricConfig、BaseMetricOperator）
    - binary_classification: 二分类指标（离散标签输入）
    - binary_curve: 二分类曲线指标（连续分数输入）
    - multi_classification: 多分类指标
    - point_adjust: 点调整指标（PA-F1，时序异常检测）
    - self_evaluation: 无标签自评估指标
    - regular: 旧实现（deprecated，保留向后兼容）

使用示例::

    from bianque.engine.operator.evaluation import (
        BinaryClassificationMetric,
        BinaryClassificationCurve,
        MultipleClassificationMetric,
        PointAdjust,
        SelfEvaluation,
    )

    # 二分类离散标签
    op = BinaryClassificationMetric()
    result = op.run((y_truth, y_predict))
    print(result.f1, result.far)

    # 二分类连续分数（曲线指标）
    op = BinaryClassificationCurve()
    result = op.run((labels, scores))
    print(result.auc_roc, result.best_f1)

    # HPO 集成
    op = BinaryClassificationMetric(main_scores={"f1": "f1"})
    scores = op.scores((y_truth, y_predict))  # -> {"f1": 0.85}
"""

from bianque.engine.operator.evaluation.base import (
    MR,
    MC,
    BaseMetricConfig,
    BaseMetricOperator,
)

from bianque.engine.operator.evaluation.binary_classification import (
    BinaryClassificationResult,
    BinaryClassificationConfig,
    BinaryClassificationMetric,
)

from bianque.engine.operator.evaluation.binary_curve import (
    BinaryClassificationCurveResult,
    BinaryClassificationCurveConfig,
    BinaryClassificationCurve,
)

from bianque.engine.operator.evaluation.multi_classification import (
    PerLabelMetricResult,
    MultiClassificationMetricResult,
    MultiClassificationMetricConfig,
    MultipleClassificationMetric,
)

from bianque.engine.operator.evaluation.point_adjust import (
    PointAdjustResult,
    PointAdjustConfig,
    PointAdjust,
)

from bianque.engine.operator.evaluation.self_evaluation import (
    SelfEvaluationConfig,
    SelfEvaluation,
)

# 旧实现（deprecated，保留向后兼容）
# 注意：旧 BinaryClassificationMetric / MultipleClassificationMetric 与新算子同名，
#       以 Legacy 前缀导出以避免覆盖新算子
from bianque.engine.v0.operator.evaluation.regular import (
    BinaryClassificationMetricBasic as LegacyBinaryClassificationMetricBasic,
    BinaryClassificationMetric as LegacyBinaryClassificationMetric,
    BinaryClassificationMetricCurve as LegacyBinaryClassificationMetricCurve,
    MultipleClassificationMetric as LegacyMultipleClassificationMetric,
    binary_classification_metric,
    binary_classification_metric_curve,
    multiple_classification_metric,
)

__all__ = [
    # 基础类型
    'MR',
    'MC',
    'BaseMetricConfig',
    'BaseMetricOperator',

    # 二分类离散标签指标
    'BinaryClassificationResult',
    'BinaryClassificationConfig',
    'BinaryClassificationMetric',

    # 二分类曲线指标
    'BinaryClassificationCurveResult',
    'BinaryClassificationCurveConfig',
    'BinaryClassificationCurve',

    # 多分类指标
    'PerLabelMetricResult',
    'MultiClassificationMetricResult',
    'MultiClassificationMetricConfig',
    'MultipleClassificationMetric',

    # 点调整指标
    'PointAdjustResult',
    'PointAdjustConfig',
    'PointAdjust',

    # 无标签自评估
    'SelfEvaluationConfig',
    'SelfEvaluation',

    # 旧实现（deprecated，Legacy 前缀）
    'LegacyBinaryClassificationMetricBasic',
    'LegacyBinaryClassificationMetric',
    'LegacyBinaryClassificationMetricCurve',
    'LegacyMultipleClassificationMetric',
    'binary_classification_metric',
    'binary_classification_metric_curve',
    'multiple_classification_metric',
]