# -*- coding: utf-8 -*-

"""
特征构造算子包。

该包提供用于时间序列特征工程的构造算子，
支持从原始数据中提取、构建和转换特征。

包含:
- 基类: base.py 中定义的 8 个编排基类
- 简单特征: simple_feature.py 中的基础实现
- 信号特征: signal_feature.py 中的 31 个预测性维护特征
"""

from bianque.engine.operator.feature.construction.base import (
    BaseFeatureConfig,
    WindowFeatureConfig,
    Alignment,
    Padding,
    IndependentMapFeature,
    IndependentWindowFeature,
    JointMapFeature,
    JointWindowFeature,
    LearnableIndependentMapFeature,
    LearnableIndependentWindowFeature,
    LearnableJointMapFeature,
    LearnableJointWindowFeature,
)

from bianque.engine.operator.feature.construction.simple_feature import (
    SquareConfig,
    SquareFeature,
    PolynomialConfig,
    PolynomialFeature,
    RollingMeanConfig,
    RollingMeanFeature,
    ColumnMedianConfig,
    ColumnMedianFeature,
    PCAConfig,
    PCAState,
    PCAFeature,
)

from bianque.engine.operator.feature.construction.signal_feature import (
    SampleRateFeatureConfig,
    BandFeatureConfig,
    AverageKurtosisConfig,
    # Group A: 简单统计特征
    MeanSquareFeature,
    VarianceFeature,
    RmsFeature,
    PeakPeakFeature,
    ShapeFactorFeature,
    CrestFeature,
    ImpulseFeature,
    ClearanceFeature,
    SkewnessFeature,
    KurtosisFeature,
    GiniIndexFeature,
    # Group B: 需要采样率的特征
    SpectralEntropyFeature,
    RoughnessFeature,
    SharpnessFeature,
    # Group C: 频域特征
    SpectralCentroidFeature,
    MeanSquareFrequencyFeature,
    RmsFrequencyFeature,
    FrequencyVarianceFeature,
    FrequencyStdFeature,
    # Group D: 复合特征
    EnvelopeRmsFeature,
    AverageKurtosisFeature,
    HnrFeature,
    # Group E: 频带特征
    BandKurtosisFeature,
    BandRmsFeature,
    BandHnrFeature,
)

__all__ = [
    # 基类 Config
    'BaseFeatureConfig',
    'WindowFeatureConfig',
    'Alignment',
    'Padding',
    # 基类
    'IndependentMapFeature',
    'IndependentWindowFeature',
    'JointMapFeature',
    'JointWindowFeature',
    'LearnableIndependentMapFeature',
    'LearnableIndependentWindowFeature',
    'LearnableJointMapFeature',
    'LearnableJointWindowFeature',
    # 简单特征
    'SquareConfig',
    'SquareFeature',
    'PolynomialConfig',
    'PolynomialFeature',
    'RollingMeanConfig',
    'RollingMeanFeature',
    'ColumnMedianConfig',
    'ColumnMedianFeature',
    'PCAConfig',
    'PCAState',
    'PCAFeature',
    # 信号特征 Config
    'SampleRateFeatureConfig',
    'BandFeatureConfig',
    'AverageKurtosisConfig',
    # Group A: 简单统计特征
    'MeanSquareFeature',
    'VarianceFeature',
    'RmsFeature',
    'PeakPeakFeature',
    'ShapeFactorFeature',
    'CrestFeature',
    'ImpulseFeature',
    'ClearanceFeature',
    'SkewnessFeature',
    'KurtosisFeature',
    'GiniIndexFeature',
    # Group B: 需要采样率的特征
    'SpectralEntropyFeature',
    'RoughnessFeature',
    'SharpnessFeature',
    # Group C: 频域特征
    'SpectralCentroidFeature',
    'MeanSquareFrequencyFeature',
    'RmsFrequencyFeature',
    'FrequencyVarianceFeature',
    'FrequencyStdFeature',
    # Group D: 复合特征
    'EnvelopeRmsFeature',
    'AverageKurtosisFeature',
    'HnrFeature',
    # Group E: 频带特征
    'BandKurtosisFeature',
    'BandRmsFeature',
    'BandHnrFeature',
]
