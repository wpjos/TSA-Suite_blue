"""
基础统计特征构造算子实现模块

提供 5 个面向通用统计分析的 reduce 标量算子，均基于 IndependentMapFeature + Map 模式。
数据模型：每个 DataFrame 格子存放一个信号段（list/ndarray），compute 逐格计算产生 float 标量。

特征分组：
- Group A: 基础统计量 (5) — max, min, mean, std, sum

所有算法逻辑源自 bqlib ops.reduce 系列（max/min/mean/std/sum），
按 TSA-Suite 规范重写为 IndependentMapFeature 子类，保留 numpy 后端。
TSA-Suite 不依赖 bqlib。
"""

import numpy as np

from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig,
    IndependentMapFeature,
)
from tsas.engine.operator.feature.construction.signal_feature import _apply_per_cell

__all__ = [
    'MaxFeature',
    'MinFeature',
    'MeanFeature',
    'StdFeature',
    'SumFeature',
]


def _max_1d(sig: np.ndarray) -> float:
    """最大值。空输入抛出 ValueError。"""
    if sig.size == 0:
        raise ValueError("max of empty array")
    return float(np.max(sig))


def _min_1d(sig: np.ndarray) -> float:
    """最小值。空输入抛出 ValueError。"""
    if sig.size == 0:
        raise ValueError("min of empty array")
    return float(np.min(sig))


def _mean_1d(sig: np.ndarray) -> float:
    """算术平均。空输入返回 0.0。"""
    if sig.size == 0:
        return 0.0
    return float(np.mean(sig))


def _std_1d(sig: np.ndarray) -> float:
    """样本标准差（ddof=1）。空输入或单元素返回 0.0。"""
    if sig.size == 0:
        return 0.0
    result = float(np.std(sig, ddof=1))
    if np.isnan(result):
        return 0.0
    return result


def _sum_1d(sig: np.ndarray) -> float:
    """总和。空输入返回 0.0。"""
    if sig.size == 0:
        return 0.0
    return float(np.sum(sig))


# ============================================================================
# Group A: 基础统计量 (5)
# ============================================================================

class MaxFeature(IndependentMapFeature[BaseFeatureConfig]):
    """最大值特征：每列信号段取 max。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "max_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _max_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "max")


class MinFeature(IndependentMapFeature[BaseFeatureConfig]):
    """最小值特征：每列信号段取 min。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "min_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _min_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "min")


class MeanFeature(IndependentMapFeature[BaseFeatureConfig]):
    """算术平均特征：每列信号段取 mean。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "mean_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _mean_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "mean")


class StdFeature(IndependentMapFeature[BaseFeatureConfig]):
    """样本标准差特征（ddof=1）：每列信号段取 std。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "std_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _std_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "std")


class SumFeature(IndependentMapFeature[BaseFeatureConfig]):
    """总和特征：每列信号段取 sum。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "sum_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _sum_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "sum")
