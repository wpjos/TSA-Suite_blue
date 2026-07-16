"""
基础变换特征构造算子实现模块

提供 4 个面向通用信号变换的 transform 算子，均基于 IndependentMapFeature + Map 模式。
数据模型：每个 DataFrame 格子存放一个信号段（list/ndarray），compute 逐格计算产生等长序列（object ndarray 输出）。

特征分组：
- Group A: 逐元素变换 (2) — abs, sqrt
- Group B: 序列变换 (2) — diff, uni

所有算法逻辑源自 bqlib ops.transform 系列（abs/sqrt/diff/uni），
按 TSA-Suite 规范重写为 IndependentMapFeature 子类，保留 numpy 后端。
TSA-Suite 不依赖 bqlib。
"""

import numpy as np

from tsas.engine.operator.feature.construction._array_helpers import _apply_per_cell_array
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig,
    IndependentMapFeature,
)

__all__ = [
    'AbsFeature',
    'SqrtFeature',
    'DiffFeature',
    'UniFeature',
]


def _abs_1d(sig: np.ndarray) -> np.ndarray:
    """逐元素绝对值。"""
    return np.abs(np.asarray(sig, dtype=float))


def _sqrt_1d(sig: np.ndarray) -> np.ndarray:
    """逐元素平方根。空输入返回空数组。"""
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return np.array([])
    return np.sqrt(sig)


def _diff_1d(sig: np.ndarray) -> np.ndarray:
    """一阶差分（前置 x[0]，y[0]=0），输出长度与输入一致。"""
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return np.array([])
    prepend = np.take(sig, [0])
    return np.diff(sig, prepend=prepend)


def _uni_1d(sig: np.ndarray) -> np.ndarray:
    """逐元素除以总元素数（uniform normalization）。空输入返回空数组。"""
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return np.array([])
    return sig / sig.size


# ============================================================================
# Group A: 逐元素变换 (2)
# ============================================================================

class AbsFeature(IndependentMapFeature[BaseFeatureConfig]):
    """绝对值特征：每列信号段取 |x|，输出等长数组。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是对应信号段的 |x|
    """

    @classmethod
    def name(cls) -> str:
        return "abs_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell_array(x, _abs_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "abs")


class SqrtFeature(IndependentMapFeature[BaseFeatureConfig]):
    """平方根特征：每列信号段逐元素 sqrt，输出等长数组。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是对应信号段的 sqrt
    """

    @classmethod
    def name(cls) -> str:
        return "sqrt_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell_array(x, _sqrt_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "sqrt")


# ============================================================================
# Group B: 序列变换 (2)
# ============================================================================

class DiffFeature(IndependentMapFeature[BaseFeatureConfig]):
    """一阶差分特征：diff(x)，前值补 0 保持等长，输出 y[0]=0。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是对应信号段的差分（等长，首值 0）
    """

    @classmethod
    def name(cls) -> str:
        return "diff_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell_array(x, _diff_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "diff")


class UniFeature(IndependentMapFeature[BaseFeatureConfig]):
    """Uniform 归一化特征：每列信号段逐元素除以总元素数，输出等长数组。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是对应信号段的 x/N
    """

    @classmethod
    def name(cls) -> str:
        return "uni_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell_array(x, _uni_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "uni")
