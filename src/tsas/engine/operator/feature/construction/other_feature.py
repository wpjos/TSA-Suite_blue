"""
其他特征构造算子实现模块

提供 2 个面向特殊信号分析需求的 reduce 标量算子，均基于 IndependentMapFeature + Map 模式。
数据模型：每个 DataFrame 格子存放一个信号段（list/ndarray），compute 逐格计算产生 float 标量。

特征分组：
- Group A: 趋势分析 (1) — slope（OLS 回归斜率，量化信号线性趋势）
- Group B: 包络谐波比 (1) — ehr（包络自相关比，检测调制周期性）

所有算法逻辑源自 bqlib ops.reduce 系列（slope/ehr），
按 TSA-Suite 规范重写为 IndependentMapFeature 子类，保留 numpy 后端。
TSA-Suite 不依赖 bqlib。
"""

import warnings

import numpy as np

from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig,
    IndependentMapFeature,
)
from tsas.engine.operator.feature.construction.signal_feature import _apply_per_cell

__all__ = [
    'SlopeFeature',
    'EhrFeature',
]


def _slope_1d(y: np.ndarray) -> float:
    """OLS 回归斜率（针对 x = [0, 1, ..., n-1]）。

    闭式简化:
        x̄ = (n-1)/2
        denominator = n(n²-1)/12
        numerator = dot(x, y) - n · x̄ · ȳ
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    denominator = n * (n * n - 1) / 12.0
    y_mean = np.mean(y)
    numerator = np.dot(np.arange(n, dtype=float), y) - n * x_mean * y_mean

    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def _hilbert_envelope_numpy(x: np.ndarray) -> np.ndarray:
    """Hilbert 包络（scipy 不可用时的 numpy 回退实现）。"""
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[1:n // 2] = 2
        h[n // 2] = 1
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * h))


def _hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """计算解析信号包络（优先 scipy，缺失时回退 numpy FFT）。"""
    try:
        from scipy.signal import hilbert  # noqa: PLC0415
        return np.abs(hilbert(x))
    except ImportError:
        warnings.warn(
            "scipy not available; falling back to numpy FFT Hilbert transform. "
            "Install scipy for better performance.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _hilbert_envelope_numpy(x)


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    """归一化自相关函数（仅正滞后）。"""
    n = len(x)
    x = x - np.mean(x)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    f = np.fft.fft(x, n=fft_size)
    acf = np.real(np.fft.ifft(f * np.conj(f)))[:n]
    if acf[0] == 0:
        return np.zeros(n)
    return acf / acf[0]


def _ehr_1d(x: np.ndarray) -> float:
    """包络相关比（Envelope-to-Harmonic Ratio）。

    检测包络信号的周期性：取包络自相关函数在第一个负相关之后的峰值，
    衡量包络的能量谐波比（取值 [0, 1]）。
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0

    envelope = _hilbert_envelope(x)
    if np.std(envelope) == 0:
        return 0.0

    ac_env = _autocorrelation(envelope)

    neg_mask = ac_env < 0
    if not neg_mask.any():
        return 0.0
    first_neg = int(np.argmax(neg_mask))
    if first_neg >= len(ac_env) - 1:
        return 0.0

    return float(ac_env[first_neg:].max())


# ============================================================================
# Group A: 趋势分析 (1)
# ============================================================================

class SlopeFeature(IndependentMapFeature[BaseFeatureConfig]):
    """OLS 回归斜率特征：量化每列信号段的线性趋势。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "slope_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _slope_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "slope")


# ============================================================================
# Group B: 包络谐波比 (1)
# ============================================================================

class EhrFeature(IndependentMapFeature[BaseFeatureConfig]):
    """包络谐波比特征（Envelope-to-Harmonic Ratio）：衡量包络自相关周期性。

    取值 [0, 1]，越大表示包络信号周期性越强（典型应用：轴承故障调制检测）。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "ehr_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _ehr_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "ehr")
