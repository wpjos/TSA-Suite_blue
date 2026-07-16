"""
频域变换特征构造算子实现模块

提供 6 个面向频谱分析的 transform 算子，均基于 IndependentMapFeature + Map 模式。
数据模型：每个 DataFrame 格子存放一个信号段（list/ndarray），compute 逐格计算产生变长序列（object ndarray 输出）。

特征分组：
- Group A: 实数 FFT 族 (2) — rfft, rfftfreq
- Group B: 完整 FFT 族 (2) — fft, fftfreq
- Group C: 包络 (1) — envelope
- Group D: 谐波乘积谱 (1) — hps

所有算法逻辑源自 bqlib ops.transform 系列（rfft/rfftfreq/fft/fftfreq/envelope/hps），
按 TSA-Suite 规范重写为 IndependentMapFeature 子类，保留 numpy 后端。
TSA-Suite 不依赖 bqlib，仅直接 import numpy / scipy。
"""

import warnings
from typing import Literal

import numpy as np
from pydantic import Field

from tsas.engine.operator.feature.construction._array_helpers import _apply_per_cell_array
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig,
    IndependentMapFeature,
)

__all__ = [
    # Group A: 实数 FFT
    'RfftFeature',
    'RfftFreqFeature',
    # Group B: 完整 FFT
    'FftFeature',
    'FftFreqFeature',
    # Group C: 包络
    'EnvelopeFeature',
    # Group D: 谐波乘积谱
    'HpsFeature',
]


# ============================================================================
# Hilbert 解析信号（envelope 用，scipy 不可用时降级到 numpy FFT）
# ============================================================================

def _analytic_signal_numpy(x: np.ndarray) -> np.ndarray:
    """Hilbert 解析信号（scipy 不可用时的 numpy 回退实现）。"""
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
    return np.fft.ifft(X * h)


def _analytic_signal(x: np.ndarray) -> np.ndarray:
    """计算解析信号。优先使用 scipy.signal.hilbert，不可用时回退到 numpy FFT。"""
    try:
        from scipy.signal import hilbert  # noqa: PLC0415
        return hilbert(x)
    except ImportError:
        warnings.warn(
            "scipy not available; falling back to numpy FFT Hilbert transform. "
            "Install scipy for better performance.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _analytic_signal_numpy(x)


# ============================================================================
# Config 定义
# ============================================================================

class RfftFeatureConfig(BaseFeatureConfig):
    """RFFT 特征 Config。"""
    output: Literal["complex", "magnitude"] = Field(
        default="complex",
        description="complex 返回复数谱，magnitude 返回 |X[k]|",
    )


class FftFeatureConfig(BaseFeatureConfig):
    """完整 FFT 特征 Config。"""
    output: Literal["complex", "magnitude"] = Field(
        default="complex",
        description="complex 返回复数谱，magnitude 返回 |X[k]|",
    )


class FrequencyFeatureConfig(BaseFeatureConfig):
    """需要采样率的频轴特征 Config。"""
    sample_rate: float = Field(gt=0, description="采样率 (Hz)")


class HpsFeatureConfig(BaseFeatureConfig):
    """谐波乘积谱 Config。"""
    n_harmonics: int = Field(default=5, ge=1, description="谐波数 H（默认 5）")


# ============================================================================
# Group A: 实数 FFT (2)
# ============================================================================

class RfftFeature(IndependentMapFeature[RfftFeatureConfig]):
    """实数 FFT 特征：每列信号段做 rfft，输出 N//2+1 半谱。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是对应信号段的 rfft 结果（复数或幅值，长度 N//2+1）
    """

    @classmethod
    def name(cls) -> str:
        return "rfft_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        output = params.get('output', 'complex')

        def _rfft_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.array([])
            result = np.fft.rfft(sig)
            if output == "magnitude":
                return np.abs(result)
            return result

        return _apply_per_cell_array(x, _rfft_1d)

    def _get_compute_params(self):
        return {'output': self.config.output}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "rfft")


class RfftFreqFeature(IndependentMapFeature[FrequencyFeatureConfig]):
    """rFFT 频率轴特征：返回 rfft 对应的非负频率轴。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是长度为 N//2+1 的非负频率值数组
    """

    @classmethod
    def name(cls) -> str:
        return "rfftfreq_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 1.0)

        def _rfftfreq_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.array([])
            return np.fft.rfftfreq(len(sig), d=1.0 / sample_rate)

        return _apply_per_cell_array(x, _rfftfreq_1d)

    def _get_compute_params(self):
        return {'sample_rate': self.config.sample_rate}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "rfftfreq")


# ============================================================================
# Group B: 完整 FFT (2)
# ============================================================================

class FftFeature(IndependentMapFeature[FftFeatureConfig]):
    """完整 FFT 特征：每列信号段做 fft，输出 N 点完整频谱。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是对应信号段的 fft 结果（复数或幅值，长度 N）
    """

    @classmethod
    def name(cls) -> str:
        return "fft_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        output = params.get('output', 'complex')

        def _fft_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.array([])
            result = np.fft.fft(sig)
            if output == "magnitude":
                return np.abs(result)
            return result

        return _apply_per_cell_array(x, _fft_1d)

    def _get_compute_params(self):
        return {'output': self.config.output}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "fft")


class FftFreqFeature(IndependentMapFeature[FrequencyFeatureConfig]):
    """FFT 频率轴特征：返回 fft 对应的完整频率轴（含负频率）。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是长度为 N 的频率值数组
    """

    @classmethod
    def name(cls) -> str:
        return "fftfreq_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 1.0)

        def _fftfreq_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.array([])
            return np.fft.fftfreq(len(sig), d=1.0 / sample_rate)

        return _apply_per_cell_array(x, _fftfreq_1d)

    def _get_compute_params(self):
        return {'sample_rate': self.config.sample_rate}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "fftfreq")


# ============================================================================
# Group C: 包络 (1)
# ============================================================================

class EnvelopeFeature(IndependentMapFeature[BaseFeatureConfig]):
    """Hilbert 包络特征：每列信号段返回解析信号的幅值（与输入等长）。

    与 EnvelopeRmsFeature 的区别：本算子返回完整包络序列（等长数组），
    供后续统计（如包络峭度）使用；EnvelopeRmsFeature 直接返回包络 RMS 标量。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是对应信号段的 Hilbert 包络（与输入等长）
    """

    @classmethod
    def name(cls) -> str:
        return "envelope_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        def _envelope_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.array([])
            return np.abs(_analytic_signal(sig))

        return _apply_per_cell_array(x, _envelope_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "envelope")


# ============================================================================
# Group D: 谐波乘积谱 (1)
# ============================================================================

class HpsFeature(IndependentMapFeature[HpsFeatureConfig]):
    """谐波乘积谱（log 域）特征。

    公式: log HPS(f) = Σ_{h=1}^{H} log|S(h·f)|
    输出长度 = len(spectrum) // n_harmonics。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是对应信号段的 HPS log 谱（变长）
    """

    @classmethod
    def name(cls) -> str:
        return "hps_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        n_harmonics = params.get('n_harmonics', 5)

        def _hps_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.array([], dtype=float)
            if sig.size == 1:
                return np.array([0.0])
            spectrum = np.abs(np.fft.rfft(sig))
            out_len = spectrum.size // n_harmonics
            if out_len == 0:
                return np.array([0.0])
            log_spec = np.log(np.maximum(spectrum[:out_len], 1e-30))
            result = log_spec.copy()
            for h in range(2, n_harmonics + 1):
                downsampled = spectrum[::h][:out_len]
                if downsampled.size < out_len:
                    break
                result += np.log(np.maximum(downsampled, 1e-30))
            return result

        return _apply_per_cell_array(x, _hps_1d)

    def _get_compute_params(self):
        return {'n_harmonics': self.config.n_harmonics}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "hps")
