"""
心理声学特征构造算子实现模块

提供 2 个面向心理声学分析的 transform 算子，均基于 IndependentMapFeature + Map 模式。
数据模型：每个 DataFrame 格子存放一个信号段或 Bark 功率谱，compute 逐格计算产生等长序列。

特征分组：
- Group A: Bark 频段功率谱 (1) — bark_spectrum（时域信号 → 24 维 Bark 功率）
- Group B: 特定响度 (1) — specific_loudness（Bark 功率谱 → 24 维响度 N'(z)）

实现复用 signal_feature.py 中已存在的 Bark 频段辅助函数
（_hz_to_bark / _bark_band_edges / _bark_to_hz_approx / _bark_spectrum_from_signal /
 _power_to_specific_loudness）。算子行为对齐 bqlib ops.transform.bark_spectrum /
 bqlib.ops.transform.specific_loudness 的 numpy 后端。
TSA-Suite 不依赖 bqlib。
"""

import numpy as np
from pydantic import Field

from tsas.engine.operator.feature.construction._array_helpers import _apply_per_cell_array
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig,
    IndependentMapFeature,
)
from tsas.engine.operator.feature.construction.signal_feature import (
    _bark_spectrum_from_signal,
    _power_to_specific_loudness,
)

__all__ = [
    'BarkSpectrumFeature',
    'SpecificLoudnessFeature',
]


# ============================================================================
# Config 定义
# ============================================================================

class BarkSpectrumConfig(BaseFeatureConfig):
    """Bark 功率谱 Config。"""
    sample_rate: float = Field(gt=0, description="采样率 (Hz)")
    n_barks: int = Field(default=24, ge=1, description="Bark 频带数量（默认 24）")


class SpecificLoudnessConfig(BaseFeatureConfig):
    """特定响度 Config。"""
    ref_power: float = Field(default=1e-12, gt=0, description="dB SPL 参考功率（默认 1e-12）")


# ============================================================================
# Group A: Bark 频段功率谱 (1)
# ============================================================================

class BarkSpectrumFeature(IndependentMapFeature[BarkSpectrumConfig]):
    """Bark 功率谱特征：时域信号 → 24 维 Bark 频段功率。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        object ndarray，每格是长度为 n_barks 的 Bark 功率数组
    """

    @classmethod
    def name(cls) -> str:
        return "bark_spectrum_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        n_barks = params.get('n_barks', 24)

        def _bark_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.zeros(n_barks)
            return _bark_spectrum_from_signal(sig, sample_rate, n_barks)

        return _apply_per_cell_array(x, _bark_1d)

    def _get_compute_params(self):
        return {
            'sample_rate': self.config.sample_rate,
            'n_barks': self.config.n_barks,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "bark_spectrum")


# ============================================================================
# Group B: 特定响度 (1)
# ============================================================================

class SpecificLoudnessFeature(IndependentMapFeature[SpecificLoudnessConfig]):
    """特定响度特征：Bark 功率谱 → 24 维响度 N'(z)。

    通常与 BarkSpectrumFeature 串联使用（Bark 功率 → 响度）。

    Input:
        x: 特征矩阵，每格是 Bark 功率谱（1D ndarray，长度通常为 24）

    Output:
        object ndarray，每格是长度相同的响度数组
    """

    @classmethod
    def name(cls) -> str:
        return "specific_loudness_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        ref_power = params.get('ref_power', 1e-12)

        def _n_prime_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.array([])
            return _power_to_specific_loudness(sig, ref_power)

        return _apply_per_cell_array(x, _n_prime_1d)

    def _get_compute_params(self):
        return {'ref_power': self.config.ref_power}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "specific_loudness")
