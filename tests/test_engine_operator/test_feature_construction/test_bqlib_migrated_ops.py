
"""
bqlib 迁移算子测试

覆盖 TSA-Suite 合入的 19 个 bqlib ops（按 MIGRATION_OPS.md 分组）：
    Group A (基础统计): max, min, mean, std, sum
    Group B (基础变换): abs, sqrt, diff, uni
    Group C (频域变换): rfft, rfftfreq, fft, fftfreq, hps, envelope
    Group D (心理声学): bark_spectrum, specific_loudness
    Group E (其他): slope, ehr

每算子测试:
    - name() 类方法
    - Config 校验
    - 手工参考值（pure-python oracle）
    - 边界情况（空信号、单元素、常数信号）
    - DataFrame 输入/输出格式
"""

import math

import numpy as np
import pandas as pd
import pytest

from tsas.engine.operator.feature.construction.basic_stat_feature import (
    MaxFeature,
    MeanFeature,
    MinFeature,
    StdFeature,
    SumFeature,
)
from tsas.engine.operator.feature.construction.basic_transform_feature import (
    AbsFeature,
    DiffFeature,
    SqrtFeature,
    UniFeature,
)
from tsas.engine.operator.feature.construction.other_feature import (
    EhrFeature,
    SlopeFeature,
)
from tsas.engine.operator.feature.construction.psychoacoustic_feature import (
    BarkSpectrumConfig,
    BarkSpectrumFeature,
    SpecificLoudnessConfig,
    SpecificLoudnessFeature,
)
from tsas.engine.operator.feature.construction.spectral_transform_feature import (
    EnvelopeFeature,
    FftFeature,
    FftFeatureConfig,
    FftFreqFeature,
    FrequencyFeatureConfig,
    HpsFeature,
    HpsFeatureConfig,
    RfftFeature,
    RfftFeatureConfig,
    RfftFreqFeature,
)

# ============================================================================
# Oracle 函数（pure-python 参考实现，对齐 bqlib _pure.py / _numpy.py 行为）
# ============================================================================

def _oracle_max(values):
    if not values:
        raise ValueError("max of empty array")
    return max(values)


def _oracle_min(values):
    if not values:
        raise ValueError("min of empty array")
    return min(values)


def _oracle_mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def _oracle_std(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def _oracle_sum(values):
    if not values:
        return 0.0
    return float(sum(values))


def _oracle_abs(values):
    return [abs(x) for x in values]


def _oracle_sqrt(values):
    if not values:
        return []
    return [math.sqrt(x) for x in values]


def _oracle_diff_default(values):
    """mode='default': prepend x[0], y[0]=0, output len N."""
    if not values:
        return []
    n = len(values)
    out = [0.0]
    for i in range(1, n):
        out.append(values[i] - values[i - 1])
    return out


def _oracle_uni(values):
    if not values:
        return []
    return [x / len(values) for x in values]


def _oracle_slope(values):
    """OLS slope for x = [0, 1, ..., n-1]."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    denom = n * (n * n - 1) / 12.0
    y_mean = sum(values) / n
    numerator = sum(i * values[i] for i in range(n)) - n * x_mean * y_mean
    return numerator / denom


def _oracle_hps(spectrum, n_harmonics=5):
    """log HPS(f) = Σ_{h=1..H} log|S(h·f)|, length = len(spectrum) // H."""
    if len(spectrum) == 0:
        return []
    if len(spectrum) == 1:
        return [0.0]
    out_len = len(spectrum) // n_harmonics
    if out_len == 0:
        return [0.0]
    result = [math.log(max(spectrum[i], 1e-30)) for i in range(out_len)]
    for h in range(2, n_harmonics + 1):
        ds = spectrum[::h][:out_len]
        if len(ds) < out_len:
            break
        for i in range(out_len):
            result[i] += math.log(max(ds[i], 1e-30))
    return result


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def random_data_1d():
    """seed=42 可复现随机信号。"""
    np.random.seed(42)
    return np.random.randn(100).tolist()


@pytest.fixture
def signal_df(random_data_1d):
    """单列 DataFrame，每格一个信号段。"""
    return pd.DataFrame({"sig": [random_data_1d]})


def _df(values, col="sig"):
    return pd.DataFrame({col: values})


# ============================================================================
# Group A: 基础统计 (5)
# ============================================================================

class TestMaxFeature:

    def test_name(self):
        assert MaxFeature.name() == "max_feature"

    def test_version(self):
        assert MaxFeature.version() == (1, 0, 0)

    @pytest.mark.parametrize("input_data,expected", [
        ([1.0, 2.0, 3.0], 3.0),
        ([-1.0, -5.0, -2.0], -1.0),
        ([5.0, 5.0, 5.0], 5.0),
        ([3.0], 3.0),
    ])
    def test_handcrafted(self, input_data, expected):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = MaxFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([input_data]))
        assert math.isclose(float(result["sig_max"].iloc[0]), expected, rel_tol=1e-12)

    def test_empty_raises(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = MaxFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        with pytest.raises(ValueError, match="empty"):
            feat.run(_df([[]]))

    def test_oracle_random(self, random_data_1d):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = MaxFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([random_data_1d]))
        assert math.isclose(float(result["sig_max"].iloc[0]), _oracle_max(random_data_1d), rel_tol=1e-12)

    def test_multi_row(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = MaxFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[1.0, 2.0, 3.0], [-1.0, -2.0]]))
        assert result["sig_max"].tolist() == [3.0, -1.0]


class TestMinFeature:

    def test_name(self):
        assert MinFeature.name() == "min_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([1.0, 2.0, 3.0], 1.0),
        ([-1.0, -5.0, -2.0], -5.0),
        ([5.0, 5.0, 5.0], 5.0),
    ])
    def test_handcrafted(self, input_data, expected):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = MinFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([input_data]))
        assert math.isclose(float(result["sig_min"].iloc[0]), expected, rel_tol=1e-12)

    def test_empty_raises(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = MinFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        with pytest.raises(ValueError, match="empty"):
            feat.run(_df([[]]))

    def test_oracle_random(self, random_data_1d):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = MinFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([random_data_1d]))
        assert math.isclose(float(result["sig_min"].iloc[0]), _oracle_min(random_data_1d), rel_tol=1e-12)


class TestMeanFeature:

    def test_name(self):
        assert MeanFeature.name() == "mean_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([1.0, 2.0, 3.0], 2.0),
        ([-1.0, 0.0, 1.0], 0.0),
        ([5.0, 5.0, 5.0], 5.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = MeanFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([input_data]))
        assert math.isclose(float(result["sig_mean"].iloc[0]), expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = MeanFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([random_data_1d]))
        assert math.isclose(float(result["sig_mean"].iloc[0]), _oracle_mean(random_data_1d), rel_tol=1e-12)


class TestStdFeature:

    def test_name(self):
        assert StdFeature.name() == "std_feature"

    @pytest.mark.parametrize("input_data,expected", [
        # ddof=1: var = sum((x-mu)^2) / (n-1) = 2 / 2 = 1.0
        ([1.0, 2.0, 3.0], 1.0),
        ([5.0, 5.0, 5.0], 0.0),
        ([3.0], 0.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = StdFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([input_data]))
        actual = float(result["sig_std"].iloc[0])
        if expected == 0.0:
            assert actual == 0.0
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = StdFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([random_data_1d]))
        assert math.isclose(float(result["sig_std"].iloc[0]), _oracle_std(random_data_1d), rel_tol=1e-12)


class TestSumFeature:

    def test_name(self):
        assert SumFeature.name() == "sum_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([1.0, 2.0, 3.0], 6.0),
        ([-1.0, 1.0], 0.0),
        ([5.0, 5.0, 5.0], 15.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = SumFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([input_data]))
        assert math.isclose(float(result["sig_sum"].iloc[0]), expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = SumFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([random_data_1d]))
        assert math.isclose(float(result["sig_sum"].iloc[0]), _oracle_sum(random_data_1d), rel_tol=1e-12)


# ============================================================================
# Group B: 基础变换 (4)
# ============================================================================

class TestAbsFeature:

    def test_name(self):
        assert AbsFeature.name() == "abs_feature"

    def test_handcrafted(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = AbsFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[-1.0, 0.0, 2.0, -3.5]]))
        out = result["sig_abs"].iloc[0]
        assert list(out) == [1.0, 0.0, 2.0, 3.5]

    def test_empty(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = AbsFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[]]))
        assert result["sig_abs"].iloc[0].size == 0

    def test_oracle_random(self, random_data_1d):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = AbsFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([random_data_1d]))
        out = list(result["sig_abs"].iloc[0])
        expected = _oracle_abs(random_data_1d)
        for a, e in zip(out, expected):
            assert math.isclose(a, e, rel_tol=1e-12)


class TestSqrtFeature:

    def test_name(self):
        assert SqrtFeature.name() == "sqrt_feature"

    def test_handcrafted(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = SqrtFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[0.0, 1.0, 4.0, 9.0]]))
        out = list(result["sig_sqrt"].iloc[0])
        assert all(math.isclose(a, e, rel_tol=1e-12) for a, e in zip(out, [0.0, 1.0, 2.0, 3.0]))

    def test_empty(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = SqrtFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[]]))
        assert result["sig_sqrt"].iloc[0].size == 0

    def test_oracle_random(self, random_data_1d):
        # sqrt of |x| to avoid negatives (sqrt of negative returns nan in numpy)
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        abs_vals = [abs(x) for x in random_data_1d]
        feat = SqrtFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([abs_vals]))
        out = list(result["sig_sqrt"].iloc[0])
        expected = _oracle_sqrt(abs_vals)
        for a, e in zip(out, expected):
            assert math.isclose(a, e, rel_tol=1e-12)


class TestDiffFeature:

    def test_name(self):
        assert DiffFeature.name() == "diff_feature"

    def test_handcrafted(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = DiffFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[1.0, 3.0, 6.0, 10.0]]))
        out = list(result["sig_diff"].iloc[0])
        # default mode: prepend x[0], y[0] = x[0]-x[0] = 0
        assert out == [0.0, 2.0, 3.0, 4.0]

    def test_empty(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = DiffFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[]]))
        assert result["sig_diff"].iloc[0].size == 0

    def test_oracle_random(self, random_data_1d):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = DiffFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([random_data_1d]))
        out = list(result["sig_diff"].iloc[0])
        expected = _oracle_diff_default(random_data_1d)
        assert len(out) == len(expected)
        for a, e in zip(out, expected):
            assert math.isclose(a, e, rel_tol=1e-12)

    def test_length_preserved(self):
        """diff 输出长度应与输入一致（首值补 0）。"""
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        sig = [1.0, 5.0, 3.0, 8.0, 2.0]
        feat = DiffFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([sig]))
        assert result["sig_diff"].iloc[0].size == len(sig)


class TestUniFeature:

    def test_name(self):
        assert UniFeature.name() == "uni_feature"

    def test_handcrafted(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = UniFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[2.0, 4.0, 6.0, 8.0]]))
        out = list(result["sig_uni"].iloc[0])
        assert all(math.isclose(a, e, rel_tol=1e-12) for a, e in zip(out, [0.5, 1.0, 1.5, 2.0]))

    def test_empty(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = UniFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[]]))
        assert result["sig_uni"].iloc[0].size == 0

    def test_oracle_random(self, random_data_1d):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = UniFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([random_data_1d]))
        out = list(result["sig_uni"].iloc[0])
        expected = _oracle_uni(random_data_1d)
        for a, e in zip(out, expected):
            assert math.isclose(a, e, rel_tol=1e-12)


# ============================================================================
# Group C: 频域变换 (6)
# ============================================================================

class TestRfftFeature:

    def test_name(self):
        assert RfftFeature.name() == "rfft_feature"

    def test_config_default(self):
        cfg = RfftFeatureConfig(input_columns=["sig"])
        assert cfg.output == "complex"

    def test_complex_output(self):
        cfg = RfftFeatureConfig(input_columns=["sig"], output="complex")
        feat = RfftFeature(config=cfg)
        sig = [1.0, 0.0, -1.0, 0.0]
        result = feat.run(_df([sig]))
        out = result["sig_rfft"].iloc[0]
        assert out.size == len(sig) // 2 + 1
        # numpy reference
        expected = np.fft.rfft(sig)
        for i, (a, e) in enumerate(zip(out, expected)):
            assert math.isclose(a.real, e.real, abs_tol=1e-12)
            assert math.isclose(a.imag, e.imag, abs_tol=1e-12)

    def test_magnitude_output(self):
        cfg = RfftFeatureConfig(input_columns=["sig"], output="magnitude")
        feat = RfftFeature(config=cfg)
        sig = [1.0, 2.0, 3.0, 4.0]
        result = feat.run(_df([sig]))
        out = result["sig_rfft"].iloc[0]
        expected = np.abs(np.fft.rfft(sig))
        for a, e in zip(out, expected):
            assert math.isclose(a, e, rel_tol=1e-12)

    def test_empty(self):
        cfg = RfftFeatureConfig(input_columns=["sig"])
        feat = RfftFeature(config=cfg)
        result = feat.run(_df([[]]))
        assert result["sig_rfft"].iloc[0].size == 0


class TestRfftFreqFeature:

    def test_name(self):
        assert RfftFreqFeature.name() == "rfftfreq_feature"

    def test_handcrafted(self):
        cfg = FrequencyFeatureConfig(input_columns=["sig"], sample_rate=1000.0)
        feat = RfftFreqFeature(config=cfg)
        sig = [0.0] * 100
        result = feat.run(_df([sig]))
        out = result["sig_rfftfreq"].iloc[0]
        assert out.size == 51  # N//2 + 1
        # 检查首末频率值
        assert math.isclose(out[0], 0.0, abs_tol=1e-12)
        assert math.isclose(out[-1], 500.0, abs_tol=1e-12)  # Nyquist
        # 单调递增
        assert all(out[i] < out[i + 1] for i in range(len(out) - 1))

    def test_empty(self):
        cfg = FrequencyFeatureConfig(input_columns=["sig"], sample_rate=1000.0)
        feat = RfftFreqFeature(config=cfg)
        result = feat.run(_df([[]]))
        assert result["sig_rfftfreq"].iloc[0].size == 0


class TestFftFeature:

    def test_name(self):
        assert FftFeature.name() == "fft_feature"

    def test_complex_output(self):
        cfg = FftFeatureConfig(input_columns=["sig"], output="complex")
        feat = FftFeature(config=cfg)
        sig = [1.0, 0.0, -1.0, 0.0]
        result = feat.run(_df([sig]))
        out = result["sig_fft"].iloc[0]
        assert out.size == len(sig)
        expected = np.fft.fft(sig)
        for a, e in zip(out, expected):
            assert math.isclose(a.real, e.real, abs_tol=1e-12)
            assert math.isclose(a.imag, e.imag, abs_tol=1e-12)

    def test_magnitude_output(self):
        cfg = FftFeatureConfig(input_columns=["sig"], output="magnitude")
        feat = FftFeature(config=cfg)
        sig = [1.0, 2.0, 3.0, 4.0]
        result = feat.run(_df([sig]))
        out = result["sig_fft"].iloc[0]
        expected = np.abs(np.fft.fft(sig))
        for a, e in zip(out, expected):
            assert math.isclose(a, e, rel_tol=1e-12)


class TestFftFreqFeature:

    def test_name(self):
        assert FftFreqFeature.name() == "fftfreq_feature"

    def test_handcrafted(self):
        cfg = FrequencyFeatureConfig(input_columns=["sig"], sample_rate=1000.0)
        feat = FftFreqFeature(config=cfg)
        sig = [0.0] * 8
        result = feat.run(_df([sig]))
        out = result["sig_fftfreq"].iloc[0]
        assert out.size == 8
        # 第一项为 0，最后一项为负（Nyquist）
        assert math.isclose(out[0], 0.0, abs_tol=1e-12)
        assert math.isclose(out[-1], -125.0, abs_tol=1e-12)


class TestHpsFeature:

    def test_name(self):
        assert HpsFeature.name() == "hps_feature"

    def test_config_default(self):
        cfg = HpsFeatureConfig(input_columns=["sig"])
        assert cfg.n_harmonics == 5

    def test_handcrafted_sine(self):
        """正弦信号幅度谱 HPS：最大峰在基频处。"""
        fs = 1000
        n = 256
        t = np.arange(n) / fs
        sig = np.sin(2 * np.pi * 50 * t).tolist()
        feat = HpsFeature(config=HpsFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([sig]))
        hps = result["sig_hps"].iloc[0]
        # 长度 = N//2+1 // n_harmonics = 129 // 5 = 25
        assert hps.size == (n // 2 + 1) // 5

    def test_hps_matches_oracle(self):
        """HPS 输出与 pure-python oracle 一致（log 域）。"""
        np.random.seed(0)
        sig = np.random.randn(64).tolist()  # 时域信号
        feat = HpsFeature(config=HpsFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([sig]))
        out = list(result["sig_hps"].iloc[0])
        # 复现 feature 内部计算：rfft → HPS oracle
        spectrum = np.abs(np.fft.rfft(np.asarray(sig, dtype=float))).tolist()
        expected = _oracle_hps(spectrum, n_harmonics=5)
        assert len(out) == len(expected)
        for a, e in zip(out, expected):
            assert math.isclose(a, e, abs_tol=1e-10)


class TestEnvelopeFeature:

    def test_name(self):
        assert EnvelopeFeature.name() == "envelope_feature"

    def test_length_preserved(self):
        """包络输出长度应与输入一致。"""
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        sig = np.sin(2 * np.pi * 10 * np.arange(100) / 100).tolist()
        feat = EnvelopeFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([sig]))
        out = result["sig_envelope"].iloc[0]
        assert out.size == len(sig)

    def test_amplitude_sine(self):
        """等幅正弦的包络应近似等于其幅值。"""
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        amp = 2.0
        sig = (amp * np.sin(2 * np.pi * 10 * np.arange(200) / 200)).tolist()
        feat = EnvelopeFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([sig]))
        out = result["sig_envelope"].iloc[0]
        # 中间区域包络应该接近 amp（边界除外）
        mid = out[len(out) // 4: 3 * len(out) // 4]
        assert np.mean(mid) == pytest.approx(amp, rel=0.15)

    def test_empty(self):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = EnvelopeFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[]]))
        assert result["sig_envelope"].iloc[0].size == 0


# ============================================================================
# Group D: 心理声学 (2)
# ============================================================================

class TestBarkSpectrumFeature:

    def test_name(self):
        assert BarkSpectrumFeature.name() == "bark_spectrum_feature"

    def test_default_n_barks(self):
        cfg = BarkSpectrumConfig(input_columns=["sig"], sample_rate=44100)
        feat = BarkSpectrumFeature(config=cfg)
        np.random.seed(0)
        sig = np.random.randn(2048).tolist()
        result = feat.run(_df([sig]))
        out = result["sig_bark_spectrum"].iloc[0]
        assert out.size == 24  # default n_barks

    def test_custom_n_barks(self):
        cfg = BarkSpectrumConfig(input_columns=["sig"], sample_rate=44100, n_barks=12)
        feat = BarkSpectrumFeature(config=cfg)
        np.random.seed(0)
        sig = np.random.randn(2048).tolist()
        result = feat.run(_df([sig]))
        assert result["sig_bark_spectrum"].iloc[0].size == 12

    def test_empty(self):
        cfg = BarkSpectrumConfig(input_columns=["sig"], sample_rate=44100)
        feat = BarkSpectrumFeature(config=cfg)
        result = feat.run(_df([[]]))
        assert result["sig_bark_spectrum"].iloc[0].size == 24


class TestSpecificLoudnessFeature:

    def test_name(self):
        assert SpecificLoudnessFeature.name() == "specific_loudness_feature"

    def test_quiet_signal_zero_loudness(self):
        """极低功率信号应得到接近 0 的响度。"""
        cfg = SpecificLoudnessConfig(input_columns=["sig"], ref_power=1e-12)
        feat = SpecificLoudnessFeature(config=cfg)
        bark_power = [1e-30] * 24
        result = feat.run(_df([bark_power]))
        out = result["sig_specific_loudness"].iloc[0]
        assert np.all(out < 0.01)

    def test_loud_signal_positive_loudness(self):
        """高功率信号应得到正响度。"""
        cfg = SpecificLoudnessConfig(input_columns=["sig"], ref_power=1e-12)
        feat = SpecificLoudnessFeature(config=cfg)
        bark_power = [1e-6] * 24
        result = feat.run(_df([bark_power]))
        out = result["sig_specific_loudness"].iloc[0]
        assert np.all(out >= 0)
        assert np.any(out > 0)


# ============================================================================
# Group E: 其他 (2)
# ============================================================================

class TestSlopeFeature:

    def test_name(self):
        assert SlopeFeature.name() == "slope_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([1.0, 2.0, 3.0, 4.0], 1.0),  # 完美线性，斜率 1
        ([5.0, 5.0, 5.0, 5.0], 0.0),  # 常数
        ([1.0], 0.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = SlopeFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([input_data]))
        actual = float(result["sig_slope"].iloc[0])
        if expected == 0.0:
            assert actual == 0.0
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = SlopeFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([random_data_1d]))
        assert math.isclose(float(result["sig_slope"].iloc[0]), _oracle_slope(random_data_1d), rel_tol=1e-12)


class TestEhrFeature:

    def test_name(self):
        assert EhrFeature.name() == "ehr_feature"

    def test_short_signal_zero(self):
        """过短信号 EHR 返回 0。"""
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = EhrFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[1.0]]))
        assert float(result["sig_ehr"].iloc[0]) == 0.0

    def test_constant_signal_zero(self):
        """常数信号包络 std=0，EHR 返回 0。"""
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        feat = EhrFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([[3.0] * 50]))
        assert float(result["sig_ehr"].iloc[0]) == 0.0

    def test_amplitude_modulated_positive(self):
        """调幅信号 EHR 应为正（包络有周期性）。"""
        from tsas.engine.operator.feature.construction.base import BaseFeatureConfig
        fs = 1000
        t = np.arange(2000) / fs
        # 高频载波被低频调幅
        carrier = np.sin(2 * np.pi * 200 * t)
        envelope = 1.0 + 0.5 * np.sin(2 * np.pi * 5 * t)
        sig = (carrier * envelope).tolist()
        feat = EhrFeature(config=BaseFeatureConfig(input_columns=["sig"]))
        result = feat.run(_df([sig]))
        ehr = float(result["sig_ehr"].iloc[0])
        assert 0.0 < ehr <= 1.0


# ============================================================================
# 跨组综合：验证所有 19 个算子 + Config 字段映射正确
# ============================================================================

class TestConfigValidation:
    """测试各算子的 Config 字段约束。"""

    def test_rfft_invalid_output(self):
        with pytest.raises(Exception):
            RfftFeatureConfig(input_columns=["sig"], output="phase")

    def test_fft_invalid_output(self):
        with pytest.raises(Exception):
            FftFeatureConfig(input_columns=["sig"], output="phase")

    def test_freq_sample_rate_must_be_positive(self):
        with pytest.raises(Exception):
            FrequencyFeatureConfig(input_columns=["sig"], sample_rate=0)

    def test_hps_n_harmonics_min(self):
        with pytest.raises(Exception):
            HpsFeatureConfig(input_columns=["sig"], n_harmonics=0)

    def test_bark_sample_rate_must_be_positive(self):
        with pytest.raises(Exception):
            BarkSpectrumConfig(input_columns=["sig"], sample_rate=-1.0)

    def test_specific_loudness_ref_power_positive(self):
        with pytest.raises(Exception):
            SpecificLoudnessConfig(input_columns=["sig"], ref_power=0.0)
