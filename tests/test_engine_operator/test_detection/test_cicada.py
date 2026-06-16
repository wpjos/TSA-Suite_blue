# -*- coding: utf-8 -*-

"""
CICADA 检测算子单元测试

对应源文件：
- cicada.py: CICADAPredictor

测试范围：
- Config 参数验证
- fit/run 基本流程
- DataFrame/ndarray 双类型支持
- save/load 持久化
- 边界条件（数据过短、未训练先推理等）
"""

import warnings

import numpy as np
import pytest
from pandas import DataFrame
from pydantic import ValidationError

from bianque.engine.operator.detection.cicada import (
    CICADAPredictor,
    CICADAPredictorConfig,
)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_data():
    """测试用训练数据（ndarray, 200x3, float32）"""
    np.random.seed(42)
    return np.random.randn(200, 3).astype(np.float32)


@pytest.fixture
def test_data():
    """测试用测试数据（ndarray, 100x3, float32，含异常点）"""
    np.random.seed(123)
    normal = np.random.randn(80, 3).astype(np.float32)
    abnormal = (np.random.randn(20, 3) * 5 + 10).astype(np.float32)
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_data):
    """测试用训练数据（DataFrame）"""
    return DataFrame(train_data, columns=["a", "b", "c"])


@pytest.fixture
def test_df(test_data):
    """测试用测试数据（DataFrame）"""
    return DataFrame(test_data, columns=["a", "b", "c"])


def _make_predictor(**overrides):
    """创建最小配置的 CICADAPredictor（用于加速测试）"""
    defaults = dict(
        experts=["MLP"],
        win_size=10,
        num_channels=3,
        batch_size=32,
        epochs=1,
        latent_space_size=8,
        n_components=4,
    )
    defaults.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return CICADAPredictor(**defaults)


# ============================================================================
# Config 测试
# ============================================================================

class TestCICADAPredictorConfig:
    """测试 CICADAPredictorConfig 参数验证"""

    def test_config_defaults(self):
        """
        目的：验证 Config 默认值与 CICADA 一致
        输入：无参数构造
        预期：所有默认值正确
        """
        cfg = CICADAPredictorConfig()
        assert cfg.experts == ["GradPCA", "GradKPCA", "GradFreKPCA", "GradSubPCA"]
        assert cfg.win_size == 5
        assert cfg.stride == 1
        assert cfg.num_channels is None
        assert cfg.batch_size == 256
        assert cfg.epochs == 60
        assert cfg.latent_space_size == 128
        assert cfg.n_components == "auto"
        assert cfg.lr == 1e-3
        assert cfg.infer_mode == "offline"
        assert cfg.th == 0.98

    def test_config_frozen(self):
        """
        目的：验证 Config 不可变
        输入：创建后尝试修改字段
        预期：抛出异常
        """
        cfg = CICADAPredictorConfig()
        with pytest.raises(ValidationError):
            cfg.win_size = 100

    def test_config_validation_win_size(self):
        """
        目的：验证 win_size 约束
        输入：win_size=0
        预期：ValidationError
        """
        with pytest.raises(ValidationError):
            CICADAPredictorConfig(win_size=0)

    def test_config_validation_epochs(self):
        """
        目的：验证 epochs 约束
        输入：epochs=-1
        预期：ValidationError
        """
        with pytest.raises(ValidationError):
            CICADAPredictorConfig(epochs=-1)

    def test_config_infer_mode_validation(self):
        """
        目的：验证 infer_mode 枚举约束
        输入：infer_mode="invalid"
        预期：ValidationError
        """
        with pytest.raises(ValidationError):
            CICADAPredictorConfig(infer_mode="invalid")

    def test_config_custom_values(self):
        """
        目的：验证自定义参数正确传递
        输入：自定义 win_size, batch_size
        预期：Config 中值为自定义值
        """
        cfg = CICADAPredictorConfig(win_size=100, batch_size=64)
        assert cfg.win_size == 100
        assert cfg.batch_size == 64


# ============================================================================
# Fit 测试
# ============================================================================

class TestCICADAPredictorFit:
    """测试 CICADAPredictor 训练流程"""

    def test_fit_creates_model(self, train_data):
        """
        目的：验证 fit 后内部模型已创建
        输入：(200, 3) 训练数据
        预期：_model 不为 None，is_fitted 为 True
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        assert predictor._model is not None
        assert predictor.is_fitted is True

    def test_fit_auto_detect_channels(self, train_data):
        """
        目的：验证 num_channels 自动推断
        输入：num_channels=None，(200, 3) 数据
        预期：_num_channels_detected == 3
        """
        predictor = _make_predictor(num_channels=None)
        predictor.fit(train_data)
        assert predictor._num_channels_detected == 3

    def test_fit_with_explicit_channels(self, train_data):
        """
        目的：验证显式指定 num_channels 时使用指定值
        输入：num_channels=3
        预期：_num_channels_detected == 3
        """
        predictor = _make_predictor(num_channels=3)
        predictor.fit(train_data)
        assert predictor._num_channels_detected == 3

    def test_fit_data_too_short_raises(self):
        """
        目的：验证数据行数不足时报错
        输入：(5, 3) 数据，win_size=10
        预期：ValueError
        """
        predictor = _make_predictor()
        short_data = np.random.randn(5, 3).astype(np.float32)
        with pytest.raises(ValueError, match="win_size"):
            predictor.fit(short_data)

    def test_fit_1d_input_raises(self):
        """
        目的：验证 1D 输入报错
        输入：(300,) 一维数据
        预期：ValueError
        """
        predictor = _make_predictor()
        data_1d = np.random.randn(300).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            predictor.fit(data_1d)

    def test_fit_with_dataframe(self, train_df):
        """
        目的：验证 DataFrame 输入可以训练
        输入：DataFrame (200, 3)
        预期：训练成功，is_fitted 为 True
        """
        predictor = _make_predictor(num_channels=3)
        predictor.fit(train_df)
        assert predictor.is_fitted is True


# ============================================================================
# Run 测试
# ============================================================================

class TestCICADAPredictorRun:
    """测试 CICADAPredictor 推理流程"""

    def test_run_output_shape(self, train_data, test_data):
        """
        目的：验证推理输出形状与输入一致
        输入：训练数据 + (100, 3) 测试数据
        预期：输出形状 == (100, 3)
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        recon = predictor.run(test_data)
        assert recon.shape == test_data.shape

    def test_run_values_finite(self, train_data, test_data):
        """
        目的：验证重构值不含 NaN/Inf
        输入：训练数据 + 测试数据
        预期：所有值为有限数
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        recon = predictor.run(test_data)
        assert np.all(np.isfinite(recon))

    def test_run_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 报错
        输入：未训练的 predictor
        预期：RuntimeError
        """
        predictor = _make_predictor()
        with pytest.raises(RuntimeError, match="训练尚未完成"):
            predictor.run(test_data)

    def test_run_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出
        输入：DataFrame 训练 + DataFrame 测试
        预期：输出为 DataFrame，列名一致
        """
        predictor = _make_predictor(num_channels=3)
        predictor.fit(train_df)
        recon = predictor.run(test_df)
        assert isinstance(recon, DataFrame)
        assert list(recon.columns) == ["a", "b", "c"]

    def test_run_with_ndarray(self, train_data, test_data):
        """
        目的：验证 ndarray 输入输出
        输入：ndarray 训练 + ndarray 测试
        预期：输出为 ndarray
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        recon = predictor.run(test_data)
        assert isinstance(recon, np.ndarray)


# ============================================================================
# Save / Load 测试
# ============================================================================

class TestCICADAPredictorSaveLoad:
    """测试 CICADAPredictor 持久化"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save + load 后推理结果不变
        输入：训练 → save → load → run
        预期：load 后推理输出与 save 前一致
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        recon_before = predictor.run(test_data)

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        loaded = CICADAPredictor.load(save_dir)
        recon_after = loaded.run(test_data)

        np.testing.assert_allclose(recon_after, recon_before, atol=1e-5)

    def test_load_restores_fitted_state(self, train_data, tmp_path):
        """
        目的：验证 load 后 is_fitted 为 True
        输入：训练 → save → load
        预期：loaded.is_fitted == True
        """
        predictor = _make_predictor()
        predictor.fit(train_data)

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        loaded = CICADAPredictor.load(save_dir)
        assert loaded.is_fitted is True

    def test_load_restores_model(self, train_data, test_data, tmp_path):
        """
        目的：验证 load 后模型可用
        输入：训练 → save → load → run
        预期：模型不为 None，推理输出有限
        """
        predictor = _make_predictor()
        predictor.fit(train_data)

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        loaded = CICADAPredictor.load(save_dir)
        assert loaded._model is not None
        recon = loaded.run(test_data)
        assert np.all(np.isfinite(recon))

    def test_save_creates_expected_files(self, train_data, tmp_path):
        """
        目的：验证 save 生成正确的文件
        输入：训练 → save
        预期：config.json、cicada_model.pt、cicada_meta.json 均存在
        """
        predictor = _make_predictor()
        predictor.fit(train_data)

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "cicada_model.pt").exists()
        assert (save_dir / "cicada_meta.json").exists()

    def test_load_num_channels_preserved(self, train_data, tmp_path):
        """
        目的：验证 num_channels 持久化正确
        输入：训练（自动推断 num_channels=3）→ save → load
        预期：loaded._num_channels_detected == 3
        """
        predictor = _make_predictor(num_channels=None)
        predictor.fit(train_data)
        assert predictor._num_channels_detected == 3

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        loaded = CICADAPredictor.load(save_dir)
        assert loaded._num_channels_detected == 3
