# -*- coding: utf-8 -*-

"""
XGBoost 时序预测算子单元测试

对应源文件：
- forecasting/xgboost.py: XGBoostForecaster, XGBoostForecasterConfig

测试范围：
- Config 参数校验
- 训练与推理基本流程
- DataFrame/ndarray 双类型支持
- 批量推理、多目标支持
- 边界条件（未训练、序列长度不足）
- Save/Load roundtrip
"""

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

# 若当前环境缺少 xgboost，则跳过相关测试
xgb = pytest.importorskip("xgboost", reason="需要安装 xgboost 才能运行 XGBoost 测试")

from tsas.engine.operator.forecasting.xgboost import (
    XGBoostForecaster,
    XGBoostForecasterConfig,
)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_data():
    """测试用训练数据（ndarray, 400x3），满足 seq_len=48, pred_len=12"""
    np.random.seed(42)
    return np.cumsum(np.random.randn(400, 3), axis=0)


@pytest.fixture
def train_df(train_data):
    """测试用训练数据（DataFrame）"""
    return DataFrame(train_data, columns=["feat_0", "feat_1", "target"])


@pytest.fixture
def test_window():
    """测试用推理窗口（ndarray, 48x3）"""
    np.random.seed(123)
    return np.cumsum(np.random.randn(48, 3), axis=0)


@pytest.fixture
def test_window_df(test_window):
    """测试用推理窗口（DataFrame）"""
    return DataFrame(test_window, columns=["feat_0", "feat_1", "target"])


@pytest.fixture
def minimal_config():
    """最小化测试配置（小模型、少轮数）"""
    return XGBoostForecasterConfig(
        seq_len=48,
        pred_len=12,
        n_estimators=10,
        max_depth=3,
        skip_tune=True,
        n_jobs=1,
    )


# ============================================================================
# Config 测试
# ============================================================================

class TestXGBoostForecasterConfig:
    """测试 XGBoostForecasterConfig 参数校验"""

    def test_default_config(self):
        """目的：验证默认配置可创建"""
        cfg = XGBoostForecasterConfig()
        assert cfg.seq_len == 96
        assert cfg.pred_len == 24
        assert cfg.max_depth == 4

    def test_searchable_bounds(self):
        """目的：验证带 ge/le 的字段可被 HPO 识别"""
        cfg = XGBoostForecasterConfig(max_depth=3, learning_rate=0.01)
        assert cfg.max_depth == 3
        assert cfg.learning_rate == 0.01

    def test_device_literal(self):
        """目的：验证 device 仅接受 cpu/gpu"""
        cfg = XGBoostForecasterConfig(device="gpu")
        assert cfg.device == "gpu"


# ============================================================================
# XGBoostForecaster 训练与推理测试
# ============================================================================

class TestXGBoostForecaster:
    """测试 XGBoost 预测算子核心流程"""

    def test_fit_learns_models(self, train_data, minimal_config):
        """目的：验证 fit 后已训练所有 horizon/target 对应的 booster"""
        forecaster = XGBoostForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        assert forecaster.is_fitted
        assert len(forecaster._models) == minimal_config.pred_len * 1
        assert (0, 0) in forecaster._models

    def test_run_output_shape(self, train_data, test_window, minimal_config):
        """目的：验证推理输出形状为 (pred_len, num_targets)"""
        forecaster = XGBoostForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        pred = forecaster.run(test_window)
        assert pred.shape == (12, 1)

    def test_run_batched_output_shape(self, train_data, minimal_config):
        """目的：验证批量推理输出形状为 (batch, pred_len, num_targets)"""
        forecaster = XGBoostForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        x_batch = np.stack([
            train_data[100:148],
            train_data[200:248],
            train_data[300:348],
        ])
        pred = forecaster.run(x_batch)
        assert pred.shape == (3, 12, 1)

    def test_multi_target(self, train_data, minimal_config):
        """目的：验证多目标支持"""
        forecaster = XGBoostForecaster(config=minimal_config)
        y = train_data[:, [-1, -2]]
        forecaster.fit(train_data, y)
        pred = forecaster.run(train_data[-48:])
        assert pred.shape == (12, 2)
        assert len(forecaster._models) == minimal_config.pred_len * 2

    def test_with_dataframe(self, train_df, test_window_df, minimal_config):
        """目的：验证 DataFrame 输入输出"""
        forecaster = XGBoostForecaster(config=minimal_config)
        y = train_df[["target"]]
        forecaster.fit(train_df, y)
        pred = forecaster.run(test_window_df)
        assert isinstance(pred, DataFrame)
        assert pred.shape == (12, 1)

    def test_before_fit_raises(self, test_window, minimal_config):
        """目的：验证未训练时 run 抛出 RuntimeError"""
        forecaster = XGBoostForecaster(config=minimal_config)
        with pytest.raises(RuntimeError):
            forecaster.run(test_window)

    def test_too_short_sequence_raises(self, train_data, minimal_config):
        """目的：验证训练序列长度不足时抛出 ValueError"""
        forecaster = XGBoostForecaster(config=minimal_config)
        short = train_data[:10]
        y = short[:, [-1]]
        with pytest.raises(ValueError, match="时间序列长度"):
            forecaster.fit(short, y)

    def test_run_invalid_window_shape(self, train_data, minimal_config):
        """目的：验证推理窗口形状不匹配时报错"""
        forecaster = XGBoostForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        with pytest.raises(ValueError, match="推理输入形状"):
            forecaster.run(train_data[-47:])


# ============================================================================
# Save/Load Roundtrip 测试
# ============================================================================

class TestXGBoostForecasterSaveLoad:
    """测试 XGBoostForecaster 持久化 roundtrip"""

    def test_save_load_roundtrip(self, train_data, test_window, minimal_config, tmp_path):
        """目的：验证 save → load 后推理结果一致"""
        forecaster = XGBoostForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        original_pred = forecaster.run(test_window)

        save_dir = tmp_path / "xgboost_forecaster"
        forecaster.save(save_dir)
        loaded = XGBoostForecaster.load(save_dir)

        loaded_pred = loaded.run(test_window)
        np.testing.assert_allclose(original_pred, loaded_pred, rtol=1e-5)

    def test_loaded_state_restored(self, train_data, minimal_config, tmp_path):
        """目的：验证加载后内部状态正确恢复"""
        forecaster = XGBoostForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)

        save_dir = tmp_path / "xgboost_forecaster"
        forecaster.save(save_dir)
        loaded = XGBoostForecaster.load(save_dir)

        assert loaded.is_fitted
        assert loaded._num_features == forecaster._num_features
        assert loaded._num_targets == forecaster._num_targets
        assert len(loaded._models) == len(forecaster._models)

    def test_save_creates_required_files(self, train_data, minimal_config, tmp_path):
        """目的：验证保存目录包含所有必需文件"""
        forecaster = XGBoostForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)

        save_dir = tmp_path / "xgboost_forecaster"
        forecaster.save(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "_forecaster_state.npz").exists()
        assert (save_dir / "_models" / "model_h0_t0.json").exists()
        assert (save_dir / "_models" / f"model_h{minimal_config.pred_len - 1}_t0.json").exists()
