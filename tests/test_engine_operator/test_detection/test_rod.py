"""ROD 异常检测算子单元测试"""
import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.rod import (
    RODDetector,
    RODScorer,
    RODScorerConfig,
)


@pytest.fixture
def train_data():
    np.random.seed(42)
    return np.random.randn(60, 3)


@pytest.fixture
def test_data():
    np.random.seed(123)
    normal = np.random.randn(18, 3)
    abnormal = np.random.randn(2, 3) + 10
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_5d():
    np.random.seed(42)
    return np.random.randn(60, 5)


@pytest.fixture
def test_5d():
    np.random.seed(123)
    normal = np.random.randn(18, 5)
    abnormal = np.random.randn(2, 5) + 10
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_data):
    return DataFrame(train_data, columns=["a", "b", "c"])


@pytest.fixture
def test_df(test_data):
    return DataFrame(test_data, columns=["a", "b", "c"])


class TestRODScorer:
    def test_config_defaults(self):
        cfg = RODScorerConfig()
        assert isinstance(cfg, RODScorerConfig)

    def test_config_frozen(self):
        cfg = RODScorerConfig()
        # RODScorerConfig has no fields, frozen should still allow construction
        assert hasattr(cfg, 'model_config')

    def test_fit_runs(self, train_data):
        scorer = RODScorer()
        scorer.fit(train_data)
        # ROD 是无状态算子, fit 不修改状态

    def test_run_scores_shape_3d(self, train_data, test_data):
        scorer = RODScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert scores.shape == (20,)

    def test_run_scores_shape_high_dim(self, train_5d, test_5d):
        scorer = RODScorer()
        scorer.fit(train_5d)
        scores = scorer.run(test_5d)
        assert scores.shape == (20,)

    def test_with_dataframe(self, train_df, test_df):
        scorer = RODScorer()
        scorer.fit(train_df)
        scores = scorer.run(test_df)
        assert isinstance(scores, DataFrame)

    def test_run_without_fit(self, test_data):
        # ROD 无状态, 允许在未 fit 时直接 run (algorithm 自包含)
        scorer = RODScorer()
        scores = scorer.run(test_data)
        assert scores.shape == (20,)


class TestRODDetector:
    def test_fit_and_run(self, train_data, test_data):
        detector = RODDetector(percentile=90.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        assert labels.shape == (20,)
        assert set(labels.tolist()).issubset({0, 1})

    def test_with_dataframe(self, train_df, test_df):
        detector = RODDetector(percentile=90.0)
        detector.fit(train_df)
        labels = detector.run(test_df)
        assert isinstance(labels, DataFrame)

    def test_high_dim_fit_and_run(self, train_5d, test_5d):
        detector = RODDetector(percentile=90.0)
        detector.fit(train_5d)
        labels = detector.run(test_5d)
        assert labels.shape == (20,)
        assert set(labels.tolist()).issubset({0, 1})


class TestRODDetectorSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        detector = RODDetector(percentile=90.0)
        detector.fit(train_data)
        original = detector.run(test_data)
        save_dir = tmp_path / 'rod_detector'
        detector.save(save_dir)
        loaded = RODDetector.load(save_dir)
        np.testing.assert_array_equal(loaded.run(test_data), original)
