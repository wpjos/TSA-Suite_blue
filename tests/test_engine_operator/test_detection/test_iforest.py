"""IForest 异常检测算子单元测试"""
import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.iforest import (
    IForestDetector,
    IForestScorer,
    IForestScorerConfig,
)


@pytest.fixture
def train_data():
    np.random.seed(42)
    return np.random.randn(100, 3)


@pytest.fixture
def test_data():
    np.random.seed(123)
    normal = np.random.randn(18, 3)
    abnormal = np.random.randn(2, 3) + 10
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_data):
    return DataFrame(train_data, columns=["a", "b", "c"])


@pytest.fixture
def test_df(test_data):
    return DataFrame(test_data, columns=["a", "b", "c"])


class TestIForestScorer:
    def test_config_defaults(self):
        cfg = IForestScorerConfig()
        assert cfg.n_estimators == 100
        assert cfg.contamination == 0.1

    def test_config_frozen(self):
        cfg = IForestScorerConfig()
        with pytest.raises(Exception):
            cfg.n_estimators = 200  # type: ignore[misc]

    def test_fit_builds_model(self, train_data):
        scorer = IForestScorer()
        scorer.fit(train_data)
        assert scorer._model is not None

    def test_run_scores_shape(self, train_data, test_data):
        scorer = IForestScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert scores.shape == (20,)

    def test_with_dataframe(self, train_df, test_df):
        scorer = IForestScorer()
        scorer.fit(train_df)
        scores = scorer.run(test_df)
        assert isinstance(scores, DataFrame)

    def test_before_fit_raises(self, test_data):
        scorer = IForestScorer()
        with pytest.raises(RuntimeError):
            scorer.run(test_data)


class TestIForestDetector:
    def test_fit_and_run(self, train_data, test_data):
        detector = IForestDetector(percentile=95.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        assert labels.shape == (20,)
        assert set(labels.tolist()).issubset({0, 1})

    def test_with_dataframe(self, train_df, test_df):
        detector = IForestDetector(percentile=95.0)
        detector.fit(train_df)
        labels = detector.run(test_df)
        assert isinstance(labels, DataFrame)

    def test_before_fit_raises(self, test_data):
        detector = IForestDetector()
        with pytest.raises(RuntimeError):
            detector.run(test_data)


class TestIForestScorerSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        scorer = IForestScorer(n_estimators=50, random_state=42)
        scorer.fit(train_data)
        original = scorer.run(test_data)
        save_dir = tmp_path / 'iforest_scorer'
        scorer.save(save_dir)
        loaded = IForestScorer.load(save_dir)
        np.testing.assert_allclose(loaded.run(test_data), original)


class TestIForestDetectorSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        detector = IForestDetector(percentile=95.0)
        detector.fit(train_data)
        original = detector.run(test_data)
        save_dir = tmp_path / 'iforest_detector'
        detector.save(save_dir)
        loaded = IForestDetector.load(save_dir)
        np.testing.assert_array_equal(loaded.run(test_data), original)
