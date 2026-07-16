"""COPOD 异常检测算子单元测试"""
import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.copod import (
    COPODDetector,
    COPODScorer,
    COPODScorerConfig,
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


class TestCOPODScorer:
    def test_config_defaults(self):
        cfg = COPODScorerConfig()
        assert cfg.contamination == 0.1

    def test_config_frozen(self):
        cfg = COPODScorerConfig()
        with pytest.raises(Exception):
            cfg.contamination = 0.2  # type: ignore[misc]

    def test_fit_builds_model(self, train_data):
        scorer = COPODScorer()
        scorer.fit(train_data)
        assert scorer._model is not None

    def test_run_scores_shape(self, train_data, test_data):
        scorer = COPODScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert scores.shape == (20,)

    def test_with_dataframe(self, train_data, test_data):
        df_train = DataFrame(train_data, columns=["a", "b", "c"])
        df_test = DataFrame(test_data, columns=["a", "b", "c"])
        scorer = COPODScorer()
        scorer.fit(df_train)
        scores = scorer.run(df_test)
        assert isinstance(scores, DataFrame)

    def test_before_fit_raises(self, test_data):
        scorer = COPODScorer()
        with pytest.raises(RuntimeError):
            scorer.run(test_data)


class TestCOPODDetector:
    def test_fit_and_run(self, train_data, test_data):
        detector = COPODDetector(percentile=95.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        assert labels.shape == (20,)
        assert set(labels.tolist()).issubset({0, 1})

    def test_before_fit_raises(self, test_data):
        detector = COPODDetector()
        with pytest.raises(RuntimeError):
            detector.run(test_data)


class TestCOPODScorerSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        scorer = COPODScorer()
        scorer.fit(train_data)
        original = scorer.run(test_data)
        save_dir = tmp_path / 'copod_scorer'
        scorer.save(save_dir)
        loaded = COPODScorer.load(save_dir)
        np.testing.assert_allclose(loaded.run(test_data), original)


class TestCOPODDetectorSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        detector = COPODDetector(percentile=95.0)
        detector.fit(train_data)
        original = detector.run(test_data)
        save_dir = tmp_path / 'copod_detector'
        detector.save(save_dir)
        loaded = COPODDetector.load(save_dir)
        np.testing.assert_array_equal(loaded.run(test_data), original)
