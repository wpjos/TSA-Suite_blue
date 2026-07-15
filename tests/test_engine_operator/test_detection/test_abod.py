"""ABOD 异常检测算子单元测试"""
import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.abod import (
    ABODDetector,
    ABODScorer,
    ABODScorerConfig,
)


@pytest.fixture
def train_data():
    np.random.seed(42)
    return np.random.randn(40, 2)


@pytest.fixture
def test_data():
    np.random.seed(123)
    normal = np.random.randn(8, 2)
    abnormal = np.random.randn(2, 2) + 10
    return np.vstack([normal, abnormal])


class TestABODScorer:
    def test_config_defaults(self):
        cfg = ABODScorerConfig()
        assert cfg.method == "fast"
        assert cfg.n_neighbors == 5

    def test_config_frozen(self):
        cfg = ABODScorerConfig()
        with pytest.raises(Exception):
            cfg.n_neighbors = 10  # type: ignore[misc]

    def test_fit_builds_model(self, train_data):
        scorer = ABODScorer()
        scorer.fit(train_data)
        assert scorer._model is not None

    def test_run_scores_shape(self, train_data, test_data):
        scorer = ABODScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert scores.shape == (10,)

    def test_with_dataframe(self, train_data, test_data):
        df_train = DataFrame(train_data, columns=["a", "b"])
        df_test = DataFrame(test_data, columns=["a", "b"])
        scorer = ABODScorer()
        scorer.fit(df_train)
        scores = scorer.run(df_test)
        assert isinstance(scores, DataFrame)

    def test_before_fit_raises(self, test_data):
        scorer = ABODScorer()
        with pytest.raises(RuntimeError):
            scorer.run(test_data)


class TestABODDetector:
    def test_fit_and_run(self, train_data, test_data):
        detector = ABODDetector(percentile=95.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        assert labels.shape == (10,)
        assert set(labels.tolist()).issubset({0, 1})

    def test_before_fit_raises(self, test_data):
        detector = ABODDetector()
        with pytest.raises(RuntimeError):
            detector.run(test_data)


class TestABODScorerSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        scorer = ABODScorer()
        scorer.fit(train_data)
        original = scorer.run(test_data)
        save_dir = tmp_path / 'abod_scorer'
        scorer.save(save_dir)
        loaded = ABODScorer.load(save_dir)
        np.testing.assert_allclose(loaded.run(test_data), original)


class TestABODDetectorSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        detector = ABODDetector(percentile=95.0)
        detector.fit(train_data)
        original = detector.run(test_data)
        save_dir = tmp_path / 'abod_detector'
        detector.save(save_dir)
        loaded = ABODDetector.load(save_dir)
        np.testing.assert_array_equal(loaded.run(test_data), original)
