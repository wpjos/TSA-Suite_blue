"""LOF 异常检测算子单元测试"""
import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.lof import (
    LOFDetector,
    LOFScorer,
    LOFScorerConfig,
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


class TestLOFScorer:
    def test_config_defaults(self):
        cfg = LOFScorerConfig()
        assert cfg.n_neighbors == 20

    def test_config_frozen(self):
        cfg = LOFScorerConfig()
        with pytest.raises(Exception):
            cfg.n_neighbors = 10  # type: ignore[misc]

    def test_fit_builds_model(self, train_data):
        scorer = LOFScorer()
        scorer.fit(train_data)
        assert scorer._model is not None

    def test_run_scores_shape(self, train_data, test_data):
        scorer = LOFScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert scores.shape == (20,)

    def test_with_dataframe(self, train_data, test_data):
        df_train = DataFrame(train_data, columns=["a", "b", "c"])
        df_test = DataFrame(test_data, columns=["a", "b", "c"])
        scorer = LOFScorer()
        scorer.fit(df_train)
        scores = scorer.run(df_test)
        assert isinstance(scores, DataFrame)

    def test_before_fit_raises(self, test_data):
        scorer = LOFScorer()
        with pytest.raises(RuntimeError):
            scorer.run(test_data)


class TestLOFDetector:
    def test_fit_and_run(self, train_data, test_data):
        detector = LOFDetector(percentile=95.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        assert labels.shape == (20,)
        assert set(labels.tolist()).issubset({0, 1})

    def test_before_fit_raises(self, test_data):
        detector = LOFDetector()
        with pytest.raises(RuntimeError):
            detector.run(test_data)


class TestLOFScorerSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        scorer = LOFScorer()
        scorer.fit(train_data)
        original = scorer.run(test_data)
        save_dir = tmp_path / 'lof_scorer'
        scorer.save(save_dir)
        loaded = LOFScorer.load(save_dir)
        np.testing.assert_allclose(loaded.run(test_data), original)


class TestLOFDetectorSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        detector = LOFDetector(percentile=95.0)
        detector.fit(train_data)
        original = detector.run(test_data)
        save_dir = tmp_path / 'lof_detector'
        detector.save(save_dir)
        loaded = LOFDetector.load(save_dir)
        np.testing.assert_array_equal(loaded.run(test_data), original)
