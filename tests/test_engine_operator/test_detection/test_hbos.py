"""HBOS 异常检测算子单元测试"""
import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.hbos import (
    HBOSDetector,
    HBOSScorer,
    HBOSScorerConfig,
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


class TestHBOSScorer:
    def test_config_defaults(self):
        cfg = HBOSScorerConfig()
        assert cfg.n_bins == 10
        assert cfg.alpha == 0.1

    def test_config_frozen(self):
        cfg = HBOSScorerConfig()
        with pytest.raises(Exception):
            cfg.n_bins = 20  # type: ignore[misc]

    def test_fit_builds_model(self, train_data):
        scorer = HBOSScorer()
        scorer.fit(train_data)
        assert scorer._model is not None

    def test_run_scores_shape(self, train_data, test_data):
        scorer = HBOSScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert scores.shape == (20,)

    def test_with_dataframe(self, train_data, test_data):
        df_train = DataFrame(train_data, columns=["a", "b", "c"])
        df_test = DataFrame(test_data, columns=["a", "b", "c"])
        scorer = HBOSScorer()
        scorer.fit(df_train)
        scores = scorer.run(df_test)
        assert isinstance(scores, DataFrame)

    def test_before_fit_raises(self, test_data):
        scorer = HBOSScorer()
        with pytest.raises(RuntimeError):
            scorer.run(test_data)


class TestHBOSDetector:
    def test_fit_and_run(self, train_data, test_data):
        detector = HBOSDetector(percentile=95.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        assert labels.shape == (20,)
        assert set(labels.tolist()).issubset({0, 1})

    def test_before_fit_raises(self, test_data):
        detector = HBOSDetector()
        with pytest.raises(RuntimeError):
            detector.run(test_data)


class TestHBOSScorerSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        scorer = HBOSScorer()
        scorer.fit(train_data)
        original = scorer.run(test_data)
        save_dir = tmp_path / 'hbos_scorer'
        scorer.save(save_dir)
        loaded = HBOSScorer.load(save_dir)
        np.testing.assert_allclose(loaded.run(test_data), original)


class TestHBOSDetectorSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        detector = HBOSDetector(percentile=95.0)
        detector.fit(train_data)
        original = detector.run(test_data)
        save_dir = tmp_path / 'hbos_detector'
        detector.save(save_dir)
        loaded = HBOSDetector.load(save_dir)
        np.testing.assert_array_equal(loaded.run(test_data), original)
