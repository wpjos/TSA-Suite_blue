"""SOS 异常检测算子单元测试"""
import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.sos import (
    SOSDetector,
    SOSScorer,
    SOSScorerConfig,
)


@pytest.fixture
def train_data():
    np.random.seed(42)
    return np.random.randn(30, 3)


@pytest.fixture
def test_data():
    np.random.seed(123)
    normal = np.random.randn(8, 3)
    abnormal = np.random.randn(2, 3) + 10
    return np.vstack([normal, abnormal])


class TestSOSScorer:
    def test_config_defaults(self):
        cfg = SOSScorerConfig()
        assert cfg.perplexity == 4.5

    def test_config_frozen(self):
        cfg = SOSScorerConfig()
        with pytest.raises(Exception):
            cfg.perplexity = 10.0  # type: ignore[misc]

    def test_fit_builds_scores(self, train_data):
        scorer = SOSScorer()
        scorer.fit(train_data)
        assert scorer._train_scores is not None
        assert scorer._train_scores.shape == (30,)

    def test_run_scores_shape(self, train_data, test_data):
        scorer = SOSScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert scores.shape == (10,)

    def test_with_dataframe(self, train_data, test_data):
        df_train = DataFrame(train_data, columns=["a", "b", "c"])
        df_test = DataFrame(test_data, columns=["a", "b", "c"])
        scorer = SOSScorer()
        scorer.fit(df_train)
        scores = scorer.run(df_test)
        assert isinstance(scores, DataFrame)

    def test_before_fit_raises(self, test_data):
        scorer = SOSScorer()
        with pytest.raises(RuntimeError):
            scorer.run(test_data)


class TestSOSDetector:
    def test_fit_and_run(self, train_data, test_data):
        detector = SOSDetector(percentile=95.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        assert labels.shape == (10,)
        assert set(labels.tolist()).issubset({0, 1})

    def test_before_fit_raises(self, test_data):
        detector = SOSDetector()
        with pytest.raises(RuntimeError):
            detector.run(test_data)


class TestSOSScorerSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        scorer = SOSScorer()
        scorer.fit(train_data)
        original = scorer.run(test_data)
        save_dir = tmp_path / 'sos_scorer'
        scorer.save(save_dir)
        loaded = SOSScorer.load(save_dir)
        np.testing.assert_allclose(loaded.run(test_data), original)


class TestSOSDetectorSaveLoad:
    def test_roundtrip(self, train_data, test_data, tmp_path):
        detector = SOSDetector(percentile=95.0)
        detector.fit(train_data)
        original = detector.run(test_data)
        save_dir = tmp_path / 'sos_detector'
        detector.save(save_dir)
        loaded = SOSDetector.load(save_dir)
        np.testing.assert_array_equal(loaded.run(test_data), original)
