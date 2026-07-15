"""EIF 异常检测算子单元测试"""
import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.eif import (
    EIFDetector,
    EIFScorer,
    EIFScorerConfig,
)


@pytest.fixture
def train_data():
    rng = np.random.default_rng(42)
    return rng.normal(size=(30, 3))


@pytest.fixture
def test_data():
    rng = np.random.default_rng(123)
    return np.vstack([rng.normal(size=(8, 3)), rng.normal(10, 1, size=(2, 3))])


def test_config_frozen():
    config = EIFScorerConfig()
    with pytest.raises(Exception):
        config.n_estimators = 20


def test_scorer_fit_run_and_dataframe(train_data, test_data):
    scorer = EIFScorer(n_estimators=10, random_state=42)
    with pytest.raises(RuntimeError):
        scorer.run(test_data)

    scorer.fit(train_data)
    scores = scorer.run(test_data)
    assert scores.shape == (10,)
    assert np.isfinite(scores).all()

    frame_scores = scorer.run(DataFrame(test_data, columns=["a", "b", "c"]))
    assert isinstance(frame_scores, DataFrame)


def test_detector_fit_and_run(train_data, test_data):
    detector = EIFDetector(n_estimators=10, random_state=42, percentile=95.0)
    detector.fit(train_data)
    labels = detector.run(test_data)
    assert labels.shape == (10,)
    assert set(labels.tolist()).issubset({0, 1})


def test_scorer_save_load(train_data, test_data, tmp_path):
    scorer = EIFScorer(n_estimators=10, random_state=42)
    scorer.fit(train_data)
    original = scorer.run(test_data)
    scorer.save(tmp_path / "eif_scorer")

    loaded = EIFScorer.load(tmp_path / "eif_scorer")
    np.testing.assert_allclose(loaded.run(test_data), original)


def test_detector_save_load(train_data, test_data, tmp_path):
    detector = EIFDetector(n_estimators=10, random_state=42, percentile=95.0)
    detector.fit(train_data)
    original = detector.run(test_data)
    detector.save(tmp_path / "eif_detector")

    loaded = EIFDetector.load(tmp_path / "eif_detector")
    np.testing.assert_array_equal(loaded.run(test_data), original)
