"""sklearn 检测算子通用测试。"""
import importlib

import numpy as np
import pytest
from pandas import DataFrame


@pytest.fixture
def train_data():
    rng = np.random.RandomState(42)
    return rng.randn(60, 3)


@pytest.fixture
def test_data():
    rng = np.random.RandomState(123)
    return np.vstack([rng.randn(18, 3), rng.randn(2, 3) + 10])


@pytest.mark.parametrize(
    ("module_name", "class_name", "kwargs", "frozen_field", "frozen_value"),
    [
        ("loda", "LODA", {"n_bins": 10, "n_random_cuts": 10, "random_state": 42}, "n_bins", 20),
        ("mcd", "MCD", {"random_state": 42}, "store_precision", False),
        ("gmm", "GMM", {"n_components": 2, "random_state": 42}, "n_components", 1),
        ("kde", "KDE", {"bandwidth": 1.0}, "bandwidth", 2.0),
    ],
)
def test_scorer_and_detector(
    module_name,
    class_name,
    kwargs,
    frozen_field,
    frozen_value,
    train_data,
    test_data,
    tmp_path,
):
    module = importlib.import_module(f"tsas.engine.operator.detection.{module_name}")
    scorer_class = getattr(module, f"{class_name}Scorer")
    config_class = getattr(module, f"{class_name}ScorerConfig")
    detector_class = getattr(module, f"{class_name}Detector")

    config = config_class(**kwargs)
    with pytest.raises(Exception):
        setattr(config, frozen_field, frozen_value)

    scorer = scorer_class(**kwargs)
    with pytest.raises(RuntimeError):
        scorer.run(test_data)
    scorer.fit(train_data)
    scores = scorer.run(test_data)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()

    frame_scores = scorer.run(DataFrame(test_data, columns=["a", "b", "c"]))
    assert isinstance(frame_scores, DataFrame)

    scorer_path = tmp_path / f"{module_name}_scorer"
    scorer.save(scorer_path)
    loaded_scorer = scorer_class.load(scorer_path)
    np.testing.assert_allclose(loaded_scorer.run(test_data), scores)

    detector = detector_class(percentile=95.0, **kwargs)
    with pytest.raises(RuntimeError):
        detector.run(test_data)
    detector.fit(train_data)
    labels = detector.run(test_data)
    assert labels.shape == (20,)
    assert set(labels.tolist()).issubset({0, 1})

    detector_path = tmp_path / f"{module_name}_detector"
    detector.save(detector_path)
    loaded_detector = detector_class.load(detector_path)
    np.testing.assert_array_equal(loaded_detector.run(test_data), labels)
