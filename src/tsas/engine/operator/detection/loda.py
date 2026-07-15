"""LODA 异常检测算子，不依赖 第三方库。"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sklearn.random_projection import SparseRandomProjection

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import PercentileDecider, PercentileDeciderConfig


class LODAScorerConfig(BaseModel):
    """LODA 评分器参数。"""
    model_config = {"frozen": True}
    n_bins: int | str = Field(default=10, description="直方图箱数或 auto")
    n_random_cuts: int = Field(default=100, ge=1, description="随机投影数量")
    random_state: int | None = Field(default=None, description="随机种子")

    @field_validator("n_bins")
    @classmethod
    def validate_n_bins(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value >= 1:
            return value
        if isinstance(value, str) and value.lower() == "auto":
            return "auto"
        raise ValueError("n_bins must be a positive integer or 'auto'")


class LODAScorer(SingleScorerMixin[None], UnsupervisedNumericOperatorMixin[None], NumericOperator[None, LODAScorerConfig, None]):
    """随机稀疏投影与直方图集成评分器。"""
    _MODEL_FILE = "_model.pkl"

    @classmethod
    def name(cls) -> str:
        return "loda_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: dict | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        projection = SparseRandomProjection(
            n_components=self.config.n_random_cuts,
            density="auto",
            random_state=self.config.random_state,
        )
        projected = projection.fit_transform(x)
        projected = np.asarray(projected)
        histograms = []
        limits = []
        for column in projected.T:
            histogram, edges = np.histogram(column, bins=self.config.n_bins, density=False)
            probability = histogram.astype(float) + 1e-12
            probability /= probability.sum()
            histograms.append(probability)
            limits.append(edges)
        self._model = {"projection": projection, "histograms": histograms, "limits": limits}

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        projected = np.asarray(self._model["projection"].transform(x))
        scores = np.zeros(projected.shape[0], dtype=float)
        for column, histogram, edges in zip(projected.T, self._model["histograms"], self._model["limits"]):
            indices = np.searchsorted(edges[1:-1], column, side="left")
            scores -= np.log(histogram[indices])
        return scores / self.config.n_random_cuts

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        with open(path / self._MODEL_FILE, "wb") as file:
            pickle.dump(self._model, file)

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        with open(path / self._MODEL_FILE, "rb") as file:
            self._model = pickle.load(file)
        self._fitted = True


class LODADetectorConfig(LODAScorerConfig):
    """LODA 检测器参数。"""
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class LODADetector(UnsupervisedNumericOperatorMixin[None], BaseDeciderMixin[None], NumericOperator[None, LODADetectorConfig, None]):
    """LODA 检测器。"""

    @classmethod
    def name(cls) -> str:
        return "loda_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = LODAScorer(config=LODAScorerConfig(
            n_bins=self.config.n_bins,
            n_random_cuts=self.config.n_random_cuts,
            random_state=self.config.random_state,
        ))
        self._decider = PercentileDecider(config=PercentileDeciderConfig(percentile=self.config.percentile))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._scorer.fit(x)
        scores = self._scorer.run(x)
        self._decider.fit(scores)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        scores = self._scorer.run(x)
        labels, _ = self._decider.run(scores)
        return labels

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        self._scorer.save(path / "_scorer")
        self._decider.save(path / "_decider")

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        self._scorer = LODAScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
