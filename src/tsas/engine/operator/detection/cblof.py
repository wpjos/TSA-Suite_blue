"""CBLOF 异常检测算子，不依赖 第三方库，直接使用 PyOD。"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pyod.models.cblof import CBLOF

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import PercentileDecider, PercentileDeciderConfig

__all__ = ["CBLOFScorerConfig", "CBLOFScorer", "CBLOFDetectorConfig", "CBLOFDetector"]


class CBLOFScorerConfig(BaseModel):
    model_config = {"frozen": True}
    n_clusters: int = Field(default=8, ge=2)
    contamination: float = Field(default=0.1, gt=0, le=0.5)
    alpha: float = Field(default=0.9, gt=0, lt=1)
    beta: float = Field(default=5, gt=1)
    use_weights: bool = False
    check_estimator: bool = False
    random_state: int | None = None
    n_jobs: int = Field(default=1, ge=-1)


class CBLOFScorer(SingleScorerMixin[None], UnsupervisedNumericOperatorMixin[None], NumericOperator[None, CBLOFScorerConfig, None]):
    _MODEL_FILE = "_model.pkl"

    @classmethod
    def name(cls) -> str:
        return "cblof_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: CBLOF | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = CBLOF(**self.config.model_dump())
        self._model.fit(x)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        return self._model.decision_function(x)

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        with open(path / self._MODEL_FILE, "wb") as file:
            pickle.dump(self._model, file)

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        with open(path / self._MODEL_FILE, "rb") as file:
            self._model = pickle.load(file)
        self._fitted = True


class CBLOFDetectorConfig(CBLOFScorerConfig):
    model_config = {"frozen": True}
    percentile: float = Field(default=95.0, ge=50.0, le=99.9)


class CBLOFDetector(UnsupervisedNumericOperatorMixin[None], BaseDeciderMixin[None], NumericOperator[None, CBLOFDetectorConfig, None]):
    @classmethod
    def name(cls) -> str:
        return "cblof_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        values = self.config.model_dump(exclude={"percentile"})
        self._scorer = CBLOFScorer(config=CBLOFScorerConfig(**values))
        self._decider = PercentileDecider(config=PercentileDeciderConfig(percentile=self.config.percentile))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._scorer.fit(x)
        self._decider.fit(self._scorer.run(x))

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        labels, _ = self._decider.run(self._scorer.run(x))
        return labels

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        self._scorer.save(path / "_scorer")
        self._decider.save(path / "_decider")

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        self._scorer = CBLOFScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
