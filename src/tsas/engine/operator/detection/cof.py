"""COF 异常检测算子，不依赖 第三方库，直接使用 PyOD。"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pyod.models.cof import COF

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import PercentileDecider, PercentileDeciderConfig

__all__ = ["COFScorerConfig", "COFScorer", "COFDetectorConfig", "COFDetector"]


class COFScorerConfig(BaseModel):
    model_config = {"frozen": True}
    contamination: float = Field(default=0.1, gt=0, le=0.5)
    n_neighbors: int = Field(default=20, ge=1)
    method: str = Field(default="fast", pattern="^(fast|memory)$")


class COFScorer(SingleScorerMixin[None], UnsupervisedNumericOperatorMixin[None], NumericOperator[None, COFScorerConfig, None]):
    _MODEL_FILE = "_model.pkl"

    @classmethod
    def name(cls) -> str:
        return "cof_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: COF | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = COF(**self.config.model_dump())
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


class COFDetectorConfig(COFScorerConfig):
    model_config = {"frozen": True}
    percentile: float = Field(default=95.0, ge=50.0, le=99.9)


class COFDetector(UnsupervisedNumericOperatorMixin[None], BaseDeciderMixin[None], NumericOperator[None, COFDetectorConfig, None]):
    @classmethod
    def name(cls) -> str:
        return "cof_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        values = self.config.model_dump(exclude={"percentile"})
        self._scorer = COFScorer(config=COFScorerConfig(**values))
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
        self._scorer = COFScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
