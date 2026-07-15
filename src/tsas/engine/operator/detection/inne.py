"""INNE 异常检测算子，不依赖 第三方库，直接使用 PyOD。"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pyod.models.inne import INNE

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
)

__all__ = ["INNEScorerConfig", "INNEScorer", "INNEDetectorConfig", "INNEDetector"]


class INNEScorerConfig(BaseModel):
    """INNE 评分器实例参数。"""

    model_config = {"frozen": True}
    n_estimators: int = Field(default=200, ge=1, description="估计器数量")
    max_samples: int | float | str = Field(default="auto", description="每个估计器的采样数")
    contamination: float = Field(default=0.1, gt=0, le=0.5, description="异常比例阈值")
    random_state: int | None = Field(default=None, description="随机种子")


class INNEScorer(
    SingleScorerMixin[None],
    UnsupervisedNumericOperatorMixin[None],
    NumericOperator[None, INNEScorerConfig, None],
):
    """INNE 直接评分器。"""

    _MODEL_FILE = "_model.pkl"

    @classmethod
    def name(cls) -> str:
        return "inne_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: INNE | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = INNE(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
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


class INNEDetectorConfig(INNEScorerConfig):
    """INNE 检测器实例参数。"""

    model_config = {"frozen": True}
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class INNEDetector(
    UnsupervisedNumericOperatorMixin[None],
    BaseDeciderMixin[None],
    NumericOperator[None, INNEDetectorConfig, None],
):
    """INNE 检测器，由评分器和百分位决策器组成。"""

    @classmethod
    def name(cls) -> str:
        return "inne_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = INNEScorer(config=INNEScorerConfig(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        ))
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
        self._scorer = INNEScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
