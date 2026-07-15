"""MCD异常检测算子，不依赖 第三方库。"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.covariance import MinCovDet

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import PercentileDecider, PercentileDeciderConfig


class MCDScorerConfig(BaseModel):
    """MCD评分器参数。"""
    model_config = {"frozen": True}
    store_precision: bool = Field(default=True, description="是否存储精度矩阵")
    assume_centered: bool = Field(default=False, description="是否假设数据已中心化")
    support_fraction: float | None = Field(default=None, gt=0, le=1, description="支持集比例")
    random_state: int | None = Field(default=None, description="随机种子")


class MCDScorer(SingleScorerMixin[None], UnsupervisedNumericOperatorMixin[None], NumericOperator[None, MCDScorerConfig, None]):
    """MCD直接评分器。"""
    _MODEL_FILE = "_model.pkl"

    @classmethod
    def name(cls) -> str:
        return "mcd_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: MinCovDet | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = MinCovDet(
            store_precision=self.config.store_precision,
            assume_centered=self.config.assume_centered,
            support_fraction=self.config.support_fraction,
            random_state=self.config.random_state,
        )
        self._model.fit(x)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        return self._model.mahalanobis(x)

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        with open(path / self._MODEL_FILE, "wb") as file:
            pickle.dump(self._model, file)

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        with open(path / self._MODEL_FILE, "rb") as file:
            self._model = pickle.load(file)
        self._fitted = True


class MCDDetectorConfig(MCDScorerConfig):
    """MCD检测器参数。"""
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class MCDDetector(UnsupervisedNumericOperatorMixin[None], BaseDeciderMixin[None], NumericOperator[None, MCDDetectorConfig, None]):
    """MCD检测器。"""

    @classmethod
    def name(cls) -> str:
        return "mcd_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = MCDScorer(config=MCDScorerConfig(
            store_precision=self.config.store_precision,
            assume_centered=self.config.assume_centered,
            support_fraction=self.config.support_fraction,
            random_state=self.config.random_state
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
        self._scorer = MCDScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
