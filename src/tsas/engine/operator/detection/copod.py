"""COPOD 异常检测算子

算法逻辑源自对应公开实现的 copod.COPOD, 按 TSA-Suite 算子规范重写。
不依赖 第三方库, 直接使用 pyod.models.copod.COPOD。

核心思想: Copula-Based Outlier Detection, 无参数, 用经验 copula 函数拟合联合分布。
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pyod.models.copod import COPOD

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
)

__all__ = ['COPODScorerConfig', 'COPODScorer', 'COPODDetectorConfig', 'COPODDetector']


class COPODScorerConfig(BaseModel):
    """COPOD 评分器实例参数"""
    model_config = {"frozen": True}
    contamination: float = Field(default=0.1, gt=0, le=0.5, description="异常比例阈值")
    n_jobs: int = Field(default=1, ge=-1, description="并行数")


class COPODScorer(SingleScorerMixin[None],
                  UnsupervisedNumericOperatorMixin[None],
                  NumericOperator[None, COPODScorerConfig, None]):
    """COPOD 直接评分器"""
    _MODEL_FILE = '_model.pkl'

    @classmethod
    def name(cls) -> str:
        return "copod_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: COPOD | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = COPOD(
            contamination=self.config.contamination,
            n_jobs=self.config.n_jobs,
        )
        self._model.fit(x)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        return self._model.decision_function(x)

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        with open(path / self._MODEL_FILE, "wb") as f:
            pickle.dump(self._model, f)

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        with open(path / self._MODEL_FILE, "rb") as f:
            self._model = pickle.load(f)
        self._fitted = True


class COPODDetectorConfig(BaseModel):
    """COPOD 检测器实例参数"""
    model_config = {"frozen": True}
    contamination: float = Field(default=0.1, gt=0, le=0.5, description="异常比例阈值")
    n_jobs: int = Field(default=1, ge=-1, description="并行数")
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class COPODDetector(UnsupervisedNumericOperatorMixin[None],
                    BaseDeciderMixin[None],
                    NumericOperator[None, COPODDetectorConfig, None]):
    """COPOD 检测器 = COPODScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "copod_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = COPODScorer(config=COPODScorerConfig(
            contamination=self.config.contamination,
            n_jobs=self.config.n_jobs,
        ))
        self._decider = PercentileDecider(config=PercentileDeciderConfig(
            percentile=self.config.percentile,
        ))

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
        self._scorer = COPODScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
