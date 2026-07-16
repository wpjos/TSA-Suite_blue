"""HBOS 异常检测算子

算法逻辑源自对应公开实现的 hbos.HBOS, 按 TSA-Suite 算子规范重写。
不依赖 第三方库, 直接使用 pyod.models.hbos.HBOS。

核心思想: 假设特征独立, 对每个特征构建直方图, 异常分数为低密度 bin 的累积对数概率。
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pyod.models.hbos import HBOS

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
)

__all__ = ['HBOSScorerConfig', 'HBOSScorer', 'HBOSDetectorConfig', 'HBOSDetector']


class HBOSScorerConfig(BaseModel):
    """HBOS 评分器实例参数"""
    model_config = {"frozen": True}
    n_bins: int | str = Field(default=10, description="直方图箱数, int 或 'auto'")
    alpha: float = Field(default=0.1, gt=0, lt=1, description="正则化参数, 避免零概率")
    tol: float = Field(default=0.5, gt=0, lt=1, description="容差倍数")
    contamination: float = Field(default=0.1, gt=0, le=0.5, description="异常比例阈值")


class HBOSScorer(SingleScorerMixin[None],
                 UnsupervisedNumericOperatorMixin[None],
                 NumericOperator[None, HBOSScorerConfig, None]):
    """HBOS 直接评分器"""
    _MODEL_FILE = '_model.pkl'

    @classmethod
    def name(cls) -> str:
        return "hbos_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: HBOS | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = HBOS(
            n_bins=self.config.n_bins,
            alpha=self.config.alpha,
            tol=self.config.tol,
            contamination=self.config.contamination,
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


class HBOSDetectorConfig(BaseModel):
    """HBOS 检测器实例参数"""
    model_config = {"frozen": True}
    n_bins: int | str = Field(default=10, description="直方图箱数")
    alpha: float = Field(default=0.1, gt=0, lt=1, description="正则化参数")
    tol: float = Field(default=0.5, gt=0, lt=1, description="容差倍数")
    contamination: float = Field(default=0.1, gt=0, le=0.5, description="异常比例阈值")
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class HBOSDetector(UnsupervisedNumericOperatorMixin[None],
                   BaseDeciderMixin[None],
                   NumericOperator[None, HBOSDetectorConfig, None]):
    """HBOS 检测器 = HBOSScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "hbos_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = HBOSScorer(config=HBOSScorerConfig(
            n_bins=self.config.n_bins,
            alpha=self.config.alpha,
            tol=self.config.tol,
            contamination=self.config.contamination,
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
        self._scorer = HBOSScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
