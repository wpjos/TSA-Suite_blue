"""
Isolation Forest 异常检测算子

算法逻辑源自对应公开实现的 iforest.IForest, 按 TSA-Suite 算子规范重写。
不依赖 第三方库, 直接使用 sklearn.ensemble.IsolationForest。

核心思想: 异常点更容易被随机切分隔离, 正常点需要更多切分才能隔离。
异常分数 = -score_samples(X), 越高越异常。

包含:
    - IForestScorer: 直接评分器, 输出异常分数
    - IForestDetector: 端到端检测器, 组合 IForestScorer + PercentileDecider
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
)

__all__ = [
    'IForestScorerConfig',
    'IForestScorer',
    'IForestDetectorConfig',
    'IForestDetector',
]


class IForestScorerConfig(BaseModel):
    """IForest 评分器实例参数"""
    model_config = {"frozen": True}
    n_estimators: int = Field(default=100, ge=10, le=500, description="树的数量")
    max_samples: int | float | str = Field(default="auto", description="每棵树采样数")
    max_features: int | float = Field(default=1.0, gt=0, le=1.0, description="每棵树特征采样比例")
    bootstrap: bool = Field(default=False, description="是否有放回采样")
    contamination: float = Field(default=0.1, gt=0, le=0.5, description="异常比例阈值")
    n_jobs: int = Field(default=1, ge=-1, description="并行数")
    random_state: int | None = Field(default=None, description="随机种子")


class IForestScorer(SingleScorerMixin[None],
                    UnsupervisedNumericOperatorMixin[None],
                    NumericOperator[None, IForestScorerConfig, None]):
    """Isolation Forest 直接评分器"""
    _MODEL_FILE = '_model.pkl'

    @classmethod
    def name(cls) -> str:
        return "iforest_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: IsolationForest | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            contamination=self.config.contamination,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )
        self._model.fit(x)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        return -self._model.score_samples(x)

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        with open(path / self._MODEL_FILE, "wb") as f:
            pickle.dump(self._model, f)

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        with open(path / self._MODEL_FILE, "rb") as f:
            self._model = pickle.load(f)
        self._fitted = True


class IForestDetectorConfig(BaseModel):
    """IForest 检测器实例参数"""
    model_config = {"frozen": True}
    n_estimators: int = Field(default=100, ge=10, le=500, description="树的数量")
    max_samples: int | float | str = Field(default="auto", description="每棵树采样数")
    max_features: int | float = Field(default=1.0, gt=0, le=1.0, description="每棵树特征采样比例")
    bootstrap: bool = Field(default=False, description="是否有放回采样")
    contamination: float = Field(default=0.1, gt=0, le=0.5, description="异常比例阈值")
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class IForestDetector(UnsupervisedNumericOperatorMixin[None],
                     BaseDeciderMixin[None],
                     NumericOperator[None, IForestDetectorConfig, None]):
    """IForest 检测器 = IForestScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "iforest_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = IForestScorer(config=IForestScorerConfig(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
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
        self._scorer = IForestScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
