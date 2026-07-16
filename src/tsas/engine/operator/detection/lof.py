"""LOF 异常检测算子

算法逻辑源自对应公开实现的 lof.LOF, 按 TSA-Suite 算子规范重写。
不依赖 第三方库, 直接使用 sklearn.neighbors.LocalOutlierFactor。

核心思想: 异常点的局部密度显著低于其邻居, LOF 越大越异常。
异常分数 = -score_samples(X), 越高越异常。
novelty=True 才能对新数据打分。
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.neighbors import LocalOutlierFactor

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
)

__all__ = ['LOFScorerConfig', 'LOFScorer', 'LOFDetectorConfig', 'LOFDetector']


class LOFScorerConfig(BaseModel):
    """LOF 评分器实例参数"""
    model_config = {"frozen": True}
    n_neighbors: int = Field(default=20, ge=1, description="近邻数")
    algorithm: str = Field(default="auto", description="近邻搜索算法")
    leaf_size: int = Field(default=30, ge=1, description="树叶节点大小")
    metric: str = Field(default="minkowski", description="距离度量")
    p: int = Field(default=2, ge=1, description="Minkowski 距离参数")
    metric_params: dict | None = Field(default=None, description="距离度量参数")
    n_jobs: int = Field(default=1, ge=-1, description="并行数")
    contamination: float = Field(default=0.1, gt=0, le=0.5, description="异常比例阈值")


class LOFScorer(SingleScorerMixin[None],
                UnsupervisedNumericOperatorMixin[None],
                NumericOperator[None, LOFScorerConfig, None]):
    """LOF 直接评分器"""
    _MODEL_FILE = '_model.pkl'

    @classmethod
    def name(cls) -> str:
        return "lof_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: LocalOutlierFactor | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = LocalOutlierFactor(
            n_neighbors=self.config.n_neighbors,
            algorithm=self.config.algorithm,
            leaf_size=self.config.leaf_size,
            metric=self.config.metric,
            p=self.config.p,
            metric_params=self.config.metric_params,
            contamination=self.config.contamination,
            n_jobs=self.config.n_jobs,
            novelty=True,
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


class LOFDetectorConfig(BaseModel):
    """LOF 检测器实例参数"""
    model_config = {"frozen": True}
    n_neighbors: int = Field(default=20, ge=1, description="近邻数")
    algorithm: str = Field(default="auto", description="近邻搜索算法")
    leaf_size: int = Field(default=30, ge=1, description="树叶节点大小")
    metric: str = Field(default="minkowski", description="距离度量")
    p: int = Field(default=2, ge=1, description="Minkowski 距离参数")
    metric_params: dict | None = Field(default=None, description="距离度量参数")
    n_jobs: int = Field(default=1, ge=-1, description="并行数")
    contamination: float = Field(default=0.1, gt=0, le=0.5, description="异常比例阈值")
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class LOFDetector(UnsupervisedNumericOperatorMixin[None],
                  BaseDeciderMixin[None],
                  NumericOperator[None, LOFDetectorConfig, None]):
    """LOF 检测器 = LOFScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "lof_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = LOFScorer(config=LOFScorerConfig(
            n_neighbors=self.config.n_neighbors,
            algorithm=self.config.algorithm,
            leaf_size=self.config.leaf_size,
            metric=self.config.metric,
            p=self.config.p,
            metric_params=self.config.metric_params,
            n_jobs=self.config.n_jobs,
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
        self._scorer = LOFScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
