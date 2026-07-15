"""KDE异常检测算子，不依赖 第三方库。"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.neighbors import KernelDensity

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import PercentileDecider, PercentileDeciderConfig


class KDEScorerConfig(BaseModel):
    """KDE评分器参数。"""
    model_config = {"frozen": True}
    bandwidth: float = Field(default=1.0, gt=0, description="核带宽")
    algorithm: str = Field(default="auto", description="密度估计算法")
    kernel: str = Field(default="gaussian", description="核函数")
    metric: str = Field(default="euclidean", description="距离度量")
    atol: float = Field(default=0.0, ge=0, description="绝对容差")
    rtol: float = Field(default=0.0, ge=0, description="相对容差")
    breadth_first: bool = Field(default=True, description="是否广度优先搜索")
    leaf_size: int = Field(default=40, ge=1, description="叶节点大小")


class KDEScorer(SingleScorerMixin[None], UnsupervisedNumericOperatorMixin[None], NumericOperator[None, KDEScorerConfig, None]):
    """KDE直接评分器。"""
    _MODEL_FILE = "_model.pkl"

    @classmethod
    def name(cls) -> str:
        return "kde_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: KernelDensity | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = KernelDensity(
            bandwidth=self.config.bandwidth,
            algorithm=self.config.algorithm,
            kernel=self.config.kernel,
            metric=self.config.metric,
            atol=self.config.atol,
            rtol=self.config.rtol,
            breadth_first=self.config.breadth_first,
            leaf_size=self.config.leaf_size,
        )
        self._model.fit(x)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        return -self._model.score_samples(x)

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        with open(path / self._MODEL_FILE, "wb") as file:
            pickle.dump(self._model, file)

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        with open(path / self._MODEL_FILE, "rb") as file:
            self._model = pickle.load(file)
        self._fitted = True


class KDEDetectorConfig(KDEScorerConfig):
    """KDE检测器参数。"""
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class KDEDetector(UnsupervisedNumericOperatorMixin[None], BaseDeciderMixin[None], NumericOperator[None, KDEDetectorConfig, None]):
    """KDE检测器。"""

    @classmethod
    def name(cls) -> str:
        return "kde_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = KDEScorer(config=KDEScorerConfig(
            bandwidth=self.config.bandwidth,
            algorithm=self.config.algorithm,
            kernel=self.config.kernel,
            metric=self.config.metric,
            atol=self.config.atol,
            rtol=self.config.rtol,
            breadth_first=self.config.breadth_first,
            leaf_size=self.config.leaf_size
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
        self._scorer = KDEScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
