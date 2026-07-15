"""GMM异常检测算子，不依赖 第三方库。"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.mixture import GaussianMixture

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import PercentileDecider, PercentileDeciderConfig


class GMMScorerConfig(BaseModel):
    """GMM评分器参数。"""
    model_config = {"frozen": True}
    n_components: int = Field(default=1, ge=1, description="高斯成分数")
    covariance_type: str = Field(default="full", description="协方差类型")
    tol: float = Field(default=1e-3, gt=0, description="收敛阈值")
    reg_covar: float = Field(default=1e-6, ge=0, description="协方差正则项")
    max_iter: int = Field(default=100, ge=1, description="最大迭代次数")
    n_init: int = Field(default=1, ge=1, description="初始化次数")
    init_params: str = Field(default="kmeans", description="初始化方法")
    random_state: int | None = Field(default=None, description="随机种子")
    warm_start: bool = Field(default=False, description="是否热启动")


class GMMScorer(SingleScorerMixin[None], UnsupervisedNumericOperatorMixin[None], NumericOperator[None, GMMScorerConfig, None]):
    """GMM直接评分器。"""
    _MODEL_FILE = "_model.pkl"

    @classmethod
    def name(cls) -> str:
        return "gmm_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: GaussianMixture | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = GaussianMixture(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            tol=self.config.tol,
            reg_covar=self.config.reg_covar,
            max_iter=self.config.max_iter,
            n_init=self.config.n_init,
            init_params=self.config.init_params,
            random_state=self.config.random_state,
            warm_start=self.config.warm_start,
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


class GMMDetectorConfig(GMMScorerConfig):
    """GMM检测器参数。"""
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class GMMDetector(UnsupervisedNumericOperatorMixin[None], BaseDeciderMixin[None], NumericOperator[None, GMMDetectorConfig, None]):
    """GMM检测器。"""

    @classmethod
    def name(cls) -> str:
        return "gmm_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = GMMScorer(config=GMMScorerConfig(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            tol=self.config.tol,
            reg_covar=self.config.reg_covar,
            max_iter=self.config.max_iter,
            n_init=self.config.n_init,
            init_params=self.config.init_params,
            random_state=self.config.random_state,
            warm_start=self.config.warm_start
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
        self._scorer = GMMScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
