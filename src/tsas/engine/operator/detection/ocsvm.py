"""OCSVM 异常检测算子

算法逻辑源自对应公开实现的 ocsvm.OCSVM, 按 TSA-Suite 算子规范重写。
不依赖 第三方库, 直接使用 sklearn.svm.OneClassSVM。

核心思想: 在特征空间中找到一个超平面, 将所有正常点包围在内, 落在外的为异常。
异常分数 = -score_samples(X), 越高越异常。
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.svm import OneClassSVM

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
)

__all__ = ['OCSVMScorerConfig', 'OCSVMScorer', 'OCSVMDetectorConfig', 'OCSVMDetector']


class OCSVMScorerConfig(BaseModel):
    """OCSVM 评分器实例参数"""
    model_config = {"frozen": True}
    kernel: str = Field(default="rbf", description="核函数类型")
    degree: int = Field(default=3, ge=1, description="多项式核阶数")
    gamma: str | float = Field(default="scale", description="核系数")
    coef0: float = Field(default=0.0, description="核函数独立项")
    tol: float = Field(default=1e-3, gt=0, description="收敛阈值")
    nu: float = Field(default=0.5, gt=0, le=1.0, description="异常比例上界")
    shrinking: bool = Field(default=True, description="是否使用收缩启发式")
    cache_size: float = Field(default=200.0, gt=0, description="核缓存大小 MB")
    max_iter: int = Field(default=-1, description="最大迭代次数, -1 为无限制")
    random_state: int | None = Field(default=None, description="随机种子")


class OCSVMScorer(SingleScorerMixin[None],
                  UnsupervisedNumericOperatorMixin[None],
                  NumericOperator[None, OCSVMScorerConfig, None]):
    """OCSVM 直接评分器"""
    _MODEL_FILE = '_model.pkl'

    @classmethod
    def name(cls) -> str:
        return "ocsvm_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: OneClassSVM | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = OneClassSVM(
            kernel=self.config.kernel,
            degree=self.config.degree,
            gamma=self.config.gamma,
            coef0=self.config.coef0,
            tol=self.config.tol,
            nu=self.config.nu,
            shrinking=self.config.shrinking,
            cache_size=self.config.cache_size,
            max_iter=self.config.max_iter,
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


class OCSVMDetectorConfig(BaseModel):
    """OCSVM 检测器实例参数"""
    model_config = {"frozen": True}
    kernel: str = Field(default="rbf", description="核函数类型")
    degree: int = Field(default=3, ge=1, description="多项式核阶数")
    gamma: str | float = Field(default="scale", description="核系数")
    coef0: float = Field(default=0.0, description="核函数独立项")
    tol: float = Field(default=1e-3, gt=0, description="收敛阈值")
    nu: float = Field(default=0.5, gt=0, le=1.0, description="异常比例上界")
    shrinking: bool = Field(default=True, description="是否使用收缩启发式")
    cache_size: float = Field(default=200.0, gt=0, description="核缓存大小 MB")
    max_iter: int = Field(default=-1, description="最大迭代次数, -1 为无限制")
    random_state: int | None = Field(default=None, description="随机种子")
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class OCSVMDetector(UnsupervisedNumericOperatorMixin[None],
                   BaseDeciderMixin[None],
                   NumericOperator[None, OCSVMDetectorConfig, None]):
    """OCSVM 检测器 = OCSVMScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "ocsvm_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = OCSVMScorer(config=OCSVMScorerConfig(
            kernel=self.config.kernel,
            degree=self.config.degree,
            gamma=self.config.gamma,
            coef0=self.config.coef0,
            tol=self.config.tol,
            nu=self.config.nu,
            shrinking=self.config.shrinking,
            cache_size=self.config.cache_size,
            max_iter=self.config.max_iter,
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
        self._scorer = OCSVMScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
