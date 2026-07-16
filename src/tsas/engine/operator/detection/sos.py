"""SOS 异常检测算子

算法逻辑源自对应公开实现的 sos.SOS, 按 TSA-Suite 算子规范重写。
不依赖 第三方库, 纯 numpy 实现 SOS 算法 (Janssens, 2012)。

核心思想: 用高斯核把样本间相异度转为亲和力, 归一化为绑定概率,
异常分数 = 其他所有样本都不与它绑定的联合概率。
"""
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
)

__all__ = ['SOSScorerConfig', 'SOSScorer', 'SOSDetectorConfig', 'SOSDetector']


class SOSScorerConfig(BaseModel):
    """SOS 评分器实例参数"""
    model_config = {"frozen": True}
    perplexity: float = Field(default=4.5, gt=1.0, description="有效近邻数的平滑度量")
    metric: str = Field(default="euclidean", description="距离度量")
    eps: float = Field(default=1e-5, gt=0.0, description="浮点误差容忍阈值")


class SOSScorer(SingleScorerMixin[None],
                UnsupervisedNumericOperatorMixin[None],
                NumericOperator[None, SOSScorerConfig, None]):
    """SOS 直接评分器 - 纯 numpy 实现"""
    _LEARNED_STATE_FILE = '_learned_state.npz'

    @classmethod
    def name(cls) -> str:
        return "sos_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._train_scores: np.ndarray | None = None

    def _x2d(self, X: np.ndarray) -> np.ndarray:
        """计算相异度矩阵 (欧氏距离)。"""
        sumX = np.sum(np.square(X), axis=1)
        D = np.sqrt(np.abs(np.add(np.add(-2 * np.dot(X, X.T), sumX).T, sumX)))
        return D

    def _get_perplexity(self, D: np.ndarray, beta: float) -> tuple:
        A = np.exp(-D * beta)
        sumA = np.sum(A)
        H = np.log(sumA) + beta * np.sum(D * A) / sumA
        return H, A

    def _d2a(self, D: np.ndarray) -> np.ndarray:
        """通过二分搜索调整高斯核精度, 返回亲和力矩阵。"""
        n = D.shape[0]
        A = np.zeros((n, n))
        beta = np.ones(n)
        logU = np.log(self.config.perplexity)
        for i in range(n):
            betamin = -np.inf
            betamax = np.inf
            idx = np.concatenate([np.r_[:i], np.r_[i + 1:n]])
            Di = D[i, idx]
            H, thisA = self._get_perplexity(Di, beta[i])
            Hdiff = H - logU
            tries = 0
            while (np.isnan(Hdiff) or np.abs(Hdiff) > self.config.eps) and tries < 5000:
                if np.isnan(Hdiff):
                    beta[i] = beta[i] / 10.0
                elif Hdiff > 0:
                    betamin = beta[i]
                    if np.isinf(betamax):
                        beta[i] = beta[i] * 2.0
                    else:
                        beta[i] = (beta[i] + betamax) / 2.0
                else:
                    betamax = beta[i]
                    if np.isinf(betamin):
                        beta[i] = beta[i] / 2.0
                    else:
                        beta[i] = (beta[i] + betamin) / 2.0
                H, thisA = self._get_perplexity(Di, beta[i])
                Hdiff = H - logU
                tries += 1
            A[i, idx] = thisA
        return A

    def _a2b(self, A: np.ndarray) -> np.ndarray:
        return A / A.sum(axis=1)[:, np.newaxis]

    def _b2o(self, B: np.ndarray) -> np.ndarray:
        return np.prod(1 - B, axis=0)

    def _decision(self, X: np.ndarray) -> np.ndarray:
        D = self._x2d(X)
        A = self._d2a(D)
        B = self._a2b(A)
        return self._b2o(B)

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._train_scores = self._decision(x)

    def _run_data(self, x: np.ndarray, params: None, idx=None) -> np.ndarray:
        return self._decision(x)

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        np.savez(path / self._LEARNED_STATE_FILE, train_scores=self._train_scores)

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        data = np.load(path / self._LEARNED_STATE_FILE)
        self._train_scores = data['train_scores']
        self._fitted = True


class SOSDetectorConfig(BaseModel):
    """SOS 检测器实例参数"""
    model_config = {"frozen": True}
    perplexity: float = Field(default=4.5, gt=1.0, description="有效近邻数")
    metric: str = Field(default="euclidean", description="距离度量")
    eps: float = Field(default=1e-5, gt=0.0, description="浮点误差容忍阈值")
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class SOSDetector(UnsupervisedNumericOperatorMixin[None],
                  BaseDeciderMixin[None],
                  NumericOperator[None, SOSDetectorConfig, None]):
    """SOS 检测器 = SOSScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "sos_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = SOSScorer(config=SOSScorerConfig(
            perplexity=self.config.perplexity,
            metric=self.config.metric,
            eps=self.config.eps,
        ))
        self._decider = PercentileDecider(config=PercentileDeciderConfig(percentile=self.config.percentile))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._scorer.fit(x)
        scores = self._scorer.run(x)
        self._decider.fit(scores)

    def _run_data(self, x: np.ndarray, params: None, idx=None) -> np.ndarray:
        scores = self._scorer.run(x)
        labels, _ = self._decider.run(scores)
        return labels

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        self._scorer.save(path / "_scorer")
        self._decider.save(path / "_decider")

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        self._scorer = SOSScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
