# -*- coding: utf-8 -*-

"""
Normalize Score 评价指标算子（PyOD 移植）

把异常分数归一化到 ``[0, 1]``，支持两种方法：
- ``linear``：用训练集分数拟合 ``MinMaxScaler``，应用到测试分数。
- ``unify``：用训练集分数的 ``mu``/``sigma`` 做 z-score，再过 ``erf``。

归一化器只用 train_scores 拟合再 transform 到 test_scores，避免 data leakage。

核心组件:
    - NormalizeScoreResult: 归一化结果（Pydantic BaseModel）
    - NormalizeScoreConfig: 配置类（继承 BaseMetricConfig）
    - NormalizeScoreMetric: 归一化指标算子

使用示例::

    from tsas.engine.operator.evaluation import NormalizeScoreMetric

    op = NormalizeScoreMetric(method='linear')
    result = op.run((train_scores, test_scores))
    print(result.train_norm, result.test_norm)
"""

from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'NormalizeScoreResult',
    'NormalizeScoreConfig',
    'NormalizeScoreMetric',
]


class NormalizeScoreResult(BaseModel):
    """归一化结果

    Attributes:
        train_norm (np.ndarray): train_scores 的归一化结果，shape (n_train,)
        test_norm (np.ndarray): test_scores 的归一化结果，shape (n_test,)；
            仅当输入提供 test_scores 时存在
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    train_norm: np.ndarray = Field(description="train_scores 的归一化结果")
    test_norm: np.ndarray | None = Field(
        default=None,
        description="test_scores 的归一化结果；输入未提供时为 None",
    )


class NormalizeScoreConfig(BaseMetricConfig):
    """Normalize Score 评价指标配置

    Attributes:
        method (Literal['linear', 'unify']): 归一化方法，默认 ``'linear'``
        return_test (bool): 是否返回 test_scores 归一化结果，默认 ``True``
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``None``
            （归一化返回数组，无单一主评分字段）
    """
    model_config = ConfigDict(frozen=True)

    method: Literal['linear', 'unify'] = Field(
        default='linear',
        description="归一化方法：linear（MinMaxScaler）或 unify（erf z-score）",
    )
    return_test: bool = Field(
        default=True,
        description="是否返回 test_scores 的归一化结果",
    )
    main_scores: dict[str, str] | None = Field(
        default=None,
        description="主评分路径映射；归一化返回数组，无单一主评分字段，默认 None",
    )


class NormalizeScoreMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray | None],
        NormalizeScoreResult,
        NormalizeScoreConfig,
        None,
    ],
):
    """Normalize Score 评价指标算子

    Input:
        train_scores: 训练集异常分数，shape (n_train,)，用于拟合归一化器
        test_scores: 待归一化的测试分数，shape (n_test,)；``None`` 时只对 train 归一化

    Output:
        NormalizeScoreResult: train_norm + test_norm（可选）

    泛型参数:
        I: tuple[np.ndarray, np.ndarray | None] — (train_scores, test_scores)
        MR: NormalizeScoreResult — 归一化结果
        MC: NormalizeScoreConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "normalize_score_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray | None],
        *,
        params: None,
    ) -> NormalizeScoreResult:
        train_scores, test_scores = x
        train = np.asarray(train_scores, dtype=float).ravel()

        config = self.config
        method = config.method if config is not None else 'linear'
        return_test = config.return_test if config is not None else True

        train_norm = _normalize_one(train, train, method)
        test_norm: np.ndarray | None = None
        if return_test and test_scores is not None:
            test = np.asarray(test_scores, dtype=float).ravel()
            test_norm = _normalize_one(train, test, method)

        return NormalizeScoreResult(
            train_norm=np.asarray(train_norm, dtype=float),
            test_norm=(np.asarray(test_norm, dtype=float) if test_norm is not None else None),
        )


def _normalize_one(ref: np.ndarray, x: np.ndarray, method: str) -> np.ndarray:
    """用 ``ref`` 拟合归一化器，应用到 ``x``。

    Args:
        ref: 用于拟合归一化器的参考数组（一维）
        x: 待归一化的输入数组（一维）
        method: 归一化方法，``'linear'``（MinMax）或 ``'unify'``（erf）

    Returns:
        归一化到 ``[0, 1]`` 的一维数组
    """
    if method == 'linear':
        scaler = MinMaxScaler().fit(ref.reshape(-1, 1))
        out = scaler.transform(x.reshape(-1, 1)).ravel()
        return np.clip(out, 0.0, 1.0)

    mu = float(np.mean(ref))
    sigma = float(np.std(ref))
    if sigma == 0.0:
        return np.full_like(x, 0.5, dtype=float)
    pre_erf = (x - mu) / (sigma * np.sqrt(2.0))
    return np.clip(erf(pre_erf), 0.0, 1.0)