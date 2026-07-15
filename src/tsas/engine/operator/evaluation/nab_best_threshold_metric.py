# -*- coding: utf-8 -*-

"""
NAB Best Threshold 评价指标算子

试遍所有 unique y_score 取最大 NAB score。等价于
``max_T nab_score(y_true, y_score, threshold=T)``，对应 Numenta
``Sweeper.calcScoreByThreshold`` 的「max-over-thresholds」语义。

注意：返回的 best_threshold 只在该数据集上有意义，不可直接部署。

核心组件:
    - NabBestThresholdConfig: 配置类（继承 BaseMetricConfig）
    - NabBestThresholdMetric: NAB 最佳阈值上限指标算子

使用示例::

    from tsas.engine.operator.evaluation import NabBestThresholdMetric

    op = NabBestThresholdMetric()
    result = op.run((y_truth, y_score))
"""

import math
from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation._vus_utils import events_inclusive
from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)
from tsas.engine.operator.evaluation.nab_score_metric import (
    DEFAULT_PROBATION_PERCENT,
    DEFAULT_WINDOW_SIZE_RATIO,
    NAB_PROFILES,
    _assign_scorable_names,
    _compute_windows,
    _score_curve,
    _sweep_scores,
)

__all__ = [
    'NabBestThresholdConfig',
    'NabBestThresholdMetric',
]


class NabBestThresholdConfig(BaseMetricConfig):
    """NAB 最佳阈值评价指标配置

    Attributes:
        window_length (int | None): 窗口半宽；``None`` 时按 NAB 默认推算
        probation_percent (float): probaton 期比例，默认 0.15
        profile (str): NAB profile 名称，默认 ``'standard'``
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"nab_best_threshold": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    window_length: int | None = Field(
        default=None,
        description="窗口半宽；None 时按 NAB 默认 0.1*N/n 推算",
    )
    probation_percent: float = Field(
        default=DEFAULT_PROBATION_PERCENT,
        description="probaton 期比例（默认 0.15）",
    )
    profile: str = Field(
        default="standard",
        description="NAB profile：standard / reward_low_FP_rate / reward_low_FN_rate",
    )
    main_scores: dict[str, str] | None = Field(
        default={"nab_best_threshold": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class NabBestThresholdMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        NabBestThresholdConfig,
        None,
    ],
):
    """NAB Best Threshold 评价指标算子

    在所有 unique y_score 上 sweep 取最大 NAB score。
    对应算法在该数据集上「任意 threshold 可达的上限」。

    Input:
        y_truth: 0/1 真值标签向量，shape (N,)
        y_score: 连续异常得分（越大越异常），shape (N,)

    Output:
        float: 最佳 NAB score

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: float — 标量 best NAB score
        MC: NabBestThresholdConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "nab_best_threshold_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> float:
        y_true, y_score = x
        y_true = np.asarray(y_true).astype(int).ravel()
        y_score = np.asarray(y_score, dtype=float).ravel()
        if y_true.shape != y_score.shape:
            raise ValueError(
                f"y_true 和 y_score 形状不一致: {y_true.shape} vs {y_score.shape}"
            )

        config = self.config
        window_length = config.window_length if config is not None else None
        probation_percent = (
            config.probation_percent if config is not None else DEFAULT_PROBATION_PERCENT
        )
        profile = config.profile if config is not None else "standard"

        if profile not in NAB_PROFILES:
            raise ValueError(
                f"Unknown profile {profile!r}; choose from {list(NAB_PROFILES)}"
            )
        profile_weights = NAB_PROFILES[profile]

        n = len(y_true)
        if n == 0:
            return 0.0

        events = events_inclusive(y_true, pos_label=1)
        windows = _compute_windows(events, n, window_length=window_length)
        probation_length = min(
            int(math.floor(probation_percent * n)),
            int(probation_percent * 5000),
        )

        sweep, names = _sweep_scores(n, windows, profile_weights)
        names = _assign_scorable_names(names, windows, probation_length)

        score_curve = _score_curve(sweep, names, y_score, profile_weights)

        candidates = [
            score_curve[i]
            for i in range(len(score_curve))
            if score_curve[i][0] is not None
        ]
        if not candidates:
            return 0.0

        def _best_key(entry):
            t, s, _ = entry
            return (s, -t)

        _, best_score, _ = max(candidates, key=_best_key)
        return float(best_score)