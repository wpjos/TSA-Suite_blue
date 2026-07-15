# -*- coding: utf-8 -*-

"""
DTW Distance 评价指标算子（Dynamic Time Warping）

动态规划寻找两条序列之间的最优对齐路径，返回路径累积距离。
支持 Sakoe-Chiba 窗口限制。

核心组件:
    - DtwDistanceConfig: 配置类（继承 BaseMetricConfig）
    - DtwDistanceMetric: DTW 距离指标算子

使用示例::

    from tsas.engine.operator.evaluation import DtwDistanceMetric

    op = DtwDistanceMetric()
    result = op.run((x, y))
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'DtwDistanceConfig',
    'DtwDistanceMetric',
]


class DtwDistanceConfig(BaseMetricConfig):
    """DTW Distance 评价指标配置

    Attributes:
        window (int | None): Sakoe-Chiba 窗口半径，``None`` 表示无约束
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"dtw_distance": "_"}``
    """
    model_config = ConfigDict(frozen=True)

    window: int | None = Field(
        default=None,
        ge=0,
        description="Sakoe-Chiba 窗口半径；None 表示无约束",
    )
    main_scores: dict[str, str] | None = Field(
        default={"dtw_distance": "_"},
        description="主评分路径映射；float 类型 MR 使用 '_' 占位符",
    )


class DtwDistanceMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        float,
        DtwDistanceConfig,
        None,
    ],
):
    """DTW Distance 评价指标算子

    通过动态规划寻找两条序列之间的最优对齐路径，
    距离 = 路径上点间距离累积的最小值。

    Input:
        x: 序列 1，shape=(n,)
        y: 序列 2，shape=(m,)

    Output:
        float: DTW 距离，非负

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (x, y)
        MR: float — 标量 DTW 距离
        MC: DtwDistanceConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "dtw_distance_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> float:
        x_seq, y_seq = x
        x_arr = np.asarray(x_seq, dtype=float).ravel()
        y_arr = np.asarray(y_seq, dtype=float).ravel()
        if x_arr.size == 0 or y_arr.size == 0:
            raise ValueError("DTW 输入序列不能为空")

        config = self.config
        window = config.window if config is not None else None

        n, m = len(x_arr), len(y_arr)
        cost_matrix = np.full((n + 1, m + 1), float("inf"))
        cost_matrix[0, 0] = 0

        if window is not None:
            window = max(window, abs(n - m))
        else:
            window = max(n, m)

        for i in range(1, n + 1):
            start_j = max(1, i - window)
            end_j = min(m + 1, i + window + 1)
            for j in range(start_j, end_j):
                d = abs(float(x_arr[i - 1]) - float(y_arr[j - 1]))
                cost_matrix[i, j] = d + min(
                    cost_matrix[i - 1, j],
                    cost_matrix[i, j - 1],
                    cost_matrix[i - 1, j - 1],
                )

        return float(cost_matrix[n, m])