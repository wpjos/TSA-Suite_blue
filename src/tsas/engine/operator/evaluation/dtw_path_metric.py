# -*- coding: utf-8 -*-

"""
DTW Path 评价指标算子（Dynamic Time Warping + 对齐路径）

动态规划寻找两条序列之间的最优对齐路径，同时返回路径上所有
``(i, j)`` 索引对。支持 Sakoe-Chiba 窗口限制。

核心组件:
    - DtwPathResult: DTW 路径结果（Pydantic BaseModel）
    - DtwPathConfig: 配置类（继承 BaseMetricConfig）
    - DtwPathMetric: DTW 距离 + 路径指标算子

使用示例::

    from tsas.engine.operator.evaluation import DtwPathMetric

    op = DtwPathMetric()
    result = op.run((x, y))
    print(result.distance)
    print(result.path)  # [(0, 0), (1, 1), ...]
"""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'DtwPathResult',
    'DtwPathConfig',
    'DtwPathMetric',
]


class DtwPathResult(BaseModel):
    """DTW 路径结果

    Attributes:
        distance (float): DTW 累积距离
        path (list[tuple[int, int]]): 最优对齐路径索引对列表
    """
    model_config = ConfigDict(frozen=True)

    distance: float = Field(description="DTW 累积距离")
    path: list[tuple[int, int]] = Field(description="最优对齐路径索引对列表")


class DtwPathConfig(BaseMetricConfig):
    """DTW Path 评价指标配置

    Attributes:
        window (int | None): Sakoe-Chiba 窗口半径，``None`` 表示无约束
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"dtw_distance": "distance"}``
    """
    model_config = ConfigDict(frozen=True)

    window: int | None = Field(
        default=None,
        ge=0,
        description="Sakoe-Chiba 窗口半径；None 表示无约束",
    )
    main_scores: dict[str, str] | None = Field(
        default={"dtw_distance": "distance"},
        description="主评分路径映射；float 字段使用对应属性名",
    )


class DtwPathMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        DtwPathResult,
        DtwPathConfig,
        None,
    ],
):
    """DTW Path 评价指标算子

    通过动态规划寻找两条序列之间的最优对齐路径，同时返回
    路径上所有 ``(i, j)`` 索引对。

    Input:
        x: 序列 1，shape=(n,)
        y: 序列 2，shape=(m,)

    Output:
        DtwPathResult: DTW 距离 + 对齐路径

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (x, y)
        MR: DtwPathResult — DTW 距离与路径
        MC: DtwPathConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "dtw_path_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> DtwPathResult:
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

        traceback = np.zeros((n + 1, m + 1), dtype=int)

        for i in range(1, n + 1):
            start_j = max(1, i - window)
            end_j = min(m + 1, i + window + 1)
            for j in range(start_j, end_j):
                d = abs(float(x_arr[i - 1]) - float(y_arr[j - 1]))
                choices = [
                    (cost_matrix[i - 1, j], 0),
                    (cost_matrix[i, j - 1], 1),
                    (cost_matrix[i - 1, j - 1], 2),
                ]
                min_cost, tb = min(choices, key=lambda c: c[0])
                cost_matrix[i, j] = d + min_cost
                traceback[i, j] = tb

        i, j = n, m
        path: list[tuple[int, int]] = []
        while i > 0 or j > 0:
            path.append((i - 1, j - 1))
            tb = traceback[i, j]
            if tb == 0:
                i -= 1
            elif tb == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.reverse()

        return DtwPathResult(
            distance=float(cost_matrix[n, m]),
            path=path,
        )