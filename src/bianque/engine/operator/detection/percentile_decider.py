# -*- coding: utf-8 -*-
"""
百分位阈值决策器模块

训练阶段学习训练分数的指定百分位数作为阈值，
推理阶段将异常分数与该阈值比较，严格大于阈值则判定为异常。

示例用法::

    from bianque.engine.operator.detection.percentile_decider import PercentileDecider

    # 创建百分位决策器，使用默认第 95 百分位
    decider = PercentileDecider(oid="my_decider")

    # 训练阶段：从训练分数中学习百分位阈值
    train_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
    decider.fit(train_scores)

    # 推理阶段：比较分数 > 阈值，输出异常标签
    test_scores = np.array([0.4, 0.8, 1.0, 1.5])
    labels, eo = decider.run(test_scores)

主要组件:
    - PercentileDeciderConfig: 百分位决策器配置
    - PercentileDeciderExtraOutput: 百分位决策器额外输出
    - PercentileDecider: 百分位阈值决策器
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from bianque.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from bianque.engine.operator.detection.base import BaseDeciderMixin

__all__ = [
    'PercentileDeciderConfig',
    'PercentileDeciderExtraOutput',
    'PercentileDecider',
]


class PercentileDeciderConfig(BaseModel):
    """
    百分位阈值决策器配置

    Attributes:
        percentile: 百分位数，训练分数的此百分位将作为阈值。
            取值范围 (0, 100)，默认 95.0。
    """
    percentile: float = Field(default=95.0, gt=0, lt=100, description="百分位数，训练分数的此百分位将作为阈值")


class PercentileDeciderExtraOutput(BaseModel):
    """
    百分位阈值决策器额外输出

    Attributes:
        threshold: 训练阶段学习到的百分位阈值
    """
    threshold: float
    """学习到的百分位阈值"""


class PercentileDecider(BaseDeciderMixin[None],
                        UnsupervisedNumericOperatorMixin[None],
                        NumericOperator[PercentileDeciderExtraOutput, PercentileDeciderConfig, None]):
    """
    百分位阈值决策器（第 3 层 Decider）

    继承 BaseDeciderMixin、UnsupervisedNumericOperatorMixin 与 NumericOperator，
    用于在异常检测管线中将连续异常分数转换为离散异常标签。

    训练阶段：计算训练分数展平后的指定百分位数作为阈值。
    推理阶段：将分数严格大于阈值（>）的样本判定为异常（label=1），否则判定为正常（label=0）。

    注意：使用"严格大于"比较，即等于阈值的样本判定为正常。

    泛型参数:
        EO: PercentileDeciderExtraOutput，包含学习到的阈值
        C: PercentileDeciderConfig，百分位决策器配置
        RP: None，无运行参数
    """

    @classmethod
    def name(cls) -> str:
        """
        返回算子名称

        Returns:
            str: 固定返回 "percentile_decider"
        """
        return "percentile_decider"

    def __init__(self, *, oid: str | None = None, config: PercentileDeciderConfig | None = None, **kwargs):
        """
        初始化百分位阈值决策器

        Args:
            oid: 算子标识符，默认为 None
            config: 百分位决策器配置，默认为 None（使用默认配置）
            **kwargs: 透传给基类的参数
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._threshold: float | None = None
        """训练阶段学习到的百分位阈值"""

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练阶段：从训练分数中计算百分位阈值

        将训练分数展平为一维数组后，使用 np.percentile 计算指定百分位数作为阈值，
        结果存储在 self._threshold 中。

        Args:
            x: 训练分数数组，形状任意（内部会展平为一维）
            params: None，无训练参数
        """
        # 计算百分位阈值
        self._threshold = float(np.percentile(x.ravel(), self.config.percentile))

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray | tuple[np.ndarray, PercentileDeciderExtraOutput]:
        """
        推理阶段：比较分数与阈值，输出异常标签

        将输入异常分数严格大于阈值（>）的样本判定为异常（label=1），
        否则判定为正常（label=0）。输出为一维整数数组及包含阈值的额外输出。

        Args:
            x: 异常分数数组，形状 (n_samples,) 或 (n_samples, 1)
            params: None，无运行参数

        Returns:
            tuple[np.ndarray, PercentileDeciderExtraOutput]:
                - labels: 标签数组，形状 (n_samples,)，1=异常 / 0=正常
                - eo: 额外输出，包含学习到的百分位阈值
        """
        # 严格大于阈值判定为异常
        labels = (x > self._threshold).astype(int).ravel()
        eo = PercentileDeciderExtraOutput(threshold=self._threshold)
        return labels, eo
