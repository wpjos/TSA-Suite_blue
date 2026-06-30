# -*- coding: utf-8 -*-

"""
时序预测算子包

提供面向工业时序预测的算子实现，遵循与异常检测算子一致的架构风格：

- ``BaseForecaster``: 预测器基类，支持 3-D 时间序列窗口输入输出
- ``ITransformerForecaster``: 基于 iTransformer + KAN + 残差预测的预测算子
- ``ForecastingMetrics``: 时序预测评价指标算子（位于 ``tsas.engine.operator.evaluation``）
"""

from tsas.engine.operator.forecasting.base import BaseForecaster, ForecastExtraOutput

try:
    from tsas.engine.operator.forecasting.itransformer import (
        ITransformerForecaster,
        ITransformerForecasterConfig,
    )
except ImportError:  # pragma: no cover
    ITransformerForecaster = None  # type: ignore
    ITransformerForecasterConfig = None  # type: ignore

try:
    from tsas.engine.operator.forecasting.lightgbm import (
        LightGBMForecaster,
        LightGBMForecasterConfig,
    )
except ImportError:  # pragma: no cover
    LightGBMForecaster = None  # type: ignore
    LightGBMForecasterConfig = None  # type: ignore

try:
    from tsas.engine.operator.forecasting.lightgbm_mimo import (
        LightGBMMIMOForecaster,
        LightGBMMIMOForecasterConfig,
    )
except ImportError:  # pragma: no cover
    LightGBMMIMOForecaster = None  # type: ignore
    LightGBMMIMOForecasterConfig = None  # type: ignore

try:
    from tsas.engine.operator.forecasting.xgboost import (
        XGBoostForecaster,
        XGBoostForecasterConfig,
    )
except ImportError:  # pragma: no cover
    XGBoostForecaster = None  # type: ignore
    XGBoostForecasterConfig = None  # type: ignore

try:
    from tsas.engine.operator.forecasting.xgboost_mimo import (
        XGBoostMIMOForecaster,
        XGBoostMIMOForecasterConfig,
    )
except ImportError:  # pragma: no cover
    XGBoostMIMOForecaster = None  # type: ignore
    XGBoostMIMOForecasterConfig = None  # type: ignore

__all__ = [
    'BaseForecaster',
    'ForecastExtraOutput',
    'ITransformerForecaster',
    'ITransformerForecasterConfig',
    'LightGBMForecaster',
    'LightGBMForecasterConfig',
    'LightGBMMIMOForecaster',
    'LightGBMMIMOForecasterConfig',
    'XGBoostForecaster',
    'XGBoostForecasterConfig',
    'XGBoostMIMOForecaster',
    'XGBoostMIMOForecasterConfig',
]
