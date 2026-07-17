# 时序预测算子开发指南 (Forecasting)

本文档旨在指导开发者基于 `forecasting` 模块开发新的时序预测算子。

## 1. 核心设计理念

### 1.1 架构定位

预测算子位于时序分析管线的前端，负责基于历史数据预测未来值。基类 `BaseForecaster` 通过 `LearnableOperatorMixin + BaseOperator` 多重继承实现，所有预测算子都是可训练的。

```
历史数据 (x, y)
        ↓ fit
BaseForecaster._fit() → _fit_data()
        ↓
训练好的模型
        ↓ run
BaseForecaster._run() → _run_data()
        ↓
预测结果 (pred_len, num_targets)
```

**注册条件**：预测算子类放置在 `tsas.engine.operator.forecasting` 包下，继承 `BaseForecaster` 并实现 `name()` 类方法，即可被 CLI 自动发现。

### 1.2 输入输出约定

预测算子的 `fit` 和 `run` 遵循统一的形状约定：

```
# 训练
fit(x, y):
    x: (timesteps, num_features)   DataFrame 或 ndarray
    y: (timesteps, num_targets)    DataFrame 或 ndarray

# 推理
run(x):
    x: (seq_len, num_features) 或 (batch, seq_len, num_features)
    返回: (pred_len, num_targets) 或 (batch, pred_len, num_targets)
```

子类只需实现纯 ndarray 的 `_fit_data` 和 `_run_data`，基类自动完成类型转换和维度校验。

### 1.3 泛型参数

`BaseForecaster` 使用四个泛型参数 `[EO, C, RP, FP]`：

| 参数   | 含义             | 常见填写                           |
|------|----------------|--------------------------------|
| `EO` | 附加输出类型         | `ForecastExtraOutput` 或 `None` |
| `C`  | 实例参数类型（Config） | `BaseModel` 子类                 |
| `RP` | 运行参数类型         | `None`                         |
| `FP` | 训练参数类型         | `None`                         |

### 1.4 Config 体系

预测算子的 Config 使用标准 Pydantic `BaseModel`（无特殊基类要求）。建议：

- 窗口结构参数（`seq_len`、`pred_len`）使用 `Field(ge=1, le=..., description="...")`
- 模型超参数使用 `Field(ge/le)` 约束搜索范围，便于 HPO 自动搜索
- 离散选项使用 `Literal` 或 `str, Enum`（选项 ≤ 3 个可用 `Literal`，否则用 `Enum`）
- 所有字段添加 `description` 以便 CLI `show` 命令展示参数说明

### 1.5 DataFrame + ndarray 双类型支持

基类自动处理 DataFrame ↔ ndarray 转换：

- **训练输入**：`_validate_fit_input` 校验 x/y 维度一致性，统一转为 ndarray 后传入 `_fit_data`
- **推理输入**：`_validate_run_input` 校验维度（2D 或 3D），转为 ndarray 后传入 `_run_data`
- **推理输出**：`_to_dataframe` 将 ndarray 结果转回 DataFrame（如原始输入为 DataFrame），输出列名由 `_name_output_columns` 决定

---

## 2. 算子开发

编写算子类 Docstring 时，应包含输入输出约定说明（`fit(x, y)` 和 `run(x)` 的形状），这些信息会被 CLI `show` 命令提取。

**典型示例 — LightGBM 预测算子**：

```python
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from tsas.engine.operator.forecasting.base import BaseForecaster, ForecastExtraOutput


class MyForecasterConfig(BaseModel):
    """预测算子配置"""
    seq_len: int = Field(default=96, ge=1, le=4096, description="输入历史窗口长度")
    pred_len: int = Field(default=24, ge=1, le=500, description="预测未来步长")
    # 模型特有超参数...
    n_estimators: int = Field(default=200, ge=1, le=10000, description="提升轮数")


class MyForecaster(BaseForecaster[ForecastExtraOutput, MyForecasterConfig, None, None]):
    """基于 LightGBM 的时序预测算子

    通过窗口展平构造监督学习样本，支持 Direct 和 MIMO 两种多步预测策略。

    输入输出约定::

        fit(x, y):
            x: (timesteps, num_features)  DataFrame / ndarray
            y: (timesteps, num_targets)   DataFrame / ndarray

        run(x):
            x: (seq_len, num_features) 或 (batch, seq_len, num_features)
            返回: (pred_len, num_targets) 或 (batch, pred_len, num_targets)
    """

    @classmethod
    def name(cls) -> str:
        return "my_forecaster"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._models = {}
        self._num_features = None
        self._num_targets = None

    def _fit_data(self, x: np.ndarray, y: np.ndarray, *, params) -> None:
        """核心训练逻辑

        Args:
            x: 训练输入，形状 (timesteps, num_features)
            y: 训练目标，形状 (timesteps, num_targets)
            params: 训练参数（当前为 None）
        """
        self._num_features = x.shape[1]
        self._num_targets = y.shape[1]
        # 1. 从时序数据构造监督学习样本
        # 2. 训练模型
        # 3. 保存模型到 self._models

    def _run_data(self, x: np.ndarray, *, params) -> np.ndarray:
        """核心推理逻辑

        Args:
            x: 推理输入，形状 (seq_len, num_features) 或 (batch, seq_len, num_features)
            params: 运行参数（当前为 None）

        Returns:
            np.ndarray: 预测结果，形状 (pred_len, num_targets) 或 (batch, pred_len, num_targets)
        """
        # 使用 self._models 进行预测
        pass
```

---

## 3. 持久化

预测算子通常需要保存训练好的模型权重。推荐覆写 `_save_fit_state` / `_load_fit_state` 钩子方法：

```python
def _save_fit_state(self, path: Path) -> None:
    """保存模型权重和结构参数"""
    super()._save_fit_state(path)
    # 保存模型文件（如 pickle、joblib、np.savez 等）
    np.savez(
        path / "model_state.npz",
        num_features=self._num_features,
        num_targets=self._num_targets,
    )


def _load_fit_state(self, path: Path) -> None:
    """恢复模型权重和结构参数"""
    super()._load_fit_state(path)
    data = np.load(path / "model_state.npz")
    self._num_features = int(data["num_features"])
    self._num_targets = int(data["num_targets"])
    # 恢复模型文件
    self._fitted = True
```

> **重要**：`_fitted` 状态不会自动恢复，`_load_fit_state` 中**必须**手动设置 `self._fitted = True`。

---

## 4. 关键注意事项

1. **输入维度校验**：基类 `_validate_fit_input` 要求 x 为 2D、y 为 1D 或 2D，且时间步数一致；`_validate_run_input` 要求 x 为 2D 或 3D。子类通常无需重复校验。
2. **`_fitted` 标志**：基类 `_fit` 模板方法在调用 `_fit_data` 后自动设置 `self._fitted = True`，子类的 `_fit_data` 无需手动设置。
3. **输出列名**：默认生成 `forecast_0`、`forecast_1` 等列名。覆写 `_name_output_columns` 可自定义（如沿用目标变量名）。
4. **可选依赖**：若算子依赖第三方库（如 `lightgbm`、`xgboost`），应将 import 放在模块顶层。注册中心在 `ImportError` 时会自动跳过该算子并记录 warning，不影响其他算子的注册。
5. **批量预测**：`run()` 支持 3D 输入 `(batch, seq_len, num_features)`，子类的 `_run_data` 应能正确处理批量维度。
