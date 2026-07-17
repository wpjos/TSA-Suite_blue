# 特征选择器算子开发指南 (Feature Selection)

本文档旨在指导开发者基于 `feature/selection` 模块开发新的特征选择器算子。

## 1. 核心设计理念

### 1.1 架构定位

特征选择器（Selector）继承数值算子管线（`NumericOperator`），支持 `DataFrame` 与 `ndarray` 双类型输入。主输出为选择后的特征数据，附加输出 EO 强制包含 `selected_indices`，记录输出列到完整输入列位置的映射。

基类体系：

```
BaseFeatureSelectorMixin[FSEO, FSC, RP]        ← 领域混入（列解析、索引映射）
    ├── BaseFeatureSelector[FSEO, FSC]          ← 不可训练选择器
    ├── UnsupervisedFeatureSelector[FSEO, FSC, FP]  ← 无监督训练选择器
    └── SupervisedFeatureSelector[FSEO, FSC, FP]    ← 有监督训练选择器
```

**注册条件**：选择器类放置在 `tsas.engine.operator.feature.selection` 包下，继承上述基类之一并实现 `name()` 类方法，即可被 CLI 自动发现。

### 1.2 Config 体系

所有选择器的 Config 必须继承 `BaseFeatureSelectorConfig`：

```python
class BaseFeatureSelectorConfig(BaseModel):
    input_columns: list[str] | list[int] | None = None
```

`input_columns` 规则：

- `None`：完整输入的全部列都是候选特征
- `list[str]`：按 DataFrame 列名选择候选列（仅适用于 DataFrame 输入）
- `list[int]`：按完整输入列位置选择候选列（适用于 DataFrame 与 ndarray 输入）
- 不允许 `str` 与 `int` 混用，也不允许重复项

子类在此基础上添加选择器特有的参数（如方差阈值等），建议使用 `Field(ge/le/description="...")` 约束参数。

### 1.3 附加输出（EO）约定

所有选择器都必须返回 EO，且 EO 必须继承 `FeatureSelectorExtraOutput`：

```python
class FeatureSelectorExtraOutput(BaseModel):
    selected_indices: list[int]  # 输出列到完整输入列位置的映射
```

子类可扩展 EO 以附加更多信息（如各候选列的方差值）。非候选列不透传到输出。

---

## 2. 算子开发

编写算子类 Docstring 时，应包含 **Input** 和 **Output** 段，描述输入数据的含义和形状、输出数据的含义和形状。这些信息会被 CLI `show` 命令自动提取并渲染为帮助文档。

### 2.1 新增不可训练选择器

**适用场景**：选择逻辑是确定性的，不需要训练（如按列名直接选择）。

继承 `BaseFeatureSelector[FSEO, FSC]`，实现 `_run_data` 方法：

```python
import numpy as np
import pandas as pd
from tsas.engine.operator.feature.selection.base import (
    BaseFeatureSelector,
    BaseFeatureSelectorConfig,
    FeatureSelectorExtraOutput,
)


class ColumnSelectorConfig(BaseFeatureSelectorConfig):
    """静态列选择器配置，仅需 input_columns"""
    pass


class ColumnSelectorExtraOutput(FeatureSelectorExtraOutput):
    """静态列选择器附加输出"""
    pass


class ColumnSelector(BaseFeatureSelector[ColumnSelectorExtraOutput, ColumnSelectorConfig]):
    """静态按列名或列索引选择特征

    不需要训练，直接将候选列作为最终输出列。

    Input:
        x: 候选特征矩阵，形状 (n_samples, n_features)

    Output:
        选择后的特征矩阵，形状 (n_samples, n_selected)，列由 Config 的 input_columns 指定
    """

    @classmethod
    def name(cls) -> str:
        return "column_selector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x: np.ndarray, params, idx: pd.Index | None = None):
        # 候选列已全部由 _filter_data 筛选好，这里直接保留全部
        local_indices = list(range(x.shape[1]))
        selected_indices = self._to_global_indices(local_indices)
        eo = ColumnSelectorExtraOutput(selected_indices=selected_indices)
        return self._select_columns(x, local_indices, eo)
```

**关键工具方法**（由 `BaseFeatureSelectorMixin` 提供）：

- `_to_global_indices(local_indices)` — 将候选列局部索引转换为完整输入全局索引
- `_select_columns(x, local_indices, eo)` — 根据局部索引生成主输出和 EO

### 2.2 新增可训练选择器

**适用场景**：选择逻辑需要基于训练数据学习（如方差阈值、相关性筛选等）。

继承 `UnsupervisedFeatureSelector[FSEO, FSC, FP]` 或 `SupervisedFeatureSelector[FSEO, FSC, FP]`，实现 `_fit_data` 和 `_run_data`：

```python
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field
from tsas.engine.operator.feature.selection.base import (
    UnsupervisedFeatureSelector,
    BaseFeatureSelectorConfig,
    FeatureSelectorExtraOutput,
)


class VarianceThresholdSelectorConfig(BaseFeatureSelectorConfig):
    """方差阈值选择器配置"""
    threshold: float = Field(default=0.0, ge=0.0, description="方差阈值，保留方差严格大于该阈值的特征")


class VarianceThresholdSelectorExtraOutput(FeatureSelectorExtraOutput):
    """方差阈值选择器附加输出"""
    variances: list[float] = Field(description="训练阶段候选特征方差")


class VarianceThresholdSelector(
    UnsupervisedFeatureSelector[
        VarianceThresholdSelectorExtraOutput,
        VarianceThresholdSelectorConfig,
        None
    ]
):
    """根据训练集方差阈值保留特征

    训练阶段计算候选特征的方差，保留方差严格大于 threshold 的特征列。
    推理阶段直接按训练结果选择列。

    Input:
        x: 候选特征矩阵，形状 (n_samples, n_features)

    Output:
        选择后的特征矩阵，形状 (n_samples, n_selected)，保留方差大于阈值的特征列
    """

    @classmethod
    def name(cls) -> str:
        return "variance_threshold_selector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._variances = None
        self._selected_local_indices = None
        self._selected_global_indices = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._variances = np.var(x, axis=0)
        self._selected_local_indices = [
            idx for idx, v in enumerate(self._variances.tolist())
            if v > self.config.threshold
        ]
        self._selected_global_indices = self._to_global_indices(self._selected_local_indices)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None):
        eo = VarianceThresholdSelectorExtraOutput(
            selected_indices=list(self._selected_global_indices),
            variances=[float(v) for v in self._variances],
        )
        return self._select_columns(x, list(self._selected_local_indices), eo)

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        np.savez(
            path / "variance_state.npz",
            variances=self._variances,
            selected_local=np.array(self._selected_local_indices or [], dtype=int),
            selected_global=np.array(self._selected_global_indices or [], dtype=int),
        )

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        data = np.load(path / "variance_state.npz")
        self._variances = data["variances"]
        self._selected_local_indices = data["selected_local"].astype(int).tolist()
        self._selected_global_indices = data["selected_global"].astype(int).tolist()
        self._fitted = True
```

---

## 3. 关键注意事项

1. **`selected_indices` 正确性**：EO 的 `selected_indices` 必须按输出列顺序记录每个输出列对应完整输入中的原始列位置，使用 `_to_global_indices()` 确保映射正确。
2. **空选择处理**：如果没有任何特征满足选择条件，算子不应抛错，应返回零列数据并记录 `WARNING`，此时 `eo.selected_indices == []`。
3. **`_fitted` 标志**：可训练选择器在 `_load_fit_state` 中**必须**手动设置 `self._fitted = True`。
4. **持久化模式**：推荐覆写 `_save_fit_state(path)` / `_load_fit_state(path)` 钩子方法保存训练状态，并在内部先调用 `super()._save_fit_state(path)` 以确保 MRO 链完整。
5. **DataFrame 输出列名恢复**：`BaseFeatureSelectorMixin._name_output_columns` 会自动根据 `selected_indices` 恢复原始列名，子类通常无需覆写。
