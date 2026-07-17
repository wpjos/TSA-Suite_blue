# 特征构造算子模块开发指南 (Feature Construction)

本文档旨在指导开发者基于 `feature/construction` 模块开发和使用新的特征构造算子。

## 1. 架构概览

特征构造模块基于统一的算子接口（`BaseOperator` 和 `LearnableOperatorMixin`），采用了**多维度正交组合混入类型（Mixin）+ 模板方法模式**的设计。整体包含两个核心组件：

1. **特征算子基类体系 (`base.py`)**：通过组合"列关系"、"行关系"和"是否可训练"这三个维度，提供了 8 个供开发者直接继承的编排基类。基类自动处理了参数校验、列遍历、滑动窗口切片、填充（Padding）与对齐（Alignment）等底层逻辑。
2. **统一命令行 (`cli/`)**：提供基于包扫描的自动算子注册发现机制和统一命令行调用入口，通过声明式配置批量协调并执行特征算子。

**注册条件**：只要特征算子类放置在 `tsas.engine.operator.feature.construction` 包或其子包下，并正确实现了 `name()` 类方法，CLI 工具会在启动时自动扫描并注册。多算子的批量调用通过 CLI 声明式调度，详见 `cli/README.md`。

---

## 2. 核心概念与基类选择

在开发新的特征算子时，你需要根据算子的计算特性，从以下 8 个编排基类中选择最合适的一个进行继承。

### 2.1 维度解析

* **列关系（Column Strategy）**:
    * **独立单列 (`Independent`)**：表示输出中每列只与输入中的一列相关的语义约定。框架将 `input_columns` 对应的**全部列作为一个完整 ndarray** 传入 `compute`（不会按列拆分逐个调用），列间独立性由 `compute` 实现保证（推荐利用 NumPy 的 `axis` 参数沿列方向进行独立计算）。框架负责根据输入列数和输出列数自动分组命名（输出列数量必须为输入列数量的整数倍）。
    * **多列联合 (`Joint`)**：计算依赖于多列的联合信息（如协方差、距离等）。`compute` 接收多列的 NumPy 数组，输出列名通过 `_name_output_columns` 方法统一命名。
* **行关系（Row Strategy）**:
    * **逐行映射 (`Map`)**：一行输入对应一行输出，行与行之间不产生时序依赖。
    * **滑动窗口 (`Window`)**：计算基于时序滑动窗口。基类自动根据配置的 `window_size`、`padding` 和 `alignment` 进行滑动切窗。
* **是否可训练（Learnability）**:
    * **不可训练 (`BaseFeature`)**：纯计算算子（如求均值、对数、差分等）。
    * **可训练 (`LearnableFeature`)**：需要根据训练数据进行拟合学习（Fit）的算子（如 PCA、Scaler 等）。

### 2.2 八个核心编排基类

开发者可以直接继承以下基类（泛型参数中的 `C` 代表你的 Config 类；`FS` 代表特征状态类型，无状态算子传 `None`）：

1. **不可训练类**（泛型参数仅为 `[C]`）：
    * `IndependentMapFeature[C]`
    * `IndependentWindowFeature[C]`
    * `JointMapFeature[C]`
    * `JointWindowFeature[C]`
2. **可训练类**（泛型参数为 `[C, FS]`）：
    * `LearnableIndependentMapFeature[C, FS]`
    * `LearnableIndependentWindowFeature[C, FS]`
    * `LearnableJointMapFeature[C, FS]`
    * `LearnableJointWindowFeature[C, FS]`

### 2.3 Config 体系

特征的配置类分为两个基类，根据行关系选择：

- **Map 类型**：继承 `BaseFeatureConfig`，基类提供 `input_columns: list[str]` 字段（至少包含一列）
- **Window 类型**：继承 `WindowFeatureConfig`，在 `BaseFeatureConfig` 基础上增加：
    - `window_size: int` — 滑动窗口大小（`ge=1`）
    - `padding: Padding | float | int | None` — 填充模式（默认 `None` 不填充）
    - `alignment: Alignment` — 窗口对齐方式（默认 `Alignment.RIGHT`）

Config 应设置 `frozen=True` 确保不可变。子类特有参数建议使用 `Field(description="...")` 添加描述，这些信息会被 CLI `show` 命令的参数表自动提取。

---

## 3. 算子开发

所有的特征算子都必须实现静态方法 `compute(x: np.ndarray, *, state=None, **params)` 和输出列命名方法。

编写算子类 Docstring 时，应包含 **Input** 和 **Output** 段，描述输入数据的含义和形状、输出数据的含义和形状。这些信息会被 CLI `show` 命令自动提取并渲染为帮助文档。

### 3.1 不可训练算子

**独立映射算子示例 (SquareFeature)**：

```python
import numpy as np
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig, IndependentMapFeature
)


class SquareConfig(BaseFeatureConfig):
    pass  # 仅需 input_columns，无额外参数


class SquareFeature(IndependentMapFeature[SquareConfig]):
    """逐元素平方特征

    对每个输入列独立计算平方值，一列输入产出一列输出。

    输出列名格式: ``{源列名}_square``

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 input_columns 选取

    Output:
        平方变换后的特征矩阵，形状 (n_samples, n_features)，列数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "square_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        # x 是 input_columns 对应列的完整 NumPy 数组（全部行，可能多列）
        # Independent 模式下列间独立性由 compute 自行保证（本例利用广播机制）
        return x ** 2

    def _name_output_column(self, input_col: str, output_val) -> str:
        # 生成标准化的输出列名 "{源列名}_{特征名}"
        return self._make_output_column_name(input_col, "square")
```

### 3.2 可训练算子

继承 `Learnable...` 基类的算子需要额外实现：

1. **`compute` 静态方法**：接收 `state` 参数用于推理
2. **`train` 静态方法**：覆写默认训练逻辑，返回状态对象（Pydantic BaseModel 子类）
3. **`_get_train_params` 方法**（可选）：向 `train` 传递额外参数
4. **`_name_output_columns` 方法**：定义输出列名
5. **`save` / `load` 方法**：持久化训练状态

可训练特征算子的训练流程由基类 `LearnableFeature._fit` 模板方法自动编排：输入校验 → 列筛选 → 数据解包 → 调用 `train` → 保存状态到 `_state` → 标记已训练。开发者无需覆写 `_fit`，只需实现 `train` 静态方法即可。

**多列联合可训练算子示例 (PCAFeature)**：

```python
from typing import Self
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import DataFrameMeta
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig, LearnableJointMapFeature
)


class PCAConfig(BaseFeatureConfig):
    """PCA 降维特征的 Config"""
    n_components: int = Field(ge=1, description="降维目标维度数")


class PCAState(BaseModel):
    """PCA 训练状态"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: np.ndarray
    components: np.ndarray


class PCAFeature(LearnableJointMapFeature[PCAConfig, PCAState]):
    """PCA 降维特征

    基于训练数据学习主成分方向，推理时将输入数据投影到主成分空间。
    多列输入，输出列数由 n_components 决定。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 input_columns 选取

    Output:
        降维后的特征矩阵，形状 (n_samples, n_components)
    """

    @classmethod
    def name(cls) -> str:
        return "pca_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def train(x: np.ndarray, **params) -> PCAState:
        """基于训练数据学习 PCA 状态（由基类自动调用）"""
        n_components = params.get("n_components", 2)
        mean = x.mean(axis=0)
        centered = x - mean
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        components = eigenvectors[:, idx]
        return PCAState(mean=mean, components=components)

    def _get_train_params(self):
        return {"n_components": self.config.n_components}

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        if state is None:
            raise ValueError("PCA 需要先训练")
        centered = x - state.mean
        return centered @ state.components

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: None) -> list[str]:
        n_components = output_data.shape[1] if output_data.ndim > 1 else 1
        return [f"pca_{i}" for i in range(n_components)]

    def save(self, path: str | Path):
        super().save(path)
        path = Path(path)
        if self._state is not None:
            np.save(path / "pca_mean.npy", self._state.mean)
            np.save(path / "pca_components.npy", self._state.components)

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None) -> Self:
        instance = super().load(path, name=name)
        path = Path(path)
        mean_file = path / "pca_mean.npy"
        components_file = path / "pca_components.npy"
        if mean_file.exists() and components_file.exists():
            instance._state = PCAState(
                mean=np.load(mean_file),
                components=np.load(components_file)
            )
            instance._fitted = True
        return instance
```

---

## 4. 关键注意事项

1. **`compute` 必须是静态方法且无副作用**：所有可训练算子的状态通过 `state` 参数传入，不应在 `compute` 内部访问 `self`。
2. **`_name_output_column` vs `_name_output_columns`**：
    - `Independent` 模式：实现 `_name_output_column(input_col, output_val)` 为每个输入列生成一个列名，框架自动按列分组调用。
    - `Joint` 模式或需要自定义多列命名：覆写 `_name_output_columns(output_data, meta, params)` 返回完整列名列表。
    - 默认命名模板：`_make_output_column_name(source_col, feature_name)` 生成 `{源列名}_{特征名}` 格式。
3. **Window 模式的 padding + alignment 组合效果**：
    - `padding=None` 时输出行数减少（`n_samples - window_size + 1`），`alignment` 决定索引截取位置。
    - `padding` 非 `None` 时输出行数与输入一致，`alignment` 决定填充方向（右对齐填充头部，左对齐填充尾部）。
    - 镜像填充（`REFLECT`）和循环填充（`RING`）要求数据长度 > `window_size - 1`。
4. **Independent 模式的输出列数约束**：输出列数必须是输入列数的整数倍，否则框架会抛出 `ValueError`。
5. **参数注入**：通过 `_get_compute_params()` 和 `_get_train_params()` 向 `compute` / `train` 传递 Config 中的参数，保持静态方法的纯粹性。

---

## 5. 持久化

可训练特征算子需要覆写 `save()` / `load()` 以持久化训练状态（如模型权重、统计参数等）：

- **`save()`**：先调用 `super().save(path)` 保存 config 等基类信息，再保存 `_state` 中的自有状态。
- **`load()`**：先调用 `super().load(path, name=name)` 恢复基类信息，再恢复状态并设置 `self._fitted = True`。

> **重要**：`_fitted` 状态不会自动恢复，`load` 中**必须**手动设置 `self._fitted = True`。
