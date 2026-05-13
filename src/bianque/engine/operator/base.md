# 算子基类开发指南

本文档旨在指导开发者如何基于 `base.py` 模块中的算子基础类型开发新的算子（Operator）。模块提供三个层次的基类：

- **BaseOperator** — 所有算子的基类，提供参数管理、唯一标识、持久化接口和 `run` 模板方法
- **LearnableOperatorMixin** — 可训练能力混入，与 BaseOperator 多重继承后获得 `fit` 训练能力
- **NumericOperator** — 单输入数值算子基类，继承 BaseOperator，面向 `pd.DataFrame` / `np.ndarray` 数据提供细粒度模板方法管线
- **BiNumericOperator** — 双输入数值算子基类（比较器），接收 `tuple[x_real, x_pred]`，度量预测值与真实值之间的关系
- **SupervisedNumericOperatorMixin** — 有监督数值训练混入，训练时接收 (x, y)
- **UnsupervisedNumericOperatorMixin** — 无监督数值训练混入，训练时仅接收 x
- **NumericBatchRunMixin** — 数值算子批量推理混入，提供分批生成器推理能力

算子基类采用 **三类参数分离原则** 和 **Pydantic 参数验证机制**，提供了高度类型安全、易于扩展的开发体验。

三者之间的继承关系如下：

```
BaseOperator[I, O, C, RP]                              # 通用算子基类
    ├── NumericOperator[EO, C, RP]                     # 单输入数值算子（自动绑定 I=O=NumericData）
    └── BiNumericOperator[EO, C, RP]                   # 双输入数值算子（比较器基类）

LearnableOperatorMixin[I, T, FP]                       # 可训练能力混入（需与 BaseOperator 多重继承）
    ├── SupervisedNumericOperatorMixin[FP]             # 有监督数值训练混入
    └── UnsupervisedNumericOperatorMixin[FP]           # 无监督数值训练混入
```

## 1. 核心设计理念

### 1.1 三类参数分离

算子在生命周期的不同阶段接收不同类型的参数，它们被严格分离：

- **类型1：实例参数（Config）**：通过算子 `__init__` 传入。定义算子的固有属性（如超参数、网络结构参数等）。创建后即不可变（**强烈建议在 Pydantic 模型中设置 `model_config = ConfigDict(frozen=True)` 以确保其不可变性**）。
- **类型2：训练参数（FitParams）**：通过算子 `fit` 传入（仅可训练算子）。定义单次训练过程的参数（如 epochs, batch_size），每次调用之间独立不影响。
- **类型3：运行/推理参数（RunParams）**：通过算子 `run` 传入。定义单次执行或推理的参数（如是否返回详细得分），每次调用之间独立不影响。

### 1.2 Pydantic 强类型验证

所有三类参数均通过 Pydantic `BaseModel` 定义。基类会自动在运行时进行类型提取和参数验证，开发者在实现核心逻辑时，拿到的始终是**验证后的强类型参数对象**，无需在内部进行繁琐的类型或合法性检查。

### 1.3 模板方法模式

基类对外暴露公开的 `run` 和 `fit` 方法，这些方法内部自动完成前置校验和参数验证，随后调用子类需要实现的 `_run` 和 `_fit` 模板抽象方法。其中：

- `run` 模板方法内部流程：参数验证 → `_can_run()` 前置校验 → `_run` 核心逻辑 → 记录参数
- `fit` 模板方法内部流程：增训许可检查 → 参数验证 → `_fit` 核心逻辑 → 记录参数

### 1.4 类继承体系

模块提供三个核心类，形成两套独立的继承路线：

1. **通用算子路线**（直接继承 `BaseOperator`）：
    - 适用于输入/输出为任意类型（list、dict、str 等）的算子
    - 子类只需实现 `_run` 方法

2. **数值算子路线**（继承 `NumericOperator` 或 `BiNumericOperator`）：
    - `NumericOperator[EO, C, RP]` — 单输入数值算子，`BaseOperator` 的特化子类
    - `BiNumericOperator[EO, C, RP]` — 双输入数值算子（比较器），接收 `tuple[x_real, x_pred]`
    - 自动处理 `pd.DataFrame` ↔ `np.ndarray` 的转换
    - 输入输出类型固定为 `NumericData`（即 `pd.DataFrame` 或数值 `np.ndarray`）
    - 子类需实现 `_run_data`（纯 ndarray 计算，签名包含 `idx: pd.Index | None = None`）和 `_name_output_columns`（输出列名）
    - 提供 6 步模板方法管线：`_validate_input → _filter_data → _unwrap_data → _adjust_data → _run_data → _validate_and_wrap_output`
    - **附加输出（EO）机制**：泛型参数 `EO` 非 `None` 时，`_run_data` 返回 `tuple[np.ndarray, EO]`，`run()` 返回 `tuple[NumericData, EO]`；可通过 `has_extra_output()` 预判返回形态

3. **可训练能力混入**（`LearnableOperatorMixin`）：
    - 不是独立基类，而是一个 Mixin，通过多重继承为任意 BaseOperator 子类添加 `fit` 训练能力
    - 可以与 `BaseOperator` 或 `NumericOperator` 组合使用
    - 通过覆写 `_can_run()` 钩子自动拦截未训练时的 `run` 调用

4. **数值训练混入**（`SupervisedNumericOperatorMixin` / `UnsupervisedNumericOperatorMixin`）：
    - 不是独立基类，而是 Mixin，通过多重继承为 NumericOperator 子类添加训练能力
    - `SupervisedNumericOperatorMixin[FP]`：训练时接收 `(x, y)`，子类实现 `_fit_data(x, y, params)`
    - `UnsupervisedNumericOperatorMixin[FP]`：训练时仅接收 `x`，子类实现 `_fit_data(x, params)`
    - **自动设置 `_fitted`**：模板方法在调用 `_fit_data` 后自动将 `_fitted` 置为 `True`，子类无需手动设置

此外，模块还提供以下辅助类型：

- **DataFrameMeta** — DataFrame 元信息快照（列名、数据类型、索引），用于数值算子内部的 ndarray → DataFrame 反向转换
- **ArrayN** — 数值 ndarray 类型别名（约束 dtype 为 integer 或 floating）
- **NumericData** — `pd.DataFrame | ArrayN` 联合类型别名

### 1.5 `_can_run()` 前置校验机制

`BaseOperator` 提供了 `_can_run()` 钩子方法（默认不做任何校验），由 `run` 模板方法在参数验证之后、调用 `_run` 之前自动调用。子类可以覆写此方法以添加推理前置条件检查。

`LearnableOperatorMixin` 正是通过覆写 `_can_run()` 来检查训练状态：未训练时抛出 `RuntimeError("训练尚未完成，无法执行推理")`。因此可训练算子在训练完成前调用 `run` 会被自动拦截。

---

## 2. 开发无需训练的算子 (BaseOperator)

如果你的算子不需要训练过程（如基础统计计算、规则匹配等），只需要继承 `BaseOperator` 并实现 `_run` 方法。

### 2.1 定义参数模型

根据需要，定义 Config 和 RunParams 模型。如果不需要某类参数，可以使用 `None` 作为泛型参数。

```python
from pydantic import BaseModel, ConfigDict, Field


class ThresholdConfig(BaseModel):
    """实例参数（Config）"""
    # 强烈建议：设置 frozen=True 确保实例参数不可变
    model_config = ConfigDict(frozen=True)

    threshold: float = Field(default=0.5, gt=0.0)


class ThresholdRunParams(BaseModel):
    """运行参数（RunParams）"""
    return_raw_score: bool = False
```

### 2.2 继承与实现

继承 `BaseOperator[I, O, C, RP]`，其中泛型参数依次为：

- `I` — 输入数据类型
- `O` — 输出数据类型
- `C` — Config 类型（`None` 表示无实例参数）
- `RP` — RunParams 类型（`None` 表示无运行参数）

```python
from bianque.engine.operator.base import BaseOperator


# 泛型参数依次为：输入类型, 输出类型, Config类型, RunParams类型
class ThresholdOperator(BaseOperator[list[float], list[bool], ThresholdConfig, ThresholdRunParams]):

    def _run(self, x: list[float], *, params: ThresholdRunParams | None) -> list[bool]:
        """
        核心执行逻辑
        x: 输入数据
        params: 验证后的运行参数实例（keyword-only，可能为 None）
        """
        # 通过 self.config 获取实例参数
        threshold = self.config.threshold

        # 处理核心逻辑
        results = [val > threshold for val in x]

        # 结合运行参数处理
        if params and params.return_raw_score:
            # 执行额外的逻辑（比如附加原始得分）
            pass

        return results
```

> **注意**：`_run` 的 `params` 参数是 **keyword-only** 的（前方有 `*` 标记），确保调用意图明确。

### 2.3 使用算子

```python
# 实例化（自动触发 ThresholdConfig 验证）
op = ThresholdOperator(threshold=0.8)

# 运行（自动触发 ThresholdRunParams 验证）
result = op.run([0.1, 0.5, 0.9], return_raw_score=True)
```

---

## 3. 开发可训练算子 (LearnableOperatorMixin + BaseOperator)

如果待开发的算子需要基于数据进行训练（如机器学习模型），请使用 **多重继承** 方式：将 `LearnableOperatorMixin` 与 `BaseOperator` 组合，并实现 `_fit` 和 `_run`。

### 3.1 多重继承模式

可训练算子通过 `LearnableOperatorMixin[I, T, FP]` + `BaseOperator[I, O, C, RP]` 多重继承实现：

- `LearnableOperatorMixin` 提供训练能力（`fit` 模板方法、训练状态管理、训练参数持久化）
- `BaseOperator` 提供推理能力（`run` 模板方法、参数管理、持久化）

两者通过 MRO（方法解析顺序）自动协作，**通常不需要覆写 `run`、`save`、`load`**。

```python
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    # 强烈建议：设置 frozen=True 确保实例参数不可变
    model_config = ConfigDict(frozen=True)

    dim: int = Field(gt=0)


class ModelFitParams(BaseModel):
    epochs: int = Field(default=10, gt=0)


class ModelRunParams(BaseModel):
    verbose: bool = False
```

### 3.2 继承与实现

继承时，`LearnableOperatorMixin` 放在 `BaseOperator` **前面**（确保 MRO 中 Mixin 优先）。
而且需要注意共有 **7个** 泛型参数需要设置，其中两个类各自的第1个泛型参数应该 **完全相同**。

```python
from typing import Self
from bianque.engine.operator.base import BaseOperator, LearnableOperatorMixin


class MyPredictor(
    LearnableOperatorMixin[list[float], list[float], ModelFitParams],
    BaseOperator[list[float], list[float], ModelConfig, ModelRunParams]
):
    """泛型参数说明:
    - LearnableOperatorMixin[I, T, FP]: I=输入类型, T=训练目标类型, FP=训练参数类型
    - BaseOperator[I, O, C, RP]: I=输入类型, O=输出类型, C=实例参数类型, RP=运行参数类型
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = None

    def _fit(self, x: list[float], y: list[float], *, params: ModelFitParams | None) -> None:
        """
        核心训练逻辑
        x: 输入训练数据
        y: 训练目标数据
        params: 验证后的训练参数实例（keyword-only，可能为 None）
        """
        epochs = params.epochs if params else 10
        dim = self.config.dim

        print(f"Training for {epochs} epochs with dim {dim}...")
        # 模拟训练逻辑...

        # 【重要】训练完成后，必须将 _fitted 标志置为 True
        # 否则后续调用 run 时会抛出 RuntimeError("训练尚未完成，无法执行推理")
        self._fitted = True

    def _run(self, x: list[float], *, params: ModelRunParams | None) -> list[float]:
        """
        核心推理逻辑
        此时基类已通过 _can_run() 自动检查训练状态，无需手动校验
        """
        verbose = params.verbose if params else False
        if verbose:
            print("Running inference...")
        return [val * self.config.dim for val in x]
```

> **关键注意**：`_fit` 签名中有 **`y` 参数**（训练目标数据），且 `params` 是 keyword-only 的。

### 3.3 使用可训练算子

```python
# 实例化
op = MyPredictor(dim=128)

# 训练（自动触发 ModelFitParams 验证）
op.fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], epochs=20)

# 推理（自动触发 _can_run 训练状态检查 + ModelRunParams 验证）
result = op.run([1.5, 2.5], verbose=True)

# 未训练时直接调用 run 会抛出 RuntimeError
op2 = MyPredictor(dim=64)
# op2.run([1.0])  # → RuntimeError: 训练尚未完成，无法执行推理
```

### 3.4 增量训练（可选）

默认情况下，可训练算子训练完成后不支持再次调用 `fit`，否则会抛出 `RuntimeError("训练已完成，且该算子不支持增训")`。如果你的模型支持增量训练，请覆写 `can_additional_fit` 属性：

```python
@property
def can_additional_fit(self) -> bool:
    return True
```

---

## 4. 开发数值算子 (NumericOperator)

如果算子需要处理 `pd.DataFrame` 或 `np.ndarray` 类型的数值数据，可以继承 `NumericOperator`。
它提供了细粒度的模板方法管线，自动完成 DataFrame↔ndarray 转换，子类只需关注纯 ndarray 计算。

### 4.1 管线流程

`NumericOperator._run` 内部按以下顺序编排：

```
_validate_input → _filter_data → _unwrap_data → _adjust_data (带 idx)
→ _run_data (带 idx) → _wrap_data（含 _adjust_index + _name_output_columns）
```

其中 `_run_data`（核心计算）和 `_name_output_columns`（输出列名）是**必须实现**的抽象方法。其他步骤均有默认实现，可按需覆写。

### 4.2 继承与实现

```python
import numpy as np
import pandas as pd
from pydantic import BaseModel
from bianque.engine.operator.base import NumericOperator


class ScaleConfig(BaseModel):
    scale: float = 1.0


class ScaleOperator(NumericOperator[None, ScaleConfig, None]):
    """将输入数值乘以 scale 的简单示例算子"""

    def _run_data(self, x: np.ndarray, params, idx: pd.Index | None = None) -> np.ndarray:
        """核心计算：纯 ndarray 乘法"""
        scale = self._resolve_param(params, 'scale', default=1.0)
        return x * scale

    def _name_output_columns(self, output_data, meta, params) -> list[str]:
        """确定输出列名（仅 DataFrame 输入时调用）"""
        if meta is not None:
            return [f"scaled_{c}" for c in meta.column_names]
        return [f"col_{i}" for i in range(output_data.shape[1])]
```

### 4.3 使用数值算子

```python
op = ScaleOperator(scale=2.0)

# DataFrame 输入 → DataFrame 输出（自动保留索引和列名信息）
df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
result = op.run(df)
# result = pd.DataFrame({'scaled_a': [2.0, 4.0], 'scaled_b': [6.0, 8.0]}, index=df.index)

# ndarray 输入 → ndarray 输出
arr = np.array([[1.0, 2.0]])
result = op.run(arr)
# result = np.array([[2.0, 4.0]])
```

### 4.4 可覆写的模板方法

| 方法                          | 默认行为   | 典型覆写场景              |
|-----------------------------|--------|---------------------|
| `_validate_dataframe_input` | 无操作    | 检查空 DataFrame、列名合法性 |
| `_validate_ndarray_input`   | 无操作    | 检查维度、dtype 约束       |
| `_filter_data`              | 原样返回   | 列选择、列顺序调整           |
| `_adjust_data`              | 原样返回   | 标准化、归一化等预处理         |
| `_adjust_index`             | 沿用输入索引 | 截断、扩展输出行索引          |
| `_validate_and_wrap_output` | 根据 `_eo_type` 校验输出并打包 | 通常不需覆写，框架自动处理 EO 校验和 DataFrame 回包 |

### 4.5 开发双输入数值算子 (BiNumericOperator)

双输入数值算子用于比较两组数据（如预测值与真实值），输入为 `tuple[x_real, x_pred]`。

```python
import numpy as np
from pydantic import BaseModel
from bianque.engine.operator.base import BiNumericOperator


class ResidualConfig(BaseModel):
    metric: str = "mse"


class ResidualScorer(BiNumericOperator[None, ResidualConfig, None]):
    """残差评分器 — 比较预测值与真实值"""

    def _run_data(self, x_real: np.ndarray, x_pred: np.ndarray, params, idx: pd.Index | None = None) -> np.ndarray:
        diff = x_real - x_pred
        if self.config.metric == "mse":
            return np.mean(diff ** 2, axis=1, keepdims=True)
        return np.abs(diff)

    def _name_output_columns(self, output_data, meta, params) -> list[str]:
        return ["score"]
```

使用方式：

```python
scorer = ResidualScorer(metric="mse")
result = scorer.run((x_real, x_pred))
```

---

## 5. 高级特性

### 5.1 参数优先级解析 (`_resolve_param`)

有时候，在开发算子时，你可能希望运行参数可以覆盖实例参数（例如默认 threshold 设在 Config 中，但某次 Run 想用不同的 threshold）。这时可以使用基类提供的 `_resolve_param`，它的解析优先级为：**运行参数 → 实例参数 → 默认值**。

值为 `None` 被视为"未指定"，将回退到下一优先级。

```python
def _run(self, x: list[float], *, params):
    # 解析优先级：params.threshold -> self.config.threshold -> 0.5
    threshold = self._resolve_param(params, "threshold", default=0.5)
```

### 5.1.1 附加输出（EO）与 `has_extra_output()`

数值算子（`NumericOperator` / `BiNumericOperator`）支持通过泛型参数 `EO` 声明附加输出：

- `EO = None`（默认）：`_run_data` 返回 `np.ndarray`，`run()` 返回 `NumericData`
- `EO = SomeModel(BaseModel)`：`_run_data` 返回 `tuple[np.ndarray, EO]`，`run()` 返回 `tuple[NumericData, EO]`

调用方可在实例化前通过 `has_extra_output()` 预判返回形态：

```python
if MyScorer.has_extra_output():
    result, eo = scorer.run(data)
else:
    result = scorer.run(data)
```

`_eo_type` 由 `__init_subclass__` 自动从泛型参数中提取，开发者无需手动设置。

### 5.2 持久化 (Save / Load)

基类已经实现了标准的 `save` 和 `load` 方法：

- **BaseOperator**：自动保存/恢复 `config`（`config.json`）和 `last_run_params`（`last_run_params.json`）
- **LearnableOperatorMixin**：通过 MRO super() 链，额外保存/恢复 `last_fit_params`（`last_fit_params.json`）

> **重要**：`_fitted` 训练状态标志**不会**被自动恢复，开发者需要在 `load` 中根据实际情况手动设置。

如果你的算子有额外状态（例如保存的模型权重、训练好的判定边界等），需要覆写 `save` 和 `load`：

```python
from pathlib import Path
from typing import Self


class MyPredictor(
    LearnableOperatorMixin[list[float], list[float], ModelFitParams],
    BaseOperator[list[float], list[float], ModelConfig, ModelRunParams]
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = None

    def save(self, path: str | Path) -> None:
        # 1. 必须先调用父类保存基础参数
        # super().save() 会通过 MRO 链依次调用 LearnableOperatorMixin.save → BaseOperator.save
        super().save(path)

        # 2. 保存算子自有状态
        target_dir = Path(path)
        if self.weights is not None:
            (target_dir / "weights.bin").write_bytes(self.weights)

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        # 1. 必须先调用父类恢复基础参数并创建算子实例
        instance = super().load(path, oid=oid)

        # 2. 恢复算子自有状态
        target_dir = Path(path)
        weights_file = target_dir / "weights.bin"
        if weights_file.exists():
            instance.weights = weights_file.read_bytes()
            # 手动恢复训练状态标志
            instance._fitted = True

        return instance
```

### 5.3 Pydantic 直接实例化与复用

除了支持通过 `**kwargs` 自动完成实例化之外，还支持直接传入已构造好的 Pydantic 实例。这对于参数复用或从外部 JSON 文件加载参数后初始化非常有用：

```python
# 提前构造验证好的实例
cfg = ModelConfig(dim=128)

# 直接传入 config 实例
predictor = MyPredictor(config=cfg)

run_p = ModelRunParams(verbose=True)
# 同样支持直接传入 params 实例
predictor.run(data, params=run_p)

# fit 也支持直接传入 FitParams 实例
fit_p = ModelFitParams(epochs=20)
predictor.fit(x_data, y_data, params=fit_p)
```

### 5.4 唯一标识管理

每个算子实例都有一个全局唯一标识（oid），通过 `oid` 属性访问。格式为 `{name()}${oid_suffix}`，其中 `name()` 是类方法返回算子类型名，`oid_suffix` 是实例标识后缀：

- 不指定 `oid` 时自动生成，格式为 `ClassName$RANDOM_ID`（如 `MyPredictor$A3F8B2C1`）
- 可通过 `__init__(oid="my-model")` 自定义

### 5.5 无参数算子

如果算子不需要任何参数，可以将所有泛型参数设为 `None`：

```python
class SimpleOp(BaseOperator[str, str, None, None]):
    def _run(self, x: str, *, params) -> str:
        return x.upper()


op = SimpleOp()  # 无需传入任何参数
```

---

## 6. 最佳实践

1. **拥抱 Pydantic 约束**：在定义 Config / Params 模型时，尽量利用 `Field(gt=0, lt=1)` 等约束条件，将防御性编程交给 Pydantic。
2. **保持 Config 不可变**：实例参数一旦在 `__init__` 中确定，就不应在后续代码中修改。如有动态调整需要，应通过 RunParams 和 FitParams 传递。
3. **正确设置 `_fitted`**：
    - 直接继承 `LearnableOperatorMixin` 的算子，在 `_fit` 方法中训练成功后**必须**设置 `self._fitted = True`
    - 使用 `SupervisedNumericOperatorMixin` / `UnsupervisedNumericOperatorMixin` 的算子，模板方法会**自动**设置 `_fitted = True`，子类无需手动设置
4. **Mixin 在前，Base 在后**：多重继承时 `LearnableOperatorMixin` 必须放在 `BaseOperator` **前面**，以确保 MRO 中 Mixin 的 `_can_run` 能正确覆写 BaseOperator 的默认实现。
5. **使用 `super().save()` / `super().load()`**：覆写持久化方法时，务必先调用 `super()`，不要直接指名调用某个基类的方法，以确保 MRO 链完整执行。
6. **`_run` 和 `_fit` 的 `params` 是 keyword-only**：签名中 `params` 前方有 `*` 标记，不可作为位置参数传入。
