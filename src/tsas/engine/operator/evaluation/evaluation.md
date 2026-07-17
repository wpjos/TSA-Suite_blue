# 评价指标算子开发指南

本模块实现了**评价指标算子基础类型**（BaseMetricOperator），基于 BaseOperator 扩展，新增 `scores()` 方法供 HPO 单目标/多目标优化统一调用。

本文档旨在指导开发者基于 `base.py` 框架开发新的评价指标算子。

**目录结构**：

```
src/tsas/engine/operator/evaluation/
├── __init__.py             # 包导出（MR, MC, BaseMetricConfig, BaseMetricOperator）
├── base.py                 # 基础类型定义（开发核心）
├── binary_classification.py # 二分类离散标签指标
├── binary_curve.py         # 二分类曲线指标（连续分数输入）
├── multi_classification.py # 多分类指标
├── point_adjust.py         # 点调整指标（PA-F1）
├── self_evaluation.py      # 无标签自评估指标
└── evaluation.md           # 本文档
```

---

## 1. 核心设计理念

### 1.1 架构定位

评价指标算子位于检测/预测管线的末端，负责对算法输出进行量化评估。它是一个**无状态纯函数**——不需要训练能力，不使用 `LearnableOperatorMixin`。

```
输入数据（真实值 + 预测值）
        ↓
BaseMetricOperator._run()
        ↓
指标结果（MR: float 或 BaseModel）
        ↓  （可选）
BaseMetricOperator.scores()  ← HPO 优化目标提取
        ↓
dict[str, float]  ← 供 Optuna 等优化框架使用
```

### 1.2 两种 MR 形态

指标结果类型（MR）支持两种形态：

| MR 形态          | 适用场景                 | `main_scores` 路径 | 示例                                       |
|----------------|----------------------|------------------|------------------------------------------|
| `float`        | 单一标量指标（F1、AUC、变异系数等） | `"_"` 占位符        | `main_scores={"f1": "_"}`                |
| `BaseModel` 子类 | 结构化指标（含多字段的完整评估结果）   | 属性路径（支持点分嵌套）     | `main_scores={"f1": "f1", "far": "far"}` |

> **CLI Help 自动渲染**：当 MR 是 `BaseModel` 子类时，算子的 `_output_type`
> 类属性会自动填充为该类型（由 `BaseOperator.__init_subclass__` 通过多层泛型
> 追踪从 MR 等价于 `BaseOperator` 的 O 位置提取，无需 `BaseMetricOperator`
> 定制）。CLI Help 会渲染 `### 主输出 ({MR 类名})` 标题，并在其后追加
> `**结构**：` 字段表，开发者无需在 docstring 中重复字段定义。
> 详见 `base.md` 的 "5.1.2 主输入/输出类型推断" 小节。

### 1.3 泛型参数

算子使用四个泛型参数 **`[I, MR, MC, RP]`**：

```python
class MyMetricOp(BaseMetricOperator[I, MR, MC, RP]):
    ...
```

| 参数   | 含义             | 约束                                    | 常见填写                                           |
|------|----------------|---------------------------------------|------------------------------------------------|
| `I`  | 输入类型           | 无约束                                   | `tuple[np.ndarray, np.ndarray]` 或 `np.ndarray` |
| `MR` | 指标结果类型         | bound `Union[float, BaseModel]`       | `float` 或 Pydantic `BaseModel` 子类              |
| `MC` | 实例参数类型（Config） | bound `Union[BaseMetricConfig, None]` | `BaseMetricConfig` 子类                          |
| `RP` | 运行参数类型         | 无约束                                   | `None`                                         |

### 1.4 Config 体系

所有评价指标算子的 Config 必须继承 `BaseMetricConfig`：

```python
from tsas.engine.operator.evaluation import BaseMetricConfig


class MyConfig(BaseMetricConfig):
    # 子类特有参数
    positive_label: int = 1
    decimals: int = 6
    
    # 重写 main_scores 默认值
    main_scores: dict[str, str] | None = {"f1": "f1", "far": "far"}
```

**`BaseMetricConfig` 关键字段**：

| 字段            | 类型                       | 默认值    | 说明                                                       |
|---------------|--------------------------|--------|----------------------------------------------------------|
| `main_scores` | `dict[str, str] \| None` | `None` | 主评分路径映射。`None` 时 `scores()` 返回 None；非 None 时按路径从 MR 提取标量 |

**配置实例为 frozen 模式**（`ConfigDict(frozen=True)`），创建后不可修改。

### 1.5 `scores()` 方法与 HPO 集成

`scores()` 方法是评价指标算子的核心创新，用于从完整指标结果中提取 HPO 所需的标量字典：

```python
def scores(self, x, *, params=None, **kwargs) -> dict[str, float] | None:
    if self.config is None or self.config.main_scores is None:
        return None
    result = self.run(x, params=params, **kwargs)
    return self._extract_scores(result, self.config.main_scores)
```

**关键行为**：

- `config` 为 `None` 或 `config.main_scores` 为 `None` → 返回 `None`
- `config.main_scores` 非 `None` → 调用 `run()` 后按映射提取，返回 `dict[str, float]`

**路径提取规则**（`_resolve_path`）：

- `"_"` → 直接返回结果对象本身（适用于 `float` MR）
- `"f1"` → 返回 `obj.f1`（单层属性）
- `"macro.f1"` → 返回 `obj.macro.f1`（点分嵌套属性）

---

## 2. 算子开发

### 2.1 新增简单标量指标算子（MR=float）

**适用场景**：指标结果为单一标量值（如变异系数、均值误差）。

```python
import numpy as np
from tsas.engine.operator.evaluation import BaseMetricConfig, BaseMetricOperator


class CVConfig(BaseMetricConfig):
    """变异系数指标配置"""
    # main_scores 路径为 "_"，因为 MR=float
    main_scores: dict[str, str] | None = {"cv": "_"}


class CVMetricOp(BaseMetricOperator[np.ndarray, float, CVConfig, None]):
    """变异系数指标算子

    Input:
        x: 一维时序数据

    Output:
        变异系数值（标准差 / 均值），经 Sigmoid 映射到 (0, 1)
    """

    @classmethod
    def name(cls) -> str:
        return "cv_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: np.ndarray, *, params) -> float:
        return float(np.std(x) / np.mean(x))
```

**使用方式**：

```python
op = CVMetricOp()
result = op.run(np.array([1.0, 2.0, 3.0]))  # -> 0.4082...
scores = op.scores(np.array([1.0, 2.0, 3.0]))  # -> {"cv": 0.4082...}
```

> 已实现的 `SelfEvaluation` 算子（`self_evaluation.py`）遵循类似模式，MR 为 `float`，`main_scores` 路径使用 `"_"` 占位符。

### 2.2 新增结构化指标算子（MR=BaseModel）

**适用场景**：指标结果包含多个字段（如二分类指标含 TP、FP、TN、FN、F1、FAR 等）。

```python
from pydantic import BaseModel
import numpy as np
from tsas.engine.operator.evaluation import BaseMetricConfig, BaseMetricOperator


class BinaryResult(BaseModel):
    """二分类指标结果"""
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float


class BinaryConfig(BaseMetricConfig):
    """二分类指标配置"""
    positive_label: int = 1
    decimals: int = 6
    # 指定 HPO 优化目标
    main_scores: dict[str, str] | None = {"f1": "f1", "far": "far"}


class BinaryMetricOp(BaseMetricOperator[tuple[np.ndarray, np.ndarray], BinaryResult, BinaryConfig, None]):
    """二分类评价指标算子

    Input:
        y_truth: 真实标签数组
        y_predict: 预测标签数组

    Output:
        BinaryResult 结构化指标，可通过 f1/far/mcc 等属性访问各项指标值，
        也可通过 main_scores 提取 f1/far 用于 HPO 优化
    """

    @classmethod
    def name(cls) -> str:
        return "binary_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: tuple[np.ndarray, np.ndarray], *, params) -> BinaryResult:
        y_truth, y_predict = x
        # ... 计算逻辑 ...
        return BinaryResult(
            tp=tp, fp=fp, tn=tn, fn=fn,
            accuracy=accuracy, precision=precision,
            recall=recall, f1=f1, far=far,
        )
```

**使用方式**：

```python
op = BinaryMetricOp()
result = op.run((y_truth, y_predict))  # -> BinaryResult(...)
scores = op.scores((y_truth, y_predict))  # -> {"f1": 0.85, "far": 0.12}
```

> 已实现的 `BinaryClassificationMetric` 算子（`binary_classification.py`）遵循类似模式，MR 为 `BinaryClassificationResult`，CLI `show` 命令会自动渲染其字段表。

> **其他变体**：当算子无需配置参数时，可将 MC 设为 `None`（此时 `scores()` 始终返回 `None`）；当 MR 包含层级结构时，`main_scores` 支持点分路径提取嵌套属性（如 `"macro.f1"`）。具体写法参见源码中的 `self_evaluation.py` 和 `multi_classification.py`。

---

## 3. HPO 集成

### 3.1 单目标优化

```python
from tsas.engine.operator.evaluation import BaseMetricConfig, BaseMetricOperator


class F1Config(BaseMetricConfig):
    main_scores: dict[str, str] | None = {"f1": "f1"}


class F1Op(BaseMetricOperator[tuple, float, F1Config, None]):

    @classmethod
    def name(cls) -> str:
        return "f1_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return 0.85  # 实际计算

op = F1Op()
scores = op.scores(x)
objective = scores["f1"]  # 供 Optuna maximize 使用
```

### 3.2 多目标优化（Pareto 前沿）

```python
class BinaryConfig(BaseMetricConfig):
    main_scores: dict[str, str] | None = {"f1": "f1", "far": "far"}


class BinaryOp(BaseMetricOperator[tuple, BinaryResult, BinaryConfig, None]):

    @classmethod
    def name(cls) -> str:
        return "binary_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return BinaryResult(f1=0.85, far=0.12, ...)

op = BinaryOp()
scores = op.scores(x)
# Optuna directions=["maximize", "minimize"]
objectives = (scores["f1"], -scores["far"])
```

### 3.3 自定义 main_scores 覆盖默认

`main_scores` 可在实例化时覆盖 Config 的默认值：

```python
# Config 默认提取 f1 + far，但用户只想优化 precision
op = BinaryMetricOp(main_scores={"precision": "precision"})
scores = op.scores(x)  # -> {"precision": 0.75}
```

---

## 4. 继承与多层派生

### 4.1 通过中间抽象类派生

支持通过中间抽象类进行多层继承，Config 类型会自动从具体类的泛型参数中提取：

```python
from typing import TypeVar, Generic
from abc import ABC

from tsas.engine.operator.evaluation import BaseMetricOperator, BaseMetricConfig

C = TypeVar("C", bound=BaseMetricConfig)
RP = TypeVar("RP")


class AbstractBinaryOp(
    BaseMetricOperator[tuple[list, list], BinaryResult, C, RP],
    Generic[C, RP],
    ABC,
):
    """中间抽象类，延迟绑定 Config 和 RP"""
    pass


class ConcreteBinaryOp(AbstractBinaryOp[BinaryConfig, None]):
    """具体实现类"""

    @classmethod
    def name(cls) -> str:
        return "concrete_binary_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return BinaryResult(f1=0.9, far=0.1, ...)


# ConcreteBinaryOp._config_type 自动提取为 BinaryConfig
```

### 4.2 Mixin 组合

评价指标算子支持与普通 Mixin 类组合，不影响 Config 提取：

```python
class LoggingMixin:
    """日志 Mixin"""
    pass


class LoggedOp(
    LoggingMixin,
    BaseMetricOperator[list[float], float, MyConfig, None],
):

    @classmethod
    def name(cls) -> str:
        return "logged_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return float(sum(x))
```

---

## 5. 关键注意事项

### 5.1 Config 类型约束

- Config 必须继承 `BaseMetricConfig`（运行时由 `__init_subclass__` 验证）
- Config 为 `None` 时不验证（`MC=None` 合法）
- 使用非 `BaseMetricConfig` 子类的 Pydantic `BaseModel` 会导致 `TypeError`

### 5.2 `scores()` 返回 None 的场景

以下场景 `scores()` 返回 `None`，HPO 编排层应做相应处理：

1. `MC=None`（算子无 Config）
2. `config.main_scores=None`（未配置提取路径）
3. 实例化时显式传入 `main_scores=None`

此时应直接使用 `run()` 获取完整指标结果。

### 5.3 `_run` 方法签名

子类必须实现 `_run` 方法，签名为：

```python
def _run(self, x, *, params) -> MR:
    ...
```

- `x` 为输入数据（类型由 `I` 泛型参数决定）
- `params` 为运行参数（类型由 `RP` 泛型参数决定，通常为 `None`）
- 返回值为指标结果（类型由 `MR` 泛型参数决定）

### 5.4 `main_scores` 路径的正确性

- `float` MR 只能使用 `"_"` 路径
- `BaseModel` MR 的路径必须与结果类属性对应
- 无效路径会在运行时抛出 `AttributeError`
- 建议在单元测试中覆盖 `scores()` 的正常和异常路径

### 5.5 算子实例化方式

评价指标算子的实例化方式继承自 `BaseOperator`：

```python
# 使用默认 Config（自动实例化）
op = MyMetricOp()

# 覆盖 Config 默认值
op = MyMetricOp(config=MyConfig(main_scores={"f1": "f1"}))

# 传入自定义 Config 实例
op = MyMetricOp(config=MyConfig(main_scores={"f1": "f1"}, positive_label=1))

# 显式指定算子实例标识后缀
op = MyMetricOp(oid="my_metric", config=MyConfig(main_scores={"f1": "f1"}))
```
