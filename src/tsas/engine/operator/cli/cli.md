# CLI 模块开发指南

本文档面向 CLI 模块的开发者和需要将新算子接入 CLI 的算子开发者，介绍 CLI 的内部架构、算子注册机制、Help 文档自动渲染机制，以及新算子接入 CLI 的通用指南。

## 1. 模块概述

CLI 模块 `tsas.engine.operator.cli` 提供统一的命令行入口，支持特征构造、特征选择、时序异常检测、评价指标和时序预测五个子模块的算子调用。

核心组件：

| 组件       | 文件                          | 职责                                   |
|----------|-----------------------------|--------------------------------------|
| 统一入口     | `__main__.py`               | 命令行解析、子模块分发、编码处理                     |
| 注册中心     | `registry.py`               | 算子自动发现、版本管理、名称查找                     |
| Help 生成器 | `help_generator.py`         | 从算子元信息自动生成结构化 Markdown 帮助文档          |
| 配置加载器    | `config_loader.py`          | JSON/JSON5/YAML 配置文件解析               |
| IO 工具    | `io.py`                     | CSV/TSV 等数据文件读写                      |
| 子模块命令    | `feature_construction.py` 等 | 各子模块的 `list`/`show`/`run`/`fit` 命令实现 |

---

## 2. 算子注册机制

### 2.1 动态包扫描原理

CLI 启动时，各子模块会实例化一个 `OperatorRegistry`，配置要扫描的包路径和基类过滤条件：

```python
from tsas.engine.operator.cli.registry import OperatorRegistry

registry = OperatorRegistry(
    base_class=BaseOperator,  # 基类过滤
    scan_packages=['tsas.engine.operator.detection'],  # 扫描路径
)
registry.discover()  # 执行扫描
```

`discover()` 的扫描流程：

1. 使用 `pkgutil.walk_packages` 递归遍历指定包及其子包下的所有模块
2. 对每个模块，通过 `inspect.getmembers` 提取所有类
3. 过滤条件：必须是 `base_class` 的非抽象子类，且有 `name` 类方法
4. 以 `cls.name()` 返回值为键注册到内部字典

### 2.2 各模块的基类要求

| 子模块                  | 扫描包路径                                       | 基类要求                                                             |
|----------------------|---------------------------------------------|------------------------------------------------------------------|
| detection            | `tsas.engine.operator.detection`            | `BaseOperator` 子类（含 Mixin 组合的 Predictor/Scorer/Decider/Detector） |
| evaluation           | `tsas.engine.operator.evaluation`           | `BaseMetricOperator` 子类                                          |
| feature_construction | `tsas.engine.operator.feature.construction` | `BaseFeatureMixin` 子类                                            |
| feature_selection    | `tsas.engine.operator.feature.selection`    | `BaseFeatureSelectorMixin` 子类                                    |
| forecasting          | `tsas.engine.operator.forecasting`          | `BaseForecaster` 子类                                              |

### 2.3 版本覆盖策略

同名算子（`name()` 返回值相同）按版本号自动去重：

- 高版本始终胜出，低版本被覆盖
- 自动扫描发生覆盖时记录 `debug` 级别日志
- 同名同版本不同类 → 抛出 `ValueError`

### 2.4 依赖缺失时的降级行为

当某个子模块因第三方依赖未安装导致 `ImportError` 时：

- 该模块被跳过，记录 `warning` 日志（包含模块名和失败原因）
- 不影响其他模块的正常扫描和注册
- 如果发现某算子未出现在 CLI `list` 列表中，应先检查日志中是否有相关 warning

---

## 3. 新算子接入 CLI 通用指南

### 3.1 编写 Config

Config 的 Pydantic Field 定义直接决定 CLI Help 参数表和 HPO 搜索空间的内容：

- **`description`**：每个字段都应添加中文描述，CLI Help 参数表的"说明"列直接取自此值
- **`ge`/`le`/`gt`/`lt`**：数值型字段的边界约束，HPO 自动提取为搜索范围；无约束的字段不参与搜索
- **`str, Enum`**：离散选项的首选方式，HPO 自动提取为 `choices` 列表，映射为 Optuna `suggest_categorical`
- **`Literal`**：仅 ≤ 3 个选项且无复用需求时使用，HPO 行为与 `Enum` 一致
- **裸 `str`**：**绝对禁止**使用裸 `str` 配合 `description` 文本说明合法值

### 3.2 编写 Docstring

算子类的 Docstring 会被 CLI `show` 命令提取并渲染为帮助文档。关键约定：

**Input 段**：

- 推荐写 `变量名: 描述` 格式（不写类型，CLI 自动从泛型参数推断并补全类型信息）
- 多变量场景每行一个变量（如 `x_real: 真实值\nx_pred: 预测值`）
- 如果输入类型是 `tuple[T1, T2]`，CLI 会自动拆解为多行

**Output 段**：

- 输出是 `BaseModel` 的算子：推荐写语义/用法说明，不重复字段定义（CLI 会自动渲染字段表）
- 输出是 `ndarray` 等的算子：必填，描述形状和语义

**类级描述**：

Docstring 的第一段（首行或首段）会被提取为算子的"简介"，显示在 `list` 命令的表格中。

### 3.3 验证注册生效

将算子文件放在对应模块包路径下后，运行以下命令验证：

```bash
# 查看算子是否出现在列表中
python -m tsas.engine.operator.cli <模块名> list

# 查看算子的详细帮助文档
python -m tsas.engine.operator.cli <模块名> show <算子名称>
```

如果 `list` 中未出现，检查：

1. 日志中是否有 `ImportError` 相关的 warning
2. 算子类是否继承了对应模块要求的基类
3. `name()` 方法是否正确实现并返回了唯一名称

---

## 4. Help 文档自动渲染机制

`help_generator.py` 从算子类的元信息中自动提取并生成结构化 Markdown 帮助文档。

### 4.1 信息来源

| 信息    | 提取来源                                               |
|-------|----------------------------------------------------|
| 算子名称  | `cls.name()`                                       |
| 功能描述  | `cls.__doc__`（含 Input/Output 结构化段落）                |
| 版本号   | `cls.version()`（点分字符串格式）                           |
| 实例参数表 | `cls._config_type.model_fields`（Pydantic Field 定义） |
| 训练参数表 | `cls._fit_params_type`                             |
| 运行参数表 | `cls._run_params_type`                             |
| 附加输出表 | `cls._eo_type`                                     |
| 基础分类  | `issubclass` 检查（角色/可训练/监督类型/分批推理等）                 |
| 输入类型  | `cls._input_type`（多层泛型追踪自动提取）                      |
| 主输出类型 | `cls._output_type`（含 `T \| tuple[T, EO]` 自动简化）     |

### 4.2 列表模式（`list` 命令）

按分组（管线组件算子、端到端检测器算子、组合管线算子）展示，每组一个 CJK 对齐的 Markdown 表格，包含名称、类型、可训练性和简介四列。

### 4.3 详情模式（`show` 命令）

按以下顺序输出完整帮助文档：

1. 标题 + 功能描述（来自 Docstring）
2. 版本号
3. 基础分类（类型、可训练性、监督类型、是否支持分批推理）
4. 输入段（变量名 + 自动推断类型 + Docstring 描述）
5. 主输出段（标题带类型 + Docstring 描述 + BaseModel 字段表）
6. 附加输出段（EO 字段表）
7. 实例参数表（Config Field 提取，含类型、默认值、值域/候选值、说明）
8. 训练参数表
9. 运行参数表

---

## 5. 关键注意事项

1. **`help` 子命令兼容**：旧版 `help` 子命令仍以别名形式保留（渐进式废弃），推荐使用 `list` 和 `show`。
2. **`--encoding` 全局参数**：解决 Windows 终端中文乱码问题，不指定时自动检测并尝试设置为 UTF-8。
3. **参数表格值域提取**：数值型字段从 `Field(ge/le)` 提取为 `[min, max]` 范围；`Enum`/`Literal` 提取为候选值列表。
4. **BaseModel 字段表渲染**：当输入或主输出类型为 `BaseModel` 子类时，Help 会自动渲染 `**结构**：` 字段表，开发者无需在 Docstring 中重复字段定义。
5. **组合算子配置**：`composite_scorer`/`composite_detector` 的配置文件支持 `operators` 字段嵌套编排多算子管线，每个子算子也是独立的配置字典。
