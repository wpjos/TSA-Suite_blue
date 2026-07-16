# 计划 C：bqlib ops -> TSA-Suite

> 迁移 bqlib `ops/` 下 18 个原子算子到 TSA-Suite `engine/operator/feature/construction/`。
> **策略：源码融合，TSA-Suite 不依赖 bqlib，不保留 bqlib 结构**。
> 创建日期：2026-07-16。

## 完成记录（2026-07-16）

全部 19 个算子已合入（含 envelope_feature 共 19 个；原文档 C1 写 18 是漏数），

| 组 | 算子数 | 状态 | 文件 |
|---|---|---|---|
| Helper | 1 | ✅ 完成 | `_array_helpers.py`（新增） |
| 第 1 组 skill 依赖 | 3 | ✅ 完成 | `spectral_transform_feature.py`（rfft / rfftfreq / envelope） |
| 第 2 组 基础统计 | 5 | ✅ 完成 | `basic_stat_feature.py`（max / min / mean / std / sum） |
| 第 3 组 基础变换 | 4 | ✅ 完成 | `basic_transform_feature.py`（abs / sqrt / diff / uni） |
| 第 4 组 频域扩展 | 3 | ✅ 完成 | `spectral_transform_feature.py`（fft / fftfreq / hps） |
| 第 5 组 心理声学 | 2 | ✅ 完成 | `psychoacoustic_feature.py`（bark_spectrum / specific_loudness） |
| 第 6 组 其他 | 2 | ✅ 完成 | `other_feature.py`（slope / ehr） |
| **合计** | **19** | **✅ 全部完成** | 6 个新文件 |

**完成度**：

- ✅ 19 个算子均按 TSA-Suite `IndependentMapFeature` 规范重写，源码融合、不依赖 bqlib
- ✅ TSA-Suite 算子总数 40 → **59**（反超 bqlib 的 43 个 ops+features）
- ✅ 9 个 skill 全部可用（feature_skills.md 中所有依赖项已合入）
- ✅ 测试覆盖：每个算子 3-6 个测试用例（手工 oracle / 边界 / config 校验 / 多列），共 94 个新测试
- ✅ Ruff check：6 个新源文件 + 1 个新测试文件全部通过
- ✅ CLI 集成：通过 `feature_construction help` 可查到所有 19 个算子

**关键发现（与计划略有差异）**：

1. **envelope_feature 数量**：原计划 C1 写 18，C11 写 19。实际合入 19 个（含 envelope）。TSA-Suite 已有 `envelope_rms_feature`，但 `envelope_analysis` skill 需要纯包络信号（算包络峭度），所以新增 envelope_feature 是必要的。
2. **rfft/rfftfreq/fft/fftfreq 输入**：bqlib 用 N（信号长度）作为入参，TSA-Suite 按列处理 → 改为对每列 signal 段计算，长度由输入决定。
3. **YAML 格式**：`input_columns` 应放在 `config:` 内（不在顶层），与 `feature_construction run` CLI 的 `instantiate_operator` 行为一致。原计划 C9 模板示例需修正（见下）。
4. **TSA-Suite 现有 helpers 已复用**：`psychoacoustic_feature` 直接复用 `signal_feature.py` 中的 `_bark_spectrum_from_signal` / `_power_to_specific_loudness`，避免重复实现。

**YAML 格式修正**（C9 示例应改为）：

```yaml
operators:
  - name: max_feature
    alias: max_val
    config:
      input_columns: [sensor_1, sensor_2]
  - name: rfft_feature
    alias: spectrum
    config:
      input_columns: [signal]
      output: magnitude
  - name: rfftfreq_feature
    alias: freqs
    config:
      input_columns: [signal]
      sample_rate: 10240.0
```

`input_columns` 必须嵌套在 `config` 内（CLI `instantiate_operator` 只读 `op_spec.config`）。

## C1. 迁移目标

把 bqlib `ops/` 下 **TSA-Suite 缺失的 18 个原子算子**融合到 TSA-Suite `engine/operator/feature/construction/`，使其通过 `python -m tsas.engine.operator.cli feature_construction run` 调用。

**硬约束**：
- TSA-Suite 不能引入 bqlib 依赖
- **所有代码按 TSA-Suite 规范融合，不保留 bqlib 命名/结构**（不搬 `_numpy.py` / `_pure.py` / `_meta.py` / `__init__.py` 等子包结构）
- 算法逻辑直接写在 TSA-Suite feature 算子的 `compute` 方法里
- 不建 `ops/` 子目录，算子按类型平铺到 `feature/construction/` 现有文件或新增文件

**背景**：bqlib 43 个 ops 中，25 个已在 TSA-Suite 有对应（如 `rms` -> `rms_feature`），18 个缺失。本计划只迁这 18 个。TSA-Suite 另有 15 个 bqlib 没有的细化特征（如 `band_*` / `frequency_*`），不在本计划范围。

## C2. 迁移策略：源码融合（不依赖 bqlib，不保留 bqlib 痕迹）

**重写 + 融合模式**。把 bqlib `ops/` 下的函数体提取出来，按 TSA-Suite `IndependentMapFeature` 规范重写为算子类。**不保留 bqlib 的子包结构、多后端分发（`_numpy` / `_pure`）、metadata 注册**。

**不保留 bqlib 依赖，也不保留 bqlib 痕迹**。理由：
- TSA-Suite 必须独立演进，不能被 bqlib 版本绑定
- 代码风格统一到 TSA-Suite 规范，避免 "bqlib 遗留区" 认知负担
- bqlib ops 是纯函数（`max(arr) -> float`），无类状态、无外部依赖、无副作用，融合成本极低
- 函数体通常 10-40 行，提取算法核心逻辑写进 TSA-Suite 算子的 `compute` 方法即可

**实现方式**：
- 算法逻辑直接写在 `IndependentMapFeature` 子类的 `compute` 静态方法里
- 依赖改为 numpy / scipy 直接 import（ops 主要用 numpy，envelope 用 scipy.signal.hilbert）
- **不建 `ops/` 子目录**，算子平铺在 `feature/construction/` 下
- bqlib 的多后端分发（`_numpy` / `_pure`）废弃，只保留 numpy 后端（TSA-Suite 已依赖 numpy）

## C3. 协议映射

| bqlib op 函数 | TSA-Suite feature 算子 |
|---|---|
| `def f(arr, *, axis=None) -> float`（reduce） | `class FFeature(IndependentMapFeature[BaseFeatureConfig])`，`compute` 返回标量 |
| `def f(arr, *, axis=None) -> ndarray`（transform 等长） | 同上，`compute` 返回与输入等长数组 |
| `def f(arr, *, axis=None, output='complex') -> ndarray`（transform 变长） | 同上，`compute` 返回变长数组（如 N//2+1） |
| 函数参数 `output='complex'/'magnitude'` | Config 字段 `output: Literal["complex", "magnitude"]` |
| 函数参数 `axis` | 不暴露（TSA-Suite 算子按列独立处理，axis 固定为 -1） |
| bqlib `_numpy.py` / `_pure.py` 多后端 | **废弃**，只保留 numpy 后端 |
| bqlib `_meta.py` 的 `OpMeta` | **不搬**。Config 字段直接写进 TSA-Suite Pydantic `BaseModel` |
| bqlib `__init__.py` 的注册 side-effect | **不搬**。TSA-Suite 用包扫描自动注册 |
| `class_path` | 自动包扫描（`scan_packages=['tsas.engine.operator.feature.construction']`） |

### 输出类型适配（关键架构点）

TSA-Suite `IndependentMapFeature` 现有算子（如 `rms_feature`）输出**标量**（每列规约成一个数）。bqlib 18 个 ops 分 3 类输出：

| 输出类型 | bqlib ops | TSA-Suite 适配方式 |
|---|---|---|
| **标量**（reduce） | max, min, mean, std, sum, slope, ehr | 直接用现有 `_apply_per_cell`，输出 float ndarray |
| **等长数组**（transform） | abs, sqrt, diff, uni, envelope | 用新 helper `_apply_per_cell_array`，输出 object ndarray（每格 1D 数组） |
| **变长数组**（transform 频域） | fft, rfft, fftfreq, rfftfreq, hps, bark_spectrum, specific_loudness | 同上，输出 object ndarray（每格长度可变） |

**需新增的 helper**（`feature/construction/_array_helpers.py`）：
- `_apply_per_cell_array(x, func) -> np.ndarray`：对 object ndarray 每格应用数组函数，返回 object ndarray
- `_name_output_column_array(self, input_col) -> str`：数组输出的列命名（不含 output_val）

## C4. 分组迁移

### 第 1 组：9 个 skill 直接依赖（3 个，1 天）- 优先级最高

**目标**：解锁 `freq_diagnosis` 和 `envelope_analysis` 两个 skill。

| bqlib op | 类型 | TSA-Suite 算子名 | 输出 | 依赖 | 代码行数 |
|---|---|---|---|---|---|
| `rfft` | transform 变长 | `rfft_feature` | 复数谱 N//2+1 | numpy | 125 |
| `rfftfreq` | transform 变长 | `rfftfreq_feature` | 频率轴 N//2+1 | numpy | 59 |
| `envelope` | transform 等长 | `envelope_feature` | 包络信号 N | numpy + scipy | 175 |

**说明**：TSA-Suite 已有 `envelope_rms_feature`（包络的 RMS 标量），但 `envelope_analysis` skill 需要纯包络信号（用于算包络峭度），所以新增 `envelope_feature` 返回等长数组。

### 第 2 组：基础统计 reduce（5 个，0.5 天）

**目标**：补齐通用基础统计量。

| bqlib op | TSA-Suite 算子名 | 输出 | 代码行数 |
|---|---|---|---|
| `max` | `max_feature` | 标量 | 120 |
| `min` | `min_feature` | 标量 | 120 |
| `mean` | `mean_feature` | 标量 | 115 |
| `std` | `std_feature` | 标量 | 131 |
| `sum` | `sum_feature` | 标量 | 113 |

### 第 3 组：基础变换 transform-等长（4 个，0.5 天）

**目标**：补齐基础信号变换。

| bqlib op | TSA-Suite 算子名 | 输出 | 代码行数 |
|---|---|---|---|
| `abs` | `abs_feature` | 等长数组 | 105 |
| `sqrt` | `sqrt_feature` | 等长数组 | 108 |
| `diff` | `diff_feature` | 等长数组（首值补 0） | 188 |
| `uni` | `uni_feature` | 等长数组 | 116 |

### 第 4 组：频域扩展 transform-变长（3 个，1.5 天）

**目标**：补齐频域分析基础（`freq_diagnosis` skill 依赖 rfft/rfftfreq 已在第 1 组）。

| bqlib op | TSA-Suite 算子名 | 输出 | 依赖 | 代码行数 |
|---|---|---|---|---|
| `fft` | `fft_feature` | 复数谱 N | numpy | 126 |
| `fftfreq` | `fftfreq_feature` | 频率轴 N | numpy | 59 |
| `hps` | `hps_feature` | 谐波乘积谱 | numpy | 290 |

### 第 5 组：心理声学（2 个，1 天）

**目标**：补齐心理声学特征（`predictive_32_features` skill 依赖 sharpness/roughness，TSA-Suite 已有；bark_spectrum / specific_loudness 是其基础）。

| bqlib op | TSA-Suite 算子名 | 输出 | 依赖 | 代码行数 |
|---|---|---|---|---|
| `bark_spectrum` | `bark_spectrum_feature` | Bark 频段谱 | numpy | 231 |
| `specific_loudness` | `specific_loudness_feature` | Bark 频段响度 | numpy | 177 |

### 第 6 组：其他（2 个，0.5 天）

| bqlib op | TSA-Suite 算子名 | 输出 | 代码行数 |
|---|---|---|---|
| `slope` | `slope_feature` | 标量（线性趋势斜率） | 175 |
| `ehr` | `ehr_feature` | 标量（能量谐波比） | 288 |

## C5. 迁移模板

### 模板 1：reduce 标量类（以 `max` 为例）

```python
# src/tsas/engine/operator/feature/construction/basic_stat_feature.py
"""基础统计特征。算法逻辑源自 bqlib ops max/min/mean/std/sum，按 TSA-Suite 规范重写。"""
import numpy as np
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig, IndependentMapFeature,
)
from tsas.engine.operator.feature.construction.signal_feature import _apply_per_cell


class MaxFeature(IndependentMapFeature[BaseFeatureConfig]):
    """最大值特征：每列信号段取 max。

    输入: object ndarray，每格 1D 信号段
    输出: float ndarray，每格为对应信号段的 max
    """

    @classmethod
    def name(cls) -> str:
        return "max_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        def _max_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                raise ValueError("max of empty array")
            return float(np.max(sig))
        return _apply_per_cell(x, _max_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "max")


class MinFeature(IndependentMapFeature[BaseFeatureConfig]):
    """最小值特征。"""

    @classmethod
    def name(cls) -> str:
        return "min_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        def _min_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                raise ValueError("min of empty array")
            return float(np.min(sig))
        return _apply_per_cell(x, _min_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "min")


# MeanFeature / StdFeature / SumFeature 同模式
```

### 模板 2：transform 等长数组类（以 `abs` 为例）

```python
# src/tsas/engine/operator/feature/construction/basic_transform_feature.py
"""基础变换特征。算法逻辑源自 bqlib ops abs/sqrt/diff/uni，按 TSA-Suite 规范重写。"""
import numpy as np
from pydantic import BaseModel, Field
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig, IndependentMapFeature,
)
from tsas.engine.operator.feature.construction._array_helpers import (
    _apply_per_cell_array, _name_output_column_array,
)


class AbsFeature(IndependentMapFeature[BaseFeatureConfig]):
    """绝对值特征：每列信号段取 |x|，输出等长数组。

    输入: object ndarray，每格 1D 信号段
    输出: object ndarray，每格为对应信号段的 |x|
    """

    @classmethod
    def name(cls) -> str:
        return "abs_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        def _abs_1d(sig):
            return np.abs(np.asarray(sig, dtype=float))
        return _apply_per_cell_array(x, _abs_1d)

    def _name_output_column(self, input_col, output_val):
        return _name_output_column_array(self, input_col, "abs")


class DiffFeature(IndependentMapFeature[BaseFeatureConfig]):
    """差分特征：diff(x)，首值补 0 保持等长。

    输入: object ndarray，每格 1D 信号段
    输出: object ndarray，每格为对应信号段的差分（等长，首值 0）
    """

    @classmethod
    def name(cls) -> str:
        return "diff_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        def _diff_1d(sig):
            sig = np.asarray(sig, dtype=float)
            d = np.diff(sig)
            return np.concatenate([[0.0], d])  # 首值补 0 保持等长
        return _apply_per_cell_array(x, _diff_1d)

    def _name_output_column(self, input_col, output_val):
        return _name_output_column_array(self, input_col, "diff")


# SqrtFeature / UniFeature 同模式
```

### 模板 3：transform 变长数组类（以 `rfft` 为例）

```python
# src/tsas/engine/operator/feature/construction/spectral_transform_feature.py
"""频域变换特征。算法逻辑源自 bqlib ops rfft/fftfreq/fft/hps，按 TSA-Suite 规范重写。"""
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig, IndependentMapFeature,
)
from tsas.engine.operator.feature.construction._array_helpers import (
    _apply_per_cell_array, _name_output_column_array,
)


class RfftFeatureConfig(BaseFeatureConfig):
    output: Literal["complex", "magnitude"] = Field(
        default="complex", description="complex 返回复数谱，magnitude 返回 |X[k]|"
    )


class RfftFeature(IndependentMapFeature[RfftFeatureConfig]):
    """实数 FFT 特征：每列信号段做 rfft，输出 N//2+1 半谱。

    输入: object ndarray，每格 1D 信号段
    输出: object ndarray，每格为对应信号段的 rfft 结果（复数或幅值）
    """

    @classmethod
    def name(cls) -> str:
        return "rfft_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, output="complex", **params):
        def _rfft_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.array([])
            result = np.fft.rfft(sig)
            if output == "magnitude":
                return np.abs(result)
            return result
        return _apply_per_cell_array(x, _rfft_1d)

    def _name_output_column(self, input_col, output_val):
        return _name_output_column_array(self, input_col, "rfft")


class RfftFreqFeature(IndependentMapFeature[BaseFeatureConfig]):
    """rFFT 频率轴特征：返回 rfft 对应的频率轴。

    输入: object ndarray，每格 1D 信号段
    输出: object ndarray，每格为对应长度信号的 rfft 频率轴
    """

    @classmethod
    def name(cls) -> str:
        return "rfftfreq_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, fs=1.0, **params):
        def _rfftfreq_1d(sig):
            sig = np.asarray(sig, dtype=float)
            if sig.size == 0:
                return np.array([])
            return np.fft.rfftfreq(len(sig), d=1.0 / fs)
        return _apply_per_cell_array(x, _rfftfreq_1d)

    def _name_output_column(self, input_col, output_val):
        return _name_output_column_array(self, input_col, "rfftfreq")


# FftFeature / FftFreqFeature / HpsFeature 同模式
```

### 模板 4：需新增的 helper（`_array_helpers.py`）

```python
# src/tsas/engine/operator/feature/construction/_array_helpers.py
"""数组输出 helper。供 transform 类 feature 算子使用。"""
import numpy as np


def _apply_per_cell_array(x: np.ndarray, func) -> np.ndarray:
    """对 object ndarray 每格应用数组函数，返回 object ndarray。

    与 signal_feature._apply_per_cell 的区别：后者返回 float ndarray（标量），
    本函数返回 object ndarray（每格是 1D 数组，长度可变）。

    Args:
        x: 输入 ndarray，每格是 1D 信号段
        func: 数组函数，接收 1D float ndarray，返回 1D ndarray

    Returns:
        object ndarray，形状与 x 相同，每格是 func 的输出
    """
    result = np.empty(x.shape, dtype=object)
    for idx in np.ndindex(x.shape):
        sig = np.asarray(x[idx], dtype=float)
        result[idx] = func(sig)
    return result


def _name_output_column_array(feature, input_col: str, op_name: str) -> str:
    """数组输出的列命名（不含 output_val，因为每格是数组）。

    Args:
        feature: 算子实例（用其 _make_output_column_name 方法）
        input_col: 输入列名
        op_name: 算子名（如 "rfft"）

    Returns:
        输出列名，如 "sensor_1_rfft"
    """
    return feature._make_output_column_name(input_col, op_name)
```

## C6. 依赖管理

`pyproject.toml` 新增可选依赖（**不含 bqlib**）：

```toml
[project.optional-dependencies]
scipy = ["scipy>=1.11"]  # envelope_feature 用 scipy.signal.hilbert
```

**import 失败降级**：`envelope_feature` 用 try/except 包 scipy import，失败时回退到 numpy FFT Hilbert（bqlib 原版已有此降级逻辑，一并融合）。

**绝不引入 bqlib 依赖**。所有算法实现必须直接依赖 numpy / scipy。

## C7. Config 字段映射

bqlib ops 函数参数 -> TSA-Suite `Field`：

| bqlib 参数 | TSA-Suite Field | 示例 |
|---|---|---|
| `output='complex'` | `output: Literal["complex", "magnitude"] = "complex"` | rfft_feature |
| `fs=1.0`（采样率） | `fs: float = Field(default=1.0, gt=0, description="采样率 Hz")` | rfftfreq_feature |
| `axis=None` | **不暴露**（TSA-Suite 按列独立，axis 固定 -1） | - |
| 无参数 | 用 `BaseFeatureConfig`（仅 input_columns） | max_feature |

## C8. 测试策略

每个迁移算子至少 3 个测试：

1. `test_compute_basic`：典型输入 + 期望输出（数值断言）
2. `test_numerical_regression`：**关键** - 固定输入，断言输出数值与 baseline 一致（容差 1e-10）。baseline 首次迁移时录制到 `tests/baselines/<op_name>.json`
3. `test_column_naming`：输出列名正确

**baseline 录制流程**：
1. 首次迁移时，在 bqlib 环境跑原版函数，记录输出到 baseline JSON
2. TSA-Suite 算子跑同样输入，对比 baseline
3. 差异 < 1e-10 才算通过
4. baseline 文件提交到 git，作为回归保险

**数组输出测试**：object ndarray 的每格数组需逐格对比，不能整体 `np.allclose`。

## C9. CLI 注册

**零配置**。TSA-Suite 用包扫描自动注册（`scan_packages=['tsas.engine.operator.feature.construction']`）。新算子文件放 `src/tsas/engine/operator/feature/construction/<name>.py`，继承 `IndependentMapFeature`，`feature_construction help` 自动列出。

**feature_construction YAML 调用**：

```yaml
# 基础统计（标量输出）
operators:
  - name: max_feature
    alias: max_val
    input_columns: [sensor_1, sensor_2]
  - name: mean_feature
    alias: mean_val
    input_columns: [sensor_1, sensor_2]

# 频域变换（数组输出）
operators:
  - name: rfft_feature
    alias: spectrum
    input_columns: [sensor_1]
    config:
      output: magnitude
  - name: rfftfreq_feature
    alias: freqs
    input_columns: [sensor_1]
    config:
      fs: 10240.0
```

## C10. 风险与降级

| 风险 | 缓解 |
|---|---|
| 源码融合引入 bug | 第 2 个测试（`test_numerical_regression`）用 baseline 锁定输出，回归立即失败 |
| 数组输出与现有标量输出体系冲突 | 新增 `_array_helpers.py`，数组算子用 `_apply_per_cell_array`，不污染现有 `_apply_per_cell` |
| 变长数组列存储（object dtype）性能差 | 可接受：feature_construction 是离线特征工程，非实时路径 |
| scipy 不可用时 envelope 降级 | 融合 bqlib 的 numpy FFT Hilbert 回退逻辑 |
| bqlib ops 升级后 TSA-Suite 不跟随 | 可接受：TSA-Suite 独立演进，必要时手动 sync |
| 心理声学算法（bark_spectrum）依赖经验参数 | baseline 测试锁定，参数从 bqlib 原版照搬 |
| `diff` 输出长度问题（numpy diff 返回 N-1） | 模板里首值补 0 保持等长，baseline 测试覆盖 |

## C11. 工作量估算

| 组 | 算子数 | 单算子工时 | 总工时 |
|---|---|---|---|
| 前置：`_array_helpers.py` helper | - | - | 0.25 天 |
| 第 1 组 skill 依赖 | 3 | 0.4 天 | 1 天 |
| 第 2 组 基础统计 | 5 | 0.1 天 | 0.5 天 |
| 第 3 组 基础变换 | 4 | 0.15 天 | 0.5 天 |
| 第 4 组 频域扩展 | 3 | 0.5 天 | 1.5 天 |
| 第 5 组 心理声学 | 2 | 0.5 天 | 1 天 |
| 第 6 组 其他 | 2 | 0.25 天 | 0.5 天 |
| 测试 + baseline 录制 + 文档 | - | - | 1.5 天 |
| **合计** | **19**（含 envelope） | - | **~6.75 天（约 1.5 周）** |

**与计划 A/B 对比**：计划 C 工时最短（~7 天 vs A ~66.5 天 / B ~17 天），因为 ops 是纯函数，无基类融合、无 DL 复杂度。

## C12. 建议执行顺序

1. **前置**：`_array_helpers.py` helper（0.25 天）- 数组输出算子的基础
2. 第 1 组 skill 依赖（1 天）- 立即解锁 freq_diagnosis / envelope_analysis 两个 skill
3. 第 2 组 基础统计（0.5 天）- 简单标量，收益快
4. 第 3 组 基础变换（0.5 天）- 简单数组，收益快
5. 第 4 组 频域扩展（1.5 天）- 补齐频域基础
6. 第 5 组 心理声学（1 天）- 按需
7. 第 6 组 其他（0.5 天）- 按需

建议与计划 B（scorers）并行：C 风险低、工时短，可先做 C 第 1-3 组，再做 B。

## C13. 与其他计划的关系

| 维度 | 计划 A（deciders） | 计划 B（scorers） | 计划 C（ops） |
|---|---|---|---|
| 算子数 | 43 | 45 | 19 |
| 策略 | 源码融合 | 源码融合 | 源码融合 |
| 前置工作 | bqlib utils 按需融合（1.5 天） | 无 | `_array_helpers.py`（0.25 天） |
| 工时 | ~66.5 天 | ~17 天 | ~6.75 天 |
| 风险 | 中（DL 易错） | 低（纯函数） | 低（纯函数 + 数组输出适配） |
| 优先级 | 高（解锁工作流 2/3） | 中（评估指标补齐） | 中（特征算子补齐） |
| 建议先后 | 最后做 | 第二做 | **第一做**（工时最短，解锁 skill 文档） |

**统一策略**：三个计划都是源码融合、不依赖 bqlib、不保留 bqlib 命名/结构。TSA-Suite 完全独立。

## C14. 合入后差异更新

| 维度 | bqlib | TSA-Suite（当前） | TSA-Suite（合入后） | 缺口（合入后） |
|---|---|---|---|---|
| preprocessing | 2 | 0 | 0（暂不合入） | -2 |
| ops + features | 43 + 9 skill | 40 + 0 skill | **59 + 0 skill**（40+19） | **+16**（反超 bqlib） |
| selectors | 1 | 2 | 2 | +1 |
| deciders | 94 | 11 | 54（计划 A） | -40 |
| scorers | 44 | 5 | 45（计划 B） | +1 |
| thresholds | 8 | 2（内嵌） | 2（内嵌） | -6 |

**注意**：ops + features 合入后 TSA-Suite 反超 bqlib（59 vs 43），因为 TSA-Suite 原有 15 个 bqlib 没有的细化特征 + 新增 19 个 = 74 个，减去与 bqlib 重叠的 15 个 = 59 个独有算子。9 个 skill 待第 2 步写入 Claude Code skill 文档。
