# 计划 B：bqlib scorers -> TSA-Suite

> 迁移 bqlib `scorers/` 44 个评估指标函数到 TSA-Suite `engine/operator/evaluation/`。
> **策略：源码融合，TSA-Suite 不依赖 bqlib，不保留 bqlib 结构**。
> 创建日期：2026-07-16。

## B1. 迁移目标

把 bqlib `scorers/` 下 44 个函数融合到 TSA-Suite `engine/operator/evaluation/`，使其通过 `python -m tsas.engine.operator.cli evaluation run` 调用。

**硬约束**：
- TSA-Suite 不能引入 bqlib 依赖
- **所有代码按 TSA-Suite 规范融合，不保留 bqlib 命名/结构**
- 算法逻辑直接写在 TSA-Suite 算子的 `_run` 方法里
- 不建 `scorers/` 子目录，指标算子直接平铺在 `evaluation/` 下

## B2. 迁移策略：源码融合（不依赖 bqlib，不保留 bqlib 痕迹）

**重写 + 融合模式**。把 bqlib `scorers/` 下的函数体提取出来，按 TSA-Suite `BaseMetricOperator` 规范重写为算子类。**不保留 bqlib 的函数命名、模块结构、helper 组织**。

**不保留 bqlib 依赖，也不保留 bqlib 痕迹**。理由：
- TSA-Suite 必须独立演进，不能被 bqlib 版本绑定
- 代码风格统一到 TSA-Suite 规范，避免 "bqlib 遗留区" 认知负担
- bqlib scorer 是纯函数（`roc_auc(y_true, y_score) -> float`），无类状态、无外部依赖、无副作用，融合成本极低
- 函数体通常 5-30 行，提取算法核心逻辑写进 TSA-Suite 算子的 `_run` 方法即可

**实现方式**：
- 算法逻辑直接写在 `BaseMetricOperator` 子类的 `_run` 方法里
- 依赖改为 numpy / sklearn 直接 import（scorer 主要用 numpy）
- bqlib 内部 helper（如 `_check_y_true` / `_validate_scores`）-> 融合到 `evaluation/utils.py`，不保留 bqlib 命名
- **不建 `scorers/` 子目录**，指标算子平铺在 `evaluation/<name>.py`

## B3. 协议映射

| bqlib scorer 函数 | TSA-Suite 评估算子 |
|---|---|
| `def f(y_true, y_score) -> float` | `class FOp(BaseMetricOperator[tuple, float, FConfig, None])` |
| `def f(y_true, y_score) -> dict` | MR 用 `BaseModel` 子类，多字段 |
| 函数参数 `pos_label=1` | Config 字段 `positive_label: int = 1` |
| 函数参数 `k` | Config 字段 `k: int` |
| 返回 float | MR=float，`main_scores={"_": "_"}`（路径占位） |
| 返回多字段 | MR=BaseModel，`main_scores={"f1": "f1", "far": "far"}` |
| bqlib helper 函数 | 融合到 `evaluation/utils.py`，不保留原命名 |
| 无对应（bqlib 是函数） | CLI 自动渲染 `### 主输出 (XxxResult)` + `**结构**` 字段表 |

## B4. 分组迁移

### 第 1 组：anomaly 排序类（7 个，2 天）

**目标**：补齐工作流 1 无监督场景的 ranking 评估。

| bqlib 函数 | TSA-Suite 算子名 | MR 类型 | 备注 |
|---|---|---|---|
| `roc_auc` | `roc_auc_metric` | float | 已含在 `binary_classification_curve` 里，**拆出来独立算子** |
| `pr_auc` | `pr_auc_metric` | float | 同上 |
| `average_precision` | `average_precision_metric` | float | 独立算子 |
| `precision_at_k` | `precision_at_k_metric` | float | Config 含 `k` |
| `recall_at_k` | `recall_at_k_metric` | float | Config 含 `k` |
| `vus_roc` | `vus_roc_metric` | float | 体积下 ROC |
| `vus_pr` | `vus_pr_metric` | float | 体积下 PR |

### 第 2 组：anomaly 事件类（9 个，3 天）

**目标**：补齐时序异常段式评估。

| bqlib 函数 | TSA-Suite 算子名 | MR 类型 | 备注 |
|---|---|---|---|
| `point_adjust_f1` | `point_adjust_f1_metric` | BaseModel（pa_f1/pa_recall/pa_precision） | TSA-Suite 已有 `point_adjust`，**对齐字段名** |
| `event_based_f1` | `event_based_f1_metric` | BaseModel | 事件级 F1 |
| `soft_event_based_f1` | `soft_event_based_f1_metric` | BaseModel | 软事件级 |
| `detection_delay` | `detection_delay_metric` | float | 检测延迟 |
| `affiliation_f1` | `affiliation_metric` | BaseModel（affiliation_p/r/f1） | Affiliation 系列 |
| `affiliation_precision` | 合并到 `affiliation_metric` | - | 不单独迁 |
| `affiliation_recall` | 合并到 `affiliation_metric` | - | 不单独迁 |
| `range_auc_roc` | `range_auc_roc_metric` | float | Range-based |
| `range_auc_pr` | `range_auc_pr_metric` | float | Range-based |

### 第 3 组：NAB + 聚类（6 个，2 天）

| bqlib 函数 | TSA-Suite 算子名 | MR 类型 |
|---|---|---|
| `nab_score` | `nab_score_metric` | BaseModel（nab_score/profile) |
| `nab_best_threshold` | `nab_best_threshold_metric` | float |
| `adjusted_rand_index` | `ari_metric` | float |
| `normalized_mutual_info` | `nmi_metric` | float |
| `mutual_info_score` | `mi_metric` | float |
| `entropy` | `entropy_metric` | float |

### 第 4 组：classification 扩展（10 个，2 天）

**目标**：补齐 `binary_classification` 没单独暴露的指标。

| bqlib 函数 | TSA-Suite 算子名 | MR 类型 | 备注 |
|---|---|---|---|
| `fbeta_score` | `fbeta_metric` | float | Config 含 `beta` |
| `macro_f1` | `macro_f1_metric` | float | 多分类 |
| `micro_f1` | `micro_f1_metric` | float | 多分类 |
| `weighted_f1` | `weighted_f1_metric` | float | 多分类 |
| `cohen_kappa` | `cohen_kappa_metric` | float | |
| `hamming_loss` | `hamming_loss_metric` | float | 多标签 |
| `exact_match_ratio` | `exact_match_ratio_metric` | float | 多标签 |
| `multi_label_accuracy` | `multi_label_accuracy_metric` | float | 多标签 |
| `accuracy` | 已在 `binary_classification` / `multi_classification` 里 | - | 不单独迁 |
| `precision` / `recall` / `f1_score` / `mcc` | 同上 | - | 不单独迁 |

### 第 5 组：regression（8 个，2 天）

**目标**：补齐 forecasting 场景评估。

| bqlib 函数 | TSA-Suite 算子名 | MR 类型 |
|---|---|---|
| `mae` | `mae_metric` | float |
| `mse` | `mse_metric` | float |
| `rmse` | `rmse_metric` | float |
| `mape` | `mape_metric` | float |
| `smape` | `smape_metric` | float |
| `r2` | `r2_metric` | float |
| `mase` | `mase_metric` | float |
| `theils_u` | `theils_u_metric` | float |

### 第 6 组：time_series + 工具（5 个，3 天）

| bqlib 函数 | TSA-Suite 算子名 | MR 类型 | 备注 |
|---|---|---|---|
| `dtw_distance` | `dtw_distance_metric` | float | 配置含 `window` / `delta` |
| `dtw_path` | `dtw_path_metric` | BaseModel（distance + path） | |
| `confidence_scores` | `confidence_metric` | BaseModel | 置信度 |
| `normalize_scores` | `normalize_score_metric` | ndarray | 不是 metric 是 transform，单独处理 |
| `plot_results` | 不迁 | - | 可视化走外部脚本 |

**不迁**：`plot_results`（viz 不走 CLI）。

## B5. 每个算子的迁移模板

以 `precision_at_k` 为例（第 1 组）：

```python
# src/tsas/engine/operator/evaluation/precision_at_k.py
"""Precision@k 指标。算法逻辑源自 bqlib precision_at_k，按 TSA-Suite 规范重写。"""
import numpy as np
from pydantic import BaseModel, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig, BaseMetricOperator,
)


class PrecisionAtKConfig(BaseMetricConfig):
    k: int = Field(default=10, ge=1, description="前 k 个样本")
    pos_label: int = Field(default=1, description="正例标签值")
    main_scores: dict[str, str] | None = {"patk": "_"}


class PrecisionAtKMetric(
    BaseMetricOperator[tuple[np.ndarray, np.ndarray], float, PrecisionAtKConfig, None]
):
    """Precision@k 指标。

    输入: (y_truth, y_score) 标签 + 连续分数
    输出: float，前 k 个高分样本中的正例比例
    """

    @classmethod
    def name(cls) -> str:
        return "precision_at_k_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: tuple[np.ndarray, np.ndarray], *, params) -> float:
        y_truth, y_score = x
        y_truth = np.asarray(y_truth)
        y_score = np.asarray(y_score)
        k = self.config.k
        pos_label = self.config.pos_label

        if k <= 0 or k > len(y_truth):
            raise ValueError(f"k must be between 1 and {len(y_truth)}")

        top_indices = np.argsort(y_score)[::-1][:k]
        top_true = y_truth[top_indices]
        return float(np.sum(top_true == pos_label) / k)
```

**多字段 MR 示例**（`affiliation_metric`）：

```python
class AffiliationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    affiliation_precision: float
    affiliation_recall: float
    affiliation_f1: float


class AffiliationConfig(BaseMetricConfig):
    pos_label: int = 1
    main_scores: dict[str, str] | None = {
        "affiliation_f1": "affiliation_f1",
    }


class AffiliationMetric(
    BaseMetricOperator[tuple[np.ndarray, np.ndarray], AffiliationResult, AffiliationConfig, None]
]):
    """Affiliation 系列（合并 bqlib 的 3 个函数）"""

    @classmethod
    def name(cls) -> str:
        return "affiliation_metric"

    def _run(self, x, *, params) -> AffiliationResult:
        y_truth, y_score = x
        # 算法逻辑直接写在这里，不调用 bqlib
        return AffiliationResult(
            affiliation_precision=p,
            affiliation_recall=r,
            affiliation_f1=f1,
        )
```

**关键约定**：
- **不 import bqlib**。所有依赖改为 numpy / sklearn 直接 import
- **不保留 bqlib 结构**：算法逻辑直接写在 `_run` 方法里，不建独立函数文件
- bqlib 内部 helper 融合到 `evaluation/utils.py`，不保留原命名
- 算子文件头注释标注算法出处：`# 算法逻辑源自 bqlib.scorers.<module>.<func>，按 TSA-Suite 规范重写`

## B6. main_scores 路径设置

每个算子的 `main_scores` 默认值决定 HPO 优化目标。约定：

| MR 类型 | main_scores 路径 | 示例 |
|---|---|---|
| `float` | `"_"` | `{"patk": "_"}` |
| 单字段 BaseModel | 属性名 | `{"auc": "auc_roc"}` |
| 多字段 BaseModel | 多属性 | `{"f1": "f1", "far": "far"}` |

HPO 用法：

```python
op = PrecisionAtKMetric(main_scores={"patk": "_"})
scores = op.scores((y_truth, y_score))  # -> {"patk": 0.85}
```

## B7. 输出类型 MR 决策

| 场景 | MR 类型 | 理由 |
|---|---|---|
| 单一标量（AUC / F1 / MAE） | `float` | 简单，main_scores 用 `"_"` |
| 多字段（混淆矩阵衍生 / Affiliation） | `BaseModel` | 一次算完所有相关字段，避免重复计算 |
| 距离 + 路径（DTW） | `BaseModel` | path 是 list，不能是 float |
| ndarray（normalize_scores） | 特殊处理，不走 BaseMetricOperator | 改用 NumericOperator |

## B8. 测试策略

每个迁移算子至少 3 个测试：

1. `test_run_basic`：典型输入 + 期望输出（数值断言）
2. `test_numerical_regression`：**关键** - 固定输入，断言输出数值与 baseline 一致（容差 1e-10）。baseline 首次迁移时录制到 `tests/baselines/<op_name>.json`
3. `test_scores_hpo`：`main_scores` 提取正确

**baseline 录制流程**：
1. 首次迁移时，在 bqlib 环境跑原版函数，记录输出到 baseline JSON
2. TSA-Suite 算子跑同样输入，对比 baseline
3. 差异 < 1e-10 才算通过
4. baseline 文件提交到 git，作为回归保险

第 2 个测试替代了 `test_against_bqlib`。不再需要 bqlib 作为测试依赖。

**测试数据**：从 bqlib 测试套件提取 fixture 逻辑，融合到 TSA-Suite 测试风格。

## B9. CLI 注册

零配置。文件放 `src/tsas/engine/operator/evaluation/<name>.py`，继承 `BaseMetricOperator`，`evaluation help` 自动列出。

**evaluation YAML 调用**：

```yaml
operators:
  - name: precision_at_k_metric
    alias: patk
    truth_columns: [label]
    predict_columns: [anomaly_score]
    config:
      k: 50
      pos_label: 1
```

## B10. 风险与降级

| 风险 | 缓解 |
|---|---|
| 源码融合引入 bug | 第 2 个测试（`test_numerical_regression`）用 baseline 锁定输出，回归立即失败 |
| 多字段 MR 的 main_scores 路径写错 | 第 3 个测试 `test_scores_hpo` 立即失败 |
| `normalize_scores` 不走 BaseMetricOperator | 单独走 `NumericOperator` 协议，放 `evaluation/normalize.py` |
| `plot_results` 不能迁 | 文档说明走外部 Python 脚本 |
| TSA-Suite 已有算子字段名不一致 | 第 2 组 `point_adjust` 对齐 bqlib 字段名（`pa_f1` 等），不另起 |
| bqlib 内部 helper 命名冲突 | 融合到 `evaluation/utils.py` 时重命名为 TSA-Suite 风格 |

## B11. 工作量估算

| 组 | 算子数 | 单算子工时 | 总工时 |
|---|---|---|---|
| 第 1 组 anomaly 排序 | 7 | 0.3 天 | 2 天 |
| 第 2 组 anomaly 事件 | 9 | 0.4 天 | 3 天 |
| 第 3 组 NAB + 聚类 | 6 | 0.3 天 | 2 天 |
| 第 4 组 classification 扩展 | 10 | 0.2 天 | 2 天 |
| 第 5 组 regression | 8 | 0.25 天 | 2 天 |
| 第 6 组 time_series + 工具 | 5 | 0.6 天 | 3 天 |
| 测试 + baseline 录制 + 文档 | - | - | 3 天 |
| **合计** | **45** | - | **~17 天（约 3 周）** |

## B12. 建议执行顺序

1. 第 1 组 anomaly 排序（2 天）- 立即补齐工作流 1 ranking 评估
2. 第 4 组 classification 扩展（2 天）- 简单函数融合，收益快
3. 第 5 组 regression（2 天）- 补齐 forecasting 评估
4. 第 2 组 anomaly 事件（3 天）- 时序段式评估
5. 第 3 组 NAB + 聚类（2 天）- 长尾评估
6. 第 6 组 time_series + 工具（3 天）- DTW 等

建议与计划 A（deciders 融合）并行：B 风险低、收益快，可先做 B 再做 A，或 B 第 1-2 组与 A 前置 + 第 1 批并行。

## B13. 与计划 A 对比

| 维度 | 计划 A（deciders，融合） | 计划 B（scorers，融合） |
|---|---|---|
| 算子数 | 43（迁）+ 51（不迁） | 45 |
| 策略 | 源码融合（无 bqlib 依赖，不保留 bqlib 结构） | 源码融合（无 bqlib 依赖，不保留 bqlib 结构） |
| 前置工作 | bqlib utils 按需融合（1.5 天） | 无 |
| 工时 | ~66.5 天 | ~17 天 |
| 风险 | 中（源码融合量大，DL 易错） | 低（纯函数融合） |
| 优先级 | 高（解锁工作流 2/3） | 中（评估指标补齐） |
| 建议先后 | **先 B 后 A** | B 风险低、收益快，先做 |

**统一策略**：两个计划都是源码融合、不依赖 bqlib、不保留 bqlib 命名/结构。TSA-Suite 完全独立。
