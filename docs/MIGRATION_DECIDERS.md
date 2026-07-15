# 计划 A：bqlib deciders -> TSA-Suite

> 迁移 bqlib `deciders/models/` 94 个异常检测模型到 TSA-Suite `engine/operator/detection/`。
> **策略：源码融合，TSA-Suite 不依赖 bqlib，不保留 bqlib 结构**。
> 创建日期：2026-07-16。

## A1. 迁移目标

把 bqlib `deciders/models/` 下 94 个模型源码迁到 TSA-Suite `engine/operator/detection/`，使其通过 `python -m tsas.engine.operator.cli detection fit/run` 调用。

**硬约束**：
- TSA-Suite 不能引入 bqlib 依赖
- **所有代码按 TSA-Suite 规范融合，不保留 bqlib 命名/结构**（不搬 `_impl.py` / `_meta.py` / `BaseDetector` / `SklearnOutlierAdapter` 等概念）
- 算法逻辑直接写在 TSA-Suite 算子的 `_fit_data` / `_run_data` 里
- bqlib utils 里有用的通用工具按需融合到 TSA-Suite 现有目录，**不建 `_bqlib_compat/`**

**关键约束**：TSA-Suite 当前 `SupervisedNumericOperatorMixin` 只有基类无实现，本次需顺手补齐监督算子协议。

## A2. 迁移策略：源码融合（不依赖 bqlib，不保留 bqlib 痕迹）

**重写 + 融合模式**。把 bqlib `deciders/models/` 下的算法逻辑提取出来，按 TSA-Suite 算子规范重写，融合进 `engine/operator/detection/` 体系。**不保留 bqlib 的目录结构、命名、基类概念**。

**不保留 bqlib 依赖，也不保留 bqlib 痕迹**。理由：
- TSA-Suite 必须独立演进，不能被 bqlib 版本绑定
- 代码风格统一到 TSA-Suite 规范，避免 "bqlib 遗留区" 认知负担
- bqlib 94 个模型大部分是 sklearn / pyod / torch 的薄壳，真正自研逻辑集中在 DL 时序模型
- 融合后 TSA-Suite 可自由修改 / 优化算法

**实现方式**：
- **sklearn 系**（20 个）：直接 `from sklearn...`，算法逻辑写在算子的 `_fit_data` / `_run_data`
- **pyod 系**：直接 `from pyod...`（TSA-Suite 可选依赖 pyod）
- **torch 系**（10 个 ts_*）：网络结构定义 + 训练循环直接写在算子文件或 TSA-Suite DL 基类里
- **bqlib 基础设施**（`deciders/utils/` 2615 行）：**不整体迁移**。按需把通用工具融合到 TSA-Suite 现有目录，不保留 bqlib 文件结构（见 A4）

## A3. 协议映射

| bqlib BaseDetector | TSA-Suite 对应 |
|---|---|
| `__init__(contamination=0.1, **model_params)` | `Config`（Pydantic BaseModel，含 `contamination` + 模型参数） |
| `fit(X, y=None)` | `_fit_data(x, params)`（无监督）或 `_fit_data(x, y, params)`（监督，需扩 Mixin） |
| `decision_function(X) -> scores` | `_run_data(x, params) -> np.ndarray`（Scorer 类型） |
| `predict(X) -> labels` | `_run_data(x, params) -> np.ndarray`（Decider / Detector 类型） |
| `predict_proba(X) -> [n, 2]` | 同上，但返回 `[:, 1]`（监督场景） |
| `decision_scores_` / `threshold_` / `labels_` | 内部缓存，不暴露（TSA-Suite 阈值走独立 Decider 算子） |
| `SklearnOutlierAdapter` 基类 | TSA-Suite `UnsupervisedNumericOperatorMixin + SingleScorerMixin`（不搬原类） |
| `SklearnClassifierAdapter` 基类 | TSA-Suite `SupervisedNumericOperatorMixin`（需补齐实现，不搬原类） |
| `BaseDetector` 基类（186 行） | TSA-Suite `NumericOperator`（已有，不搬） |
| `BaseDL` / `TSBase` 基类（DL 训练循环） | 融合到 `detection/dl_base.py` / `ts_base.py`，按 TSA-Suite 规范重写 |
| `_meta.py` 的 `ModelMeta` / `ParameterMeta` | **不搬**。Config 字段直接写进 TSA-Suite Pydantic `BaseModel`，metadata 通过 `Field(description=...)` + docstring 暴露 |
| `_impl.py` 的算法实现 | **不保留 _impl 结构**。算法逻辑直接写在算子类的 `_fit_data` / `_run_data` |
| `class_path` | 自动包扫描（`scan_packages=['tsas.engine.operator.detection']`） |

## A4. 前置工作：算法依赖按需融合

在迁模型之前，把 bqlib `deciders/utils/` 里 **真正被模型依赖** 的工具按需融合到 TSA-Suite 现有目录。**不建 `_bqlib_compat/`，不保留 bqlib 命名**。

**融合映射**（按需，不全搬）：

| bqlib 文件 | 行数 | TSA-Suite 融合目标 | 处理方式 |
|---|---|---|---|
| `utils/base_detector.py` | 186 | 不搬 | TSA-Suite `NumericOperator + Mixin` 体系替代 |
| `utils/sklearn_adapters.py` | 140 | 不搬 | TSA-Suite Mixin 替代（`SklearnOutlierAdapter` / `SklearnClassifierAdapter` 概念废弃） |
| `utils/utility.py` | 228 | `detection/utils.py` | 按需挑通用函数（`invert_order` / `check_array` / `_validate_ndarray`），重写为 TSA-Suite 风格 |
| `utils/ts_utils.py` | 143 | `detection/ts_utils.py` | 时序工具（sliding window 等）按需融合 |
| `utils/confidence_mixin.py` | 57 | `detection/base.py`（作为 Mixin） | 置信度评分 mixin，融合进现有 base |
| `utils/neighbor_mixin.py` | 63 | `detection/base.py`（作为 Mixin） | 邻域查询 mixin（KNN / LOF 算子用） |
| `utils/proba_mixin.py` | 64 | `detection/base.py`（作为 Mixin） | 概率输出 mixin（监督算子用） |
| `utils/reconstruction_mixin.py` | 57 | `detection/base.py`（作为 Mixin） | 重构误差 mixin（AE 系算子用） |
| `utils/dl/base_dl.py` | 407 | `detection/dl_base.py` | DL 训练基类，按 TSA-Suite 规范重写为 DL 算子基类 |
| `utils/dl/data_module.py` | 196 | `detection/data_module.py` | DL 数据模块，重写 |
| `utils/dl/nn_utils.py` | 417 | `detection/nn_utils.py` | NN 工具（层定义 / loss），重写 |
| `utils/dl/ts_base.py` | 127 | `detection/ts_base.py` | TS 算子基类，重写 |
| `utils/dl/ae.py` | 115 | `detection/ae_base.py` | AE 算子基类，重写 |
| `utils/dl/device.py` | 59 | 融合进 `detection/dl_utils.py` | 设备管理 |
| `utils/dl/masked_collate.py` | 142 | 融合进 `detection/dl_utils.py` | masked collate |
| `utils/_fallback.py` | 127 | 不搬 | bqlib 特有降级机制，TSA-Suite 用 `pytest.importorskip` 替代 |

**融合原则**：
- 去掉 bqlib 命名（`SklearnOutlierAdapter` / `BaseDetector` 等概念不保留）
- 去掉 bqlib 特有内部状态（`decision_scores_` / `threshold_` / `labels_` 内部缓存 -> TSA-Suite 不暴露，阈值走独立 Decider 算子）
- 代码风格对齐 TSA-Suite（类型注解 / docstring / pydantic Config / frozen model_config）
- **不保留 `_impl.py` / `_meta.py` 文件结构**。算法逻辑直接写在算子类里
- 每个融合文件头标注算法出处：`# 算法逻辑源自 bqlib.deciders.utils.xxx，按 TSA-Suite 规范重写`

## A5. 分批迁移（按优先级 + 类别）

### 第 1 批：核心 sklearn 无监督（20 个，~2 周）

**目标**：覆盖工作流 1（无监督）80% 场景。

| 模型 | bqlib 类 | 依赖 | 备注 |
|---|---|---|---|
| `iforest_detector` | `IForest` | sklearn | 业内 baseline，必迁 |
| `lof_detector` | `LOF` | sklearn | 局部密度，必迁 |
| `knn_detector` | `KNN` | sklearn / pyod | bqlib 已有，TSA-Suite 也已有，**对齐 config 字段** |
| `ocsvm_detector` | `OCSVM` | sklearn | 一类 SVM，必迁 |
| `pca_detector` | `PCA` | sklearn | bqlib/TSA-Suite 都有，**对齐 config 字段** |
| `hbos_detector` | `HBOS` | pyod | 直方图，快 |
| `abod_detector` | `ABOD` | pyod | 角度异常 |
| `copod_detector` | `COPOD` | pyod | 无参数，快 |
| `ecod_detector` | `ECOD` | pyod | 无参数，快 |
| `eif_detector` | `EIF` | 第三方 eif 库 | 扩展 IForest |
| `inne_detector` | `INNE` | pyod | 邻域异常 |
| `loda_detector` | `LODA` | sklearn | 稀疏投影 |
| `mcd_detector` | `MCD` | sklearn | 鲁棒协方差 |
| `rod_detector` | `ROD` | numpy | 鲁棒深度 |
| `sos_detector` | `SOS` | numpy | 随机离群选择 |
| `sod_detector` | `SOD` | pyod | 子空间异常 |
| `gmm_detector` | `GMM` | sklearn | 概率密度 |
| `kde_detector` | `KDE` | sklearn | 核密度 |
| `cblof_detector` | `CBLOF` | pyod | 聚类局部异常 |
| `cof_detector` | `COF` | pyod | 链式离群 |

### 第 2 批：监督分类器（3 个，~4 天）

**目标**：解锁工作流 2（supervised）真监督能力。

| 模型 | bqlib 类 | 依赖 | 备注 |
|---|---|---|---|
| `random_forest_detector` | `RandomForest` | sklearn | 必迁，工作流 2 默认模型 |
| `svm_detector` | `SVM` | sklearn | 必迁 |
| `lgbm_detector` | `LGBM` | lightgbm | lightgbm 依赖 |

**前置工作**：补齐 TSA-Suite `SupervisedNumericOperatorMixin` 的 `fit(x, y)` 实现协议（当前只有基类无具体子类验证）。

### 第 3 批：DL 时序（10 个，~5 周）

**目标**：覆盖工作流 1 的时序场景。

| 模型 | bqlib 类 | 依赖 | 备注 |
|---|---|---|---|
| `ts_donut_detector` | `TSDonut` | torch | 网络结构 ~300 行 |
| `ts_tranad_detector` | `TSTranAD` | torch | 网络结构 ~500 行 |
| `ts_omni_anomaly_detector` | `TSOmniAnomaly` | torch | 网络结构 ~600 行 |
| `ts_usad_detector` | `TSUSAD` | torch | 网络结构 ~300 行 |
| `ts_lstm_detector` | `TSLSTM` | torch | 网络结构 ~200 行 |
| `ts_timesnet_detector` | `TSTimesNet` | torch | 网络结构 ~700 行 |
| `ts_anomaly_transformer_detector` | `TSAnomalyTransformer` | torch | 网络结构 ~500 行 |
| `ts_spectral_residual_detector` | `TSSpectralResidual` | numpy | 纯 numpy |
| `ts_matrix_profile_detector` | `TSMatrixProfile` | numpy | 纯 numpy |
| `ts_fft_detector` | `TSFFT` | numpy | 纯 numpy |

**DL 模型融合要点**：
- 网络结构定义（`nn.Module` 子类）直接写在算子文件或 TSA-Suite DL 基类里
- 训练循环（`fit` 逻辑）融合进算子的 `_fit_data`
- 依赖 TSA-Suite 融合后的 DL 基础设施（`detection/dl_base.py` / `data_module.py` / `nn_utils.py`）
- `torch.save(state_dict)` 而非 pickle 整个对象（避免 save/load 兼容性问题）

**剩余 24 个 ts_* 模型**作为长尾，按需再迁（chronos / moirai / timesfm 等大模型基座优先级低）。

### 第 4 批：集成 / 高级（10 个，~2 周）

| 模型 | bqlib 类 | 依赖 | 备注 |
|---|---|---|---|
| `feature_bagging_detector` | `FeatureBagging` | sklearn / pyod | 必迁 |
| `lscp_detector` | `LSCP` | pyod | 集成 |
| `suod_detector` | `SUOD` | pyod | 集成 |
| `xgbod_detector` | `XGBOD` | xgboost + pyod | xgboost 依赖 |
| `mo_gaal_detector` | `MOGAAL` | tensorflow / torch | GAN（需确认 bqlib 实现用的哪个） |
| `so_gaal_detector` | `SOGAAL` | tensorflow / torch | GAN |
| `lunar_detector` | `LUNAR` | torch | torch |
| `devnet_detector` | `DevNet` | torch | 半监督 |
| `dif_detector` | `DIF` | torch | 深度 IForest |
| `deep_svdd_detector` | `DeepSVDD` | torch | torch |

### 第 5 批：长尾不迁（51 个）

`ae1svm` / `alad` / `anogan` / `cicada_semi` / `cicada_sup` / `deep_svdd`（非 ts_） / `donut`（非 ts_） / `embedding` / `fft`（非 ts_） / `hdbscan` / `kpca` / `lmdd` / `loci` / `mad` / `poly` / `qmcd` / `rgraph` / `robust_pca` / `rod` / `sampling` / `so_gaal_new` / `ts_charm` / `ts_chronos` / `ts_cicada` / `ts_cicada_sup` / `ts_cnn` / `ts_dada` / `ts_fits` / `ts_kmeans_ad` / `ts_kshape` / `ts_lag_llama` / `ts_left_stampi` / `ts_m2n2` / `ts_mmpad` / `ts_moirai` / `ts_moment` / `ts_od` / `ts_ofa` / `ts_poly` / `ts_sand` / `ts_sliding_window` / `ts_time_moe` / `ts_time_rcd`（已有）/ `ts_timesfm` / `ts_tspulse` / `vae` / `ts_cicada`（已有 cicada_predictor）

**不迁理由**：
- 与已迁算子重叠（如 `ts_cicada` vs `cicada_predictor`）
- 依赖过重（`ts_chronos` / `ts_moirai` / `ts_timesfm` 等大模型基座）
- 算法冷门（`qmcd` / `lmdd` / `loci`）

按用户需求再补。

## A6. 每个算子的迁移模板

以 IForest 为例（第 1 批，sklearn 系薄壳）：

```python
# src/tsas/engine/operator/detection/iforest.py
"""IForest 检测器。算法逻辑源自 bqlib IForest，按 TSA-Suite 算子规范重写。

异常分数 = -score_samples(X)，越高越异常。
"""
import pickle
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import (
    SingleScorerMixin, BaseDeciderMixin,
)
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider, PercentileDeciderConfig,
)


class IForestScorerConfig(BaseModel):
    model_config = {"frozen": True}
    n_estimators: int = Field(default=100, ge=10, le=500, description="树的数量")
    max_samples: int | float | str = Field(default="auto", description="每棵树采样数")
    max_features: int | float = Field(default=1.0, gt=0, le=1.0)
    bootstrap: bool = False
    contamination: float = Field(default=0.1, gt=0, le=0.5)
    n_jobs: int = Field(default=1, ge=-1)
    random_state: int | None = None


class IForestScorer(SingleScorerMixin[None],
                    UnsupervisedNumericOperatorMixin[None],
                    NumericOperator[None, IForestScorerConfig, None]):
    """Isolation Forest 直接评分器。"""

    @classmethod
    def name(cls) -> str:
        return "iforest_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: IsolationForest | None = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            contamination=self.config.contamination,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )
        self._model.fit(x)

    def _run_data(self, x: np.ndarray, params: None) -> np.ndarray:
        # 与 bqlib 一致：异常分数 = -score_samples(X)
        return -self._model.score_samples(x)

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        with open(path / "_model.pkl", "wb") as f:
            pickle.dump(self._model, f)

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        with open(path / "_model.pkl", "rb") as f:
            self._model = pickle.load(f)
        self._fitted = True


class IForestDetectorConfig(BaseModel):
    model_config = {"frozen": True}
    n_estimators: int = Field(default=100, ge=10, le=500)
    max_samples: int | float | str = "auto"
    contamination: float = Field(default=0.1, gt=0, le=0.5)
    percentile: float = Field(default=95.0, ge=50.0, le=99.9)


class IForestDetector(UnsupervisedNumericOperatorMixin[None],
                      BaseDeciderMixin[None],
                      NumericOperator[None, IForestDetectorConfig, None]):
    """IForest 检测器 = IForestScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "iforest_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scorer = IForestScorer(config=IForestScorerConfig(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
        ))
        self._decider = PercentileDecider(config=PercentileDeciderConfig(
            percentile=self.config.percentile,
        ))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._scorer.fit(x)
        scores, _ = self._scorer.run(x)
        self._decider.fit(scores)

    def _run_data(self, x: np.ndarray, params: None) -> np.ndarray:
        scores, _ = self._scorer.run(x)
        labels, _ = self._decider.run(scores)
        return labels

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        self._scorer.save(path / "_scorer")
        self._decider.save(path / "_decider")

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        self._scorer = IForestScorer.load(path / "_scorer")
        self._decider = PercentileDecider.load(path / "_decider")
        self._fitted = True
```

**关键约定**：
- 每个 bqlib 模型迁过来生成 2 个 TSA-Suite 算子：`<name>_scorer`（输出分数）+ `<name>_detector`（输出分数 + 标签）
- **不 import bqlib**。所有依赖改为 sklearn / pyod / torch / numpy 直接 import
- **不保留 bqlib 结构**：算法逻辑直接写在 `_fit_data` / `_run_data`，不建 `_impl.py` / `_meta.py`
- 通用 helper（如 `invert_order`）从 `detection/utils.py` import，不依赖 bqlib 命名
- `_save_fit_state` / `_load_fit_state` 用 pickle 序列化模型（sklearn 系）或 `torch.save(state_dict)`（DL 系）
- 算子文件头注释标注算法出处：`# 算法逻辑源自 bqlib.deciders.models.<name>，按 TSA-Suite 规范重写`

## A7. 监督算子协议补齐（第 2 批前置）

TSA-Suite `SupervisedNumericOperatorMixin` 当前无实现。第 2 批迁移前需：

1. 在 `tsas/engine/operator/base.py` 补 `SupervisedNumericOperatorMixin.fit(x, y)` 模板方法（仿 `UnsupervisedNumericOperatorMixin`）
2. 加 `_validate_supervised_input(x, y)` 校验
3. 写一个 `RandomForestDetector` 跑通端到端，验证协议
4. CLI 的 `detection fit` 子命令加 `--label-columns` 参数指定 y 列

## A8. 依赖管理

`pyproject.toml` 新增可选依赖（**不含 bqlib**）：

```toml
[project.optional-dependencies]
sklearn = ["scikit-learn>=1.3"]
pyod = ["pyod>=2.0"]
torch = ["torch>=2.0"]
lightgbm = ["lightgbm>=4.0"]
xgboost = ["xgboost>=1.7.6"]
eif = ["eif>=1.0"]
all = ["tsa-suite[sklearn,pyod,torch,lightgbm,xgboost,eif]"]
```

**import 失败降级**：算子模块用 try/except 包 sklearn / pyod / torch import，失败时 warning + 不注册（CLI help 自动跳过）。TSA-Suite 已有此机制（detection.md §4.1）。

**绝不引入 bqlib 依赖**。所有算法实现必须直接依赖 sklearn / pyod / torch / numpy。

## A9. Config 字段映射

bqlib `_meta.py` 的 `ParameterMeta` -> TSA-Suite `Field`：

| bqlib ParameterMeta | TSA-Suite Field |
|---|---|
| `name` | Config 字段名 |
| `type="int"` | `int` |
| `type="float"` | `float` |
| `type="bool"` | `bool` |
| `type="any"` | `int | float | str`（union） |
| `default` | `default=...` |
| `range="(0, 0.5]"` | `gt=0, le=0.5` |
| `enum=["auto"]` | `Literal["auto"]` 或 `Enum` |
| `description` | `description=...` |

**半自动化**：写一个脚本扫 bqlib `_meta.py`，生成 TSA-Suite Config 骨架（手工校对）。

## A10. 测试策略

每个迁移算子至少 4 个测试：

1. `test_config`：默认值 / frozen / 字段校验
2. `test_fit_run`：fit + run 输出形状 + 值有限
3. `test_save_load`：roundtrip 一致性
4. `test_numerical_regression`：**关键** - 固定输入 + 固定 `random_state`，断言输出数值与 baseline 一致（容差 1e-6）。baseline 首次迁移时录制到 `tests/baselines/<op_name>.json`

**baseline 录制流程**：
1. 首次迁移时，在 bqlib 环境跑原版算子，记录输出到 baseline JSON
2. TSA-Suite 算子跑同样输入，对比 baseline
3. 差异 < 1e-6 才算通过
4. baseline 文件提交到 git，作为回归保险

第 4 个测试替代了 Adapter 模式下的 `test_against_bqlib`。不再需要 bqlib 作为测试依赖。

## A11. CLI 注册

**零配置**。TSA-Suite 用包扫描自动注册（`scan_packages=['tsas.engine.operator.detection']`）。新算子文件放 `src/tsas/engine/operator/detection/<name>.py`，继承正确 Mixin，`detection help` 自动列出。

## A12. 风险与降级

| 风险 | 缓解 |
|---|---|
| 源码融合引入 bug | 第 4 个测试（`test_numerical_regression`）用 baseline 锁定输出，回归立即失败 |
| torch 模型 save/load 不兼容 | DL 算子用 `torch.save(state_dict)` 而非 pickle 整个对象 |
| 监督算子协议补不齐 | 第 2 批先做 1 个 RandomForest 跑通端到端，再批量迁 |
| bqlib 升级后 TSA-Suite 不跟随 | 可接受：TSA-Suite 独立演进，必要时手动 sync |
| pyod 算子依赖 pyod 版本 | 锁版本 `pyod>=2.0,<3.0`，baseline 测试验证 |
| bqlib utils 融合量大（2615 行，但按需不全搬） | A4 前置工作先做，1.5 天挑通用工具融合完，再开始迁模型 |
| 融合时丢失 bqlib 原版优化 | baseline 测试覆盖，数值差 > 1e-6 立即失败 |
| DL 网络结构复杂、易抄错 | DL 算子逐行对比 bqlib 原版，baseline 测试覆盖 |
| 长尾算子需求突增 | 第 5 批按需迁，不预先做 |

## A13. 工作量估算

| 批次 | 算子数 | 单算子工时 | 总工时 |
|---|---|---|---|
| 前置：bqlib utils 按需融合 | - | - | 1.5 天 |
| 第 1 批 sklearn 无监督 | 20 | 0.75 天 | 15 天 |
| 第 2 批监督分类器 | 3 | 1.5 天（含协议补齐） | 4.5 天 |
| 第 3 批 DL 时序 | 10 | 2.5 天 | 25 天 |
| 第 4 批集成高级 | 10 | 1.25 天 | 12.5 天 |
| 测试 + baseline 录制 + 文档 | - | - | 8 天 |
| **合计** | **43** | - | **~66.5 天（约 3.2 个月）** |

第 5 批 51 个长尾按需再估。

**与 Adapter 模式对比**：Adapter 模式 ~43 天，源码融合模式 ~66.5 天，多 23.5 天（55%）。
**收益**：TSA-Suite 完全独立，可自由演进，不被 bqlib 版本绑定，代码风格统一。

## A14. 建议执行顺序

1. **前置**：bqlib utils 按需融合（1.5 天）- 先把通用工具融合进 TSA-Suite 现有目录
2. 第 1 批 sklearn 无监督（15 天）- 立即扩大工作流 1 算子选择
3. 第 2 批监督分类器（4.5 天）- 解锁工作流 2 真监督能力
4. 第 3 批 DL 时序（25 天）- 扩大工作流 1 时序场景
5. 第 4 批集成高级（12.5 天）- 按需

建议与计划 B（scorers 迁移）并行：B 风险低、收益快，可先做 B 再做 A，或 B 第 1-2 组与 A 前置 + 第 1 批并行。

## A15. 与计划 B 对比

| 维度 | 计划 A（deciders，融合） | 计划 B（scorers，融合） |
|---|---|---|
| 算子数 | 43（迁）+ 51（不迁） | 45 |
| 策略 | 源码融合（无 bqlib 依赖，不保留 bqlib 结构） | 源码融合（无 bqlib 依赖） |
| 前置工作 | bqlib utils 按需融合（1.5 天） | 无 |
| 工时 | ~66.5 天 | ~17 天 |
| 风险 | 中（源码融合量大，DL 易错） | 低（纯函数融合） |
| 优先级 | 高（解锁工作流 2/3） | 中（评估指标补齐） |
| 建议先后 | **先 B 后 A** | B 风险低、收益快，先做 |

**统一策略**：两个计划都是源码融合、不依赖 bqlib、不保留 bqlib 命名/结构。TSA-Suite 完全独立。
