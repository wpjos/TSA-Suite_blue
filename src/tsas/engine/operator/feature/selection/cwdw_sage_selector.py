# -*- coding: utf-8 -*-

"""CWDW + SAGE 特征优选算子的 CLI 适配包装。

将 ``feature_selection_by_cwdw_sage.py`` 中的完整流水线
SISFilter → CWDMSelector → ConvergenceFinder → 代理模型训练 → SAGE 评估
包装为 ``SupervisedFeatureSelector`` 子类，注册到 ``feature_selection`` CLI。
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tsas.engine.operator.feature.selection.base import (
    BaseFeatureSelectorConfig,
    FeatureSelectorExtraOutput,
    SupervisedFeatureSelector,
)

__all__ = [
    'CWDWSageSelectorConfig',
    'CWDWSageSelectorExtraOutput',
    'RankedFeatureItem',
    'CWDWSageSelector',
]


# 与 feature_selection_by_cwdw_sage.py 内部默认值保持一致
_SIS_THRESHOLD: float = 0.05
_CWDM_N_ITERATIONS: int = 100
_CWDM_FINAL_K_MAX: int = 150
_CWDM_N_BLOCKS: int = 5
_CWDM_DATA_THRESHOLD: int = 10_000_000
_CONVERGENCE_STABILITY_THRESHOLD: float = 1e-3
_CONVERGENCE_ACCURACY_THRESHOLD: float = 0.75
_REGRESSION_LABEL_THRESHOLD: int = 20
_SMOTE_RATIO_THRESHOLD: float = 0.35
_SMOTE_RANDOM_STATE: int = 42


class CWDWSageSelectorConfig(BaseFeatureSelectorConfig):
    """cwdw_sage 算子配置。

    Attributes:
        input_columns: 候选特征列名或列位置索引。``None`` 表示使用全部特征列。
        task: 任务类型，'Classification' | 'Regression' | 'auto'。
            为 'auto' 时根据标签唯一值数量自动判断。
        sis_threshold: SIS 独立性筛选阈值。
        cwdm_n_iterations: CWDM 迭代次数。
        cwdm_final_k: CWDM 最终保留特征数上限。
        cwdm_n_blocks: 大数据量时分块数。
        cwdm_data_threshold: 普通/分块策略阈值（样本数×特征数）。
        convergence_stability_threshold: 收敛稳定性阈值。
        convergence_accuracy_threshold: 收敛准确率阈值。
        regression_label_threshold: 自动判断回归任务时标签唯一值阈值。
        proxy_model: 分类代理模型名称。
        auto_selected_model: 是否自动选择分类代理模型（使用 LazyPredict）。
        sample_balanced: 是否对训练数据做 SMOTE 样本均衡。
        sage_batch_size: SAGE 评估批大小。
        sage_thresh: SAGE 收敛阈值。
        sage_n_jobs: SAGE 评估并行任务数。
        sage_bar: 是否显示 SAGE 进度条。
        visualization: 是否在 fit 阶段生成可视化图片。
        generate_csv: 是否在 fit 阶段生成结果 CSV。
        output_dir: 可视化图片与 CSV 输出目录。为 ``None`` 时使用当前工作目录下的 FS_Results。
        random_state: 随机种子。
    """

    task: str = Field(default='auto', description="任务类型：'Classification'、'Regression' 或 'auto'")
    sis_threshold: float = Field(default=_SIS_THRESHOLD, ge=0.0, description='SIS 筛选阈值')
    cwdm_n_iterations: int = Field(default=_CWDM_N_ITERATIONS, ge=1, description='CWDM 迭代次数')
    cwdm_final_k: int = Field(default=_CWDM_FINAL_K_MAX, ge=1, description='CWDM 最终特征数上限')
    cwdm_n_blocks: int = Field(default=_CWDM_N_BLOCKS, ge=1, description='CWDM 分块数')
    cwdm_data_threshold: int = Field(default=_CWDM_DATA_THRESHOLD, ge=1, description='CWDM 数据量阈值')
    convergence_stability_threshold: float = Field(
        default=_CONVERGENCE_STABILITY_THRESHOLD,
        ge=0.0,
        description='收敛稳定性阈值',
    )
    convergence_accuracy_threshold: float = Field(
        default=_CONVERGENCE_ACCURACY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description='收敛准确率阈值',
    )
    regression_label_threshold: int = Field(
        default=_REGRESSION_LABEL_THRESHOLD,
        ge=2,
        description='标签唯一值大于该值时视为回归',
    )
    proxy_model: str = Field(default='LGBMClassifier', description='分类代理模型名称')
    auto_selected_model: bool = Field(default=False, description='是否自动选择分类代理模型')
    sample_balanced: bool = Field(default=False, description='是否对训练数据做 SMOTE 样本均衡')
    sage_batch_size: int = Field(default=512, ge=1, description='SAGE 批大小')
    sage_thresh: float = Field(default=0.05, ge=0.0, description='SAGE 收敛阈值')
    sage_n_jobs: int = Field(default=1, ge=1, description='SAGE 并行任务数')
    sage_bar: bool = Field(default=False, description='是否显示 SAGE 进度条')
    visualization: bool = Field(default=False, description='是否在 fit 阶段生成可视化图片')
    generate_csv: bool = Field(default=False, description='是否在 fit 阶段生成结果 CSV')
    output_dir: str | None = Field(default=None, description='可视化图片与 CSV 输出目录')
    random_state: int | None = Field(default=42, description='随机种子')


class RankedFeatureItem(BaseModel):
    """按 SAGE 重要性排序后的特征信息项。"""

    feat_name: str = Field(description='特征名')
    indices: int = Field(description='在完整输入中的全局列索引')
    weight: float = Field(description='SAGE 重要性分数')


class CWDWSageSelectorExtraOutput(FeatureSelectorExtraOutput):
    """cwdw_sage 附加输出。"""

    sis_selected_indices: list[int] = Field(description='SIS 筛选后保留的候选特征局部索引')
    cwdm_selected_features: list[int] = Field(description='CWDM 选出的候选特征局部索引')
    convergence_points: dict[str, int] = Field(description='各模型收敛点')
    final_k: int = Field(description='最终保留特征数')
    task: str = Field(description='实际使用的任务类型')
    proxy_model_name: str = Field(description='代理模型名称')
    sage_values: dict[int, float] = Field(description='全局索引到 SAGE 值的映射')
    ranked_features: list[RankedFeatureItem] = Field(description='按 SAGE 值排序的特征列表')
    feature_names: list[str] = Field(description='按 SAGE 值排序的特征名')


class CWDWSageSelector(SupervisedFeatureSelector[CWDWSageSelectorExtraOutput, CWDWSageSelectorConfig, None]):
    """CWDW + SAGE 有监督特征优选算子（CLI 包装）。

    内部执行：
        1. SISFilter 独立性筛选；
        2. CWDMSelector 快速初筛；
        3. ConvergenceFinder 稳定收敛点截取；
        4. 训练分类/回归代理模型；
        5. SAGE 评估特征重要性并输出最终排序。

    属于可训练算子，需要先 ``fit(x, y)`` 再 ``run(x)``，或加载已保存模型后 ``run``。
    """

    _eo_type = CWDWSageSelectorExtraOutput
    _config_type = CWDWSageSelectorConfig
    _STATE_FILE: str = 'cwdw_sage_state.json'
    _MODEL_FILE_SKLEARN: str = 'proxy_model.joblib'
    _MODEL_FILE_XGB: str = 'proxy_model.xgb'

    def __init__(
        self,
        *,
        oid: str | None = None,
        config: CWDWSageSelectorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(oid=oid, config=config, **kwargs)
        self._selected_candidate_indices: list[int] | None = None
        self._sis_selected_indices: list[int] | None = None
        self._cwdm_selected_features: list[int] | None = None
        self._convergence_points: dict[str, int] | None = None
        self._final_k: int | None = None
        self._task: str | None = None
        self._proxy_model: object | None = None
        self._proxy_model_name: str | None = None
        self._sage_values: dict[int, float] | None = None
        self._ranked_features: list[RankedFeatureItem] | None = None
        self._feature_names_ranked: list[str] | None = None
        self._candidate_feature_names: list[str] | None = None

    @classmethod
    def name(cls) -> str:
        return 'cwdw_sage'

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 1, 0)

    def _filter_fit_data(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        params: None,
    ) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
        """按候选列筛选训练输入，并保留 DataFrame 列名供后续使用。"""
        filtered_x, filtered_y = super()._filter_fit_data(x, y, params=params)
        if isinstance(filtered_x, pd.DataFrame):
            self._candidate_feature_names = list(filtered_x.columns)
        else:
            self._candidate_feature_names = None
        return filtered_x, filtered_y

    def _fit_data(self, x: np.ndarray, y: np.ndarray, params: None) -> None:
        """训练：执行 SIS + CWDM + ConvergenceFinder + 代理模型 + SAGE。"""
        from imblearn.over_sampling import SMOTE

        from tsas.engine.operator.feature.selection.feature_selection_by_cwdw_sage import (
            ClassificationTrainer,
            ClassificationTrainerConfig,
            ConvergenceFinder,
            ConvergenceFinderConfig,
            CWDMSelector,
            CWDMSelectorConfig,
            SAGEEvaluator,
            SAGEEvaluatorConfig,
            SISFilter,
            SISFilterConfig,
            data_preprocessing,
        )

        config = self._selector_config()
        if config.random_state is not None:
            np.random.seed(config.random_state)

        task = self._resolve_task(y, config)

        # 1. 样本均衡（与原脚本 DataLoader 行为一致：在划分前对整个数据集做 SMOTE）
        data_x, data_y = x, y.ravel()
        if (
            config.sample_balanced
            and task == 'Classification'
            and len(np.unique(data_y)) >= 2
        ):
            values, counts = np.unique(data_y, return_counts=True)
            ratio = np.min(counts) / np.max(counts)
            if ratio < _SMOTE_RATIO_THRESHOLD:
                k_neighbors = max(1, np.min(counts) - 2)
                smote = SMOTE(
                    sampling_strategy='auto',
                    random_state=_SMOTE_RANDOM_STATE,
                    k_neighbors=k_neighbors,
                )
                data_x, data_y = smote.fit_resample(data_x, data_y)

        # 2. 数据集划分与标准化
        train, val, test, y_train, y_val, y_test = data_preprocessing(
            data_x,
            data_y,
            normalization_method='z-score',
            task=task,
        )

        # 3. SIS 独立性筛选
        sis = SISFilter(config=SISFilterConfig(threshold=config.sis_threshold))
        if self._candidate_feature_names is not None:
            feature_names = np.array(self._candidate_feature_names)
        else:
            feature_names = np.arange(x.shape[1]).astype(str)

        train, val, test, y_train, y_val, y_test, feature_names, sis_eo = sis.filter(
            train,
            val,
            test,
            y_train,
            y_val,
            y_test,
            feature_names,
        )

        # 4. CWDM 快速初筛
        final_k = min(config.cwdm_final_k, train.shape[1])
        cwdm = CWDMSelector(
            config=CWDMSelectorConfig(
                n_iterations=config.cwdm_n_iterations,
                k_features='auto',
                final_k=final_k,
                random_state=config.random_state,
                n_blocks=config.cwdm_n_blocks,
                data_threshold=config.cwdm_data_threshold,
            )
        )
        cwdm_eo = cwdm.select(train, y_train)
        selected_features = cwdm_eo.selected_features  # 相对于 SIS 输出

        # 5. ConvergenceFinder 收敛点截取
        train_cwdm = train[:, selected_features]
        val_cwdm = val[:, selected_features]
        test_cwdm = test[:, selected_features]
        finder = ConvergenceFinder(
            config=ConvergenceFinderConfig(
                task=task,
                stability_threshold=config.convergence_stability_threshold,
                accuracy_threshold=config.convergence_accuracy_threshold,
            )
        )
        conv_eo = finder.find(
            train_cwdm,
            val_cwdm,
            y_train,
            y_val,
            np.arange(len(selected_features)),
        )
        final_stable_count = len(conv_eo.final_stable_features)

        # 映射到候选列索引（相对于 input_columns 过滤后的候选矩阵）
        candidate_indices_convergence_order = [
            sis_eo.selected_indices[selected_features[i]]
            for i in range(final_stable_count)
        ]
        feature_names_convergence_order = [
            str(feature_names[selected_features[i]])
            for i in range(final_stable_count)
        ]

        # 6. 训练代理模型
        train_final = train_cwdm[:, conv_eo.final_stable_features]
        val_final = val_cwdm[:, conv_eo.final_stable_features]
        test_final = test_cwdm[:, conv_eo.final_stable_features]

        if task == 'Classification':
            proxy_arg = 1 if config.auto_selected_model else config.proxy_model
            trainer = ClassificationTrainer(
                config=ClassificationTrainerConfig(proxy_model=proxy_arg)
            )
            trainer_eo = trainer.train(
                train_final, val_final, test_final,
                y_train, y_val, y_test,
            )
            proxy_model = trainer_eo.best_model
            proxy_model_name = trainer_eo.best_model_name
        else:
            from tsas.engine.operator.feature.selection.feature_selection_by_cwdw_sage import (
                RegressionTrainer,
                RegressionTrainerConfig,
            )
            trainer = RegressionTrainer(config=RegressionTrainerConfig())
            trainer_eo = trainer.train(
                train_final, val_final, test_final,
                y_train, y_val, y_test,
            )
            proxy_model = trainer_eo.best_model
            proxy_model_name = trainer_eo.best_model_name

        # 7. SAGE 评估
        sage_evaluator = SAGEEvaluator(
            config=SAGEEvaluatorConfig(
                batch_size=config.sage_batch_size,
                thresh=config.sage_thresh,
                task=task,
                n_jobs=config.sage_n_jobs,
                bar=config.sage_bar,
            )
        )
        sage_train = sage_evaluator.evaluate(proxy_model, train_final, y_train).sage_values
        sage_val = sage_evaluator.evaluate(proxy_model, val_final, y_val).sage_values
        sage_test = sage_evaluator.evaluate(proxy_model, test_final, y_test).sage_values

        # 8. 按 train SAGE 值排序（与原脚本一致）
        sage_train_values = np.asarray(sage_train.values)
        importance_index_local = np.argsort(-sage_train_values)

        self._selected_candidate_indices = [
            candidate_indices_convergence_order[i]
            for i in importance_index_local
        ]
        self._feature_names_ranked = [
            feature_names_convergence_order[i]
            for i in importance_index_local
        ]
        self._sage_values = {
            self._to_global_indices([self._selected_candidate_indices[i]])[0]: float(
                sage_train_values[importance_index_local[i]]
            )
            for i in range(len(importance_index_local))
        }
        self._ranked_features = [
            RankedFeatureItem(
                feat_name=self._feature_names_ranked[i],
                indices=self._to_global_indices([self._selected_candidate_indices[i]])[0],
                weight=float(sage_train_values[importance_index_local[i]]),
            )
            for i in range(len(importance_index_local))
        ]

        # 9. 保存训练状态
        self._sis_selected_indices = sis_eo.selected_indices
        self._cwdm_selected_features = selected_features
        self._convergence_points = conv_eo.convergence_points
        self._final_k = final_stable_count
        self._task = task
        self._proxy_model = proxy_model
        self._proxy_model_name = proxy_model_name

        # 10. 可选产物输出
        if config.visualization or config.generate_csv:
            self._generate_artifacts(
                sage_train=sage_train,
                sage_val=sage_val,
                sage_test=sage_test,
                feature_importance_index=importance_index_local,
                feature_names_convergence_order=feature_names_convergence_order,
                model=proxy_model,
                best_model_name=proxy_model_name,
                x_train=train_final,
                x_val=val_final,
                x_test=test_final,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
            )

    def _run_data(
        self,
        x: np.ndarray,
        params: None,
        idx: pd.Index | None = None,
    ) -> tuple[np.ndarray, CWDWSageSelectorExtraOutput]:
        """推理：按训练得到的 SAGE 排名索引选择特征。"""
        if self._selected_candidate_indices is None:
            raise RuntimeError('CWDWSageSelector 尚未训练，请先 fit 或 load')

        selected_global = self._to_global_indices(self._selected_candidate_indices)
        eo = CWDWSageSelectorExtraOutput(
            selected_indices=selected_global,
            sis_selected_indices=list(self._sis_selected_indices or []),
            cwdm_selected_features=list(self._cwdm_selected_features or []),
            convergence_points=dict(self._convergence_points or {}),
            final_k=self._final_k or 0,
            task=self._task or 'Unknown',
            proxy_model_name=self._proxy_model_name or 'Unknown',
            sage_values=dict(self._sage_values or {}),
            ranked_features=list(self._ranked_features or []),
            feature_names=list(self._feature_names_ranked or []),
        )
        return self._select_columns(x, self._selected_candidate_indices, eo)

    def _save_fit_state(self, path: Path) -> None:
        """保存训练状态到模型目录。"""
        super()._save_fit_state(path)
        state = {
            'selected_candidate_indices': [int(i) for i in (self._selected_candidate_indices or [])],
            'sis_selected_indices': [int(i) for i in (self._sis_selected_indices or [])],
            'cwdm_selected_features': [int(i) for i in (self._cwdm_selected_features or [])],
            'convergence_points': {k: int(v) for k, v in (self._convergence_points or {}).items()},
            'final_k': int(self._final_k or 0),
            'task': self._task,
            'proxy_model_name': self._proxy_model_name,
            'feature_names_ranked': list(self._feature_names_ranked or []),
            'sage_values': {str(k): float(v) for k, v in (self._sage_values or {}).items()},
            'ranked_features': [
                item.model_dump()
                for item in (self._ranked_features or [])
            ],
        }
        (path / self._STATE_FILE).write_text(json.dumps(state), encoding='utf-8')
        self._save_proxy_model(path)

    def _load_fit_state(self, path: Path) -> None:
        """从模型目录加载训练状态。"""
        super()._load_fit_state(path)
        state = json.loads((path / self._STATE_FILE).read_text(encoding='utf-8'))
        self._selected_candidate_indices = state['selected_candidate_indices']
        self._sis_selected_indices = state['sis_selected_indices']
        self._cwdm_selected_features = state['cwdm_selected_features']
        self._convergence_points = state['convergence_points']
        self._final_k = state['final_k']
        self._task = state['task']
        self._proxy_model_name = state['proxy_model_name']
        self._feature_names_ranked = state['feature_names_ranked']
        self._sage_values = {int(k): float(v) for k, v in state['sage_values'].items()}
        self._ranked_features = [RankedFeatureItem(**item) for item in state['ranked_features']]
        self._load_proxy_model(path)
        self._fitted = True

    def _save_proxy_model(self, path: Path) -> None:
        """保存代理模型。"""
        import joblib

        if self._proxy_model is None or self._proxy_model_name is None:
            return
        if self._proxy_model_name == 'xgboost':
            self._proxy_model.save_model(str(path / self._MODEL_FILE_XGB))
        else:
            joblib.dump(self._proxy_model, path / self._MODEL_FILE_SKLEARN)

    def _load_proxy_model(self, path: Path) -> None:
        """加载代理模型。"""
        import joblib

        if self._proxy_model_name == 'xgboost':
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(str(path / self._MODEL_FILE_XGB))
            self._proxy_model = model
        else:
            self._proxy_model = joblib.load(path / self._MODEL_FILE_SKLEARN)

    def _generate_artifacts(
        self,
        *,
        sage_train: object,
        sage_val: object,
        sage_test: object,
        feature_importance_index: np.ndarray,
        feature_names_convergence_order: list[str],
        model: object,
        best_model_name: str,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """生成可视化图片与结果 CSV（副作用，仅在配置开启时调用）。"""
        from tsas.engine.operator.feature.selection.feature_selection_by_cwdw_sage import (
            fig_show_classification,
            fig_show_regression,
            plot_feature_importance_comparison,
        )

        config = self._selector_config()
        output_dir = Path(config.output_dir) if config.output_dir else Path.cwd() / 'FS_Results'
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_names = np.array(feature_names_convergence_order)

        if config.visualization:
            plot_feature_importance_comparison(
                sage_train,
                sage_val,
                sage_test,
                feature_names,
                output_dir,
                'cwdw_sage',
            )
            if self._task == 'Classification':
                fig_show_classification(
                    feature_importance_index,
                    model,
                    best_model_name,
                    x_train,
                    x_val,
                    x_test,
                    y_train,
                    y_val,
                    y_test,
                    output_dir,
                )
            elif self._task == 'Regression':
                fig_show_regression(
                    feature_importance_index,
                    model,
                    best_model_name,
                    x_train,
                    x_val,
                    x_test,
                    y_train,
                    y_val,
                    y_test,
                    output_dir,
                )

        if config.generate_csv:
            import pandas as pd

            result = pd.DataFrame(
                [
                    {
                        'feat_name': item.feat_name,
                        'indices': item.indices,
                        'weight': item.weight,
                    }
                    for item in (self._ranked_features or [])
                ]
            )
            result.to_csv(output_dir / '特征优选_result_特征重要性.csv', index=False)

    def _resolve_task(self, y: np.ndarray, config: CWDWSageSelectorConfig) -> str:
        if config.task != 'auto':
            return config.task
        return (
            'Regression'
            if len(np.unique(y)) > config.regression_label_threshold
            else 'Classification'
        )
