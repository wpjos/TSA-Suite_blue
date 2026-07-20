# -*- coding: utf-8 -*-

"""CWDW + SAGE 三步特征优选算子的 CLI 适配包装。

将 ``feature_selection_by_cwdw_sage.py`` 中的
SISFilter → CWDMSelector → ConvergenceFinder 流水线包装为
``BaseFeatureSelector`` 子类，注册到 ``feature_selection`` CLI。
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from tsas.engine.operator.feature.selection.base import (
    BaseFeatureSelector,
    BaseFeatureSelectorConfig,
    FeatureSelectorExtraOutput,
)

__all__ = [
    'CWDWSageSelectorConfig',
    'CWDWSageSelectorExtraOutput',
    'CWDWSageSelector',
]

_SIS_THRESHOLD: float = 0.05
_CWDM_N_ITERATIONS: int = 100
_CWDM_FINAL_K_MAX: int = 150
_CWDM_N_BLOCKS: int = 5
_CWDM_DATA_THRESHOLD: int = 10_000_000
_CONVERGENCE_STABILITY_THRESHOLD: float = 1e-3
_CONVERGENCE_ACCURACY_THRESHOLD: float = 0.75
_REGRESSION_LABEL_THRESHOLD: int = 20


class CWDWSageSelectorConfig(BaseFeatureSelectorConfig):
    """cwdw_sage 算子配置。

    Attributes:
        label_column: 输入数据中标签列的列名（DataFrame）或列索引（ndarray）。
            该列仅用于内部训练/选择，不会出现在输出特征中。
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
    """

    label_column: str | int = Field(default='label', description='标签列名或列索引')
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


class CWDWSageSelectorExtraOutput(FeatureSelectorExtraOutput):
    """cwdw_sage 附加输出。"""

    sis_selected_indices: list[int] = Field(description='SIS 筛选后保留的候选特征局部索引')
    cwdm_selected_features: list[int] = Field(description='CWDM 选出的候选特征局部索引')
    convergence_points: dict[str, int] = Field(description='各模型收敛点')
    final_k: int = Field(description='最终保留特征数')
    task: str = Field(description='实际使用的任务类型')


class CWDWSageSelector(BaseFeatureSelector[CWDWSageSelectorExtraOutput, CWDWSageSelectorConfig]):
    """CWDW + SAGE 特征优选算子（CLI 包装）。

    内部执行：
        1. SISFilter 独立性筛选；
        2. CWDMSelector 快速初筛；
        3. ConvergenceFinder 稳定收敛点截取。

    Input:
        包含标签列的数值 DataFrame 或 ndarray，形状 (n_samples, n_features+1)

    Output:
        选择后的特征矩阵，形状 (n_samples, n_selected)
    """

    _eo_type = CWDWSageSelectorExtraOutput
    _config_type = CWDWSageSelectorConfig

    def __init__(
        self,
        *,
        oid: str | None = None,
        config: CWDWSageSelectorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(oid=oid, config=config, **kwargs)
        self._y: np.ndarray | None = None

    @classmethod
    def name(cls) -> str:
        return 'cwdw_sage'

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _filter_data(self, x, params=None):
        """剔除标签列，缓存候选列全局索引与标签数组。

        复用 ``BaseFeatureSelectorMixin._resolve_candidate_indices`` 解析
        ``input_columns``，再把标签列从候选列中排除。
        """
        config = self._selector_config()
        label_col = config.label_column

        all_indices = self._resolve_candidate_indices(x)

        # 定位标签列位置
        if isinstance(x, pd.DataFrame):
            if isinstance(label_col, str):
                if label_col not in x.columns:
                    raise ValueError(f"输入数据缺少标签列 '{label_col}'")
                label_pos = list(x.columns).index(label_col)
            else:
                if label_col < 0 or label_col >= x.shape[1]:
                    raise ValueError(f"标签列索引 {label_col} 越界")
                label_pos = label_col
            self._y = x.iloc[:, label_pos].to_numpy()
        else:
            if not isinstance(label_col, int):
                raise TypeError("ndarray 输入时 label_column 必须为整数列索引")
            if label_col < 0 or label_col >= x.shape[1]:
                raise ValueError(f"标签列索引 {label_col} 越界")
            label_pos = label_col
            self._y = x[:, label_pos]

        candidate_indices = [i for i in all_indices if i != label_pos]
        if not candidate_indices:
            raise ValueError("剔除标签列后没有可用的候选特征")

        self._candidate_indices = candidate_indices

        if isinstance(x, pd.DataFrame):
            return x.iloc[:, candidate_indices]
        return x[:, candidate_indices]

    def _run_data(
        self,
        x: np.ndarray,
        params: None,
        idx: pd.Index | None = None,
    ) -> tuple[np.ndarray, CWDWSageSelectorExtraOutput]:
        if self._y is None:
            raise RuntimeError('标签数据未解析，请先通过 run() 传入包含标签列的完整数据')

        config = self._selector_config()
        task = self._resolve_task(self._y, config)

        # 延迟导入 heavy 依赖，避免 CLI 启动/扫描时加载
        from tsas.engine.operator.feature.selection.feature_selection_by_cwdw_sage import (
            SISFilter,
            SISFilterConfig,
            CWDMSelector,
            CWDMSelectorConfig,
            ConvergenceFinder,
            ConvergenceFinderConfig,
            data_preprocessing,
        )

        # 1. 数据集划分与标准化（仅用于内部选择）
        train, val, test, y_train, y_val, y_test = data_preprocessing(
            x, self._y, normalization_method='z-score', task=task
        )

        # 2. SIS 独立性筛选
        sis = SISFilter(config=SISFilterConfig(threshold=config.sis_threshold))
        feature_names = np.arange(x.shape[1]).astype(str)
        train, val, test, y_train, y_val, y_test, feature_names, sis_eo = sis.filter(
            train, val, test, y_train, y_val, y_test, feature_names
        )

        # 3. CWDM 快速初筛
        final_k = min(config.cwdm_final_k, train.shape[1])
        cwdm = CWDMSelector(
            config=CWDMSelectorConfig(
                n_iterations=config.cwdm_n_iterations,
                k_features='auto',
                final_k=final_k,
                random_state=None,
                n_blocks=config.cwdm_n_blocks,
                data_threshold=config.cwdm_data_threshold,
            )
        )
        cwdm_eo = cwdm.select(train, y_train)
        selected_features = cwdm_eo.selected_features  # 相对于 SIS 输出

        # 4. ConvergenceFinder 收敛点截取
        train_cwdm = train[:, selected_features]
        val_cwdm = val[:, selected_features]
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

        # 5. 映射回候选矩阵索引
        final_candidate_indices = [selected_features[i] for i in range(final_stable_count)]
        selected_global = self._to_global_indices(final_candidate_indices)

        eo = CWDWSageSelectorExtraOutput(
            selected_indices=selected_global,
            sis_selected_indices=sis_eo.selected_indices,
            cwdm_selected_features=selected_features,
            convergence_points=conv_eo.convergence_points,
            final_k=final_stable_count,
            task=task,
        )
        return self._select_columns(x, final_candidate_indices, eo)

    def _resolve_task(self, y: np.ndarray, config: CWDWSageSelectorConfig) -> str:
        if config.task != 'auto':
            return config.task
        return (
            'Regression'
            if len(np.unique(y)) > config.regression_label_threshold
            else 'Classification'
        )
