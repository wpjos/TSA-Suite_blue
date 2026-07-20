# -*- coding: utf-8 -*-

"""特征优选工具包（单文件版）：基于 SAGE + CWDM 的特征选择流水线，支持分类和回归任务。

本文件合并了 config / utils / data_loader / feature_screening / model_training /
sage_evaluator / visualization / main 的全部代码，可直接独立运行。
"""

__all__ = [
    'SIS_THRESHOLD',
    'CWDM_N_ITERATIONS',
    'CWDM_FINAL_K_MAX',
    'CWDM_N_BLOCKS',
    'CWDM_DATA_THRESHOLD',
    'CONVERGENCE_STABILITY_THRESHOLD',
    'CONVERGENCE_ACCURACY_THRESHOLD',
    'CONVERGENCE_STEP',
    'CONVERGENCE_WINDOW',
    'CONVERGENCE_FEAT_UPPER_BOUND',
    'REGRESSION_LABEL_THRESHOLD',
    'SMOTE_RATIO_THRESHOLD',
    'SMOTE_RANDOM_STATE',
    'TEST_SIZE',
    'VAL_SIZE',
    'RANDOM_STATE',
    'OPTUNA_N_TRIALS',
    'SAGE_IMPUTER_SAMPLES',
    'FIG_DPI',
    'FIG_FEATURE_IMPORTANCE_SIZE',
    'FIG_PERFORMANCE_CLASSIFICATION_SIZE',
    'FONT_SIZE',
    'FEATURE_TICK_THRESHOLD',
    'OUTPUT_DIR_NAME',
    'DEFAULT_FONT_PATH',
    'DEFAULT_FONT_FAMILY',
    'OPENBLAS_NUM_THREADS',
    'PipelineConfig',
    'recursive_update',
    'setup_environment',
    'setup_matplotlib',
    'DualOutput',
    'setup_dual_logging',
    'OUTPUT_DIR_NAME',
    'ensure_output_dir',
    'DataLoaderConfig',
    'DataLoaderExtraOutput',
    'DataLoader',
    'SISFilterConfig',
    'SISFilterExtraOutput',
    'SISFilter',
    'CWDMSelectorConfig',
    'CWDMSelectorExtraOutput',
    'CWDMSelector',
    'ConvergenceFinderConfig',
    'ConvergenceFinderExtraOutput',
    'ConvergenceFinder',
    'ClassificationTrainerConfig',
    'ClassificationTrainerExtraOutput',
    'ClassificationTrainer',
    'RegressionTrainerConfig',
    'RegressionTrainerExtraOutput',
    'RegressionTrainer',
    'SAGEEvaluatorConfig',
    'SAGEEvaluatorExtraOutput',
    'SAGEEvaluator',
    'plot_feature_importance_comparison',
    'fig_show_classification',
    'fig_show_regression',
    'parse_args',
    'run_pipeline',
    'main',
]

# ---------------------------------------------------------------------------
# 标准库导入
# ---------------------------------------------------------------------------
import copy
import os
import sys
from pathlib import Path
from collections import Counter
import time
import argparse
from multiprocessing import freeze_support

# ---------------------------------------------------------------------------
# 第三方库导入
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field
from loguru import logger
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pydantic import Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.svm import SVR
import optuna
import xgboost as xgb
from lightgbm import LGBMClassifier
from lazypredict.Supervised import LazyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sage
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)
import joblib



# ===========================================================================
# config
# ===========================================================================

SIS_THRESHOLD: float = 0.05

# CWDM 特征选择参数
CWDM_N_ITERATIONS: int = 100
CWDM_FINAL_K_MAX: int = 150
CWDM_N_BLOCKS: int = 5
CWDM_DATA_THRESHOLD: int = 10_000_000

# 收敛判断参数
CONVERGENCE_STABILITY_THRESHOLD: float = 1e-3
CONVERGENCE_ACCURACY_THRESHOLD: float = 0.75
CONVERGENCE_STEP: int = 5
CONVERGENCE_WINDOW: int = 5
CONVERGENCE_FEAT_UPPER_BOUND: int = 150

# 任务类型自动判断阈值：label 唯一值 > 此值则视为回归
REGRESSION_LABEL_THRESHOLD: int = 20

# 样本均衡参数
SMOTE_RATIO_THRESHOLD: float = 0.35
SMOTE_RANDOM_STATE: int = 42

# 数据划分参数
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.7
RANDOM_STATE: int = 42

# Optuna 超参搜索试验次数
OPTUNA_N_TRIALS: int = 50

# SAGE 评估默认样本数
SAGE_IMPUTER_SAMPLES: int = 512

# 可视化参数
FIG_DPI: int = 300
FIG_FEATURE_IMPORTANCE_SIZE: tuple[int, int] = (9, 6)
FIG_PERFORMANCE_CLASSIFICATION_SIZE: tuple[int, int] = (20, 10)
FONT_SIZE: int = 18
FEATURE_TICK_THRESHOLD: int = 30

# 输出目录名
OUTPUT_DIR_NAME: str = 'FS_Results'

# 中文字体
DEFAULT_FONT_PATH: str = 'C:/Windows/Fonts/simhei.ttf'
DEFAULT_FONT_FAMILY: str = 'SimHei'

# OpenBLAS 线程数
OPENBLAS_NUM_THREADS: str = '8'


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """特征优选流水线的完整配置。

    Attributes:
        task (str): 任务类型，可选 'Classification' | 'Regression'。
        proxy_model (str): 代理模型名称。
        auto_selected_model (bool): 是否自动选择模型。
        sample_balanced (bool): 是否对不均衡样本进行 SMOTE 过采样。
        batch_size (int): SAGE 并行处理的样本数量。
        thresh (float): SAGE 收敛阈值。
        n_jobs (int): 并行任务数。
        bar (bool): 是否显示进度条。
        cwdm (bool): 是否启用 CWDM 快速初筛。
        dataset (str): 数据集文件夹路径。
        output (str): 输出根目录。
        filename (str): 指定单个文件名，为空则扫描整个文件夹。
    """

    # 任务与模型
    task: str = Field(default='Classification', description='任务类型')
    proxy_model: str = Field(default='LGBMClassifier', description='代理模型名称')
    auto_selected_model: bool = Field(default=False, description='是否自动选择模型')
    sample_balanced: bool = Field(default=False, description='是否样本均衡')

    # SAGE 参数
    batch_size: int = Field(default=512, ge=1, description='SAGE 批大小')
    thresh: float = Field(default=0.05, ge=0.0, description='SAGE 收敛阈值')
    n_jobs: int = Field(default=8, ge=1, description='并行任务数')
    bar: bool = Field(default=False, description='是否显示进度条')

    # CWDM 开关
    cwdm: bool = Field(default=True, description='是否启用 CWDM')

    # 路径
    dataset: str = Field(default='dataset_labeled', description='数据集文件夹路径')
    output: str = Field(default='.', description='输出根目录')
    filename: str = Field(default='', description='指定单个文件名')

    model_config = {'frozen': False}

    def to_dict(self) -> dict:
        """转换为字典。

        Returns:
            dict: 配置字典。
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> 'PipelineConfig':
        """从字典创建配置，忽略未知键。

        Args:
            d (dict): 原始字典。

        Returns:
            PipelineConfig: 配置实例。
        """
        valid_keys = set(cls.model_fields.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def update(self, overrides: dict) -> 'PipelineConfig':
        """返回一个应用了覆盖值的新配置实例。

        Args:
            overrides (dict): 待覆盖的键值字典。

        Returns:
            PipelineConfig: 新配置实例。
        """
        new_dict = recursive_update(self.to_dict(), overrides)
        return PipelineConfig.from_dict(new_dict)


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def recursive_update(old: dict, new: dict) -> dict:
    """递归合并字典，new 中的值覆盖 old 中的同名键。

    Args:
        old (dict): 原始字典。
        new (dict): 待合并的覆盖字典。

    Returns:
        dict: 合并后的新字典（深拷贝，不修改原始字典）。
    """
    result = copy.deepcopy(old)
    if not isinstance(new, dict):
        return old

    for k, v in new.items():
        if isinstance(v, dict):
            result[k] = recursive_update(old.get(k, {}), v)
        else:
            result[k] = v

    return result



# ===========================================================================
# utils
# ===========================================================================

def setup_environment() -> None:
    """设置运行环境：OpenBLAS 线程数、递归限制。

    Returns:
        None: 本方法无返回值。
    """
    os.environ['OPENBLAS_NUM_THREADS'] = OPENBLAS_NUM_THREADS
    sys.setrecursionlimit(sys.getrecursionlimit() * 5)
    logger.info('utils 环境初始化完成')


def setup_matplotlib() -> None:
    """配置 matplotlib 中文字体和非交互式后端。

    Returns:
        None: 本方法无返回值。
    """
    import matplotlib
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
    plt.ion()

    matplotlib.rcParams['font.sans-serif'] = [DEFAULT_FONT_FAMILY]
    matplotlib.rcParams['axes.unicode_minus'] = False

    font_path = Path(DEFAULT_FONT_PATH)
    if font_path.exists():
        prop = fm.FontProperties(fname=str(font_path))
        plt.rcParams['font.family'] = prop.get_name()

    logger.info('utils matplotlib 配置完成')


# ---------------------------------------------------------------------------
# 日志双写
# ---------------------------------------------------------------------------

class DualOutput:
    """同时输出到控制台和文件的 stdout 替代类。

    用法::

        sys.stdout = DualOutput('log.txt')
    """

    def __init__(self, filename: str) -> None:
        """初始化双写输出。

        Args:
            filename (str): 日志文件路径。
        """
        self.file = open(filename, 'a', encoding='utf-8')
        self.console = sys.stdout

    def write(self, message: str) -> None:
        """同时写入控制台和文件。

        Args:
            message (str): 待写入的消息。

        Returns:
            None: 本方法无返回值。
        """
        self.console.write(message)
        self.file.write(message)

    def flush(self) -> None:
        """刷新两个输出流。

        Returns:
            None: 本方法无返回值。
        """
        self.console.flush()
        self.file.flush()

    def close(self) -> None:
        """关闭文件流并恢复 stdout。

        Returns:
            None: 本方法无返回值。
        """
        if sys.stdout is self:
            sys.stdout = self.console
        self.file.close()


def setup_dual_logging(log_path: str = 'log.txt') -> DualOutput:
    """设置双写日志，返回 DualOutput 实例以便后续恢复。

    Args:
        log_path (str): 日志文件路径。

    Returns:
        DualOutput: 双写输出实例。
    """
    dual = DualOutput(log_path)
    sys.stdout = dual
    logger.info('utils 双写日志已启动')
    return dual


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------

OUTPUT_DIR_NAME = 'FS_Results'


def ensure_output_dir(base_dir: str, file_stem: str) -> Path:
    """创建并返回输出目录路径。

    Args:
        base_dir (str): 输出根目录。
        file_stem (str): 数据文件名（不含扩展名），用于创建子目录。

    Returns:
        Path: 输出目录的 Path 对象。
    """
    out_path = Path(base_dir) / OUTPUT_DIR_NAME / file_stem
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f'utils 输出目录已创建: {out_path}')
    return out_path



# ===========================================================================
# data_loader
# ===========================================================================

_TIME_KEYWORDS = ['时间', 'time', 'date', '日期', 'day', 'month', 'year', 'Time', 'stamp']


# ---------------------------------------------------------------------------
# DataLoader 三件套
# ---------------------------------------------------------------------------

class DataLoaderConfig:
    """数据加载器实例参数。

    Attributes:
        dataset_folder (str): 数据集文件夹路径。
        filename (str): 指定单个文件名，为空则扫描整个文件夹。
        sample_balanced (bool): 是否对不均衡样本进行 SMOTE 过采样。
        cwdm (bool): 是否启用 CWDM（预留接口）。
    """

    def __init__(
        self,
        dataset_folder: str = './dataset_labeled',
        filename: str = '',
        sample_balanced: bool = False,
        cwdm: bool = False,
    ) -> None:
        self.dataset_folder = dataset_folder
        self.filename = filename
        self.sample_balanced = sample_balanced
        self.cwdm = cwdm


class DataLoaderExtraOutput:
    """数据加载器附加输出。

    Attributes:
        data_x_list (list[np.ndarray]): 特征矩阵列表。
        data_y_list (list[np.ndarray]): 标签数组列表。
        feature_name_list (list[np.ndarray]): 特征名列表的列表。
        files (list[str]): 有效文件名列表。
    """

    def __init__(
        self,
        data_x_list: list[np.ndarray],
        data_y_list: list[np.ndarray],
        feature_name_list: list[np.ndarray],
        files: list[str],
    ) -> None:
        self.data_x_list = data_x_list
        self.data_y_list = data_y_list
        self.feature_name_list = feature_name_list
        self.files = files


class DataLoader:
    """从文件夹加载 CSV 数据文件，执行清洗与样本均衡。"""

    def __init__(self, *, config: DataLoaderConfig | None = None, **kwargs) -> None:
        """初始化数据加载器。

        Args:
            config (DataLoaderConfig | None): 加载器配置。
            **kwargs: 透传给配置的键值参数。
        """
        if config is None:
            config = DataLoaderConfig(**kwargs)
        self._config = config

    @classmethod
    def name(cls) -> str:
        """返回算子注册名称。

        Returns:
            str: 算子名称。
        """
        return 'data_loader'

    def load(self) -> DataLoaderExtraOutput:
        """从文件夹加载所有 CSV 数据文件。

        Returns:
            DataLoaderExtraOutput: 加载后的数据与附加输出。
        """
        config = self._config

        if config.filename:
            files = [config.filename]
        else:
            files = [f for f in os.listdir(config.dataset_folder) if f.endswith('.csv')]

        files_ori = files.copy()
        data_x_list: list[np.ndarray] = []
        data_y_list: list[np.ndarray] = []
        feature_name_list: list[np.ndarray] = []

        for file in files:
            file_path = os.path.join(config.dataset_folder, file)
            df = pd.read_csv(file_path)

            logger.info(f'DataLoader 读取文件: {file}')
            logger.info(f'DataLoader 数据类型:\n{df.dtypes}')
            logger.info(f'DataLoader 数据形状: {df.shape}')

            # 将 object 列尝试转为数值
            df = df.apply(
                lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col,
            )

            # 去除时间列和非数值列
            time_columns, non_numeric_columns = _identify_columns(df)
            df = df.drop(columns=time_columns, errors='ignore')
            df = df.drop(columns=non_numeric_columns, errors='ignore')

            # 检查 label 列
            if 'label' not in df.columns:
                logger.warning(f"DataLoader '{file}' 数据集不包含 'label' 列，跳过")
                files_ori.remove(file)
                continue

            if len(set(df['label'])) < 2:
                logger.warning(f"DataLoader '{file}' label 标签类型少于 2 类，跳过")
                files_ori.remove(file)
                continue

            # 清洗
            df = df.dropna(axis=1, how='all')
            df = df.dropna(axis=0, how='any')
            df = df.drop(columns=df.columns[df.std() <= 0])

            # 提取标签
            data_y = df['label'].to_numpy().copy()
            data_y[data_y == -1] = 0

            if config.cwdm:
                logger.info('DataLoader CWDM 预处理（预留）')

            df = df.drop(columns=['label'], errors='ignore')
            feature_names = df.keys()
            data_x = df.to_numpy()

            # 样本均衡
            if data_y.sum() == 0 or data_y.sum() == data_y.size:
                raise ValueError('标签全为同一值，无法进行特征选择')

            values, counts = np.unique(data_y, return_counts=True)
            n_label_min = np.min(counts)
            n_label_max = np.max(counts)
            ratio = n_label_min / n_label_max

            if ratio < SMOTE_RATIO_THRESHOLD and config.sample_balanced:
                if (np.min(counts) - 2) < 1:
                    logger.warning('DataLoader 每个 label 的样本数不少于 3 个')

                logger.info(f'DataLoader 原始样本数: {data_x.shape}')
                smote = SMOTE(
                    sampling_strategy='auto',
                    random_state=SMOTE_RANDOM_STATE,
                    k_neighbors=np.min(counts) - 2,
                )
                data_x, data_y = smote.fit_resample(data_x, data_y)
                logger.info('DataLoader 正负样本不均衡，已进行样本均衡操作')
                logger.info(f'DataLoader 平衡后样本数: {data_x.shape}')

            data_x_list.append(data_x)
            data_y_list.append(data_y)
            feature_name_list.append(feature_names)

        eo = DataLoaderExtraOutput(
            data_x_list=data_x_list,
            data_y_list=data_y_list,
            feature_name_list=feature_name_list,
            files=files_ori,
        )
        return eo
# ---------------------------------------------------------------------------
# 列识别（内部函数）
# ---------------------------------------------------------------------------

def _identify_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """识别 CSV 数据中的时间列和非数值列。

    Args:
        df (pd.DataFrame): 输入 DataFrame。

    Returns:
        tuple[list[str], list[str]]: (time_columns, non_numeric_columns) 时间列名列表和非数值列名列表。
    """
    non_numeric_columns = [
        col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
    ]

    time_columns = []
    for col in non_numeric_columns:
        if any(kw.lower() in col.lower() for kw in _TIME_KEYWORDS):
            time_columns.append(col)

    return time_columns, non_numeric_columns


# ---------------------------------------------------------------------------
# 数据预处理（划分 + 标准化）
# ---------------------------------------------------------------------------

def data_preprocessing(
    data_x: np.ndarray,
    data_y: np.ndarray,
    normalization_method: str = 'z-score',
    task: str = 'Classification',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """数据集划分与标准化。

    train 集用于选变量，test 集用于评估选出的变量好坏。其中在 SAGE 中，
    train 会被再次分为同大小的两个子集：一个用于训练 proxy，一个用于计算 sage value；
    在评估选出的变量中，test 会被分为同大小的两个子集。

    Args:
        data_x (np.ndarray): 样本-特征矩阵 (n_samples, n_features)。
        data_y (np.ndarray): 标签数组 (n_samples,)。
        normalization_method (str): 标准化方法，可选 'z-score' | 'min-max' | 'max-abs'。
        task (str): 任务类型，可选 'Classification' | 'Regression'。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (x_train, x_val, x_test, y_train, y_val, y_test)。
    """
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=False,
    )

    # 基于 train 集进行标准化
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    logger.info(f'DataLoader 数据划分完成: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}')

    return x_train, x_val, x_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# SISFilter 三件套
# ---------------------------------------------------------------------------

class SISFilterConfig:
    """独立性筛选器实例参数。

    Attributes:
        threshold (float): 筛选阈值，低于此值的特征将被剔除。
    """

    def __init__(self, threshold: float = SIS_THRESHOLD) -> None:
        self.threshold = threshold


class SISFilterExtraOutput:
    """独立性筛选器附加输出。

    Attributes:
        selected_indices (list[int]): 保留特征的全局索引。
        sis_values (list[float]): 各特征的 SIS 统计值。
    """

    def __init__(
        self,
        selected_indices: list[int],
        sis_values: list[float],
    ) -> None:
        self.selected_indices = selected_indices
        self.sis_values = sis_values


class SISFilter:
    """基于独立性筛选去除确定不相关的特征。

    仅在二分类任务时启用（train/val/test 的 label 均恰好有两类）。
    """

    def __init__(self, *, config: SISFilterConfig | None = None, **kwargs) -> None:
        """初始化独立性筛选器。

        Args:
            config (SISFilterConfig | None): 筛选器配置。
            **kwargs: 透传给配置的键值参数。
        """
        if config is None:
            config = SISFilterConfig(**kwargs)
        self._config = config

    @classmethod
    def name(cls) -> str:
        """返回算子注册名称。

        Returns:
            str: 算子名称。
        """
        return 'sis_filter'

    def filter(
        self,
        train: np.ndarray,
        val: np.ndarray,
        test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, SISFilterExtraOutput]:
        """基于独立性筛选去除确定不相关的特征。

        Args:
            train (np.ndarray): 训练集特征。
            val (np.ndarray): 验证集特征。
            test (np.ndarray): 测试集特征。
            y_train (np.ndarray): 训练集标签。
            y_val (np.ndarray): 验证集标签。
            y_test (np.ndarray): 测试集标签。
            feature_names (np.ndarray): 特征名数组。

        Returns:
            tuple: 筛选后的 (train, val, test, y_train, y_val, y_test, feature_names, eo)。
        """
        is_binary = (
            len(np.unique(y_train)) == 2
            and len(np.unique(y_val)) == 2
            and len(np.unique(y_test)) == 2
        )

        if is_binary:
            combined_x = np.vstack([train, val])
            combined_y = np.concatenate([y_train, y_val])

            dc_value = _dc_sis(combined_x, combined_y)
            mv_value = _mv_sis(combined_x, combined_y)
            ks_value = _ks_sis(combined_x, combined_y)

            sis_matrix = np.stack([dc_value, mv_value, ks_value])
            sis_matrix = np.max(sis_matrix, axis=0)

            selected = np.where(sis_matrix > self._config.threshold)[0]
            unselected = np.where(sis_matrix <= self._config.threshold)[0]

            if len(unselected) > 0:
                unselected_names = feature_names[unselected]
                logger.info(
                    f"SISFilter 由于相关性低于 {self._config.threshold}，"
                    f"以下特征会被剔除: {unselected_names}",
                )

            train = train[:, selected]
            val = val[:, selected]
            test = test[:, selected]
            feature_names = feature_names[selected]

            eo = SISFilterExtraOutput(
                selected_indices=selected.tolist(),
                sis_values=sis_matrix.tolist(),
            )
        else:
            eo = SISFilterExtraOutput(
                selected_indices=list(range(train.shape[1])),
                sis_values=[0.0] * train.shape[1],
            )
            logger.info('SISFilter 非二分类任务，跳过独立性筛选')

        return train, val, test, y_train, y_val, y_test, feature_names, eo


# ---------------------------------------------------------------------------
# SIS 内部方法
# ---------------------------------------------------------------------------

def _dc_sis(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """基于距离相关的独立性筛选 (DC-SIS)。

    Args:
        X (np.ndarray): 特征矩阵 (n, p)。
        Y (np.ndarray): 响应变量 (n,)。

    Returns:
        np.ndarray: 每个特征的距离相关值 (p,)。
    """
    import dcor

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
        Y = Y.to_numpy()

    n, p = X.shape
    dcor_value = np.zeros(p)

    for i in range(p):
        dcor_value[i] = dcor.distance_correlation(X[:, i], Y)

    return dcor_value


def _ks_sis(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """基于 Kolmogorov-Smirnov 统计量的独立性筛选 (KS-SIS)。

    Args:
        X (np.ndarray): 特征矩阵 (n, p)。
        Y (np.ndarray): 响应变量 (n,)。

    Returns:
        np.ndarray: 每个特征的 KS 统计值 (p,)。
    """
    from statsmodels.distributions.empirical_distribution import ECDF

    n_points = 1000
    n, p = X.shape
    ks_value = np.zeros(p)
    zero_positions = np.where(Y == 0)[0]
    one_positions = np.where(Y == 1)[0]

    for i in range(p):
        x_max = np.max(X[:, i])
        x_min = np.min(X[:, i])
        points = np.linspace(x_min, x_max, n_points)

        ecdf0 = ECDF(X[zero_positions, i])
        ecdf0_values = ecdf0(points)

        ecdf1 = ECDF(X[one_positions, i])
        ecdf1_values = ecdf1(points)

        ks_value[i] = np.max(np.abs(ecdf1_values - ecdf0_values))

    return ks_value


def _mv_sis(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """基于 Kolmogorov-Smirnov 差异的独立性筛选 (MV-SIS)。

    Args:
        X (np.ndarray): 特征矩阵 (n, p)。
        Y (np.ndarray): 响应变量 (n,)。

    Returns:
        np.ndarray: 每个特征的 MV 统计值 (p,)。
    """
    from statsmodels.distributions.empirical_distribution import ECDF

    n, p = X.shape
    mv_value = np.zeros(p)
    zero_positions = np.where(Y == 0)[0]
    p0_hat = zero_positions.shape[0] / n
    one_positions = np.where(Y == 1)[0]
    p1_hat = one_positions.shape[0] / n

    for i in range(p):
        ecdf_all = ECDF(X[:, i])
        x_zero = X[zero_positions, i]
        x_one = X[one_positions, i]

        ecdf0 = ECDF(x_zero)
        part0 = (ecdf0(x_zero) / p0_hat - ecdf_all(x_zero)) ** 2

        ecdf1 = ECDF(x_one)
        part1 = (ecdf1(x_one) / p1_hat - ecdf_all(x_one)) ** 2

        mv_value[i] = (sum(part0) * p0_hat + sum(part1) * p1_hat) / n

    return mv_value


# ---------------------------------------------------------------------------
# 向后兼容的函数式接口
# ---------------------------------------------------------------------------

def load_data_from_folder(
    dataset_folder: str = './dataset_labeled',
    filename: str = '',
    sample_balanced: bool = False,
    cwdm: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    """从文件夹加载所有 CSV 数据文件。

    Args:
        dataset_folder (str): 数据集文件夹路径。
        filename (str): 指定单个文件名，为空则扫描整个文件夹。
        sample_balanced (bool): 是否对不均衡样本进行 SMOTE 过采样。
        cwdm (bool): 是否启用 CWDM（预留接口，当前无实际操作）。

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
            (data_x_list, data_y_list, feature_name_list, files)。
    """
    config = DataLoaderConfig(
        dataset_folder=dataset_folder,
        filename=filename,
        sample_balanced=sample_balanced,
        cwdm=cwdm,
    )
    loader = DataLoader(config=config)
    eo = loader.load()
    return eo.data_x_list, eo.data_y_list, eo.feature_name_list, eo.files


def sure_independence_screening(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    feature_names: np.ndarray,
    threshold: float = SIS_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """基于独立性筛选去除确定不相关的特征。

    仅在二分类任务时启用（train/val/test 的 label 均恰好有两类）。

    Args:
        train (np.ndarray): 训练集特征。
        val (np.ndarray): 验证集特征。
        test (np.ndarray): 测试集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        y_test (np.ndarray): 测试集标签。
        feature_names (np.ndarray): 特征名数组。
        threshold (float): 筛选阈值，低于此值的特征将被剔除。

    Returns:
        tuple: 筛选后的 (train, val, test, y_train, y_val, y_test, feature_names)。
    """
    config = SISFilterConfig(threshold=threshold)
    sis_filter = SISFilter(config=config)
    result = sis_filter.filter(train, val, test, y_train, y_val, y_test, feature_names)
    return result[:7]



# ===========================================================================
# feature_screening
# ===========================================================================

def _chol_it(
    Li: np.ndarray, Ri: np.ndarray, betai: np.ndarray, bi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cholesky 分解的迭代更新。

    Args:
        Li (np.ndarray): 当前下三角矩阵 (s, s)。
        Ri (np.ndarray): 当前上三角逆矩阵 (s, s)。
        betai (np.ndarray): 新增列与已选列的协方差 (s, 1)。
        bi (float): 新增列的方差。

    Returns:
        tuple[np.ndarray, np.ndarray]: (new_L, new_R) 更新后的 (s+1, s+1) 矩阵。
    """
    s = Li.shape[0]
    alphai = Ri.T @ betai
    alpha_squared = float((alphai.T @ alphai).ravel()[0])

    if bi - alpha_squared > 1e-6:
        li = np.sqrt(bi - alpha_squared)
        ri = 1 / li
    else:
        li = 0
        ri = 1e-8

    gammai = -ri * alphai

    new_L = np.zeros((s + 1, s + 1))
    new_L[:s, :s] = Li
    new_L[:s, s] = 0
    new_L[s, :s] = alphai.ravel()
    new_L[s, s] = li

    new_R = np.zeros((s + 1, s + 1))
    new_R[:s, :s] = Ri
    new_R[:s, s] = 0
    new_R[s, :s] = gammai.ravel()
    new_R[s, s] = ri

    return new_L, new_R
# ---------------------------------------------------------------------------
# CWDM 核心
# ---------------------------------------------------------------------------

def _cwdm(X: np.ndarray, Y: np.ndarray, n: int) -> np.ndarray:
    """基于相关系数矩阵的 CWDM 特征选择。

    Args:
        X (np.ndarray): 特征矩阵 (n_samples, n_features)。
        Y (np.ndarray): 响应变量 (n_samples,)。
        n (int): 选择特征数量。

    Returns:
        np.ndarray: 选中的特征索引数组 (n,)。
    """
    n_features = X.shape[1]
    B = np.corrcoef(X, rowvar=False)
    a = X.T @ Y / (Y.T @ Y)

    indexi = np.zeros(n, dtype=int)
    di = np.zeros(n)
    d_val = (a ** 2) / np.diag(B)
    idx = np.argmax(d_val)
    indexi[0] = idx
    di[0] = d_val[idx]

    Li = np.array([[np.sqrt(B[idx, idx])]])
    Ri = np.array([[1 / np.sqrt(B[idx, idx])]])
    ai = a[idx].reshape(1, 1)

    for i in range(1, n):
        id_left = [j for j in range(n_features) if j not in indexi[:i]]

        dj = np.zeros(n_features)
        for j in id_left:
            betai = B[indexi[:i], j].reshape(-1, 1)
            bj = B[j, j]

            Lj, Rj = _chol_it(Li, Ri, betai, bj)
            aj = np.concatenate([ai, a[j].reshape(1, 1)], axis=0)
            Rj_last_row = Rj[-1, :]
            dj[j] = float((Rj_last_row @ aj).ravel()[0]) ** 2

        idx = np.argmax(dj)
        betai = B[indexi[:i], idx].reshape(-1, 1)
        Li, Ri = _chol_it(Li, Ri, betai, B[idx, idx])
        ai = np.concatenate([ai, a[idx].reshape(1, 1)], axis=0)
        di[i] = di[i - 1] + dj[idx]
        indexi[i] = idx

    return indexi


def _cwdm_quick(X: np.ndarray, Y: np.ndarray, n: int) -> np.ndarray:
    """快速版 CWDM，使用列范数代替相关系数矩阵。

    Args:
        X (np.ndarray): 特征矩阵 (n_samples, n_features)。
        Y (np.ndarray): 响应变量 (n_samples,)。
        n (int): 选择特征数量。

    Returns:
        np.ndarray: 选中的特征索引数组 (n,)。
    """
    n_features = X.shape[1]
    a = X.T @ Y / (Y.T @ Y)
    col_norms = np.linalg.norm(X, axis=0) ** 2 / (Y.T @ Y)

    indexi = np.zeros(n, dtype=int)
    di = np.zeros(n)
    d_val = (a ** 2) / col_norms
    idx = np.argmax(d_val)
    indexi[0] = idx
    di[0] = d_val[idx]

    Li = np.array([[np.sqrt(col_norms[idx])]])
    Ri = np.array([[1 / np.sqrt(col_norms[idx])]])
    ai = a[idx].reshape(1, 1)

    for i in range(1, n):
        id_left = [j for j in range(n_features) if j not in indexi[:i]]

        dj = np.zeros(n_features)
        for j in id_left:
            betai = (X[:, indexi[:i]].T @ X[:, j] / (Y.T @ Y)).reshape(-1, 1)
            bj = col_norms[j]

            Lj, Rj = _chol_it(Li, Ri, betai, bj)
            aj = np.concatenate([ai, a[j].reshape(1, 1)], axis=0)
            Rj_last_row = Rj[-1, :]
            dj[j] = float((Rj_last_row @ aj).ravel()[0]) ** 2

        idx = np.argmax(dj)
        betai = (X[:, indexi[:i]].T @ X[:, idx] / (Y.T @ Y)).reshape(-1, 1)
        Li, Ri = _chol_it(Li, Ri, betai, col_norms[idx])
        ai = np.concatenate([ai, a[idx].reshape(1, 1)], axis=0)
        di[i] = di[i - 1] + dj[idx]
        indexi[i] = idx

    return indexi


# ---------------------------------------------------------------------------
# CWDMSelector 三件套
# ---------------------------------------------------------------------------

class CWDMSelectorConfig:
    """CWDM 特征选择器实例参数。

    Attributes:
        n_iterations (int): 打乱和特征选择的迭代次数。
        k_features (str | int): 每次迭代选择的特征数量，'auto' 表示选择前 150 个。
        final_k (int): 最终选择的特征数量。
        random_state (int | None): 随机种子。
        n_blocks (int): 分块数量，仅当数据量大时使用。
        data_threshold (int): 数据量阈值，超过该值使用分块方法。
    """

    def __init__(
        self,
        n_iterations: int = CWDM_N_ITERATIONS,
        k_features: str | int = 'auto',
        final_k: int = 10,
        random_state: int | None = None,
        n_blocks: int = CWDM_N_BLOCKS,
        data_threshold: int = CWDM_DATA_THRESHOLD,
    ) -> None:
        self.n_iterations = n_iterations
        self.k_features = k_features
        self.final_k = final_k
        self.random_state = random_state
        self.n_blocks = n_blocks
        self.data_threshold = data_threshold


class CWDMSelectorExtraOutput:
    """CWDM 特征选择器附加输出。

    Attributes:
        selected_features (list[int]): 选中的特征索引列表。
        feature_counts (dict[int, int]): 各特征出现次数。
        feature_ratios (dict[int, float]): 各特征出现频率。
    """

    def __init__(
        self,
        selected_features: list[int],
        feature_counts: dict[int, int],
        feature_ratios: dict[int, float],
    ) -> None:
        self.selected_features = selected_features
        self.feature_counts = feature_counts
        self.feature_ratios = feature_ratios


class CWDMSelector:
    """基于 CWDM 的特征选择器。

    根据数据量自动选择分块或普通 CWDM 方法，通过多次打乱特征顺序进行特征选择，
    基于出现频率选择重要特征。
    """

    def __init__(self, *, config: CWDMSelectorConfig | None = None, **kwargs) -> None:
        """初始化 CWDM 特征选择器。

        Args:
            config (CWDMSelectorConfig | None): 选择器配置。
            **kwargs: 透传给配置的键值参数。
        """
        if config is None:
            config = CWDMSelectorConfig(**kwargs)
        self._config = config

    @classmethod
    def name(cls) -> str:
        """返回算子注册名称。

        Returns:
            str: 算子名称。
        """
        return 'cwdm_selector'

    def select(self, X: np.ndarray, y: np.ndarray) -> CWDMSelectorExtraOutput:
        """执行 CWDM 特征选择。

        Args:
            X (np.ndarray): 特征矩阵 (n_samples, n_features)。
            y (np.ndarray): 目标变量 (n_samples,)。

        Returns:
            CWDMSelectorExtraOutput: 选择结果与附加输出。
        """
        config = self._config
        n_samples, n_features = X.shape
        total_elements = n_samples * n_features

        if total_elements > config.data_threshold:
            logger.info(
                f'CWDMSelector 数据量较大 ({total_elements} > {config.data_threshold})，'
                f'采用分块方法',
            )
            selected_features, feature_counts, feature_ratios = _block_feature_selection(
                X, y,
                n_blocks=config.n_blocks,
                n_iterations=config.n_iterations,
                k_features=config.k_features,
                final_k=config.final_k,
                random_state=config.random_state,
            )
        else:
            logger.info(
                f'CWDMSelector 数据量适中 ({total_elements} ≤ {config.data_threshold})，'
                f'采用普通 CWDM 方法',
            )
            selected_features, feature_counts, feature_ratios = _shuffle_cwdm_selection(
                X, y,
                n_iterations=config.n_iterations,
                k_features=config.k_features,
                final_k=config.final_k,
                random_state=config.random_state,
            )

        return CWDMSelectorExtraOutput(
            selected_features=selected_features,
            feature_counts=feature_counts,
            feature_ratios=feature_ratios,
        )


# ---------------------------------------------------------------------------
# 打乱选择（内部函数）
# ---------------------------------------------------------------------------

def _shuffle_cwdm_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = CWDM_N_ITERATIONS,
    k_features: str | int = 'auto',
    final_k: int = 10,
    random_state: int | None = None,
) -> tuple[list[int], dict[int, int], dict[int, float]]:
    """通过多次打乱特征顺序进行 CWDM 特征选择，基于出现频率选择重要特征。

    Args:
        X (np.ndarray): 特征矩阵 (n_samples, n_features)。
        y (np.ndarray): 目标变量 (n_samples,)。
        n_iterations (int): 打乱和特征选择的迭代次数。
        k_features (str | int): 每次迭代选择的特征数量，'auto' 表示选择前 150 个。
        final_k (int): 最终选择的特征数量。
        random_state (int | None): 随机种子。

    Returns:
        tuple[list[int], dict[int, int], dict[int, float]]:
            (selected_features, feature_counts, feature_ratios)。
    """
    n_samples, n_features = X.shape

    if k_features == 'auto':
        k = min(150, n_features)
    else:
        k = min(k_features, n_features)

    all_selected_features: list[int] = []

    if random_state is not None:
        np.random.seed(random_state)

    logger.info(f'CWDMSelector 开始进行 {n_iterations} 次特征选择...')

    for i in range(n_iterations):
        feature_indices = np.random.permutation(n_features)
        X_shuffled = X[:, feature_indices]

        selected = _cwdm(X_shuffled, y, k)

        original_selected = [feature_indices[idx] for idx in selected]
        all_selected_features.extend(original_selected)

        if (i + 1) % 20 == 0:
            logger.info(f'CWDMSelector 已完成 {i + 1} 次迭代')

    feature_counts = Counter(all_selected_features)
    feature_ratios = {f: c / n_iterations for f, c in feature_counts.items()}

    selected_features = [f for f, _ in feature_counts.most_common(final_k)]

    logger.info('CWDMSelector 特征选择完成')
    return selected_features, dict(feature_counts), feature_ratios


# ---------------------------------------------------------------------------
# 分块选择（内部函数）
# ---------------------------------------------------------------------------

def _split_data(
    X: np.ndarray, y: np.ndarray, n_blocks: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """将数据分成多个块。

    Args:
        X (np.ndarray): 特征矩阵。
        y (np.ndarray): 目标变量。
        n_blocks (int): 分块数量。

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: 数据块列表。
    """
    n_samples = X.shape[0]
    block_size = n_samples // n_blocks

    blocks = []
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < n_blocks - 1 else n_samples
        blocks.append((X[start_idx:end_idx], y[start_idx:end_idx]))

    return blocks


def _block_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_blocks: int = CWDM_N_BLOCKS,
    n_iterations: int = CWDM_N_ITERATIONS,
    k_features: str | int = 'auto',
    final_k: int = 10,
    random_state: int | None = None,
) -> tuple[list[int], dict[int, int], dict[int, float]]:
    """分块特征选择：在每个块上独立选择特征，然后统计出现次数。

    Args:
        X (np.ndarray): 特征矩阵。
        y (np.ndarray): 目标变量。
        n_blocks (int): 分块数量。
        n_iterations (int): 每个块上的迭代次数。
        k_features (str | int): 每次迭代选择的特征数量。
        final_k (int): 最终选择的特征数量。
        random_state (int | None): 随机种子。

    Returns:
        tuple[list[int], dict[int, int], dict[int, float]]:
            (selected_features, feature_counts, feature_ratios)。
    """
    blocks = _split_data(X, y, n_blocks)

    all_selected_features: list[int] = []

    logger.info(f'CWDMSelector 将数据分为 {n_blocks} 个块进行特征选择...')

    for i, (X_block, y_block) in enumerate(blocks):
        logger.info(
            f'CWDMSelector 处理第 {i + 1}/{n_blocks} 个块 '
            f'(样本数: {X_block.shape[0]})...',
        )

        block_selected, block_counts, block_ratios = _shuffle_cwdm_selection(
            X=X_block, y=y_block, n_iterations=n_iterations,
            k_features=k_features, final_k=final_k,
            random_state=random_state,
        )

        all_selected_features.extend(block_selected)

    feature_counter = Counter(all_selected_features)
    most_common = feature_counter.most_common(final_k)
    selected_features = [item[0] for item in most_common]

    frequencies = {idx: count / n_blocks for idx, count in feature_counter.items()}
    feature_counts = dict(feature_counter)
    feature_ratios = frequencies

    return selected_features, feature_counts, feature_ratios


# ---------------------------------------------------------------------------
# ConvergenceFinder 三件套
# ---------------------------------------------------------------------------

class ConvergenceFinderConfig:
    """收敛点查找器实例参数。

    Attributes:
        task (str): 任务类型，可选 'Classification' | 'Regression'。
        stability_threshold (float): 稳定性阈值。
        accuracy_threshold (float): 准确率阈值。
    """

    def __init__(
        self,
        task: str = 'Classification',
        stability_threshold: float = CONVERGENCE_STABILITY_THRESHOLD,
        accuracy_threshold: float = CONVERGENCE_ACCURACY_THRESHOLD,
    ) -> None:
        self.task = task
        self.stability_threshold = stability_threshold
        self.accuracy_threshold = accuracy_threshold


class ConvergenceFinderExtraOutput:
    """收敛点查找器附加输出。

    Attributes:
        final_stable_features (np.ndarray): 稳定特征索引。
        convergence_points (dict[str, int]): 各模型收敛点。
    """

    def __init__(
        self,
        final_stable_features: np.ndarray,
        convergence_points: dict[str, int],
    ) -> None:
        self.final_stable_features = final_stable_features
        self.convergence_points = convergence_points


class ConvergenceFinder:
    """使用多个模型找到 CWDM 特征的稳定收敛点，取收敛点最大值。"""

    def __init__(self, *, config: ConvergenceFinderConfig | None = None, **kwargs) -> None:
        """初始化收敛点查找器。

        Args:
            config (ConvergenceFinderConfig | None): 查找器配置。
            **kwargs: 透传给配置的键值参数。
        """
        if config is None:
            config = ConvergenceFinderConfig(**kwargs)
        self._config = config

    @classmethod
    def name(cls) -> str:
        """返回算子注册名称。

        Returns:
            str: 算子名称。
        """
        return 'convergence_finder'

    def find(
        self,
        train: np.ndarray,
        val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        cwdm_features: np.ndarray,
    ) -> ConvergenceFinderExtraOutput:
        """查找 CWDM 特征的稳定收敛点。

        Args:
            train (np.ndarray): 训练集特征。
            val (np.ndarray): 验证集特征。
            y_train (np.ndarray): 训练集标签。
            y_val (np.ndarray): 验证集标签。
            cwdm_features (np.ndarray): CWDM 选出的特征索引。

        Returns:
            ConvergenceFinderExtraOutput: 收敛点结果与附加输出。
        """
        config = self._config

        logger.info(f'ConvergenceFinder CWDM 选出特征数量: {len(cwdm_features)}')
        logger.info(f'ConvergenceFinder 任务类型: {config.task}')

        models = _initialize_models_by_task(config.task)

        train_cwdm = train[:, cwdm_features]
        val_cwdm = val[:, cwdm_features]

        convergence_points: dict[str, int] = {}
        accuracy_points: dict[str, float] = {}

        for model_name, model in models.items():
            logger.info(f'ConvergenceFinder 正在计算 {model_name} 的收敛点...')

            convergence_point, accuracy_point = _calculate_convergence_point(
                model, train_cwdm, val_cwdm, y_train, y_val, config.task,
            )

            convergence_points[model_name] = convergence_point
            accuracy_points[model_name] = accuracy_point
            logger.info(f'ConvergenceFinder {model_name} 收敛点: {convergence_point}')

        # 取准确率最高的模型对应的收敛点
        best_model_name = max(accuracy_points, key=accuracy_points.get)
        final_stable_point = convergence_points[best_model_name]

        logger.info(f'ConvergenceFinder 最终稳定点（最大值）: {final_stable_point}')

        eo = ConvergenceFinderExtraOutput(
            final_stable_features=cwdm_features[:final_stable_point],
            convergence_points=convergence_points,
        )
        return eo


# ---------------------------------------------------------------------------
# 收敛点判断（内部函数）
# ---------------------------------------------------------------------------

def _initialize_models_by_task(task: str) -> dict[str, object]:
    """根据任务类型初始化模型。

    Args:
        task (str): 任务类型。

    Returns:
        dict[str, object]: 模型名称到模型实例的映射。

    Raises:
        ValueError: 任务类型不支持时抛出。
    """
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression

    if task == 'Classification':
        return {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        }
    elif task == 'Regression':
        return {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(kernel='linear'),
        }
    else:
        raise ValueError("任务类型必须是 'Classification' 或 'Regression'")


def _calculate_convergence_point(
    model: object,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    task: str,
    stability_threshold: float = CONVERGENCE_STABILITY_THRESHOLD,
    accuracy_threshold: float = CONVERGENCE_ACCURACY_THRESHOLD,
) -> tuple[int, float]:
    """计算模型性能的收敛点。

    收敛条件：连续 CONVERGENCE_WINDOW 个点的标准差 < stability_threshold，
    且当前准确率 >= accuracy_threshold。

    Args:
        model (object): sklearn 兼容模型。
        X_train (np.ndarray): 训练集特征。
        X_val (np.ndarray): 验证集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        task (str): 任务类型。
        stability_threshold (float): 稳定性阈值。
        accuracy_threshold (float): 准确率阈值。

    Returns:
        tuple[int, float]: (convergence_point, current_accuracy) 收敛点特征数和当前准确率。
    """
    n_features = X_train.shape[1]
    performances: list[float] = []
    accuracy_scores: list[float] = []

    feat_up_bound = min(n_features + 1, CONVERGENCE_FEAT_UPPER_BOUND)

    for k in range(1, feat_up_bound, CONVERGENCE_STEP):
        X_train_subset = X_train[:, :k]
        X_val_subset = X_val[:, :k]

        try:
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_val_subset)

            if task == 'Classification':
                performance_score = f1_score(y_val, y_pred, average='weighted')
                accuracy_score_val = accuracy_score(y_val, y_pred)
            else:
                performance_score = r2_score(y_val, y_pred)
                accuracy_score_val = r2_score(y_val, y_pred)

            performances.append(performance_score)
            accuracy_scores.append(accuracy_score_val)

        except Exception as e:
            logger.warning(f'ConvergenceFinder 特征数 {k} 时出错: {e}')
            performances.append(0)
            accuracy_scores.append(0)

    convergence_point = n_features

    if len(performances) >= CONVERGENCE_WINDOW:
        for k in range(CONVERGENCE_WINDOW - 1, len(performances)):
            recent_performances = performances[k - CONVERGENCE_WINDOW + 1: k + 1]
            std_dev = np.std(recent_performances)
            current_accuracy = accuracy_scores[k]

            if std_dev < stability_threshold and current_accuracy >= accuracy_threshold:
                convergence_point = (k + 1) * CONVERGENCE_STEP
                logger.info(
                    f'ConvergenceFinder 找到收敛点: 特征数={convergence_point}, '
                    f'标准差={std_dev:.6f}, 准确率={current_accuracy:.4f}',
                )
                break

            elif current_accuracy >= accuracy_threshold and std_dev >= stability_threshold:
                logger.info(
                    f'ConvergenceFinder 特征数{k * CONVERGENCE_STEP + 1}: '
                    f'准确率达标({current_accuracy:.4f})但标准差较大({std_dev:.6f})',
                )

    logger.info(f'ConvergenceFinder 性能序列长度: {len(performances)}')
    logger.info(f'ConvergenceFinder 收敛点: {convergence_point}')

    if performances:
        logger.info(
            f'ConvergenceFinder 性能值范围: {min(performances):.4f} ~ {max(performances):.4f}',
        )
        logger.info(
            f'ConvergenceFinder 准确率范围: {min(accuracy_scores):.4f} ~ {max(accuracy_scores):.4f}',
        )

    if convergence_point < n_features:
        start = max(0, convergence_point // CONVERGENCE_STEP - CONVERGENCE_WINDOW + 1)
        end = convergence_point // CONVERGENCE_STEP + 1
        logger.info(f'ConvergenceFinder 收敛点附近的性能值: {performances[start:end]}')
        logger.info(f'ConvergenceFinder 收敛点附近的准确率: {accuracy_scores[start:end]}')

    if convergence_point < X_train.shape[1]:
        return convergence_point, accuracy_scores[-1]
    else:
        return X_train.shape[1], max(accuracy_scores) if accuracy_scores else 0.0
# ---------------------------------------------------------------------------
# 向后兼容的函数式接口
# ---------------------------------------------------------------------------

def smart_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = CWDM_N_ITERATIONS,
    k_features: str | int = 'auto',
    final_k: int = 10,
    random_state: int | None = None,
    n_blocks: int = CWDM_N_BLOCKS,
    data_threshold: int = CWDM_DATA_THRESHOLD,
) -> tuple[list[int], dict[int, int], dict[int, float]]:
    """智能特征选择：根据数据量自动选择分块或普通方法。

    Args:
        X (np.ndarray): 特征矩阵。
        y (np.ndarray): 目标变量。
        n_iterations (int): 打乱和特征选择的迭代次数。
        k_features (str | int): 每次迭代选择的特征数量。
        final_k (int): 最终选择的特征数量。
        random_state (int | None): 随机种子。
        n_blocks (int): 分块数量，仅当数据量大时使用。
        data_threshold (int): 数据量阈值，超过该值使用分块方法。

    Returns:
        tuple[list[int], dict[int, int], dict[int, float]]:
            (selected_features, feature_counts, feature_ratios)。
    """
    config = CWDMSelectorConfig(
        n_iterations=n_iterations,
        k_features=k_features,
        final_k=final_k,
        random_state=random_state,
        n_blocks=n_blocks,
        data_threshold=data_threshold,
    )
    selector = CWDMSelector(config=config)
    eo = selector.select(X, y)
    return eo.selected_features, eo.feature_counts, eo.feature_ratios


def fst_round_important_feature_index(
    train: np.ndarray,
    val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cwdm_features: np.ndarray,
    task: str = 'Classification',
) -> tuple[np.ndarray, dict[str, int]]:
    """使用多个模型找到 CWDM 特征的稳定收敛点，取收敛点最大值。

    Args:
        train (np.ndarray): 训练集特征。
        val (np.ndarray): 验证集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        cwdm_features (np.ndarray): CWDM 选出的特征索引。
        task (str): 任务类型。

    Returns:
        tuple[np.ndarray, dict[str, int]]: (final_stable_features, convergence_points) 稳定特征索引和各模型收敛点。
    """
    config = ConvergenceFinderConfig(task=task)
    finder = ConvergenceFinder(config=config)
    eo = finder.find(train, val, y_train, y_val, cwdm_features)
    return eo.final_stable_features, eo.convergence_points



# ===========================================================================
# model_training
# ===========================================================================

_CLASSIFIERS_LIST = [
    LogisticRegression,
    SVC,
    RandomForestClassifier,
    LGBMClassifier,
    KNeighborsClassifier,
    MLPClassifier,
]


# ---------------------------------------------------------------------------
# ClassificationTrainer 三件套
# ---------------------------------------------------------------------------

class ClassificationTrainerConfig:
    """分类模型训练器实例参数。

    Attributes:
        proxy_model (str | int): 模型名称字符串或 1（自动选择）。
    """

    def __init__(self, proxy_model: str | int = 'LGBMClassifier') -> None:
        self.proxy_model = proxy_model


class ClassificationTrainerExtraOutput:
    """分类模型训练器附加输出。

    Attributes:
        best_model (object): 训练好的模型。
        best_model_name (str): 模型名称。
    """

    def __init__(self, best_model: object, best_model_name: str) -> None:
        self.best_model = best_model
        self.best_model_name = best_model_name


class ClassificationTrainer:
    """分类代理模型训练器。

    当 proxy_model == 1 时，使用 LazyPredict 自动选择最优模型；
    否则使用指定的 proxy_model 名称。
    """

    def __init__(self, *, config: ClassificationTrainerConfig | None = None, **kwargs) -> None:
        """初始化分类模型训练器。

        Args:
            config (ClassificationTrainerConfig | None): 训练器配置。
            **kwargs: 透传给配置的键值参数。
        """
        if config is None:
            config = ClassificationTrainerConfig(**kwargs)
        self._config = config

    @classmethod
    def name(cls) -> str:
        """返回算子注册名称。

        Returns:
            str: 算子名称。
        """
        return 'classification_trainer'

    def train(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> ClassificationTrainerExtraOutput:
        """训练分类代理模型。

        Args:
            x_train (np.ndarray): 训练集特征。
            x_val (np.ndarray): 验证集特征。
            x_test (np.ndarray): 测试集特征。
            y_train (np.ndarray): 训练集标签。
            y_val (np.ndarray): 验证集标签。
            y_test (np.ndarray): 测试集标签。

        Returns:
            ClassificationTrainerExtraOutput: 训练结果与附加输出。
        """
        proxy_model = self._config.proxy_model

        if proxy_model == 1:
            clf = LazyClassifier(
                verbose=0, ignore_warnings=True, custom_metric=None,
                classifiers=_CLASSIFIERS_LIST,
            )
            models, predictions = clf.fit(x_train, x_val, y_train, y_val)
            models.sort_values(by=['Balanced Accuracy', 'Time Taken'], ascending=[False, True])
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            logger.info(f'ClassificationTrainer LazyPredict 评估的模型:\n{models}')

            best_model_name = models.index[0]
            logger.info(f'ClassificationTrainer 最优模型: {best_model_name}')
        else:
            best_model_name = proxy_model

        # 准备合并训练数据
        X_train_optuna = np.vstack([x_train, x_val]).copy()
        Y_train_optuna = np.concatenate([y_train, y_val]).copy()

        # 根据模型名称选择训练策略
        trainer = _CLASSIFIER_TRAINERS.get(best_model_name)
        if trainer is None:
            raise ValueError(f'ClassificationTrainer 不支持的分类模型: {best_model_name}')

        best_model = trainer(
            x_train, x_val, y_train, y_val,
            X_train_optuna, Y_train_optuna,
        )

        # LGBMClassifier 额外输出测试集报告
        if best_model_name == 'LGBMClassifier':
            y_pred = best_model.predict(x_test)
            f1 = f1_score(y_test, y_pred, average='macro')
            logger.info(f'ClassificationTrainer 未做特征选择前测试集 f1_score: {f1}')
            logger.info(
                f'ClassificationTrainer 未做特征选择前测试集报告:\n'
                f'{classification_report(y_test, y_pred, digits=5)}',
            )

        return ClassificationTrainerExtraOutput(
            best_model=best_model,
            best_model_name=best_model_name,
        )


# ---------------------------------------------------------------------------
# RegressionTrainer 三件套
# ---------------------------------------------------------------------------

class RegressionTrainerConfig:
    """回归模型训练器实例参数。"""

    pass


class RegressionTrainerExtraOutput:
    """回归模型训练器附加输出。

    Attributes:
        best_model (object): 训练好的模型。
        best_model_name (str): 模型名称。
    """

    def __init__(self, best_model: object, best_model_name: str) -> None:
        self.best_model = best_model
        self.best_model_name = best_model_name


class RegressionTrainer:
    """回归代理模型训练器（XGBoost）。"""

    def __init__(self, *, config: RegressionTrainerConfig | None = None, **kwargs) -> None:
        """初始化回归模型训练器。

        Args:
            config (RegressionTrainerConfig | None): 训练器配置。
            **kwargs: 透传给配置的键值参数。
        """
        if config is None:
            config = RegressionTrainerConfig(**kwargs)
        self._config = config

    @classmethod
    def name(cls) -> str:
        """返回算子注册名称。

        Returns:
            str: 算子名称。
        """
        return 'regression_trainer'

    def train(
        self,
        train: np.ndarray,
        val: np.ndarray,
        test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> RegressionTrainerExtraOutput:
        """训练回归代理模型（XGBoost）。

        Args:
            train (np.ndarray): 训练集特征。
            val (np.ndarray): 验证集特征。
            test (np.ndarray): 测试集特征。
            y_train (np.ndarray): 训练集标签。
            y_val (np.ndarray): 验证集标签。
            y_test (np.ndarray): 测试集标签。

        Returns:
            RegressionTrainerExtraOutput: 训练结果与附加输出。
        """
        dtrain = xgb.DMatrix(train, label=y_train)
        dval = xgb.DMatrix(val, label=y_val)

        param = {'max_depth': 10, 'objective': 'reg:squarederror', 'nthread': 4}
        evallist = [(dtrain, 'train'), (dval, 'val')]
        num_round = 50

        best_model = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)
        best_model_name = 'xgboost'

        logger.info('RegressionTrainer XGBoost 回归模型训练完成')

        return RegressionTrainerExtraOutput(
            best_model=best_model,
            best_model_name=best_model_name,
        )


# ---------------------------------------------------------------------------
# 各分类器的训练函数
# ---------------------------------------------------------------------------

def _train_logistic_regression(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_combined: np.ndarray,
    y_combined: np.ndarray,
) -> LogisticRegression:
    """训练 LogisticRegression。

    Args:
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        X_combined (np.ndarray): 合并后的训练数据。
        y_combined (np.ndarray): 合并后的标签。

    Returns:
        LogisticRegression: 训练好的模型。
    """
    model = LogisticRegression(penalty=None, solver='saga')

    acc = cross_val_score(model, X_combined, y_combined, cv=2, scoring='balanced_accuracy')
    logger.info(f'ClassificationTrainer LogisticRegression 二折交叉验证 balanced_accuracy: {acc}')
    acc = cross_val_score(model, X_combined, y_combined, cv=2, scoring='f1')
    logger.info(f'ClassificationTrainer LogisticRegression 二折交叉验证 f1_score: {acc}')

    model.fit(X_combined, y_combined)
    return model


def _train_lda(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_combined: np.ndarray,
    y_combined: np.ndarray,
) -> LinearDiscriminantAnalysis:
    """训练 LinearDiscriminantAnalysis。

    Args:
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        X_combined (np.ndarray): 合并后的训练数据。
        y_combined (np.ndarray): 合并后的标签。

    Returns:
        LinearDiscriminantAnalysis: 训练好的模型。
    """
    model = LinearDiscriminantAnalysis()

    acc = cross_val_score(model, X_combined, y_combined, cv=2, scoring='balanced_accuracy')
    logger.info(f'ClassificationTrainer LDA 二折交叉验证 balanced_accuracy: {acc}')
    acc = cross_val_score(model, X_combined, y_combined, cv=2, scoring='f1')
    logger.info(f'ClassificationTrainer LDA 二折交叉验证 f1_score: {acc}')

    model.fit(X_combined, y_combined)
    return model


def _train_svc(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_combined: np.ndarray,
    y_combined: np.ndarray,
) -> SVC:
    """训练 SVC，使用 Optuna 优化超参数。

    Args:
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        X_combined (np.ndarray): 合并后的训练数据。
        y_combined (np.ndarray): 合并后的标签。

    Returns:
        SVC: 训练好的模型。
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: _objective_svc(trial, X_combined, y_combined),
        n_trials=OPTUNA_N_TRIALS,
    )
    logger.info(f'ClassificationTrainer SVC 最佳超参数: {study.best_params}')

    best_model = SVC(**study.best_params, random_state=42, probability=True)

    acc = cross_val_score(best_model, X_combined, y_combined, cv=2, scoring='balanced_accuracy')
    logger.info(f'ClassificationTrainer SVC 二折交叉验证 balanced_accuracy: {acc}')
    acc = cross_val_score(best_model, X_combined, y_combined, cv=2, scoring='f1')
    logger.info(f'ClassificationTrainer SVC 二折交叉验证 f1_score: {acc}')

    best_model.fit(X_combined, y_combined)
    return best_model


def _train_lgbm(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_combined: np.ndarray,
    y_combined: np.ndarray,
) -> LGBMClassifier:
    """训练 LGBMClassifier，使用 Optuna 优化超参数。

    Args:
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        X_combined (np.ndarray): 合并后的训练数据。
        y_combined (np.ndarray): 合并后的标签。

    Returns:
        LGBMClassifier: 训练好的模型。
    """
    import lightgbm as lgb

    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: _objective_lgbm(trial, x_train, y_train, x_val, y_val),
        n_trials=OPTUNA_N_TRIALS,
    )
    logger.info(f'ClassificationTrainer LGBM 最佳超参数: {study.best_params}')

    best_model = lgb.LGBMClassifier(**study.best_params, verbose=-1)

    acc = cross_val_score(best_model, X_combined, y_combined, cv=2, scoring='balanced_accuracy')
    logger.info(f'ClassificationTrainer LGBM 二折交叉验证 balanced_accuracy: {acc}')
    acc = cross_val_score(best_model, X_combined, y_combined, cv=2, scoring='f1')
    logger.info(f'ClassificationTrainer LGBM 二折交叉验证 f1_score: {acc}')

    best_model.fit(X_combined, y_combined)
    return best_model


def _train_random_forest(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_combined: np.ndarray,
    y_combined: np.ndarray,
) -> RandomForestClassifier:
    """训练 RandomForestClassifier，使用 Optuna 优化超参数。

    Args:
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        X_combined (np.ndarray): 合并后的训练数据。
        y_combined (np.ndarray): 合并后的标签。

    Returns:
        RandomForestClassifier: 训练好的模型。
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: _objective_rf(trial, X_combined, y_combined),
        n_trials=OPTUNA_N_TRIALS,
    )
    logger.info(f'ClassificationTrainer RandomForest 最佳超参数: {study.best_params}')

    best_model = RandomForestClassifier(**study.best_params)

    acc = cross_val_score(best_model, X_combined, y_combined, cv=2, scoring='balanced_accuracy')
    logger.info(f'ClassificationTrainer RandomForest 二折交叉验证 balanced_accuracy: {acc}')
    acc = cross_val_score(best_model, X_combined, y_combined, cv=2, scoring='f1')
    logger.info(f'ClassificationTrainer RandomForest 二折交叉验证 f1_score: {acc}')

    best_model.fit(X_combined, y_combined)
    return best_model


def _train_mlp(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_combined: np.ndarray,
    y_combined: np.ndarray,
) -> MLPClassifier:
    """训练 MLPClassifier，使用 Optuna 优化超参数。

    Args:
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        X_combined (np.ndarray): 合并后的训练数据。
        y_combined (np.ndarray): 合并后的标签。

    Returns:
        MLPClassifier: 训练好的模型。
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: _objective_mlp(trial, X_combined, y_combined),
        n_trials=OPTUNA_N_TRIALS,
    )
    logger.info(f'ClassificationTrainer MLP 最佳超参数: {study.best_params}')

    best_params = study.best_params
    hidden_layer_sizes = tuple(
        best_params[f'hidden_layer_{i}'] for i in range(best_params['n_layers'])
    )

    best_model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
    )

    acc = cross_val_score(best_model, X_combined, y_combined, cv=2, scoring='balanced_accuracy')
    logger.info(f'ClassificationTrainer MLP 二折交叉验证 balanced_accuracy: {acc}')

    best_model.fit(X_combined, y_combined)
    return best_model


def _train_knn(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_combined: np.ndarray,
    y_combined: np.ndarray,
) -> KNeighborsClassifier:
    """训练 KNeighborsClassifier，使用 Optuna 优化超参数。

    Args:
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        X_combined (np.ndarray): 合并后的训练数据。
        y_combined (np.ndarray): 合并后的标签。

    Returns:
        KNeighborsClassifier: 训练好的模型。
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: _objective_knn(trial, X_combined, y_combined),
        n_trials=OPTUNA_N_TRIALS,
    )
    logger.info(f'ClassificationTrainer KNN 最佳超参数: {study.best_params}')

    best_model = KNeighborsClassifier(**study.best_params)

    acc = cross_val_score(best_model, X_combined, y_combined, cv=2, scoring='balanced_accuracy')
    logger.info(f'ClassificationTrainer KNN 二折交叉验证 balanced_accuracy: {acc}')

    best_model.fit(X_combined, y_combined)
    return best_model


# 分类器名称 -> 训练函数 映射
_CLASSIFIER_TRAINERS: dict[str, callable] = {
    'LogisticRegression': _train_logistic_regression,
    'LinearDiscriminantAnalysis': _train_lda,
    'SVC': _train_svc,
    'LGBMClassifier': _train_lgbm,
    'RandomForestClassifier': _train_random_forest,
    'MLPClassifier': _train_mlp,
    'KNeighborsClassifier': _train_knn,
}


# ---------------------------------------------------------------------------
# Optuna 目标函数
# ---------------------------------------------------------------------------

def _objective_svc(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """SVC 超参数优化目标函数。

    Args:
        trial (optuna.Trial): Optuna 试验对象。
        X (np.ndarray): 特征数据。
        y (np.ndarray): 标签数据。

    Returns:
        float: 交叉验证准确率均值。
    """
    C = trial.suggest_loguniform('C', 1e-3, 1e3)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])

    if kernel == 'linear':
        gamma = 'scale'
    else:
        gamma = trial.suggest_loguniform('gamma', 1e-3, 1e3)

    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    return cross_val_score(model, X, y, cv=2, scoring='accuracy').mean()


def _objective_lgbm(
    trial: optuna.Trial,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """LGBMClassifier 超参数优化目标函数。

    Args:
        trial (optuna.Trial): Optuna 试验对象。
        x_train (np.ndarray): 训练集特征。
        y_train (np.ndarray): 训练集标签。
        x_val (np.ndarray): 验证集特征。
        y_val (np.ndarray): 验证集标签。

    Returns:
        float: 验证集准确率。
    """
    import lightgbm as lgb

    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.2),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'num_boost_round': trial.suggest_int('num_boost_round', 1, 100),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
    }

    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

    model = lgb.train(params, train_data, valid_sets=[valid_data])
    y_pred = model.predict(x_val)
    y_pred_binary = np.round(y_pred)

    return accuracy_score(y_val, y_pred_binary)


def _objective_rf(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """RandomForestClassifier 超参数优化目标函数。

    Args:
        trial (optuna.Trial): Optuna 试验对象。
        X (np.ndarray): 特征数据。
        y (np.ndarray): 标签数据。

    Returns:
        float: 交叉验证准确率均值。
    """
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        max_features=max_features, random_state=42, n_jobs=-1,
    )
    return cross_val_score(model, X, y, cv=2).mean()


def _objective_mlp(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """MLPClassifier 超参数优化目标函数。

    Args:
        trial (optuna.Trial): Optuna 试验对象。
        X (np.ndarray): 特征数据。
        y (np.ndarray): 标签数据。

    Returns:
        float: 交叉验证准确率均值。
    """
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layer_sizes = tuple(
        trial.suggest_int(f'hidden_layer_{i}', 10, 200) for i in range(n_layers)
    )
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha, learning_rate_init=learning_rate_init,
        max_iter=500, random_state=42,
    )
    return cross_val_score(model, X, y, cv=2).mean()


def _objective_knn(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """KNeighborsClassifier 超参数优化目标函数。

    Args:
        trial (optuna.Trial): Optuna 试验对象。
        X (np.ndarray): 特征数据。
        y (np.ndarray): 标签数据。

    Returns:
        float: 交叉验证准确率均值。
    """
    n_neighbors = trial.suggest_int('n_neighbors', 3, 20)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size', 10, 50)

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights,
        algorithm=algorithm, leaf_size=leaf_size,
    )
    return cross_val_score(model, X, y, cv=2).mean()


# ---------------------------------------------------------------------------
# 向后兼容的函数式接口
# ---------------------------------------------------------------------------

def best_model_training(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    proxy_model: object,
) -> tuple[object, str]:
    """训练分类代理模型。

    当 proxy_model == 1 时，使用 LazyPredict 自动选择最优模型；
    否则使用指定的 proxy_model 名称。

    Args:
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        x_test (np.ndarray): 测试集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        y_test (np.ndarray): 测试集标签。
        proxy_model (object): 模型名称字符串或 1（自动选择）。

    Returns:
        tuple[object, str]: (best_model, best_model_name) 训练好的模型和模型名称。
    """
    config = ClassificationTrainerConfig(proxy_model=proxy_model)
    trainer = ClassificationTrainer(config=config)
    eo = trainer.train(x_train, x_val, x_test, y_train, y_val, y_test)
    return eo.best_model, eo.best_model_name


def best_model_training_regression(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[object, str]:
    """训练回归代理模型（XGBoost）。

    Args:
        train (np.ndarray): 训练集特征。
        val (np.ndarray): 验证集特征。
        test (np.ndarray): 测试集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        y_test (np.ndarray): 测试集标签。

    Returns:
        tuple[object, str]: (best_model, best_model_name) 训练好的模型和模型名称。
    """
    config = RegressionTrainerConfig()
    trainer = RegressionTrainer(config=config)
    eo = trainer.train(train, val, test, y_train, y_val, y_test)
    return eo.best_model, eo.best_model_name



# ===========================================================================
# sage_evaluator
# ===========================================================================

class SAGEEvaluatorConfig:
    """SAGE 评估器实例参数。

    Attributes:
        batch_size (int): 并行处理的样本数量，建议设为较大值。
        thresh (float): 收敛阈值。
        task (str): 任务类型，'Classification' | 'Regression'。
        n_jobs (int): 并行任务数。
        bar (bool): 是否显示进度条。
    """

    def __init__(
        self,
        batch_size: int = 128,
        thresh: float = 0.025,
        task: str = 'Regression',
        n_jobs: int = 8,
        bar: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.thresh = thresh
        self.task = task
        self.n_jobs = n_jobs
        self.bar = bar


class SAGEEvaluatorExtraOutput:
    """SAGE 评估器附加输出。

    Attributes:
        sage_values (sage.SageValues): 特征重要性值。
        elapsed_seconds (float): 计算耗时（秒）。
    """

    def __init__(
        self,
        sage_values: sage.SageValues,
        elapsed_seconds: float,
    ) -> None:
        self.sage_values = sage_values
        self.elapsed_seconds = elapsed_seconds


class SAGEEvaluator:
    """基于 SAGE 的特征重要性评估器。"""

    def __init__(self, *, config: SAGEEvaluatorConfig | None = None, **kwargs) -> None:
        """初始化 SAGE 评估器。

        Args:
            config (SAGEEvaluatorConfig | None): 评估器配置。
            **kwargs: 透传给配置的键值参数。
        """
        if config is None:
            config = SAGEEvaluatorConfig(**kwargs)
        self._config = config

    @classmethod
    def name(cls) -> str:
        """返回算子注册名称。

        Returns:
            str: 算子名称。
        """
        return 'sage_evaluator'

    def evaluate(
        self,
        model: object,
        x_data: np.ndarray,
        y: np.ndarray,
    ) -> SAGEEvaluatorExtraOutput:
        """基于 SAGE 的特征重要性排序。

        Args:
            model (object): 代理模型。
            x_data (np.ndarray): 样本-特征矩阵 (n_samples, n_features)。
            y (np.ndarray): 标签数组 (n_samples,)。

        Returns:
            SAGEEvaluatorExtraOutput: 评估结果与附加输出。
        """
        config = self._config
        begin = time.time()

        imputer = sage.MarginalImputer(model, x_data[:SAGE_IMPUTER_SAMPLES])

        if config.task == 'Regression':
            loss = 'mse'
        elif config.task == 'Classification':
            loss = 'cross entropy'
        else:
            loss = 'zero one'

        estimator = sage.PermutationEstimator(imputer, loss, n_jobs=config.n_jobs)
        sage_values = estimator(x_data, y, batch_size=config.batch_size, thresh=config.thresh, bar=config.bar)

        elapsed = time.time() - begin
        logger.info(f'SAGEEvaluator 计算耗时: {elapsed:.4f}s')

        return SAGEEvaluatorExtraOutput(
            sage_values=sage_values,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# 向后兼容的函数式接口
# ---------------------------------------------------------------------------

def sage_value(
    model: object,
    x_data: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    thresh: float = 0.025,
    task: str = 'Regression',
    n_jobs: int = 8,
    bar: bool = False,
) -> sage.SageValues:
    """基于 SAGE 的特征重要性排序。

    Args:
        model (object): 代理模型。
        x_data (np.ndarray): 样本-特征矩阵 (n_samples, n_features)。
        y (np.ndarray): 标签数组 (n_samples,)。
        batch_size (int): 并行处理的样本数量，建议设为较大值。
        thresh (float): 收敛阈值。
        task (str): 任务类型，'Classification' | 'Regression'。
        n_jobs (int): 并行任务数。
        bar (bool): 是否显示进度条。

    Returns:
        sage.SageValues: 特征重要性值。
    """
    config = SAGEEvaluatorConfig(
        batch_size=batch_size,
        thresh=thresh,
        task=task,
        n_jobs=n_jobs,
        bar=bar,
    )
    evaluator = SAGEEvaluator(config=config)
    eo = evaluator.evaluate(model, x_data, y)
    return eo.sage_values



# ===========================================================================
# visualization
# ===========================================================================

def plot_feature_importance_comparison(
    sage_train: sage.SageValues,
    sage_val: sage.SageValues,
    sage_test: sage.SageValues,
    feature_names: np.ndarray,
    save_dir: Path,
    file_stem: str,
) -> None:
    """绘制 train/val/test 三组 SAGE 特征重要性对比图。

    Args:
        sage_train (sage.SageValues): 训练集 SAGE 值。
        sage_val (sage.SageValues): 验证集 SAGE 值。
        sage_test (sage.SageValues): 测试集 SAGE 值。
        feature_names (np.ndarray): 特征名数组。
        save_dir (Path): 图表保存目录。
        file_stem (str): 数据文件名（不含扩展名）。

    Returns:
        None: 本方法无返回值。
    """
    plt.figure(figsize=FIG_FEATURE_IMPORTANCE_SIZE)
    sage.comparison_plot(
        (sage_train, sage_val, sage_test),
        ('x_train', 'x_val', 'x_test'),
        feature_names,
        colors=('tab:orange', 'tab:purple', 'tab:green'),
        title='x_train vs. x_val vs. x_test',
    )

    # 保存到子目录
    plt.savefig(str(save_dir / '特征重要性评估.jpeg'), dpi=FIG_DPI)
    # 拷一份到当前目录
    plt.savefig(f'{file_stem}_特征重要性评估.jpeg', dpi=FIG_DPI)
    plt.close()

    logger.info('visualization 特征重要性对比图已保存')


# ---------------------------------------------------------------------------
# 分类任务性能评估
# ---------------------------------------------------------------------------

def _performance_evaluation(
    model: object,
    best_model_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    selected_variable: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """评估分类模型在选定特征上的性能。

    Args:
        model (object): 代理模型。
        best_model_name (str): 模型名称。
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        x_test (np.ndarray): 测试集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        y_test (np.ndarray): 测试集标签。
        selected_variable (np.ndarray): 选定特征的索引。

    Returns:
        tuple[float, float, np.ndarray, np.ndarray]:
            (accuracy, f1_macro, precision, recall)。
    """
    from lightgbm import LGBMClassifier

    X = np.vstack([x_train, x_val])
    y = np.concatenate([y_train, y_val])
    X = X[:, selected_variable]

    if best_model_name == 'LGBMClassifier':
        eval_model = LGBMClassifier()
    else:
        eval_model = model

    eval_model.fit(X, y)
    y_prob = eval_model.predict(x_test[:, selected_variable])

    labels = np.sort(np.unique(y_test))
    rs = recall_score(y_test, y_prob, labels=labels, average=None)
    pr = precision_score(y_test, y_prob, labels=labels, average=None)
    accuracy = accuracy_score(y_test, y_prob)
    f1_macro = f1_score(y_test, y_prob, average='macro')

    return accuracy, f1_macro, pr.reshape(1, -1), rs.reshape(1, -1)


def fig_show_classification(
    feature_importance_index: np.ndarray,
    model: object,
    best_model_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    fig_path_task: Path,
) -> None:
    """绘制分类任务中每加入一个特征的模型性能表现。

    Args:
        feature_importance_index (np.ndarray): 特征重要性排序索引。
        model (object): 代理模型。
        best_model_name (str): 模型名称。
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        x_test (np.ndarray): 测试集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        y_test (np.ndarray): 测试集标签。
        fig_path_task (Path): 图表保存目录。

    Returns:
        None: 本方法无返回值。
    """
    n_features = len(feature_importance_index)
    feature_no = np.arange(1, n_features + 1, 1)

    fig, axs = plt.subplots(1, 3, figsize=FIG_PERFORMANCE_CLASSIFICATION_SIZE)

    accuracy_list: list[float] = []
    f1_macro_list: list[float] = []
    pr_matrix: np.ndarray | None = None
    rs_matrix: np.ndarray | None = None

    for i in range(1, n_features + 1):
        logger.info(f'visualization 逐一放入特征效果评估进度: {round(100 * i / n_features, 3)}%')
        selected_variable = feature_importance_index[:i]

        accuracy, f1_macro, pr, rs = _performance_evaluation(
            model, best_model_name,
            x_train, x_val, x_test,
            y_train, y_val, y_test,
            selected_variable,
        )

        accuracy_list.append(accuracy)
        f1_macro_list.append(f1_macro)

        pr_matrix = pr if pr_matrix is None else np.vstack([pr_matrix, pr])
        rs_matrix = rs if rs_matrix is None else np.vstack([rs_matrix, rs])

    n_feats = x_train.shape[1]
    x_tick_gap = 10 if n_feats > FEATURE_TICK_THRESHOLD else 1

    labels = np.sort(np.unique(y_test)).astype(int)

    # Accuracy / F1 子图
    axs[0].plot(feature_no, accuracy_list, linewidth=3, linestyle='-', marker='o', label='Accuracy')
    axs[0].plot(feature_no, f1_macro_list, linewidth=3, linestyle='-', marker='o', label='f1_score')
    axs[0].set_xticks(range(1, n_features, int(n_features / x_tick_gap)))
    axs[0].set_xlabel('#Features', fontsize=FONT_SIZE)
    axs[0].set_ylabel('Accuracy/f1_score', fontsize=FONT_SIZE)
    axs[0].grid(True)
    axs[0].set_title('Accuracy/f1_score', fontsize=FONT_SIZE)
    axs[0].legend(loc='best', fontsize=FONT_SIZE)

    # Precision 子图
    axs[1].plot(feature_no, pr_matrix, linewidth=3, label=labels)
    axs[1].set_xticks(range(1, n_features, int(n_features / x_tick_gap)))
    axs[1].set_xlabel('#Features', fontsize=FONT_SIZE)
    axs[1].set_ylabel('Precision', fontsize=FONT_SIZE)
    axs[1].grid(True)
    axs[1].set_title('Precision', fontsize=FONT_SIZE)
    axs[1].legend(loc='best', fontsize=FONT_SIZE)

    # Recall 子图
    axs[2].plot(feature_no, rs_matrix, linewidth=3, label=labels)
    axs[2].set_xticks(range(1, n_features, int(n_features / x_tick_gap)))
    axs[2].set_xlabel('#Features', fontsize=FONT_SIZE)
    axs[2].set_ylabel('RecallScore', fontsize=FONT_SIZE)
    axs[2].grid(True)
    axs[2].set_title('RecallScore', fontsize=FONT_SIZE)
    axs[2].legend(loc='best', fontsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(str(fig_path_task / '逐一放入特征精度变化.jpeg'), dpi=FIG_DPI)
    fig.savefig(f'{Path(fig_path_task).name}_逐一放入特征精度变化.jpeg', dpi=FIG_DPI)
    plt.close()

    logger.info('visualization 分类任务性能曲线图已保存')


# ---------------------------------------------------------------------------
# 回归任务性能评估
# ---------------------------------------------------------------------------

def _performance_evaluation_regression(
    model: object,
    best_model_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    selected_variable: np.ndarray,
) -> float:
    """评估回归模型在选定特征上的 R² 分数。

    Args:
        model (object): 代理模型。
        best_model_name (str): 模型名称。
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        x_test (np.ndarray): 测试集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        y_test (np.ndarray): 测试集标签。
        selected_variable (np.ndarray): 选定特征的索引。

    Returns:
        float: R² 分数。

    Raises:
        ValueError: 不支持的回归模型时抛出。
    """
    if best_model_name == 'xgboost':
        dtrain = xgb.DMatrix(x_train[:, selected_variable], label=y_train)
        dval = xgb.DMatrix(x_val[:, selected_variable], label=y_val)
        dtest = xgb.DMatrix(x_test[:, selected_variable], label=y_test)

        param = {'max_depth': 50, 'objective': 'reg:squarederror', 'nthread': 4}
        evallist = [(dtrain, 'train'), (dval, 'val')]
        num_round = 100

        xgb_model = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)
        y_prob = xgb_model.predict(dtest)
        return r2_score(y_test, y_prob)

    raise ValueError(f'不支持的回归模型: {best_model_name}')


def fig_show_regression(
    feature_importance_index: np.ndarray,
    model: object,
    best_model_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    fig_path_task: Path,
) -> None:
    """绘制回归任务中每加入一个特征的模型性能表现。

    Args:
        feature_importance_index (np.ndarray): 特征重要性排序索引。
        model (object): 代理模型。
        best_model_name (str): 模型名称。
        x_train (np.ndarray): 训练集特征。
        x_val (np.ndarray): 验证集特征。
        x_test (np.ndarray): 测试集特征。
        y_train (np.ndarray): 训练集标签。
        y_val (np.ndarray): 验证集标签。
        y_test (np.ndarray): 测试集标签。
        fig_path_task (Path): 图表保存目录。

    Returns:
        None: 本方法无返回值。
    """
    n_features = len(feature_importance_index)
    feature_no = np.arange(1, n_features + 1, 5)

    mse_list: list[float] = []

    for i in range(1, n_features + 1, 5):
        logger.info(f'visualization 逐一放入特征效果评估进度: {round(100 * i / n_features, 3)}%')
        selected_variable = feature_importance_index[:i]
        mse = _performance_evaluation_regression(
            model, best_model_name,
            x_train, x_val, x_test,
            y_train, y_val, y_test,
            selected_variable,
        )
        mse_list.append(mse)

    n_feats = x_train.shape[1]
    x_tick_gap = 10 if n_feats > FEATURE_TICK_THRESHOLD else 1

    plt.plot(feature_no, mse_list, linewidth=3, linestyle='-', marker='o', label='MSE')
    plt.xticks(range(1, n_features, int(n_features / x_tick_gap)))
    plt.xlabel('#Features', fontsize=FONT_SIZE)
    plt.ylabel('R2', fontsize=FONT_SIZE)
    plt.grid(True)
    plt.title('R2 evaluation, based on XGBoost', fontsize=FONT_SIZE)
    plt.legend(loc='best', fontsize=FONT_SIZE)

    plt.tight_layout()
    plt.savefig(str(fig_path_task / '逐一放入特征精度变化.jpeg'), dpi=FIG_DPI)
    plt.savefig(f'{Path(fig_path_task).name}_特征重要性评估.jpeg', dpi=FIG_DPI)
    plt.close()

    logger.info('visualization 回归任务性能曲线图已保存')



# ===========================================================================
# main
# ===========================================================================

def parse_args() -> PipelineConfig:
    """解析命令行参数并返回 PipelineConfig。

    注意：argparse 的 type=bool 不会按预期工作（任何非空字符串都为 True），
    因此布尔参数使用 store_true 动作。

    Returns:
        PipelineConfig: 流水线配置。
    """
    parser = argparse.ArgumentParser(description='Feature Reduction Tool')

    parser.add_argument('--task', type=str, default='Classification', help='MAIN TASK')
    parser.add_argument('--proxy_model', type=str, default='LGBMClassifier', help='PROXY MODEL')
    parser.add_argument('--batch_size', type=int, default=512, help='BATCH SIZE')
    parser.add_argument('--thresh', type=float, default=0.05, help='THRESHOLD')
    parser.add_argument('--n_jobs', type=int, default=8, help='NUMBER OF JOBS')
    parser.add_argument('--auto_selected_model', action='store_true', default=False,
                        help='AUTO MODEL SELECTION')
    parser.add_argument('--sample_balanced', action='store_true', default=False,
                        help='AUTO SAMPLE BALANCE')
    parser.add_argument('--bar', action='store_true', default=False, help='BAR?')
    parser.add_argument('--dataset', type=str, default='dataset_labeled', help='PATH INPUT')
    parser.add_argument('--output', type=str, default='.', help='PATH OUTPUT')
    parser.add_argument('--filename', type=str, default='', help='INPUT CSV NAME')
    parser.add_argument('--cwdm', action='store_true', default=True,
                        help='USE CWDM BEFORE SAGE')
    parser.add_argument('--no-cwdm', dest='cwdm', action='store_false',
                        help='DISABLE CWDM BEFORE SAGE')

    args = parser.parse_args()

    return PipelineConfig(
        task=args.task,
        proxy_model=args.proxy_model,
        batch_size=args.batch_size,
        thresh=args.thresh,
        n_jobs=args.n_jobs,
        auto_selected_model=args.auto_selected_model,
        sample_balanced=args.sample_balanced,
        bar=args.bar,
        dataset=args.dataset,
        output=args.output,
        filename=args.filename,
        cwdm=args.cwdm,
    )


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """执行特征优选流水线。

    流程：
    1. 加载数据
    2. 数据预处理（划分 + 标准化）
    3. 独立性筛选 (SIS)
    4. CWDM 快速初筛（可选）
    5. 代理模型训练
    6. SAGE 特征重要性评估
    7. 可视化与结果输出

    Args:
        config (PipelineConfig): 流水线配置。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (feature_importance_value, feature_importance_index, feature_importance_name)。
    """
    # Windows 多线程兼容
    joblib.parallel_backend('threading')

    logger.info(f'main 流水线配置: {config}')

    # 展开配置
    auto_selected_model = config.auto_selected_model
    proxy_model = config.proxy_model
    batch_size = config.batch_size
    thresh = config.thresh
    task = config.task
    n_jobs = config.n_jobs
    bar = config.bar
    cwdm = config.cwdm

    # 回归任务强制使用 XGBRegressor
    if task == 'Regression':
        proxy_model = 'XGBRegressor'
        auto_selected_model = False

    # ---- 第一步：加载数据 ----
    data_x_list, data_y_list, feature_name_list, files = load_data_from_folder(
        config.dataset, config.filename, config.sample_balanced, cwdm=config.cwdm,
    )

    if len(data_x_list) == 0:
        raise ValueError('无有效数据集，请重新检视数据集')

    data_y_list = [val.astype(float) for val in data_y_list]

    # 结果列定义
    res_cols = ['feat_name', 'indices', 'weight']
    result = pd.DataFrame([], columns=res_cols)

    feature_importance_value: np.ndarray | None = None
    feature_importance_index: np.ndarray | None = None
    feature_importance_name: np.ndarray | None = None

    for i in range(len(data_x_list)):
        data_x = data_x_list[i]
        data_y = data_y_list[i]
        feature_names = feature_name_list[i]
        filename = files[i]

        # 自动判断任务类型
        if len(np.unique(data_y)) > REGRESSION_LABEL_THRESHOLD:
            task = 'Regression'
            logger.info('main Y 的类型多于 20 种，采用 regression 回归方法进行分析')
        else:
            task = 'Classification'
            logger.info('main Y 的类型少于 20 种，采用 Classification 分类方法进行分析')

        # 准备输出目录
        file_name_temp = os.path.splitext(filename)[0]
        fig_path_sort = ensure_output_dir(config.output, file_name_temp)

        logger.info(f'main 目前正在分析: {filename}')
        logger.info(f'main 样本和特征结构: {data_x.shape}')
        logger.info(f'main label 总共有类数: {len(np.unique(data_y))}')

        # ---- 第二步：查看两两特征之间相关性 ----
        data_x_pd = pd.DataFrame(data_x, columns=feature_names)
        logger.info('main corr_matrix_DONE')

        # ---- 第三步：数据集划分和归一化 ----
        train, val, test, y_train, y_val, y_test = data_preprocessing(data_x, data_y)

        # ---- 第四步：基于 SIS 去掉确定不相关特征 ----
        train, val, test, y_train, y_val, y_test, feature_names = sure_independence_screening(
            train, val, test, y_train, y_val, y_test, feature_names,
        )
        logger.info('main sure_independence_screening_DONE')

        # ---- 第五步：基于 CWDM 进行快速初筛 ----
        if cwdm:
            selected_features, feature_counts, feature_ratios = smart_feature_selection(
                train, y_train,
                n_iterations=100,
                k_features='auto',
                final_k=min(CWDM_FINAL_K_MAX, train.shape[1]),
                random_state=None,
                n_blocks=5,
                data_threshold=10_000_000,
            )

            train_sec = train[:, selected_features]
            val_sec = val[:, selected_features]
            test_sec = test[:, selected_features]
            feature_names_sec = feature_names[selected_features]

            cwdm_features = np.arange(0, train_sec.shape[1])
            cwdm_features_2, convergence_points = fst_round_important_feature_index(
                train_sec, val_sec, y_train, y_val, cwdm_features, task,
            )

            train = train_sec[:, cwdm_features_2]
            val = val_sec[:, cwdm_features_2]
            test = test_sec[:, cwdm_features_2]
            feature_names = feature_names_sec[cwdm_features_2]

        # ---- 第六步：代理模型选取 ----
        if auto_selected_model and task == 'Classification':
            model_activation, best_model_name = best_model_training(
                train, val, test, y_train, y_val, y_test, 1,
            )
        elif not auto_selected_model and task == 'Classification':
            model_activation, best_model_name = best_model_training(
                train, val, test, y_train, y_val, y_test, proxy_model,
            )
        elif task == 'Regression':
            model_activation, best_model_name = best_model_training_regression(
                train, val, test, y_train, y_val, y_test,
            )

        # ---- 第七步：SAGE 评估 ----
        sage_train = sage_value(
            model_activation, train, y_train,
            batch_size=batch_size, thresh=thresh, task=task,
            n_jobs=n_jobs, bar=bar,
        )

        sage_val = sage_value(
            model_activation, val, y_val,
            batch_size=batch_size, thresh=thresh, task=task,
            n_jobs=n_jobs, bar=bar,
        )

        sage_test = sage_value(
            model_activation, test, y_test,
            batch_size=batch_size, thresh=thresh, task=task,
            n_jobs=n_jobs, bar=bar,
        )

        # ---- 第八步：画特征重要性对比图 ----
        plot_feature_importance_comparison(
            sage_train, sage_val, sage_test, feature_names,
            fig_path_sort, file_name_temp,
        )

        # 排序特征重要性
        feature_importance_value = -sage_train.values
        feature_importance_index = np.argsort(feature_importance_value)
        feature_importance_name = feature_names[feature_importance_index]
        feature_importance_value = sage_train.values[feature_importance_index]

        # ---- 第九步：逐特征加入性能评估图 ----
        if task == 'Classification':
            fig_show_classification(
                feature_importance_index, model_activation, best_model_name,
                train, val, test, y_train, y_val, y_test,
                fig_path_sort,
            )
        elif task == 'Regression':
            fig_show_regression(
                feature_importance_index, model_activation, best_model_name,
                train, val, test, y_train, y_val, y_test,
                fig_path_sort,
            )

        # 输出结果
        logger.info(f'main 处理文件: {filename}')
        logger.info(f'main 特征变量名(从最重要到最不重要): {feature_importance_name}')
        logger.info(f'main 特征变量序号(从重要到不重要): {feature_importance_index}')
        logger.info(f'main 特征变量重要性评估值(从重要到不重要): {feature_importance_value}')

        addition = pd.DataFrame(
            np.array([feature_importance_name, feature_importance_index, feature_importance_value]).T,
            columns=res_cols,
        )
        tail = pd.DataFrame([[np.nan] * len(res_cols)], columns=res_cols)
        result = pd.concat([addition, tail])

        result.to_csv('特征优选_result_特征重要性_' + filename + '.csv', index=False)

    return feature_importance_value, feature_importance_index, feature_importance_name


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    """命令行入口函数。

    Returns:
        None: 本方法无返回值。
    """
    freeze_support()

    # 初始化环境
    setup_environment()
    setup_matplotlib()
    setup_dual_logging()

    for i in range(1):
        logger.info(f'main 第 {i + 1} 次测试')
        config = parse_args()
        run_pipeline(config)


if __name__ == '__main__':
    main()