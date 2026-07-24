#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 reproduce_hbhd_arx.py，但将数据预处理替换为 run_sliding_window_eval.py 的滑窗逻辑。

训练数据源：/Users/panhui/Desktop/ph_hw/lanqu_code/HBHD_ARX_v2.0/data/data_episodes_for_soft_meas1.csv
推理数据源：/Users/panhui/Desktop/ph_hw/lanqu_code/HBHD_ARX_v2.0/data/total_clean_0710.csv

主要改动：
1. 使用基于时间 gap 的 chunk 切分替代 episode 切分。
2. 缺失值处理统一采用 0→NaN→按 chunk 线性插值→剩余填 0 的逻辑。
3. 目标列双指数平滑统一按 chunk 进行。
4. 滚动推理时通过 _get_valid_window_indices 过滤跨 chunk 窗口。
5. 保留对测试集的软测量修正，并增加 --no-softsensor 开关（默认开启）。
"""

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=UserWarning)

# 将 TSA-Suite src 加入当前进程 Python 路径，确保 Python API 可导入 tsas
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ==========================================
# 路径配置
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

HBHD_DIR = Path("/Users/panhui/Desktop/ph_hw/lanqu_code/HBHD_ARX_v2.0")
SOFTSENSOR_DIR = HBHD_DIR.parent / "HBHD_softsensor_v1.0"
TRAIN_CSV_DEFAULT = HBHD_DIR / "data" / "data_episodes_for_soft_meas1.csv"
TEST_CSV_DEFAULT = HBHD_DIR / "data" / "total_clean_0710.csv"

OUTPUT_DIR_DEFAULT = PROJECT_ROOT / "output" / "hbhd_arx_sliding"

# ==========================================
# 业务常量（与 HBHD 原脚本保持一致）
# ==========================================
TIME_COL_DEFAULT = "datatime"
TARGET_COL_DEFAULT = "diya_qibao_shuiwei_youxuanzhi"
INPUT_COLS = [
    "ningjie_shuiliuliang_youxuanzhi",
    "gaoya_geishuiliuliang_youxuanzhi",
    "chuyangqi_tiaojiefa_weizhifankui",
    "ranji_fuhe",
    "qiji_fuhe",
    "chuyangqi_rukou_yali",
    "diya_qibao_yali_youxuanzhi",
    "diya_zhuqiliuliang_youxuanzhi",
    "gaoya_jianwenshui_liuliang",
    "ranji_paiqi_wendu",
]

# 中文名 -> 英文名（用于软测量 JSON 构造）
JSON_TO_ENGLISH = {
    "凝结水流量优选值": "ningjie_shuiliuliang_youxuanzhi",
    "高压给水流量优选值": "gaoya_geishuiliuliang_youxuanzhi",
    "除氧器液位调节阀位置反馈": "chuyangqi_tiaojiefa_weizhifankui",
    "燃机发电机功率": "ranji_fuhe",
    "汽轮发电机有功功率": "qiji_fuhe",
    "除氧器入口压力": "chuyangqi_rukou_yali",
    "低压汽包压力优选值": "diya_qibao_yali_youxuanzhi",
    "低压主汽流量优选值": "diya_zhuqiliuliang_youxuanzhi",
    "高压减温水流量": "gaoya_jianwenshui_liuliang",
    "燃机排汽温度": "ranji_paiqi_wendu",
    "低压汽包水位优选值": TARGET_COL_DEFAULT,
    # 软测量额外依赖列
    "高压给水泵B液力耦合器勺管调节指": "gaoya_geishuibeng_b_shaoguan_tiaojiezhi",
    "高压给水压力": "gaoya_geishui_yali",
}
ENGLISH_TO_JSON = {v: k for k, v in JSON_TO_ENGLISH.items()}

HORIZONS = [30, 40, 50]
SEQ_LEN = 160
EMA_ALPHA_DEFAULT = 0.05
MAX_GAP_DEFAULT = 5.0
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# ==========================================
# 软测量模块导入
# ==========================================
try:
    sys.path.insert(0, str(SOFTSENSOR_DIR))
    from soft_infer import run_soft_sensor_inference, MODEL_CONFIG as SOFT_CONFIG
    SOFT_SENSOR_AVAILABLE = True
except ImportError as e:
    print(f"\n⚠️ 软测量模块导入失败: {e}，推理时将跳过软测量修正。")
    SOFT_SENSOR_AVAILABLE = False


# ==========================================
# 工具函数
# ==========================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cli(cmd: list[str]) -> None:
    """调用 TSAS CLI，失败时抛出 RuntimeError。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    full_cmd = [sys.executable, "-m", "tsas.engine.operator.cli", *cmd]
    print(f"\n[CLI] {' '.join(full_cmd)}")
    result = subprocess.run(full_cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"CLI 命令失败: {' '.join(full_cmd)}")
    print(result.stdout)


# ==========================================
# 预处理函数（移植/适配 run_sliding_window_eval.py）
# ==========================================
def detect_time_gaps(timestamps: pd.Series, max_gap: float) -> np.ndarray:
    """按时间差检测连续 chunk，返回每个样本的 chunk_id。"""
    ts = pd.to_datetime(timestamps, errors="coerce")
    diff = ts.diff().dt.total_seconds().fillna(0)
    diff.iloc[0] = 0.0
    chunk_ids = (diff > max_gap).cumsum()
    return chunk_ids.astype(int).to_numpy()


def impute_missing_values(data: np.ndarray, chunk_ids: np.ndarray) -> np.ndarray:
    """按 chunk 线性插补缺失值，剩余填 0。"""
    df = pd.DataFrame(data)
    df.replace(0, np.nan, inplace=True)
    df = df.groupby(chunk_ids, group_keys=False).apply(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    df.fillna(0, inplace=True)
    return df.values.astype(np.float32)


def double_ema_smooth(
    data: np.ndarray, chunk_ids: np.ndarray, target_idx: int, alpha: float
) -> np.ndarray:
    """Brown's double EMA on target column per chunk."""
    for cid in np.unique(chunk_ids):
        mask = chunk_ids == cid
        series = data[mask, target_idx]
        if len(series) == 0:
            continue
        s1 = pd.Series(series).ewm(alpha=alpha, adjust=False).mean().values
        s2 = pd.Series(s1).ewm(alpha=alpha, adjust=False).mean().values
        data[mask, target_idx] = 2 * s1 - s2
    return data


def _get_valid_window_indices(
    chunk_ids: np.ndarray, seq_len: int, pred_len: int
) -> np.ndarray:
    """返回不跨 chunk 边界的合法窗口起始索引。"""
    valid_indices = []
    unique_chunks = np.unique(chunk_ids)
    for cid in unique_chunks:
        mask = chunk_ids == cid
        chunk_row_indices = np.where(mask)[0]
        num_samples = len(chunk_row_indices) - seq_len - pred_len + 1
        if num_samples > 0:
            valid_indices.extend(chunk_row_indices[:num_samples])
    return np.array(valid_indices, dtype=int)


def _validate_preprocessed_data(
    df: pd.DataFrame,
    numeric_cols: list[str],
    split_name: str = "train",
) -> None:
    """检查预处理后的数据是否存在会导致训练失败的异常情况。"""
    values = df[numeric_cols].values.astype(np.float32)

    nan_count = np.isnan(values).sum()
    inf_count = np.isinf(values).sum()
    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"{split_name} 数据中存在 NaN/Inf: nan={nan_count}, inf={inf_count}\n"
            f"请检查原始数据或调整缺失值插补逻辑。"
        )

    std = df[numeric_cols].std()
    zero_std = std[std == 0].index.tolist()
    if zero_std:
        raise ValueError(
            f"{split_name} 数据中存在标准差为 0 的列（常量列）: {zero_std}\n"
            f"这些列会导致标准化后除以 0，从而产生 NaN。"
        )

    print(f"[VALIDATE] {split_name} 数据检查通过: 行数={len(df)}, 列数={len(numeric_cols)}")
    print(f"[VALIDATE] 各列 std 范围: [{std.min():.6f}, {std.max():.6f}]")


# ==========================================
# 软测量模块辅助函数（来自 reproduce_hbhd_arx.py）
# ==========================================
def csv_to_softsensor_payload(df: pd.DataFrame, time_col: str) -> dict:
    """将 DataFrame 转换为软测量模块期望的 JSON payload 格式。"""
    context = {}
    for chi_name, eng_name in JSON_TO_ENGLISH.items():
        if eng_name in df.columns:
            context[chi_name] = df[eng_name].tolist()

    if time_col in df.columns:
        context[SOFT_CONFIG.get("time_col", time_col)] = df[time_col].astype(str).tolist()

    return {"data": [{"context": context}]}


def apply_soft_sensor_to_csv(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """对 DataFrame 调用软测量模块，修正高压给水流量后返回。"""
    if not SOFT_SENSOR_AVAILABLE:
        print("\n⚠️ 软测量不可用，跳过修正。")
        return df

    soft_model_path = str(SOFTSENSOR_DIR / "soft_sensor" / "soft_sensor_config.json")
    if not Path(soft_model_path).exists():
        print(f"\n⚠️ 软测量模型不存在: {soft_model_path}，跳过修正。")
        return df

    print("\n🧲 [软测量前处理] 修正高压给水流量传感器异常...")
    payload = csv_to_softsensor_payload(df, time_col)
    df_fixed = run_soft_sensor_inference(payload, soft_model_path, SOFT_CONFIG)

    result = df.copy()
    for chi_name, eng_name in JSON_TO_ENGLISH.items():
        if chi_name in df_fixed.columns and eng_name in result.columns:
            result[eng_name] = df_fixed[chi_name].values

    return result


# ==========================================
# 训练数据准备
# ==========================================
def prepare_train_data(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    """读取训练 CSV，执行滑窗预处理，保存处理后的文件。"""
    print("\n[1/5] 准备训练数据...")
    processed_dir = output_dir / "processed"
    ensure_dir(processed_dir)

    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        feature_cols = INPUT_COLS.copy()

    cols_to_read = [args.time_col, args.target_col, *feature_cols]
    df = pd.read_csv(args.train_csv, usecols=cols_to_read)
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col]).reset_index(drop=True)

    for col in feature_cols + [args.target_col]:
        if col not in df.columns:
            raise ValueError(f"训练数据缺少列: {col}")

    numeric_cols = feature_cols + [args.target_col]
    target_idx = len(feature_cols)

    # gap 检测
    chunk_ids = detect_time_gaps(df[args.time_col], args.max_gap)
    n_chunks = len(np.unique(chunk_ids))
    print(f"   -> 训练数据共 {len(df)} 行，划分为 {n_chunks} 个 chunk")

    # 数值矩阵
    data = df[numeric_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

    # 缺失值插补
    data = impute_missing_values(data, chunk_ids)
    print("   -> 已完成缺失值插补（0→NaN→线性插值→填0）")

    # 目标列平滑
    if args.smooth_target:
        data = double_ema_smooth(data, chunk_ids, target_idx, args.smooth_alpha)
        print(f"   -> 已完成目标列双指数平滑（alpha={args.smooth_alpha}）")

    df[numeric_cols] = data

    # 校验
    _validate_preprocessed_data(df, numeric_cols, split_name="train")

    # 列顺序：外生变量在前，目标列在最后（配合 ridge_forecaster 默认 target_idx=-1）
    ordered_cols = [args.time_col] + numeric_cols
    df = df[ordered_cols]

    train_csv = processed_dir / "train_processed.csv"
    df.to_csv(train_csv, index=False)

    chunk_ids_csv = processed_dir / "train_chunk_ids.csv"
    pd.DataFrame({"chunk_id": chunk_ids}).to_csv(chunk_ids_csv, index=False, header=False)

    print(f"   -> 已保存: {train_csv}")
    print(f"   -> 已保存: {chunk_ids_csv}")
    return train_csv, chunk_ids_csv


# ==========================================
# 通过 CLI 训练 Ridge 模型
# ==========================================
def build_forecaster_config_yaml(args: argparse.Namespace, save_dir: Path) -> Path:
    """为指定 horizon 生成 ridge_forecaster 的 YAML 配置文件。"""
    ensure_dir(save_dir)

    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        feature_cols = INPUT_COLS.copy()

    config = {
        "operator": {
            "name": "ridge_forecaster",
            "input_columns": feature_cols + [args.target_col],
            "target_column": args.target_col,
            "config": {
                "seq_len": SEQ_LEN,
                "horizon": args.plot_horizon,
                "target_idx": -1,
                "alphas": [0.1, 1.0, 10.0, 100.0],
                "fit_intercept": True,
                "solver": "lsqr",
                "standardize": True,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "train_sample_step": max(1, SEQ_LEN // 10),
                "val_sample_step": 4,
                "seed": 42,
            },
        }
    }
    config_path = save_dir / f"ridge_config_h{args.plot_horizon}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return config_path


def train_models(train_csv: Path, chunk_ids_csv: Path, args: argparse.Namespace, output_dir: Path) -> None:
    """调用 TSAS CLI 训练指定 horizon 的 Ridge 模型。"""
    print(f"\n[2/5] 通过 TSAS CLI 训练 Ridge 模型 (horizon={args.plot_horizon})...")
    saved_models_dir = output_dir / "saved_models"
    ensure_dir(saved_models_dir)

    config_path = build_forecaster_config_yaml(args, saved_models_dir)
    model_dir = saved_models_dir / f"ridge_h{args.plot_horizon}"

    run_cli([
        "forecasting", "fit",
        "--input", str(train_csv),
        "--target", args.target_col,
        "--config", str(config_path),
        "--save", str(model_dir),
        "--chunk-ids", str(chunk_ids_csv),
        "--seed", "42",
    ])


# ==========================================
# 推理数据准备（含软测量）
# ==========================================
def prepare_test_data(args: argparse.Namespace) -> pd.DataFrame:
    """读取测试 CSV，执行软测量修正和滑窗预处理，返回处理后的 DataFrame。"""
    print("\n[3/5] 准备推理数据...")

    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        feature_cols = INPUT_COLS.copy()

    cols_to_read = [args.time_col, args.target_col, *feature_cols]
    df = pd.read_csv(args.test_csv, usecols=cols_to_read)
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col]).reset_index(drop=True)

    for col in feature_cols + [args.target_col]:
        if col not in df.columns:
            raise ValueError(f"测试数据缺少列: {col}")

    # 软测量修正（可开关）
    if args.softsensor:
        df = apply_soft_sensor_to_csv(df, args.time_col)
    else:
        print("\n⚠️ 根据参数跳过软测量修正。")

    numeric_cols = feature_cols + [args.target_col]
    target_idx = len(feature_cols)

    # gap 检测
    chunk_ids = detect_time_gaps(df[args.time_col], args.max_gap)
    n_chunks = len(np.unique(chunk_ids))
    print(f"   -> 测试数据共 {len(df)} 行，划分为 {n_chunks} 个 chunk")

    # 数值矩阵
    data = df[numeric_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

    # 缺失值插补
    data = impute_missing_values(data, chunk_ids)
    print("   -> 已完成缺失值插补")

    # 目标列平滑
    if args.smooth_target:
        data = double_ema_smooth(data, chunk_ids, target_idx, args.smooth_alpha)
        print(f"   -> 已完成目标列双指数平滑（alpha={args.smooth_alpha}）")

    df[numeric_cols] = data

    # 列顺序：时间列 + 外生变量 + 目标列
    ordered_cols = [args.time_col] + numeric_cols
    df = df[ordered_cols]

    print(f"   -> 测试数据共 {len(df)} 行")
    return df


# ==========================================
# 滚动推理
# ==========================================
def load_ridge_model(horizon: int, output_dir: Path) -> object:
    """从保存目录加载指定 horizon 的 ridge_forecaster 模型。"""
    from tsas.engine.operator.forecasting.ridge import RidgeForecaster

    model_dir = output_dir / "saved_models" / f"ridge_h{horizon}"
    return RidgeForecaster.load(model_dir)


def rolling_inference(
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对指定 horizon 做批量滚动推理，返回预测值与对应真值。"""
    model = load_ridge_model(args.plot_horizon, output_dir)

    cols = [col for col in df.columns if col != args.time_col]
    data = df[cols].to_numpy(dtype=np.float32)
    target_col_idx = cols.index(args.target_col)

    total_len = len(data)
    if total_len < SEQ_LEN + args.plot_horizon:
        raise ValueError(f"测试数据过短: {total_len} < {SEQ_LEN + args.plot_horizon}")

    # 检测 chunk 并计算合法窗口起始索引
    chunk_ids = detect_time_gaps(df[args.time_col], args.max_gap)
    valid_starts = _get_valid_window_indices(chunk_ids, SEQ_LEN, args.plot_horizon)
    if len(valid_starts) == 0:
        raise ValueError(
            f"没有可用样本：所有 chunk 的长度都小于 "
            f"seq_len + horizon = {SEQ_LEN + args.plot_horizon}"
        )

    # 按 step 子采样
    if args.step > 1:
        valid_starts = valid_starts[::args.step]

    n_windows = len(valid_starts)
    window_end_indices = valid_starts + SEQ_LEN - 1

    # 构造批量窗口输入 (n_windows, seq_len, num_features)
    x_test = np.stack([data[i : i + SEQ_LEN] for i in valid_starts], axis=0)

    # 构造真值：每个窗口未来第 horizon 步的目标值
    y_true = np.array([
        data[i + SEQ_LEN + args.plot_horizon - 1, target_col_idx]
        for i in valid_starts
    ], dtype=np.float32)

    # 批量推理
    preds = []
    n_batches = (n_windows + args.eval_batch_size - 1) // args.eval_batch_size
    for i in range(0, n_windows, args.eval_batch_size):
        batch = x_test[i : i + args.eval_batch_size]
        pred = model.run(batch)  # (batch_size, 1, 1)
        preds.append(pred.reshape(-1))
        print(f"    -> 推理进度: {min(i + args.eval_batch_size, n_windows)}/{n_windows} "
              f"(batch {i // args.eval_batch_size + 1}/{n_batches})")

    y_pred = np.concatenate(preds, axis=0)

    return y_pred, y_true, window_end_indices


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> tuple[float, float, float]:
    """计算 RMSE、MAE、MAPE。"""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / (np.abs(y_true) + epsilon))) * 100)
    return rmse, mae, mape


def apply_post_processing_ema(raw_preds: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """对流式预测结果做二阶指数平滑（与 main_roll_arx.py 一致）。"""
    raw_preds = np.asarray(raw_preds, dtype=float)
    s1 = raw_preds.copy()
    s2 = raw_preds.copy()
    smoothed = np.empty_like(raw_preds)
    smoothed[0] = raw_preds[0]

    for i in range(1, len(raw_preds)):
        s1[i] = alpha * raw_preds[i] + (1.0 - alpha) * s1[i - 1]
        s2[i] = alpha * s1[i] + (1.0 - alpha) * s2[i - 1]
        smoothed[i] = 2.0 * s1[i] - s2[i]

    return smoothed


# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="HBHD Ridge-ARX 滑窗预处理复现脚本")

    # 数据路径
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV_DEFAULT,
                        help="训练 CSV 路径")
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV_DEFAULT,
                        help="测试/推理 CSV 路径")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT,
                        help="输出目录")
    parser.add_argument("--time-col", default=TIME_COL_DEFAULT, help="时间列名")
    parser.add_argument("--target-col", default=TARGET_COL_DEFAULT, help="目标列名")
    parser.add_argument("--feature-cols", default="",
                        help="逗号分隔的特征列名，为空则使用 HBHD 默认输入列")

    # 预处理
    parser.add_argument("--max-gap", type=float, default=MAX_GAP_DEFAULT,
                        help="时间断点阈值（秒）")
    parser.add_argument("--smooth-target", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="是否对目标列做双指数平滑（默认开启）")
    parser.add_argument("--smooth-alpha", type=float, default=EMA_ALPHA_DEFAULT,
                        help="EMA 平滑系数")

    # 软测量
    parser.add_argument("--softsensor", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="是否对测试集应用软测量修正（默认开启）")

    # 训练/推理
    parser.add_argument("--plot-horizon", type=int, default=30,
                        choices=HORIZONS, help="重点评估的预测步长")
    parser.add_argument("--step", type=int, default=1, help="滚动推理步长")
    parser.add_argument("--eval-batch-size", type=int, default=256,
                        help="批量推理时每批窗口数量")
    parser.add_argument("--post-ema", type=float, default=0.1,
                        help="后处理二阶指数平滑的 Alpha 系数")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，直接加载已有模型")
    parser.add_argument("--skip-inference", action="store_true", help="跳过推理")

    args = parser.parse_args()

    output_dir = args.output_dir

    if not args.train_csv.exists():
        raise FileNotFoundError(f"训练数据不存在: {args.train_csv}")
    if not args.test_csv.exists():
        raise FileNotFoundError(f"测试数据不存在: {args.test_csv}")

    # 1. 准备训练数据
    train_csv, chunk_ids_csv = prepare_train_data(args, output_dir)

    # 2. 通过 CLI 训练模型
    if not args.skip_train:
        train_models(train_csv, chunk_ids_csv, args, output_dir)
    else:
        print("\n[2/5] 跳过训练，使用已有模型...")

    # 3. 准备推理数据
    test_df = prepare_test_data(args)

    # 4. 滚动推理
    if not args.skip_inference:
        print(f"\n[4/5] 滚动推理 (horizon={args.plot_horizon}, step={args.step})...")
        preds_raw, trues, _ = rolling_inference(test_df, args, output_dir)

        # 后处理二阶指数平滑
        preds_smoothed = apply_post_processing_ema(preds_raw, alpha=args.post_ema)

        rmse_raw, mae_raw, mape_raw = calculate_metrics(trues, preds_raw)
        rmse_smooth, mae_smooth, mape_smooth = calculate_metrics(trues, preds_smoothed)

        print(f"\n📊 全局测试集指标 (T+{args.plot_horizon})：")
        print(f"   [Raw Pred]      RMSE: {rmse_raw:.4f}, MAE: {mae_raw:.4f}, MAPE: {mape_raw:.4f}%")
        print(f"   [Smoothed Pred] RMSE: {rmse_smooth:.4f}, MAE: {mae_smooth:.4f}, MAPE: {mape_smooth:.4f}%")

        # 保存结果
        result_csv = output_dir / f"rolling_result_h{args.plot_horizon}.csv"
        pd.DataFrame({
            "true": trues,
            "pred_raw": preds_raw,
            "pred_smoothed": preds_smoothed,
        }).to_csv(result_csv, index=False)
        print(f"   -> 结果已保存: {result_csv}")

    print("\n✅ 脚本执行完毕")


if __name__ == "__main__":
    main()
