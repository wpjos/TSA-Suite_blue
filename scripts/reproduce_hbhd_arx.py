#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
复现 HBHD_ARX_v2.0 的 Ridge-ARX 训练与滚动推理流程

训练数据源：/Users/panhui/Desktop/ph_hw/lanqu_code/HBHD_ARX_v2.0/data/data_episodes_for_soft_meas1.csv
推理数据源：/Users/panhui/Desktop/ph_hw/lanqu_code/HBHD_ARX_v2.0/data/total_clean_0710.csv

本脚本职责：
1. 数据预处理（episode 切分、目标列双指数平滑）
2. 通过 TSAS CLI 调用 ridge_forecaster 训练每个 horizon 的 Ridge 模型
3. 推理前调用软测量模块修正高压给水流量
4. 使用 ridge_forecaster 对每个 horizon 做滚动推理
5. 计算并输出全局测试集指标

注意：
- 训练阶段通过 subprocess 调用 TSAS CLI ``forecasting fit``。
- 滚动推理阶段为效率起见直接通过 Python API 加载已训练模型并调用 ``run()``；
  脚本末尾会额外演示一次通过 TSAS CLI ``forecasting run`` 做单窗口推理。
"""

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 路径配置
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

HBHD_DIR = Path("/Users/panhui/Desktop/ph_hw/lanqu_code/HBHD_ARX_v2.0")
SOFTSENSOR_DIR = HBHD_DIR.parent / "HBHD_softsensor_v1.0"
TRAIN_CSV = HBHD_DIR / "data" / "data_episodes_for_soft_meas1.csv"
TEST_CSV = HBHD_DIR / "data" / "total_clean_0710.csv"

OUTPUT_DIR = PROJECT_ROOT / "output" / "hbhd_arx_reproduction"
SAVED_MODELS_DIR = OUTPUT_DIR / "saved_models"
PROCESSED_DIR = OUTPUT_DIR / "processed"

# ==========================================
# 业务常量（与 HBHD 原脚本保持一致）
# ==========================================
TIME_COL = "datatime"
TARGET_COL = "diya_qibao_shuiwei_youxuanzhi"
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
    "低压汽包水位优选值": TARGET_COL,
    # 软测量额外依赖列
    "高压给水泵B液力耦合器勺管调节指": "gaoya_geishuibeng_b_shaoguan_tiaojiezhi",
    "高压给水压力": "gaoya_geishui_yali",
}
ENGLISH_TO_JSON = {v: k for k, v in JSON_TO_ENGLISH.items()}

HORIZONS = [30, 40, 50]
SEQ_LEN = 160
EMA_ALPHA = 0.05
MAX_GAP_SECONDS = 5.0
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
def detect_episodes(timestamps: pd.Series) -> np.ndarray:
    """按时间差检测连续 episode，返回每个样本的 episode_id。"""
    time_diff = timestamps.diff().dt.total_seconds()
    new_episode = time_diff.isna() | (time_diff > MAX_GAP_SECONDS) | (time_diff <= 0)
    return new_episode.cumsum().to_numpy(dtype=int)


def double_ema_by_episode(values: np.ndarray, episode_ids: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """按 episode 分组做双指数平滑。"""
    smoothed = np.empty(len(values), dtype=float)
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        series = pd.Series(values[mask])
        s1 = series.ewm(alpha=alpha, adjust=False).mean().to_numpy()
        s2 = pd.Series(s1).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        smoothed[mask] = 2.0 * s1 - s2
    return smoothed


def apply_double_ema_global(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """全局双指数平滑。"""
    series = pd.Series(values)
    s1 = series.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    s2 = pd.Series(s1).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return 2.0 * s1 - s2


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
# 训练数据准备
# ==========================================
def prepare_train_data() -> tuple[Path, Path]:
    """读取训练 CSV，执行 episode 切分和目标列平滑，保存处理后的文件。

    Returns:
        tuple[Path, Path]: (processed_train_csv, chunk_ids_csv)
    """
    print("\n[1/5] 准备训练数据...")
    ensure_dir(PROCESSED_DIR)

    usecols = [TIME_COL, TARGET_COL, *INPUT_COLS]
    df = pd.read_csv(TRAIN_CSV, usecols=usecols)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).reset_index(drop=True)

    episode_ids = detect_episodes(df[TIME_COL])
    print(f"   -> 训练数据共 {len(df)} 行，划分为 {len(np.unique(episode_ids))} 个 episode")

    # 目标列双指数平滑（训练时按 episode）
    target_raw = df[TARGET_COL].to_numpy(dtype=float)
    target_smooth = double_ema_by_episode(target_raw, episode_ids)
    df[TARGET_COL] = target_smooth

    # 列顺序：外生变量在前，目标列在最后（配合 ridge_forecaster 默认 target_idx=-1）
    ordered_cols = [col for col in INPUT_COLS if col in df.columns] + [TARGET_COL]
    df = df[ordered_cols]

    train_csv = PROCESSED_DIR / "train_processed.csv"
    df.to_csv(train_csv, index=False)

    chunk_ids_csv = PROCESSED_DIR / "train_chunk_ids.csv"
    pd.DataFrame({"chunk_id": episode_ids}).to_csv(chunk_ids_csv, index=False, header=False)

    print(f"   -> 已保存: {train_csv}")
    print(f"   -> 已保存: {chunk_ids_csv}")
    return train_csv, chunk_ids_csv


# ==========================================
# 通过 CLI 训练 Ridge 模型
# ==========================================
def build_forecaster_config_yaml(horizon: int, save_dir: Path) -> Path:
    """为指定 horizon 生成 ridge_forecaster 的 YAML 配置文件。"""
    ensure_dir(save_dir)
    config = {
        "operator": {
            "name": "ridge_forecaster",
            "input_columns": INPUT_COLS + [TARGET_COL],
            "target_column": TARGET_COL,
            "config": {
                "seq_len": SEQ_LEN,
                "horizon": horizon,
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
    config_path = save_dir / f"ridge_config_h{horizon}.yaml"
    import yaml
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return config_path


def train_models(train_csv: Path, chunk_ids_csv: Path, horizon: int) -> None:
    """调用 TSAS CLI 训练指定 horizon 的 Ridge 模型。"""
    print(f"\n[2/5] 通过 TSAS CLI 训练 Ridge 模型 (horizon={horizon})...")
    ensure_dir(SAVED_MODELS_DIR)

    config_path = build_forecaster_config_yaml(horizon, SAVED_MODELS_DIR)
    model_dir = SAVED_MODELS_DIR / f"ridge_h{horizon}"

    run_cli([
        "forecasting", "fit",
        "--input", str(train_csv),
        "--target", TARGET_COL,
        "--config", str(config_path),
        "--save", str(model_dir),
        "--chunk-ids", str(chunk_ids_csv),
        "--seed", "42",
    ])


# ==========================================
# 推理数据准备（含软测量）
# ==========================================
def csv_to_softsensor_payload(df: pd.DataFrame) -> dict:
    """将 DataFrame 转换为软测量模块期望的 JSON payload 格式。"""
    context = {}
    for chi_name, eng_name in JSON_TO_ENGLISH.items():
        if eng_name in df.columns:
            context[chi_name] = df[eng_name].tolist()

    if TIME_COL in df.columns:
        context[SOFT_CONFIG.get("time_col", TIME_COL)] = df[TIME_COL].astype(str).tolist()

    return {"data": [{"context": context}]}


def apply_soft_sensor_to_csv(df: pd.DataFrame) -> pd.DataFrame:
    """对 DataFrame 调用软测量模块，修正高压给水流量后返回。"""
    if not SOFT_SENSOR_AVAILABLE:
        print("\n⚠️ 软测量不可用，跳过修正。")
        return df

    soft_model_path = str(SOFTSENSOR_DIR / "soft_sensor" / "soft_sensor_config.json")
    if not Path(soft_model_path).exists():
        print(f"\n⚠️ 软测量模型不存在: {soft_model_path}，跳过修正。")
        return df

    print("\n🧲 [软测量前处理] 修正高压给水流量传感器异常...")
    payload = csv_to_softsensor_payload(df)
    df_fixed = run_soft_sensor_inference(payload, soft_model_path, SOFT_CONFIG)

    # 将软测量结果中的中文列名映射回英文
    result = df.copy()
    for chi_name, eng_name in JSON_TO_ENGLISH.items():
        if chi_name in df_fixed.columns and eng_name in result.columns:
            result[eng_name] = df_fixed[chi_name].values

    return result


def prepare_test_data() -> pd.DataFrame:
    """读取测试 CSV，执行软测量修正和目标列平滑，返回处理后的 DataFrame。"""
    print("\n[3/5] 准备推理数据...")
    df = pd.read_csv(TEST_CSV)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).reset_index(drop=True)

    # 软测量修正
    df = apply_soft_sensor_to_csv(df)

    # 目标列双指数平滑（推理时全局平滑，与 main_arx.py 一致）
    target_raw = df[TARGET_COL].to_numpy(dtype=float)
    df[TARGET_COL] = apply_double_ema_global(target_raw, EMA_ALPHA)

    # 缺失值处理
    numeric_cols = [c for c in df.columns if c != TIME_COL]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # 保留需要的列，目标列放最后
    available_input_cols = [col for col in INPUT_COLS if col in df.columns]
    df = df[[TIME_COL, *available_input_cols, TARGET_COL]]

    print(f"   -> 测试数据共 {len(df)} 行")
    return df


# ==========================================
# 滚动推理
# ==========================================
def load_ridge_model(horizon: int) -> object:
    """从保存目录加载指定 horizon 的 ridge_forecaster 模型。"""
    from tsas.engine.operator.forecasting.ridge import RidgeForecaster

    model_dir = SAVED_MODELS_DIR / f"ridge_h{horizon}"
    return RidgeForecaster.load(model_dir)


def rolling_inference(
    df: pd.DataFrame,
    plot_horizon: int,
    step: int = 1,
    eval_batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对指定 horizon 做批量滚动推理，返回预测值与对应真值。

    Args:
        df: 测试数据 DataFrame
        plot_horizon: 预测步长
        step: 滚动步长
        eval_batch_size: 每次推理的窗口批量大小

    Returns:
        (preds, trues, window_end_indices)
    """
    model = load_ridge_model(plot_horizon)

    cols = [col for col in df.columns if col != TIME_COL]
    data = df[cols].to_numpy(dtype=np.float32)
    target_col_idx = cols.index(TARGET_COL)

    total_len = len(data)
    if total_len < SEQ_LEN + plot_horizon:
        raise ValueError(f"测试数据过短: {total_len} < {SEQ_LEN + plot_horizon}")

    # 构造所有窗口起始索引
    n_windows = (total_len - SEQ_LEN - plot_horizon + 1) // step
    window_starts = np.arange(n_windows, dtype=int) * step
    window_end_indices = window_starts + SEQ_LEN - 1

    # 构造批量窗口输入 (n_windows, seq_len, num_features)
    x_test = np.stack([data[i : i + SEQ_LEN] for i in window_starts], axis=0)

    # 构造真值：每个窗口未来第 horizon 步的目标值
    y_true = np.array([
        data[i + SEQ_LEN + plot_horizon - 1, target_col_idx]
        for i in window_starts
    ], dtype=np.float32)

    # 批量推理
    preds = []
    n_batches = (n_windows + eval_batch_size - 1) // eval_batch_size
    for i in range(0, n_windows, eval_batch_size):
        batch = x_test[i : i + eval_batch_size]
        pred = model.run(batch)  # (batch_size, 1, 1)
        preds.append(pred.reshape(-1))
        print(f"    -> 推理进度: {min(i + eval_batch_size, n_windows)}/{n_windows} "
              f"(batch {i // eval_batch_size + 1}/{n_batches})")

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
    """对流式预测结果做二阶指数平滑（与 main_roll_arx.py 一致）。

    Args:
        raw_preds: 原始预测序列，形状 ``(n_windows,)``
        alpha: EMA 平滑系数

    Returns:
        np.ndarray: 平滑后的预测序列，形状 ``(n_windows,)``
    """
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
    parser = argparse.ArgumentParser(description="复现 HBHD_ARX_v2.0 Ridge-ARX 训练与推理")
    parser.add_argument("--plot-horizon", type=int, default=30,
                        choices=HORIZONS, help="重点评估的预测步长")
    parser.add_argument("--step", type=int, default=1, help="滚动推理步长")
    parser.add_argument("--eval-batch-size", type=int, default=256,
                        help="批量推理时每批窗口数量")
    parser.add_argument("--post-ema", type=float, default=0.1,
                        help="后处理二阶指数平滑的 Alpha 系数")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，直接加载已有模型")
    parser.add_argument("--skip-inference", action="store_true", help="跳过推理")
    parser.add_argument("--skip-softsensor", action="store_true", help="跳过软测量修正")
    args = parser.parse_args()

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"训练数据不存在: {TRAIN_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"测试数据不存在: {TEST_CSV}")

    # 1. 准备训练数据
    train_csv, chunk_ids_csv = prepare_train_data()

    # 2. 通过 CLI 训练模型
    if not args.skip_train:
        train_models(train_csv, chunk_ids_csv, args.plot_horizon)
    else:
        print("\n[2/5] 跳过训练，使用已有模型...")

    # 3. 准备推理数据
    test_df = prepare_test_data()
    if args.skip_softsensor:
        print("\n⚠️ 根据参数跳过软测量修正（数据仍保留）")

    # 4. 滚动推理
    if not args.skip_inference:
        print(f"\n[4/5] 滚动推理 (horizon={args.plot_horizon}, step={args.step})...")
        preds_raw, trues, _ = rolling_inference(test_df, args.plot_horizon, step=args.step,
                                                eval_batch_size=args.eval_batch_size)

        # 后处理二阶指数平滑
        preds_smoothed = apply_post_processing_ema(preds_raw, alpha=args.post_ema)

        rmse_raw, mae_raw, mape_raw = calculate_metrics(trues, preds_raw)
        rmse_smooth, mae_smooth, mape_smooth = calculate_metrics(trues, preds_smoothed)

        print(f"\n📊 全局测试集指标 (T+{args.plot_horizon})：")
        print(f"   [Raw Pred]      RMSE: {rmse_raw:.4f}, MAE: {mae_raw:.4f}, MAPE: {mape_raw:.4f}%")
        print(f"   [Smoothed Pred] RMSE: {rmse_smooth:.4f}, MAE: {mae_smooth:.4f}, MAPE: {mape_smooth:.4f}%")

        # 保存结果
        result_csv = OUTPUT_DIR / f"rolling_result_h{args.plot_horizon}.csv"
        pd.DataFrame({
            "true": trues,
            "pred_raw": preds_raw,
            "pred_smoothed": preds_smoothed,
        }).to_csv(result_csv, index=False)
        print(f"   -> 结果已保存: {result_csv}")

    print("\n✅ 复现脚本执行完毕")


if __name__ == "__main__":
    main()
