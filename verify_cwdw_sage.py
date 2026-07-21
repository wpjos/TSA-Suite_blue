# -*- coding: utf-8 -*-

"""验证 cwdw_sage CLI 包装结果。

使用方式::

    PYTHONPATH=src python verify_cwdw_sage.py your_data.csv --label label

说明：
    - 本脚本会对输入 CSV 做简单清洗（全部转数值、删除含 NaN 行、删除方差为 0 的特征列）。
    - CLI 路径：先 ``fit --label-column`` 训练并保存模型，再 ``run --load`` 加载运行。
    - 输出最终返回的 ``selected_indices``（原始输入列位置，按 SAGE 重要性排序）、
      ``ranked_features``、``proxy_model_name`` 与 ``task``。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def clean_dataframe(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """对输入 DataFrame 做简单清洗，保持列顺序不变。"""
    if label_col not in df.columns:
        raise ValueError(f"输入数据缺少标签列 '{label_col}'")

    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    df = df.dropna(axis=1, how='all')

    zero_std_cols = [c for c in df.columns if c != label_col and df[c].std() <= 0]
    if zero_std_cols:
        df = df.drop(columns=zero_std_cols)

    df = df.dropna(axis=0, how='any')

    if label_col not in df.columns:
        raise ValueError(f"清洗后标签列 '{label_col}' 丢失，请检查数据")

    return df


def run_cli_path(df: pd.DataFrame, label_col: str, seed: int) -> dict:
    """通过 feature_selection CLI 先 fit 再 run，返回附加输出 JSON 内容。"""
    from tsas.engine.operator.cli import feature_selection

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / 'input.csv'
        config_path = tmp / 'config.json'
        model_dir = tmp / 'model'
        output_path = tmp / 'output.csv'
        eo_path = tmp / 'eo.json'

        df.to_csv(csv_path, index=False)
        config = {
            'operator': {
                'name': 'cwdw_sage',
                'config': {
                    'task': 'auto',
                    'sis_threshold': 0.05,
                    'cwdm_n_iterations': 100,
                    'cwdm_final_k': 150,
                    'cwdm_n_blocks': 5,
                    'cwdm_data_threshold': 10_000_000,
                    'convergence_stability_threshold': 0.001,
                    'convergence_accuracy_threshold': 0.75,
                    'sage_n_jobs': 1,
                    'random_state': seed,
                },
            }
        }
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')

        np.random.seed(seed)
        # 1. fit
        feature_selection.main([
            'fit',
            '--input', str(csv_path),
            '--config', str(config_path),
            '--model-dir', str(model_dir),
            '--label-column', label_col,
        ])

        # 2. run（加载已训练模型）
        feature_selection.main([
            'run',
            '--input', str(csv_path),
            '--config', str(config_path),
            '--load', str(model_dir),
            '--output', str(output_path),
            '--eo-output', str(eo_path),
            '--label-column', label_col,
        ])

        return json.loads(eo_path.read_text(encoding='utf-8'))


def main() -> int:
    parser = argparse.ArgumentParser(description='验证 cwdw_sage CLI 结果')
    parser.add_argument('csv', help='输入 CSV 文件路径')
    parser.add_argument('--label', default='label', help='标签列名，默认 "label"')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于 CWDM 打乱')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = clean_dataframe(df, args.label)
    print(f'清洗后数据形状: {df.shape}')

    print('\n--- CLI 路径 ---')
    cli_eo = run_cli_path(df, args.label, args.seed)
    cli_selected = cli_eo['selected_indices']
    cli_task = cli_eo['task']
    cli_proxy = cli_eo.get('proxy_model_name', 'Unknown')
    cli_ranked = cli_eo.get('ranked_features', [])
    print(f'CLI selected_indices: {cli_selected}')
    print(f'CLI task: {cli_task}')
    print(f'CLI proxy_model_name: {cli_proxy}')

    # 把 SAGE 评估值按 selected_indices 顺序对齐
    cli_sage_values_raw = cli_eo.get('sage_values', {})
    cli_sage_values = {int(k): float(v) for k, v in cli_sage_values_raw.items()}
    cli_weights = [cli_sage_values.get(i, float('nan')) for i in cli_selected]

    print('\n--- CLI selected_indices 与 SAGE 评估值汇总（按重要性从高到低） ---')
    print(f'indices={cli_selected}')
    print(f'weights={cli_weights}')

    print('\n✅ CLI 路径执行完成')
    return 0


if __name__ == '__main__':
    sys.exit(main())
