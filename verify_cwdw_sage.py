# -*- coding: utf-8 -*-

"""验证 cwdw_sage CLI 包装与 feature_selection_by_cwdw_sage.py 三步选择结果是否一致。

使用方式::

    PYTHONPATH=src python verify_cwdw_sage.py your_data.csv --label label

说明：
    - 本脚本会对输入 CSV 做简单清洗（全部转数值、删除含 NaN 行、删除方差为 0 的特征列）。
    - CLI 路径：调用 ``feature_selection`` 子命令的 ``run``。
    - 直接路径：调用 ``feature_selection_by_cwdw_sage.py`` 中的
      ``SISFilter / CWDMSelector / ConvergenceFinder`` 和 ``data_preprocessing``。
    - 比较两者最终返回的 ``selected_indices``（原始输入列位置）。
    - 注意：原脚本后续还有模型训练、SAGE 评估、可视化等步骤，本脚本只比较
      SIS + CWDM + ConvergenceFinder 这三步的选择结果。
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

    # 全部尝试转数值
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

    # 删除全为 NaN 的列
    df = df.dropna(axis=1, how='all')

    # 删除方差为 0 的特征列（保留标签列）
    zero_std_cols = [c for c in df.columns if c != label_col and df[c].std() <= 0]
    if zero_std_cols:
        df = df.drop(columns=zero_std_cols)

    # 删除包含 NaN 的行
    df = df.dropna(axis=0, how='any')

    if label_col not in df.columns:
        raise ValueError(f"清洗后标签列 '{label_col}' 丢失，请检查数据")

    return df


def run_cli_path(df: pd.DataFrame, label_col: str, seed: int) -> dict:
    """通过 feature_selection CLI 运行 cwdw_sage，返回附加输出 JSON 内容。"""
    from tsas.engine.operator.cli import feature_selection

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / 'input.csv'
        config_path = tmp / 'config.json'
        output_path = tmp / 'output.csv'
        eo_path = tmp / 'eo.json'

        df.to_csv(csv_path, index=False)
        config = {
            'operator': {
                'name': 'cwdw_sage',
                'config': {
                    'label_column': label_col,
                    'task': 'auto',
                    'sis_threshold': 0.05,
                    'cwdm_n_iterations': 100,
                    'cwdm_final_k': 150,
                    'cwdm_n_blocks': 5,
                    'cwdm_data_threshold': 10_000_000,
                    'convergence_stability_threshold': 1e-3,
                    'convergence_accuracy_threshold': 0.75,
                },
            }
        }
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')

        np.random.seed(seed)
        feature_selection.main([
            'run',
            '--input', str(csv_path),
            '--config', str(config_path),
            '--output', str(output_path),
            '--eo-output', str(eo_path),
        ])

        return json.loads(eo_path.read_text(encoding='utf-8'))


def run_direct_path(df: pd.DataFrame, label_col: str, seed: int) -> tuple[list[int], str]:
    """直接调用原脚本中的三步选择逻辑，返回原始列位置索引和任务类型。"""
    from tsas.engine.operator.feature.selection.feature_selection_by_cwdw_sage import (
        CWDMSelector,
        CWDMSelectorConfig,
        ConvergenceFinder,
        ConvergenceFinderConfig,
        SISFilter,
        SISFilterConfig,
        data_preprocessing,
    )

    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col]).to_numpy()

    task = (
        'Regression'
        if len(np.unique(y)) > 20
        else 'Classification'
    )

    np.random.seed(seed)
    train, val, test, y_train, y_val, y_test = data_preprocessing(
        X, y, normalization_method='z-score', task=task
    )

    # 1. SIS
    sis = SISFilter(config=SISFilterConfig(threshold=0.05))
    feature_names = np.arange(X.shape[1]).astype(str)
    train, val, test, y_train, y_val, y_test, feature_names, sis_eo = sis.filter(
        train, val, test, y_train, y_val, y_test, feature_names
    )

    # 2. CWDM
    final_k = min(150, train.shape[1])
    cwdm = CWDMSelector(
        config=CWDMSelectorConfig(
            n_iterations=100,
            k_features='auto',
            final_k=final_k,
            random_state=None,
            n_blocks=5,
            data_threshold=10_000_000,
        )
    )
    cwdm_eo = cwdm.select(train, y_train)
    selected_features = cwdm_eo.selected_features  # 相对于 SIS 输出

    # 3. ConvergenceFinder
    train_cwdm = train[:, selected_features]
    val_cwdm = val[:, selected_features]
    finder = ConvergenceFinder(
        config=ConvergenceFinderConfig(
            task=task,
            stability_threshold=1e-3,
            accuracy_threshold=0.75,
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

    # 映射回原始输入列位置（X 在 df 中的列位置，因为清洗后未改变列顺序）
    final_local = [selected_features[i] for i in range(final_stable_count)]
    direct_selected = [sis_eo.selected_indices[i] for i in final_local]

    return direct_selected, task


def main() -> int:
    parser = argparse.ArgumentParser(description='验证 cwdw_sage CLI 与原脚本三步选择结果一致性')
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
    print(f'CLI selected_indices: {cli_selected}')
    print(f'CLI task: {cli_task}')

    print('\n--- 直接调用原脚本路径 ---')
    direct_selected, direct_task = run_direct_path(df, args.label, args.seed)
    print(f'Direct selected_indices: {direct_selected}')
    print(f'Direct task: {direct_task}')

    print('\n--- 比较结果 ---')
    if cli_selected == direct_selected:
        print('✅ CLI 与原脚本的三步选择结果完全一致')
        return 0

    print('❌ CLI 与原脚本的三步选择结果不一致')
    cli_set = set(cli_selected)
    direct_set = set(direct_selected)
    print(f'CLI 多选: {sorted(cli_set - direct_set)}')
    print(f'原脚本多选: {sorted(direct_set - cli_set)}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
