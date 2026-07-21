# -*- coding: utf-8 -*-

"""验证 cwdw_sage CLI 包装与 feature_selection_by_cwdw_sage.py 完整流水线结果是否一致。

使用方式::

    PYTHONPATH=src python verify_cwdw_sage.py your_data.csv --label label

说明：
    - 本脚本会对输入 CSV 做简单清洗（全部转数值、删除含 NaN 行、删除方差为 0 的特征列）。
    - CLI 路径：先 ``fit --label-column`` 训练并保存模型，再 ``run --load`` 加载运行。
    - 直接路径：直接实例化 ``CWDWSageSelector``，调用 ``fit(x, y)`` 和 ``run(x)``。
    - 比较两者最终返回的 ``selected_indices``（原始输入列位置，按 SAGE 重要性排序）、
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

from tsas.engine.operator.feature.selection.cwdw_sage_selector import (
    CWDWSageSelector,
    CWDWSageSelectorConfig,
)


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
                    'random_state': 42,
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


def run_direct_path(df: pd.DataFrame, label_col: str, seed: int) -> tuple[list[int], str, str, list[dict]]:
    """直接实例化 CWDWSageSelector，fit + run，返回原始列位置索引、任务类型、代理模型名和排序结果。"""
    y = df[[label_col]]
    x = df.drop(columns=[label_col])

    op = CWDWSageSelector(
        config=CWDWSageSelectorConfig(
            task='auto',
            sage_n_jobs=1,
            random_state=42,
        )
    )

    np.random.seed(seed)
    op.fit(x, y)
    _, eo = op.run(x)

    ranked = [item.model_dump() for item in eo.ranked_features]
    return eo.selected_indices, eo.task, eo.proxy_model_name, ranked


def main() -> int:
    parser = argparse.ArgumentParser(description='验证 cwdw_sage CLI 与原脚本完整流水线结果一致性')
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

    print('\n--- 直接调用 CWDWSageSelector ---')
    direct_selected, direct_task, direct_proxy, direct_ranked = run_direct_path(df, args.label, args.seed)
    print(f'Direct selected_indices: {direct_selected}')
    print(f'Direct task: {direct_task}')
    print(f'Direct proxy_model_name: {direct_proxy}')

    print('\n--- 比较结果 ---')
    ok = True

    # 由于 SAGE / 代理模型训练存在随机性，exact 排名可能不一致，
    # 因此主要比较集合、任务类型、代理模型名和排序长度。
    cli_set = set(cli_selected)
    direct_set = set(direct_selected)
    if cli_set != direct_set:
        ok = False
        print('❌ selected_indices 集合不一致')
        print(f'CLI 多选: {sorted(cli_set - direct_set)}')
        print(f'直接调用多选: {sorted(direct_set - cli_set)}')
    else:
        print('✅ selected_indices 集合一致')

    if cli_task != direct_task:
        ok = False
        print(f'❌ task 不一致: CLI={cli_task}, Direct={direct_task}')
    else:
        print('✅ task 一致')

    if cli_proxy != direct_proxy:
        ok = False
        print(f'❌ proxy_model_name 不一致: CLI={cli_proxy}, Direct={direct_proxy}')
    else:
        print('✅ proxy_model_name 一致')

    if len(cli_ranked) != len(direct_ranked):
        ok = False
        print(f'❌ ranked_features 长度不一致: CLI={len(cli_ranked)}, Direct={len(direct_ranked)}')
    else:
        print('✅ ranked_features 长度一致')

    # 若集合相同，进一步比较 Top-3 重叠情况作为稳定性参考
    if cli_set == direct_set:
        top_k = min(3, len(cli_selected))
        cli_top = set(cli_selected[:top_k])
        direct_top = set(direct_selected[:top_k])
        common_top = cli_top & direct_top
        print(f'Top-{top_k} 重叠: {len(common_top)}/{top_k} ({sorted(common_top)})')

    if ok:
        print('\n✅ CLI 与直接调用的完整流水线结果基本一致')
        print('   注：SAGE / 模型训练存在随机性，精确排名顺序可能略有差异。')
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
