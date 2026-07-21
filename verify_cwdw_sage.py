# -*- coding: utf-8 -*-

"""验证 cwdw_sage CLI 包装、直接调用与 feature_selection_by_cwdw_sage.py 完整流水线结果是否一致。

使用方式::

    PYTHONPATH=src python verify_cwdw_sage.py your_data.csv --label label

说明：
    - 本脚本会对输入 CSV 做简单清洗（全部转数值、删除含 NaN 行、删除方差为 0 的特征列）。
    - CLI 路径：先 ``fit --label-column`` 训练并保存模型，再 ``run --load`` 加载运行。
    - 直接路径：直接实例化 ``CWDWSageSelector``，调用 ``fit(x, y)`` 和 ``run(x)``。
    - 原始脚本路径：直接调用 ``feature_selection_by_cwdw_sage.run_pipeline``。
    - 比较三者最终返回的 ``selected_indices``（原始输入列位置，按 SAGE 重要性排序）、
      ``ranked_features``、``proxy_model_name`` 与 ``task``。
"""

from __future__ import annotations

import os

# 在导入 numpy 等库之前设置线程数环境变量，保证各路径结果可复现
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

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


def run_direct_path(df: pd.DataFrame, label_col: str, seed: int) -> tuple[list[int], str, str, list[dict], dict[int, float]]:
    """直接实例化 CWDWSageSelector，fit + run，返回原始列位置索引、任务类型、代理模型名、排序结果和 SAGE 值。"""
    y = df[[label_col]]
    x = df.drop(columns=[label_col])

    op = CWDWSageSelector(
        config=CWDWSageSelectorConfig(
            task='auto',
            sage_n_jobs=1,
            random_state=seed,
        )
    )

    np.random.seed(seed)
    op.fit(x, y)
    _, eo = op.run(x)

    ranked = [item.model_dump() for item in eo.ranked_features]
    return eo.selected_indices, eo.task, eo.proxy_model_name, ranked, eo.sage_values


def run_original_path(df: pd.DataFrame, label_col: str) -> tuple[list[int], str, str, list[str], list[float]]:
    """直接调用 feature_selection_by_cwdw_sage.run_pipeline，返回结果及 SAGE 评估值。"""
    from tsas.engine.operator.feature.selection.feature_selection_by_cwdw_sage import (
        PipelineConfig,
        run_pipeline,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dataset_dir = tmp / 'dataset'
        dataset_dir.mkdir()
        output_dir = tmp / 'output'
        output_dir.mkdir()

        csv_path = dataset_dir / 'input.csv'
        df.to_csv(csv_path, index=False)

        config = PipelineConfig(
            task='Classification',  # run_pipeline 内部会根据 label 唯一值数量重新判断
            proxy_model='LGBMClassifier',
            auto_selected_model=False,
            sample_balanced=False,
            batch_size=512,
            thresh=0.05,
            n_jobs=1,
            bar=False,
            cwdm=True,
            dataset=str(dataset_dir),
            output=str(output_dir),
            filename='input.csv',
        )

        # run_pipeline 会把输出写到当前工作目录，切换到临时目录避免污染仓库
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            feature_importance_value, feature_importance_index, feature_importance_name = run_pipeline(config)
        finally:
            os.chdir(original_cwd)

        # run_pipeline 返回的是最终保留特征名，映射回原始列位置
        original_indices = [int(df.columns.get_loc(name)) for name in feature_importance_name]

        # 推断任务类型与代理模型名称（与 run_pipeline 内部逻辑保持一致）
        task = 'Regression' if len(np.unique(df[label_col])) > 20 else 'Classification'
        proxy_model_name = 'xgboost' if task == 'Regression' else 'LGBMClassifier'

        return original_indices, task, proxy_model_name, list(feature_importance_name), [float(v) for v in feature_importance_value]


def _compare_pair(
    name_a: str,
    a_selected: list[int],
    a_task: str,
    a_proxy: str,
    a_ranked: list,
    name_b: str,
    b_selected: list[int],
    b_task: str,
    b_proxy: str,
    b_ranked: list,
) -> bool:
    """严格比较两路结果（特征按重要到不重要的顺序），打印差异并返回是否一致。"""
    ok = True

    # 严格按顺序比较 selected_indices
    if a_selected == b_selected:
        print(f'✅ {name_a} vs {name_b}: selected_indices 顺序完全一致')
    else:
        ok = False
        print(f'❌ {name_a} vs {name_b}: selected_indices 顺序不一致')

        # 找出第一个不同的位置
        min_len = min(len(a_selected), len(b_selected))
        diff_found = False
        for i in range(min_len):
            if a_selected[i] != b_selected[i]:
                print(f'   第一个不同位置: index={i}, {name_a}={a_selected[i]}, {name_b}={b_selected[i]}')
                diff_found = True
                break
        if not diff_found and len(a_selected) != len(b_selected):
            print(f'   长度不一致: {name_a}={len(a_selected)}, {name_b}={len(b_selected)}')

        # 退一步检查集合是否一致
        a_set = set(a_selected)
        b_set = set(b_selected)
        if a_set == b_set:
            print(f'   ⚠️  集合相同，仅排名顺序不同')
        else:
            print(f'   {name_a} 多选: {sorted(a_set - b_set)}')
            print(f'   {name_b} 多选: {sorted(b_set - a_set)}')

    if a_task != b_task:
        ok = False
        print(f'❌ {name_a} vs {name_b}: task 不一致 ({name_a}={a_task}, {name_b}={b_task})')
    else:
        print(f'✅ {name_a} vs {name_b}: task 一致')

    if a_proxy != b_proxy:
        ok = False
        print(f'❌ {name_a} vs {name_b}: proxy_model_name 不一致 ({name_a}={a_proxy}, {name_b}={b_proxy})')
    else:
        print(f'✅ {name_a} vs {name_b}: proxy_model_name 一致')

    if len(a_ranked) != len(b_ranked):
        ok = False
        print(f'❌ {name_a} vs {name_b}: ranked_features 长度不一致 ({name_a}={len(a_ranked)}, {name_b}={len(b_ranked)})')
    else:
        print(f'✅ {name_a} vs {name_b}: ranked_features 长度一致')

    # Top-K 也按顺序比较
    top_k = min(3, len(a_selected), len(b_selected))
    if top_k > 0:
        if a_selected[:top_k] == b_selected[:top_k]:
            print(f'   Top-{top_k} 顺序完全一致: {a_selected[:top_k]}')
        else:
            print(f'   Top-{top_k} 顺序不一致: {name_a}={a_selected[:top_k]}, {name_b}={b_selected[:top_k]}')

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description='验证 cwdw_sage 三条路径结果一致性')
    parser.add_argument('csv', help='输入 CSV 文件路径')
    parser.add_argument('--label', default='label', help='标签列名，默认 "label"')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于 CWDM 打乱（原始脚本内部固定为 42）')
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
    direct_selected, direct_task, direct_proxy, direct_ranked, direct_sage_values = run_direct_path(df, args.label, args.seed)
    print(f'Direct selected_indices: {direct_selected}')
    print(f'Direct task: {direct_task}')
    print(f'Direct proxy_model_name: {direct_proxy}')

    print('\n--- 原始脚本 feature_selection_by_cwdw_sage ---')
    original_selected, original_task, original_proxy, original_ranked, original_weights = run_original_path(df, args.label)
    print(f'Original selected_indices: {original_selected}')
    print(f'Original task: {original_task}')
    print(f'Original proxy_model_name: {original_proxy}')

    # 把 SAGE 评估值按 selected_indices 顺序对齐
    cli_sage_values_raw = cli_eo.get('sage_values', {})
    cli_sage_values = {int(k): float(v) for k, v in cli_sage_values_raw.items()}
    cli_weights = [cli_sage_values.get(i, float('nan')) for i in cli_selected]

    direct_weights = [direct_sage_values[i] for i in direct_selected]

    print('\n--- 三路 selected_indices 与 SAGE 评估值汇总（按重要性从高到低） ---')
    print(f'CLI:      indices={cli_selected}')
    print(f'          weights={cli_weights}')
    print(f'Direct:   indices={direct_selected}')
    print(f'          weights={direct_weights}')
    print(f'Original: indices={original_selected}')
    print(f'          weights={original_weights}')

    print('\n--- 比较结果 ---')
    ok = True
    ok &= _compare_pair(
        'CLI', cli_selected, cli_task, cli_proxy, cli_ranked,
        'Direct', direct_selected, direct_task, direct_proxy, direct_ranked,
    )
    print()
    ok &= _compare_pair(
        'CLI', cli_selected, cli_task, cli_proxy, cli_ranked,
        'Original', original_selected, original_task, original_proxy, original_ranked,
    )
    print()
    ok &= _compare_pair(
        'Direct', direct_selected, direct_task, direct_proxy, direct_ranked,
        'Original', original_selected, original_task, original_proxy, original_ranked,
    )

    if args.seed != 42:
        print('\n⚠️  提示：原始脚本 run_pipeline 内部固定 np.random.seed(42)，'
              '当前 --seed 与原始脚本不一致，原始路径的结果可能无法与 CLI/Direct 严格对齐。')

    if ok:
        print('\n✅ 三条路径结果基本一致')
        print('   注：SAGE / 模型训练存在随机性，精确排名顺序可能略有差异。')
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
