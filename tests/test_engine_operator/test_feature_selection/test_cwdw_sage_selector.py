# -*- coding: utf-8 -*-

"""CWDW + SAGE 特征选择器单元测试。"""

import numpy as np
import pandas as pd
import pytest

from tsas.engine.operator.feature.selection.cwdw_sage_selector import (
    CWDWSageSelector,
    CWDWSageSelectorConfig,
)


def _make_classification_data(n_samples: int = 60, n_features: int = 8, seed: int = 42) -> pd.DataFrame:
    """构造一个二分类数据集：第一个特征几乎能完全区分标签。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    # 第一个特征越大越可能是类别 1
    logits = 5 * x[:, 0] + 0.1 * rng.normal(size=n_samples)
    y = (logits > 0).astype(int)
    columns = [f'feat_{i}' for i in range(n_features)]
    df = pd.DataFrame(x, columns=columns)
    df['label'] = y
    return df


def test_cwdw_sage_fit_run_returns_ranked_features() -> None:
    """测试目的：验证 fit/run 能返回按 SAGE 排序的特征与完整 EO 字段。"""
    df = _make_classification_data()
    x = df.drop(columns=['label'])
    y = df[['label']]

    selector = CWDWSageSelector(
        config=CWDWSageSelectorConfig(
            task='Classification',
            cwdm_n_iterations=5,
            cwdm_final_k=8,
            sage_batch_size=32,
            sage_thresh=0.1,
        )
    )
    selector.fit(x, y)
    output, eo = selector.run(x)

    assert output.shape[1] == len(eo.selected_indices)
    assert len(eo.selected_indices) > 0
    assert eo.selected_indices == [eo.ranked_features[i].indices for i in range(len(eo.ranked_features))]
    assert eo.feature_names == [item.feat_name for item in eo.ranked_features]
    assert eo.proxy_model_name == 'LGBMClassifier'
    assert eo.task == 'Classification'
    assert len(eo.sis_selected_indices) > 0
    assert len(eo.cwdm_selected_features) > 0
    assert eo.final_k == len(eo.selected_indices)


def test_cwdw_sage_save_load_preserves_ranking(tmp_path) -> None:
    """测试目的：验证保存/加载后 run 结果与原始结果一致。"""
    df = _make_classification_data()
    x = df.drop(columns=['label'])
    y = df[['label']]

    selector = CWDWSageSelector(
        config=CWDWSageSelectorConfig(
            task='Classification',
            cwdm_n_iterations=5,
            cwdm_final_k=8,
            sage_batch_size=32,
            sage_thresh=0.1,
        )
    )
    selector.fit(x, y)
    output, eo = selector.run(x)

    selector.save(tmp_path)
    loaded = CWDWSageSelector.load(tmp_path)
    loaded_output, loaded_eo = loaded.run(x)

    np.testing.assert_array_equal(output.values, loaded_output.values)
    assert loaded_eo.selected_indices == eo.selected_indices
    assert loaded_eo.ranked_features == eo.ranked_features
    assert loaded_eo.proxy_model_name == eo.proxy_model_name


def test_cwdw_sage_requires_fit_before_run() -> None:
    """测试目的：验证未训练时 run 会报错。"""
    selector = CWDWSageSelector(config=CWDWSageSelectorConfig(task='Classification'))

    with pytest.raises(RuntimeError):
        selector.run(np.array([[1, 2, 3]]))


def test_cwdw_sage_csv_artifact(tmp_path) -> None:
    """测试目的：验证 generate_csv=True 时能写出结果 CSV。"""
    df = _make_classification_data()
    x = df.drop(columns=['label'])
    y = df[['label']]

    selector = CWDWSageSelector(
        config=CWDWSageSelectorConfig(
            task='Classification',
            cwdm_n_iterations=5,
            cwdm_final_k=8,
            sage_batch_size=32,
            sage_thresh=0.1,
            generate_csv=True,
            output_dir=str(tmp_path),
        )
    )
    selector.fit(x, y)

    csv_path = tmp_path / '特征优选_result_特征重要性.csv'
    assert csv_path.exists()
    result = pd.read_csv(csv_path)
    assert list(result.columns) == ['feat_name', 'indices', 'weight']
    assert len(result) == len(selector.run(x)[1].selected_indices)


def test_cwdw_sage_auto_task_regression() -> None:
    """测试目的：验证标签唯一值较多时自动判断为回归任务。"""
    rng = np.random.default_rng(42)
    n_samples, n_features = 60, 6
    x = rng.normal(size=(n_samples, n_features))
    y = x[:, 0] * 3 + rng.normal(size=n_samples) * 0.1
    df = pd.DataFrame(x, columns=[f'feat_{i}' for i in range(n_features)])
    df['label'] = y

    selector = CWDWSageSelector(
        config=CWDWSageSelectorConfig(
            task='auto',
            regression_label_threshold=10,
            cwdm_n_iterations=5,
            cwdm_final_k=6,
            sage_batch_size=32,
            sage_thresh=0.1,
        )
    )
    selector.fit(df.drop(columns=['label']), df[['label']])
    _, eo = selector.run(df.drop(columns=['label']))

    assert eo.task == 'Regression'
    assert eo.proxy_model_name == 'xgboost'
    assert len(eo.selected_indices) > 0


def test_cwdw_sage_feature_name_preservation() -> None:
    """测试目的：验证 DataFrame 输入时特征名被保留到 EO。"""
    df = _make_classification_data()
    x = df.drop(columns=['label'])
    y = df[['label']]

    selector = CWDWSageSelector(
        config=CWDWSageSelectorConfig(
            task='Classification',
            cwdm_n_iterations=5,
            cwdm_final_k=8,
            sage_batch_size=32,
            sage_thresh=0.1,
        )
    )
    selector.fit(x, y)
    _, eo = selector.run(x)

    assert len(eo.feature_names) == len(eo.selected_indices)
    assert all(name in x.columns for name in eo.feature_names)
