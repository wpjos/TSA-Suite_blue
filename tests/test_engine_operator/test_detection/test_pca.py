# -*- coding: utf-8 -*-

"""
具体检测算子单元测试

对应源文件：
- pca.py: PCAPredictor

测试范围：
- 各算子的 fit/run 基本流程
- DataFrame/ndarray 双类型支持
- 边界条件（零标准差、小样本、单特征等）
- 附加输出正确性
- 参数验证
"""

import numpy as np
import pytest
from pandas import DataFrame

from bianque.engine.operator.detection.pca import (
    PCAPredictor, PCAPredictorExtraOutput,
)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_data():
    """测试用训练数据（ndarray, 100x3）"""
    np.random.seed(42)
    return np.random.randn(100, 3)


@pytest.fixture
def test_data():
    """测试用测试数据（ndarray, 60x3，含异常点）"""
    np.random.seed(123)
    normal = np.random.randn(50, 3)
    abnormal = np.random.randn(10, 3) * 5 + 10
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_data):
    """测试用训练数据（DataFrame）"""
    return DataFrame(train_data, columns=["a", "b", "c"])


@pytest.fixture
def test_df(test_data):
    """测试用测试数据（DataFrame）"""
    return DataFrame(test_data, columns=["a", "b", "c"])


# ============================================================================
# PCAPredictor 测试
# ============================================================================

class TestPCAPredictor:
    """测试 PCA 预测器"""

    def test_fit_learns_components(self, train_data):
        """
        目的：验证 fit 正确学习主成分
        输入：(100, 3) 训练数据，n_components=2
        预期：_components 形状 (2, 3)
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_data)
        assert predictor._components is not None
        assert predictor._components.shape == (2, 3)
        assert predictor._mean is not None

    def test_run_reconstruction(self, train_data):
        """
        目的：验证推理输出重构值
        输入：训练数据
        预期：重构值形状与输入相同
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_data)
        pred, eo = predictor.run(train_data)
        assert pred.shape == train_data.shape
        assert isinstance(eo, PCAPredictorExtraOutput)
        assert eo.n_components == 2
        assert len(eo.explained_variance_ratio) == 2

    def test_full_components_perfect_reconstruction(self, train_data):
        """
        目的：验证 n_components 等于特征数时重构完美
        输入：n_components=3（等于特征数）
        预期：重构值 ≈ 原始值（去中心化后）
        """
        predictor = PCAPredictor(n_components=3)
        predictor.fit(train_data)
        pred, _ = predictor.run(train_data)
        # n_components == n_features 时，重构应完美
        np.testing.assert_allclose(pred, train_data, atol=1e-10)

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出
        输入：DataFrame
        预期：输出为 DataFrame
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_df)
        pred, _ = predictor.run(test_df)
        assert isinstance(pred, DataFrame)

    def test_explained_variance_ratio(self, train_data):
        """
        目的：验证解释方差比正确
        输入：(100, 3) 训练数据
        预期：解释方差比之和 ≤ 1.0
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_data)
        total = sum(predictor._explained_variance_ratio)
        assert total <= 1.0 + 1e-10
        assert all(v >= 0 for v in predictor._explained_variance_ratio)

    def test_constant_data_degenerate_case(self):
        """
        目的：验证所有特征为常数时的退化情况
        输入：全部为常数的训练数据
        预期：解释方差比全为 0，不报错
        """
        # 所有特征为常数（方差为 0）
        train = np.ones((50, 3)) * 5.0
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train)
        # 解释方差比应全为 0
        assert all(v == 0 for v in predictor._explained_variance_ratio)

    def test_n_components_exceeds_features(self):
        """
        目的：验证 n_components 超过特征数时自动调整
        输入：2个特征的训练数据，n_components=5
        预期：实际使用 2 个主成分
        """
        train = np.random.randn(50, 2)
        predictor = PCAPredictor(n_components=5)
        predictor.fit(train)
        assert predictor._components.shape[0] == 2  # 自动调整为特征数
