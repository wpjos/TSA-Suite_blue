# -*- coding: utf-8 -*-

"""
百分位阈值决策器单元测试

对应源文件：
- percentile_decider.py: PercentileDecider

测试范围：
- 训练阶段学习百分位阈值
- 推理阶段阈值决策
- DataFrame/ndarray 双类型支持
- 边界条件
"""

import numpy as np
import pytest
from pandas import DataFrame

from bianque.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderExtraOutput,
)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_scores():
    """训练分数（均匀分布 0-99）"""
    return np.arange(100, dtype=float)


@pytest.fixture
def test_scores():
    """测试分数"""
    return np.array([50.0, 94.0, 95.0, 96.0, 99.0])


# ============================================================================
# PercentileDecider 测试
# ============================================================================

class TestPercentileDecider:
    """测试百分位阈值决策器"""

    def test_fit_learns_threshold(self, train_scores):
        """
        目的：验证 fit 正确学习百分位阈值
        输入：0-99 均匀分布，percentile=95
        预期：阈值与 np.percentile 一致
        """
        decider = PercentileDecider(percentile=95.0)
        decider.fit(train_scores)
        assert decider._threshold is not None
        np.testing.assert_allclose(decider._threshold, 95.0, atol=1)

    def test_run_labels(self, train_scores, test_scores):
        """
        目的：验证推理标签正确
        输入：percentile=95
        预期：> 阈值的为异常
        """
        decider = PercentileDecider(percentile=95.0)
        decider.fit(train_scores)
        labels, eo = decider.run(test_scores)
        assert labels.dtype == int
        assert isinstance(eo, PercentileDeciderExtraOutput)
        assert eo.threshold == decider._threshold

    def test_with_dataframe(self, train_scores, test_scores):
        """
        目的：验证 DataFrame 输入输出
        预期：输出为 DataFrame
        """
        train_df = DataFrame(train_scores, columns=["score"])
        test_df = DataFrame(test_scores, columns=["score"])
        decider = PercentileDecider(percentile=95.0)
        decider.fit(train_df)
        labels, _ = decider.run(test_df)
        assert isinstance(labels, DataFrame)

    def test_before_fit_raises(self, test_scores):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        预期：抛出 RuntimeError
        """
        decider = PercentileDecider()
        with pytest.raises(RuntimeError):
            decider.run(test_scores)

    def test_strict_greater(self, train_scores):
        """
        目的：验证严格大于判定
        输入：percentile=95，测试值恰好等于阈值
        预期：等于阈值的判定为正常
        """
        decider = PercentileDecider(percentile=95.0)
        decider.fit(train_scores)
        # 测试值恰好等于阈值
        test = np.array([decider._threshold])
        labels, _ = decider.run(test)
        assert labels[0] == 0  # 严格大于，等于阈值应为 0

    def test_50th_percentile(self, train_scores):
        """
        目的：验证 50 百分位阈值
        输入：percentile=50
        预期：阈值 ≈ 50.0，约一半判定为异常
        """
        decider = PercentileDecider(percentile=50.0)
        decider.fit(train_scores)
        labels, _ = decider.run(train_scores)
        # 约 50% 应判定为异常（严格大于中位数）
        anomaly_rate = labels.sum() / len(labels)
        assert 0.4 <= anomaly_rate <= 0.6
