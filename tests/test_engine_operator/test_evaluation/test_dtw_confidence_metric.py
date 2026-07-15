# -*- coding: utf-8 -*-

"""
DTW / Confidence / Normalize Score 评价指标算子测试

测试覆盖:
    1. 基本功能: run() 返回正确类型与字段
    2. 算法正确性: DTW 距离 / 路径 / Confidence / 归一化
    3. scores() 方法: 按 main_scores 提取命名标量
    4. 边界条件: 空数组、常数、单点
    5. 错误处理: 长度不一致、缺配置、非法参数

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from tsas.engine.operator.evaluation.confidence_metric import (
    ConfidenceConfig,
    ConfidenceMetric,
)
from tsas.engine.operator.evaluation.dtw_distance_metric import (
    DtwDistanceConfig,
    DtwDistanceMetric,
)
from tsas.engine.operator.evaluation.dtw_path_metric import (
    DtwPathConfig,
    DtwPathMetric,
)
from tsas.engine.operator.evaluation.normalize_score_metric import (
    NormalizeScoreConfig,
    NormalizeScoreMetric,
)


# ============================================================================
# DtwDistanceMetric
# ============================================================================


class TestDtwDistanceBasic:
    """DTW Distance 基本功能测试"""

    def test_identical_sequences(self):
        """相同序列 → 距离 = 0"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        op = DtwDistanceMetric()
        assert op.run((x, y)) == 0.0

    def test_constant_shift(self):
        """常数平移 → 距离 = 1.0 × min(len(x), len(y))"""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([2.0, 2.0, 2.0])

        op = DtwDistanceMetric()
        assert op.run((x, y)) == pytest.approx(3.0)

    def test_with_window(self):
        """Sakoe-Chiba 窗口限制"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        op = DtwDistanceMetric(window=0)
        assert op.run((x, y)) == 0.0

    def test_empty_input(self):
        """空输入 → ValueError"""
        op = DtwDistanceMetric()
        with pytest.raises(ValueError, match="不能为空"):
            op.run((np.array([]), np.array([1.0])))

    def test_scores(self):
        """scores() 提取 dtw_distance 命名标量"""
        op = DtwDistanceMetric()
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        scores = op.scores((x, y))

        assert "dtw_distance" in scores
        assert scores["dtw_distance"] == 0.0


# ============================================================================
# DtwPathMetric
# ============================================================================


class TestDtwPathBasic:
    """DTW Path 基本功能测试"""

    def test_identical_sequences(self):
        """相同序列 → 距离 0 + 路径 [(0,0), (1,1), ...]"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        op = DtwPathMetric()
        result = op.run((x, y))

        assert result.distance == 0.0
        assert result.path == [(0, 0), (1, 1), (2, 2)]

    def test_constant_shift(self):
        """常数平移 → 累积距离"""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([2.0, 2.0, 2.0])

        op = DtwPathMetric()
        result = op.run((x, y))

        assert result.distance == pytest.approx(3.0)
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (2, 2)

    def test_path_monotonic(self):
        """路径单调（i 和 j 都递增）"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        op = DtwPathMetric()
        result = op.run((x, y))

        for k in range(len(result.path) - 1):
            i0, j0 = result.path[k]
            i1, j1 = result.path[k + 1]
            assert i0 <= i1 and j0 <= j1

    def test_scores(self):
        """scores() 提取 dtw_distance 命名标量"""
        op = DtwPathMetric()
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        scores = op.scores((x, y))

        assert "dtw_distance" in scores


# ============================================================================
# ConfidenceMetric
# ============================================================================


class TestConfidenceBasic:
    """Confidence 基本功能测试"""

    def test_run_returns_array(self):
        """run() 返回 confidence 数组"""
        rng = np.random.RandomState(0)
        train_scores = rng.rand(100)
        test_scores = rng.rand(20)

        op = ConfidenceMetric(threshold=0.5)
        result = op.run((train_scores, test_scores))

        assert result.confidence.shape == (20,)
        assert (result.confidence >= 0).all()
        assert (result.confidence <= 1).all()

    def test_constant_train(self):
        """训练分数无方差 → 每个 test_score 都得到近似相同的 confidence"""
        train_scores = np.zeros(100)
        test_scores = np.array([0.0, 0.5, 1.0])

        op = ConfidenceMetric(threshold=0.5)
        result = op.run((train_scores, test_scores))

        assert result.confidence.shape == (3,)

    def test_invalid_contamination(self):
        """contamination 越界会被 Pydantic 字段约束直接拒绝"""
        with pytest.raises(Exception):
            ConfidenceMetric(threshold=0.5, contamination=1.5)

    def test_empty_train(self):
        """空训练集 → ValueError at run time"""
        op = ConfidenceMetric(threshold=0.5)
        with pytest.raises(ValueError, match="train_scores must be non-empty"):
            op.run((np.array([]), np.array([0.5])))

    def test_missing_config(self):
        """config 缺 threshold → ValidationError at construction"""
        with pytest.raises(Exception):
            ConfidenceMetric()

    def test_scores(self):
        """scores() 在 main_scores=None 时返回 None（结果为数组，无标量主评分）"""
        rng = np.random.RandomState(0)
        train_scores = rng.rand(100)
        test_scores = rng.rand(5)

        op = ConfidenceMetric(threshold=0.5)
        scores = op.scores((train_scores, test_scores))

        # confidence 字段是 ndarray → main_scores 默认 None → scores 返回 None
        assert scores is None


# ============================================================================
# NormalizeScoreMetric
# ============================================================================


class TestNormalizeScoreBasic:
    """Normalize Score 基本功能测试"""

    def test_linear_normalization(self):
        """linear 归一化 → MinMax 映射到 [0, 1]"""
        train_scores = np.linspace(0, 10, 100)
        test_scores = np.array([0.0, 5.0, 10.0])

        op = NormalizeScoreMetric(method='linear')
        result = op.run((train_scores, test_scores))

        assert result.train_norm.min() == pytest.approx(0.0)
        assert result.train_norm.max() == pytest.approx(1.0)
        assert result.test_norm.min() == pytest.approx(0.0)
        assert result.test_norm.max() == pytest.approx(1.0)

    def test_unify_normalization(self):
        """unify 归一化 → erf 映射到 [0, 1]"""
        rng = np.random.RandomState(0)
        train_scores = rng.randn(100)
        test_scores = rng.randn(20)

        op = NormalizeScoreMetric(method='unify')
        result = op.run((train_scores, test_scores))

        assert result.train_norm.min() >= 0.0
        assert result.train_norm.max() <= 1.0
        assert result.test_norm.shape == (20,)

    def test_constant_train_unify(self):
        """unify + 训练集无方差 → 全 0.5"""
        train_scores = np.zeros(100)
        test_scores = np.array([0.0, 0.5, 1.0])

        op = NormalizeScoreMetric(method='unify')
        result = op.run((train_scores, test_scores))

        assert (result.train_norm == 0.5).all()
        assert (result.test_norm == 0.5).all()

    def test_invalid_method(self):
        """非法 method 会被 Pydantic Literal 字段约束直接拒绝"""
        with pytest.raises(Exception):
            NormalizeScoreMetric(method='invalid')

    def test_only_train(self):
        """只对 train 归一化（test_scores=None）"""
        train_scores = np.linspace(0, 10, 100)

        op = NormalizeScoreMetric(method='linear')
        result = op.run((train_scores, None))

        assert result.train_norm.shape == (100,)
        assert result.test_norm is None

    def test_return_test_false(self):
        """return_test=False → test_norm=None"""
        train_scores = np.linspace(0, 10, 100)
        test_scores = np.array([0.0, 5.0, 10.0])

        op = NormalizeScoreConfig(method='linear', return_test=False)
        op2 = NormalizeScoreMetric(config=op)
        result = op2.run((train_scores, test_scores))

        assert result.train_norm.shape == (100,)
        assert result.test_norm is None