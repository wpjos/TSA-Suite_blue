# -*- coding: utf-8 -*-

"""
聚类 + NAB 评价指标算子测试

测试覆盖:
    1. 基本功能: run() 返回正确类型与字段
    2. 算法正确性: ARI/MI/NMI/Entropy/NAB 计算
    3. scores() 方法: 按 main_scores 提取命名标量
    4. 边界条件: 空标签、单标签、全同
    5. 错误处理: 长度不一致、缺配置、未知 profile

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
"""

import numpy as np
import pytest

from tsas.engine.operator.evaluation.ari_metric import ARIMetric
from tsas.engine.operator.evaluation.entropy_metric import EntropyMetric
from tsas.engine.operator.evaluation.mi_metric import MutualInfoMetric
from tsas.engine.operator.evaluation.nab_best_threshold_metric import (
    NabBestThresholdMetric,
)
from tsas.engine.operator.evaluation.nab_score_metric import (
    NabScoreConfig,
    NabScoreMetric,
)
from tsas.engine.operator.evaluation.nmi_metric import NMIMetric


# ============================================================================
# ARIMetric
# ============================================================================


class TestARIBasic:
    """ARI 基本功能测试"""

    def test_perfect_match(self):
        """完美匹配 → ARI = 1.0"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        op = ARIMetric()
        assert op.run((y_true, y_pred)) == 1.0

    def test_different_labels_same_clustering(self):
        """标签值不同但聚类结果相同 → ARI = 1.0"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([5, 5, 9, 9, 3, 3])

        op = ARIMetric()
        assert op.run((y_true, y_pred)) == 1.0

    def test_random_clustering(self):
        """完全不同的聚类标签（不同分区） → ARI 应 < 1.0"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])  # 完全打散的预测

        op = ARIMetric()
        ari = op.run((y_true, y_pred))
        assert ari < 1.0

    def test_length_mismatch(self):
        """长度不一致 → ValueError"""
        op = ARIMetric()
        with pytest.raises(ValueError, match="长度不一致"):
            op.run((np.array([0, 1, 0]), np.array([0, 1])))

    def test_scores(self):
        """scores() 提取 ari 命名标量"""
        op = ARIMetric()
        y = np.array([0, 0, 1, 1])
        scores = op.scores((y, y))

        assert "ari" in scores
        assert scores["ari"] == 1.0


# ============================================================================
# MutualInfoMetric
# ============================================================================


class TestMIBasic:
    """MI 基本功能测试"""

    def test_perfect_match(self):
        """完美匹配 → MI > 0"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        op = MutualInfoMetric()
        mi = op.run((y_true, y_pred))
        assert mi > 0

    def test_independent(self):
        """完全不同的标签 → MI 应较低"""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        op = MutualInfoMetric()
        mi = op.run((y_true, y_pred))
        assert mi >= 0

    def test_scores(self):
        """scores() 提取 mi 命名标量"""
        op = MutualInfoMetric()
        y = np.array([0, 1, 0, 1])
        scores = op.scores((y, y))

        assert "mi" in scores

    def test_length_mismatch(self):
        """长度不一致 → ValueError"""
        op = MutualInfoMetric()
        with pytest.raises(ValueError, match="长度不一致"):
            op.run((np.array([0, 1]), np.array([0])))


# ============================================================================
# NMIMetric
# ============================================================================


class TestNMIBasic:
    """NMI 基本功能测试"""

    def test_perfect_match(self):
        """完美匹配 → NMI = 1.0"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        op = NMIMetric()
        nmi = op.run((y_true, y_pred))
        assert nmi == pytest.approx(1.0)

    def test_both_constant(self):
        """两边都是单一类别 → NMI = 1.0（按 upstream 约定）"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])

        op = NMIMetric()
        nmi = op.run((y_true, y_pred))
        assert nmi == 1.0

    def test_one_constant(self):
        """一边是单类别 → NMI = 0.0"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 1])

        op = NMIMetric()
        nmi = op.run((y_true, y_pred))
        assert nmi == 0.0

    def test_scores(self):
        """scores() 提取 nmi 命名标量"""
        op = NMIMetric()
        y = np.array([0, 0, 1, 1])
        scores = op.scores((y, y))

        assert "nmi" in scores
        assert scores["nmi"] == pytest.approx(1.0)


# ============================================================================
# EntropyMetric
# ============================================================================


class TestEntropyBasic:
    """Entropy 基本功能测试"""

    def test_binary_equal(self):
        """二元均匀分布 → H = ln(2) ≈ 0.693"""
        labels = np.array([0, 0, 1, 1])

        op = EntropyMetric()
        h = op.run(labels)

        assert h == pytest.approx(np.log(2), abs=1e-6)

    def test_single_label(self):
        """单类别 → H = 0"""
        labels = np.array([0, 0, 0, 0])

        op = EntropyMetric()
        h = op.run(labels)

        assert h == 0.0

    def test_three_classes_equal(self):
        """三类均匀分布 → H = ln(3)"""
        labels = np.array([0, 1, 2] * 10)

        op = EntropyMetric()
        h = op.run(labels)

        assert h == pytest.approx(np.log(3), abs=1e-6)

    def test_empty(self):
        """空数组 → H = 0"""
        op = EntropyMetric()
        assert op.run(np.array([])) == 0.0

    def test_scores(self):
        """scores() 提取 entropy 命名标量"""
        op = EntropyMetric()
        scores = op.scores(np.array([0, 0, 1, 1]))

        assert "entropy" in scores


# ============================================================================
# NabScoreMetric
# ============================================================================


class TestNabScoreBasic:
    """NAB Score 基本功能测试"""

    def test_run_returns_result(self):
        """run() 返回 NabScoreResult"""
        y_true = np.zeros(100)
        y_true[50:60] = 1
        y_score = np.linspace(0, 1, 100)

        op = NabScoreMetric(threshold=0.5)
        result = op.run((y_true, y_score))

        assert result.profile == "standard"
        assert result.threshold == 0.5
        assert isinstance(result.nab_score, float)

    def test_threshold_default(self):
        """threshold=None 时取 max(y_score)/2"""
        y_true = np.zeros(100)
        y_true[50:60] = 1
        y_score = np.linspace(0, 1, 100)

        op = NabScoreMetric()
        result = op.run((y_true, y_score))

        assert result.threshold == pytest.approx(0.5)

    def test_invalid_profile(self):
        """未知 profile → ValueError"""
        op = NabScoreMetric(threshold=0.5, profile="unknown")

        with pytest.raises(ValueError, match="Unknown profile"):
            op.run((np.zeros(10), np.linspace(0, 1, 10)))

    def test_length_mismatch(self):
        """长度不一致 → ValueError"""
        op = NabScoreMetric(threshold=0.5)
        with pytest.raises(ValueError, match="形状不一致"):
            op.run((np.zeros(10), np.zeros(5)))

    def test_scores(self):
        """scores() 提取 nab_score 命名标量"""
        y_true = np.zeros(100)
        y_true[50:60] = 1
        y_score = np.linspace(0, 1, 100)

        op = NabScoreMetric(threshold=0.5)
        scores = op.scores((y_true, y_score))

        assert "nab_score" in scores


# ============================================================================
# NabBestThresholdMetric
# ============================================================================


class TestNabBestThresholdBasic:
    """NAB Best Threshold 基本功能测试"""

    def test_run_returns_score(self):
        """run() 返回 best NAB score"""
        y_true = np.zeros(100)
        y_true[50:60] = 1
        y_score = np.linspace(0, 1, 100)

        op = NabBestThresholdMetric()
        best = op.run((y_true, y_score))

        assert isinstance(best, float)

    def test_empty(self):
        """空输入 → 0.0"""
        op = NabBestThresholdMetric()
        assert op.run((np.array([]), np.array([]))) == 0.0

    def test_invalid_profile(self):
        """未知 profile → ValueError"""
        op = NabBestThresholdMetric(profile="unknown")

        with pytest.raises(ValueError, match="Unknown profile"):
            op.run((np.zeros(10), np.linspace(0, 1, 10)))

    def test_scores(self):
        """scores() 提取 nab_best_threshold 命名标量"""
        op = NabBestThresholdMetric()
        scores = op.scores((np.zeros(10), np.linspace(0, 1, 10)))

        assert "nab_best_threshold" in scores