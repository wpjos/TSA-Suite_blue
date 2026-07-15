# -*- coding: utf-8 -*-

"""
Affiliation / Range-AUC 评价指标算子测试

测试覆盖:
    1. 基本功能: run() 返回正确类型与字段
    2. 算法正确性: 验证 affiliation P/R/F1 与 Range-AUC
    3. scores() 方法: 按 main_scores 提取命名标量
    4. 边界条件: 空事件、单事件、全异常
    5. 错误处理: 长度不一致、缺配置

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
"""

import numpy as np
import pytest

from tsas.engine.operator.evaluation.affiliation_metric import (
    AffiliationConfig,
    AffiliationMetric,
)
from tsas.engine.operator.evaluation.range_auc_pr_metric import (
    RangeAucPrConfig,
    RangeAucPrMetric,
)
from tsas.engine.operator.evaluation.range_auc_roc_metric import (
    RangeAucRocConfig,
    RangeAucRocMetric,
)


# ============================================================================
# AffiliationMetric
# ============================================================================


class TestAffiliationBasic:
    """Affiliation P/R/F1 基本功能测试"""

    def test_perfect_match(self):
        """完美匹配 → P/R/F1 = 1.0"""
        y_true = np.array([0, 1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0])

        op = AffiliationMetric()
        result = op.run((y_true, y_pred))

        assert result.affiliation_precision == 1.0
        assert result.affiliation_recall == 1.0
        assert result.affiliation_f1 == 1.0

    def test_no_true_events(self):
        """无真值事件 → P/R/F1 = 0.0"""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 1, 0])

        op = AffiliationMetric()
        result = op.run((y_true, y_pred))

        assert result.affiliation_precision == 0.0
        assert result.affiliation_recall == 0.0
        assert result.affiliation_f1 == 0.0

    def test_no_pred_events(self):
        """无预测事件 → P/R/F1 = 0.0"""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 0])

        op = AffiliationMetric()
        result = op.run((y_true, y_pred))

        assert result.affiliation_precision == 0.0
        assert result.affiliation_recall == 0.0
        assert result.affiliation_f1 == 0.0

    def test_partial_overlap(self):
        """部分重叠 → P/R/F1 介于 0 和 1 之间"""
        y_true = np.array([0, 1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0])

        op = AffiliationMetric()
        result = op.run((y_true, y_pred))

        assert 0.0 < result.affiliation_f1 < 1.0
        assert 0.0 <= result.affiliation_precision <= 1.0
        assert 0.0 <= result.affiliation_recall <= 1.0

    def test_scores(self):
        """scores() 同时提取三个命名标量"""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])

        op = AffiliationMetric()
        scores = op.scores((y_true, y_pred))

        assert set(scores.keys()) == {"affiliation_precision", "affiliation_recall", "affiliation_f1"}
        assert scores["affiliation_f1"] == 1.0

    def test_length_mismatch(self):
        """长度不一致 → ValueError"""
        op = AffiliationMetric()
        with pytest.raises(ValueError, match="长度不一致"):
            op.run((np.array([0, 1, 1]), np.array([0, 1])))


# ============================================================================
# RangeAucRocMetric
# ============================================================================


class TestRangeAucRocBasic:
    """Range-AUC-ROC 基本功能测试"""

    def test_perfect_score(self):
        """完美分数 → AUC 应接近 1.0"""
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.zeros(50), np.ones(50)])

        # sliding_window=0 表示不扩展退化为标准 ROC-AUC → 完美时 AUC=1.0
        op = RangeAucRocMetric(sliding_window=0)
        auc = op.run((y_true, y_score))

        assert auc == pytest.approx(1.0, abs=1e-6)

    def test_anti_perfect_score(self):
        """完全反向分数 → AUC 接近 0"""
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.ones(50), np.zeros(50)])

        op = RangeAucRocMetric(sliding_window=5)
        auc = op.run((y_true, y_score))

        assert auc < 0.1

    def test_random_score(self):
        """随机分数 → AUC 接近 0.5"""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = rng.rand(100)

        op = RangeAucRocMetric(sliding_window=5)
        auc = op.run((y_true, y_score))

        assert 0.3 < auc < 0.7

    def test_missing_config(self):
        """config 缺 sliding_window → ValidationError at construction"""
        with pytest.raises(Exception):
            RangeAucRocMetric()

    def test_invalid_window(self):
        """sliding_window < 0 会被 Pydantic 字段约束直接拒绝"""
        # Pydantic 没对 sliding_window 加 ge 约束 → 构造时接受 → run 时报错
        op = RangeAucRocMetric(sliding_window=-1)
        with pytest.raises(ValueError, match="sliding_window must be >= 0"):
            op.run((np.array([0, 1, 0]), np.array([0.1, 0.9, 0.1])))

    def test_scores(self):
        """scores() 提取 range_auc_roc 命名标量"""
        op = RangeAucRocMetric(sliding_window=5)
        scores = op.scores((np.array([0, 1, 0]), np.array([0.1, 0.9, 0.1])))

        assert "range_auc_roc" in scores

    def test_length_mismatch(self):
        """长度不一致 → ValueError"""
        op = RangeAucRocMetric(sliding_window=5)
        with pytest.raises(ValueError, match="形状不一致"):
            op.run((np.array([0, 1, 0]), np.array([0.1, 0.9])))


# ============================================================================
# RangeAucPrMetric
# ============================================================================


class TestRangeAucPrBasic:
    """Range-AUC-PR 基本功能测试"""

    def test_perfect_score(self):
        """完美分数 → AUC 应接近 1.0"""
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.zeros(50), np.ones(50)])

        op = RangeAucPrMetric(sliding_window=0)
        auc = op.run((y_true, y_score))

        assert auc == pytest.approx(1.0, abs=1e-6)

    def test_missing_config(self):
        """config 缺 sliding_window → ValidationError at construction"""
        with pytest.raises(Exception):
            RangeAucPrMetric()

    def test_invalid_window(self):
        """sliding_window < 0 → ValueError at run time"""
        op = RangeAucPrMetric(sliding_window=-1)
        with pytest.raises(ValueError, match="sliding_window must be >= 0"):
            op.run((np.array([0, 1, 0]), np.array([0.1, 0.9, 0.1])))

    def test_invalid_num_thresholds(self):
        """num_thresholds < 2 会被 Pydantic 字段约束直接拒绝"""
        with pytest.raises(Exception):
            RangeAucPrMetric(sliding_window=5, num_thresholds=1)

    def test_scores(self):
        """scores() 提取 range_auc_pr 命名标量"""
        op = RangeAucPrMetric(sliding_window=5)
        scores = op.scores((np.array([0, 1, 0]), np.array([0.1, 0.9, 0.1])))

        assert "range_auc_pr" in scores