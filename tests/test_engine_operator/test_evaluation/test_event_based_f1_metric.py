# -*- coding: utf-8 -*-

"""
Event-Based F1 / Soft Event-Based F1 / Detection Delay 评价指标算子测试

测试覆盖:
    1. 基本功能: run() 返回正确类型与字段
    2. 算法正确性: 验证事件分割 / IoU / 延迟计算
    3. scores() 方法: 按 main_scores 提取命名标量
    4. 边界条件: 空事件、单事件、全异常
    5. 错误处理: 长度不一致、缺配置

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
"""

import numpy as np
import pytest

from tsas.engine.operator.evaluation.detection_delay_metric import (
    DetectionDelayConfig,
    DetectionDelayMetric,
)
from tsas.engine.operator.evaluation.event_based_f1_metric import (
    EventBasedF1Config,
    EventBasedF1Metric,
)
from tsas.engine.operator.evaluation.soft_event_based_f1_metric import (
    SoftEventBasedF1Config,
    SoftEventBasedF1Metric,
)


# ============================================================================
# EventBasedF1Metric
# ============================================================================


class TestEventBasedF1Basic:
    """Event-Based F1 基本功能测试"""

    def test_perfect_match(self):
        """完美匹配 → P/R/F1 = 1.0"""
        y_true = np.array([0, 1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0])

        op = EventBasedF1Metric()
        result = op.run((y_true, y_pred))

        assert result.n_true_events == 1
        assert result.n_pred_events == 1
        assert result.tp == 1
        assert result.fp == 0
        assert result.fn == 0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_no_overlap(self):
        """预测事件与真值事件完全不重叠 → F1 = 0"""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1])

        op = EventBasedF1Metric()
        result = op.run((y_true, y_pred))

        assert result.tp == 0
        assert result.fp == 2  # 两个预测事件都没匹配上
        assert result.fn == 1
        assert result.f1 == 0.0

    def test_partial_overlap_no_threshold(self):
        """默认 overlap_threshold=0：任何重叠都算匹配"""
        y_true = np.array([0, 1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 0])

        op = EventBasedF1Metric()
        result = op.run((y_true, y_pred))

        assert result.tp == 1
        assert result.fp == 0
        assert result.fn == 0
        assert result.f1 == 1.0

    def test_overlap_threshold_blocks_match(self):
        """重叠不足 → 视为不匹配"""
        y_true = np.array([0, 1, 1, 1, 1, 0, 0])  # event = [1,4]
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1])  # events = [1,1] and [5,6]

        op = EventBasedF1Metric()
        result = op.run((y_true, y_pred))
        # 默认 overlap_threshold=0 → [1,1] 与 [1,4] 重叠 1 点 → 匹配
        assert result.tp == 1

        op2 = EventBasedF1Metric(config=EventBasedF1Config(overlap_threshold=2))
        result2 = op2.run((y_true, y_pred))
        # 阈值=2 → 重叠 1 点不够 → 不匹配
        assert result2.tp == 0
        assert result2.fp == 2
        assert result2.fn == 1

    def test_multiple_events(self):
        """多事件匹配"""
        y_true = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0])

        op = EventBasedF1Metric()
        result = op.run((y_true, y_pred))

        assert result.n_true_events == 2
        assert result.n_pred_events == 2
        # 两个真值事件 [1,2] 和 [5,7] 各匹配一个预测事件 → tp=2, fn=0
        assert result.tp == 2
        assert result.fp == 0
        assert result.fn == 0
        assert result.f1 == 1.0

    def test_scores(self):
        """scores() 提取 f1 命名标量"""
        y_true = np.array([0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0])

        op = EventBasedF1Metric()
        scores = op.scores((y_true, y_pred))

        assert scores == {"event_f1": 1.0}

    def test_length_mismatch(self):
        """长度不一致 → ValueError"""
        op = EventBasedF1Metric()
        with pytest.raises(ValueError, match="长度不一致"):
            op.run((np.array([0, 1, 1]), np.array([0, 1])))


# ============================================================================
# SoftEventBasedF1Metric
# ============================================================================


class TestSoftEventBasedF1Basic:
    """Soft Event-Based F1 基本功能测试"""

    def test_perfect_match(self):
        """完美匹配 → soft P/R/F1 = 1.0"""
        y_true = np.array([0, 1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0])

        op = SoftEventBasedF1Metric()
        result = op.run((y_true, y_pred))

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_partial_overlap(self):
        """部分重叠 → soft F1 应介于 0 和 1 之间"""
        y_true = np.array([0, 1, 1, 1, 1, 0, 0])  # event [1,4]
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1])  # event [1,2] + [5,6]

        op = SoftEventBasedF1Metric()
        result = op.run((y_true, y_pred))

        # recall = IoU([1,2], [1,4]) = 2/4 = 0.5
        # precision = mean(IoU(p1, t1), IoU(p2, t1)) = mean(0.5, 0) = 0.25
        assert 0 < result.f1 < 1.0
        assert 0 < result.recall <= 1.0
        assert 0 <= result.precision <= 1.0

    def test_both_empty(self):
        """真值/预测都为空 → P/R/F1 = 1.0（完美）"""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        op = SoftEventBasedF1Metric()
        result = op.run((y_true, y_pred))

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_one_empty(self):
        """一边为空 → P/R/F1 = 0.0"""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 0])

        op = SoftEventBasedF1Metric()
        result = op.run((y_true, y_pred))

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_scores(self):
        """scores() 同时提取三个命名标量"""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])

        op = SoftEventBasedF1Metric()
        scores = op.scores((y_true, y_pred))

        assert "soft_event_f1" in scores
        assert scores["soft_event_f1"] == 1.0


# ============================================================================
# DetectionDelayMetric
# ============================================================================


class TestDetectionDelayBasic:
    """Detection Delay 基本功能测试"""

    def test_immediate_detection(self):
        """事件起点检出 → 延迟 = 0"""
        y_true = np.array([0, 1, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.1, 0.1, 0.1])

        op = DetectionDelayMetric(threshold=0.5)
        delay = op.run((y_true, y_score))

        assert delay == 0.0

    def test_no_detection(self):
        """未检出 → 延迟 = 事件长度"""
        y_true = np.array([0, 1, 1, 1, 1, 0])
        y_score = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        op = DetectionDelayMetric(threshold=0.5)
        delay = op.run((y_true, y_score))

        # 事件 [1,4] 长度=4，未检出 → delay=4
        assert delay == 4.0

    def test_partial_delay(self):
        """事件起点后 2 个点检出 → 延迟 = 2"""
        y_true = np.array([0, 1, 1, 1, 0])
        y_score = np.array([0.1, 0.1, 0.1, 0.9, 0.1])

        op = DetectionDelayMetric(threshold=0.5)
        delay = op.run((y_true, y_score))

        assert delay == 2.0

    def test_multiple_events(self):
        """多事件 → 返回平均延迟"""
        y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        y_score = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        op = DetectionDelayMetric(threshold=0.5)
        # 事件 [1,2]：检出在位置 0 → check_before_event=False → 延迟 2
        # 事件 [5,6]：未检出 → 延迟 2
        # 平均 = 2.0
        delay = op.run((y_true, y_score))
        assert delay == 2.0

    def test_return_all(self):
        """return_all=True 返回每事件延迟列表"""
        # 事件 [1,3] 在 y_score[1]=0.9 检出 → 延迟 0
        # 事件 [5,6] 在 y_score[6]=0.9 检出（位置 6 - 起点 5 = 1）→ 延迟 1
        y_true = np.array([0, 1, 1, 1, 0, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1])

        op = DetectionDelayMetric(threshold=0.5, return_all=True)
        delays = op.run((y_true, y_score))

        assert isinstance(delays, list)
        assert delays == [0.0, 1.0]

    def test_no_events(self):
        """无事件 → 平均延迟 0.0 / 空列表"""
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.1, 0.5, 0.9])

        op = DetectionDelayMetric(threshold=0.5)
        assert op.run((y_true, y_score)) == 0.0

        op2 = DetectionDelayMetric(threshold=0.5, return_all=True)
        assert op2.run((y_true, y_score)) == []

    def test_missing_config(self):
        """config 缺 threshold → ValidationError at construction"""
        # threshold 是必填字段，构造时就拒绝
        with pytest.raises(Exception):
            DetectionDelayMetric()

    def test_check_before_event(self):
        """check_before_event=True：事件开始前的检出也算有效"""
        y_true = np.array([0, 0, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.1, 0.1, 0.1])

        op_no_before = DetectionDelayMetric(threshold=0.5)
        # 事件 [2,3] 起点=2，检出在位置 1 → check_before_event=False → 延迟 2
        assert op_no_before.run((y_true, y_score)) == 2.0

        op_with_before = DetectionDelayMetric(threshold=0.5, check_before_event=True)
        # 检出在事件起点前 → 延迟 max(0, 1-2) = 0
        assert op_with_before.run((y_true, y_score)) == 0.0