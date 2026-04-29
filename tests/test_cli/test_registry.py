# -*- coding: utf-8 -*-

"""
通用算子注册中心单元测试

对应源文件：
- cli/registry.py: OperatorRegistry

测试范围：
- discover 自动扫描注册
- get 按名称查找
- list_all 列出所有
- register 手动注册
- 自动触发 discover（懒加载）
- 过滤函数 filter_fn
- 异常场景（未知名称、无 name 方法等）
"""

import pytest

from bianque.engine.operator.cli.registry import OperatorRegistry
from bianque.engine.operator.base import BaseOperator


# ============================================================================
# 辅助：用于测试的简单算子类
# ============================================================================

class _DummyOperator(BaseOperator[None, None, None, None]):
    """测试用的简单算子"""

    @classmethod
    def name(cls) -> str:
        return "dummy_op"

    def _run(self, x, *, params):
        return x


class _AnotherOperator(BaseOperator[None, None, None, None]):
    """另一个测试用算子"""

    @classmethod
    def name(cls) -> str:
        return "another_op"

    def _run(self, x, *, params):
        return x


# ============================================================================
# 测试类
# ============================================================================

class TestOperatorRegistryDiscover:
    """测试 OperatorRegistry 的 discover 扫描功能"""

    def test_discover_feature_construction(self):
        """
        目的：验证 discover 能扫描 feature.construction 包
        输入：扫描 bianque.engine.operator.feature.construction
        预期：至少发现 5 个特征算子（square_feature 等）
        """
        from bianque.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['bianque.engine.operator.feature.construction'],
        )
        registry.discover()
        operators = registry.list_all()

        assert len(operators) >= 5
        assert 'square_feature' in operators
        assert 'polynomial_feature' in operators
        assert 'rolling_mean_feature' in operators
        assert 'column_median_feature' in operators
        assert 'pca_feature' in operators

    def test_discover_detection(self):
        """
        目的：验证 discover 能扫描 detection 包并通过 filter_fn 过滤
        输入：扫描 bianque.engine.operator.detection，过滤 Scorer 和 Decider
        预期：发现 Scorer、Decider、Detector 类型的算子
        """
        from bianque.engine.operator.detection.base import (
            BaseScorerMixin, BaseDeciderMixin, BaseDetector,
        )

        def _filter(cls):
            return issubclass(cls, (BaseScorerMixin, BaseDeciderMixin, BaseDetector))

        registry = OperatorRegistry(
            base_class=BaseOperator,
            scan_packages=['bianque.engine.operator.detection'],
            filter_fn=_filter,
        )
        registry.discover()
        operators = registry.list_all()

        # 应该包含 Scorer、Detector
        assert 'knn_scorer' in operators
        assert 'knn_detector' in operators
        assert 'residual_scorer' in operators
        assert 'threshold_decider' in operators

    def test_discover_evaluation(self):
        """
        目的：验证 discover 能扫描 evaluation 包
        输入：扫描 bianque.engine.operator.evaluation
        预期：发现 5 个评价指标算子
        """
        from bianque.engine.operator.evaluation.base import BaseMetricOperator

        registry = OperatorRegistry(
            base_class=BaseMetricOperator,
            scan_packages=['bianque.engine.operator.evaluation'],
        )
        registry.discover()
        operators = registry.list_all()

        assert len(operators) >= 5
        assert 'binary_classification' in operators
        assert 'self_evaluation' in operators

    def test_discover_sets_flag(self):
        """
        目的：验证 discover 后 discovered 属性为 True
        输入：创建注册中心并调用 discover
        预期：discovered 从 False 变为 True
        """
        from bianque.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['bianque.engine.operator.feature.construction'],
        )
        assert registry.discovered is False
        registry.discover()
        assert registry.discovered is True

    def test_discover_incremental(self):
        """
        目的：验证 discover 重复调用会增量合并
        输入：先 discover 一次，手动注册一个，再 discover
        预期：两次 discover 的结果加上手动注册的都存在
        """
        from bianque.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['bianque.engine.operator.feature.construction'],
        )
        registry.discover()
        count1 = len(registry.list_all())

        # 手动注册一个
        registry.register(_DummyOperator, name="test_dummy")

        # 再次 discover
        registry.discover()
        count2 = len(registry.list_all())

        # 手动注册的仍在
        assert 'test_dummy' in registry.list_all()
        assert count2 >= count1


class TestOperatorRegistryGet:
    """测试 OperatorRegistry 的 get 查找功能"""

    def test_get_existing(self):
        """
        目的：验证 get 能找到已注册的算子
        输入：手动注册后 get
        预期：返回对应的类
        """
        registry = OperatorRegistry(
            base_class=BaseOperator,
            scan_packages=[],
        )
        registry.register(_DummyOperator)
        assert registry.get("dummy_op") is _DummyOperator

    def test_get_not_found_raises(self):
        """
        目的：验证 get 未找到时抛出 KeyError
        输入：查找一个不存在的算子名称
        预期：抛出 KeyError，错误信息包含可用算子列表
        """
        registry = OperatorRegistry(
            base_class=BaseOperator,
            scan_packages=[],
        )
        registry._discovered = True  # 跳过 discover

        with pytest.raises(KeyError, match="未找到名为"):
            registry.get("nonexistent_operator")

    def test_get_triggers_discover(self):
        """
        目的：验证 get 在未 discover 时自动触发 discover
        输入：不调用 discover 直接 get
        预期：自动触发 discover 后能找到算子
        """
        from bianque.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['bianque.engine.operator.feature.construction'],
        )
        assert registry.discovered is False

        # get 应自动触发 discover
        cls = registry.get('square_feature')
        assert cls is not None
        assert registry.discovered is True


class TestOperatorRegistryListAll:
    """测试 OperatorRegistry 的 list_all 功能"""

    def test_list_all_sorted(self):
        """
        目的：验证 list_all 返回按名称排序的字典
        输入：注册两个算子（名称逆序）
        预期：返回按名称升序排列的字典
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry._discovered = True
        registry.register(_AnotherOperator)
        registry.register(_DummyOperator)

        result = registry.list_all()
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_list_all_triggers_discover(self):
        """
        目的：验证 list_all 在未 discover 时自动触发 discover
        输入：不调用 discover 直接 list_all
        预期：自动触发 discover 后返回结果
        """
        from bianque.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['bianque.engine.operator.feature.construction'],
        )
        result = registry.list_all()
        assert len(result) >= 5
        assert registry.discovered is True


class TestOperatorRegistryRegister:
    """测试 OperatorRegistry 的 register 手动注册功能"""

    def test_register_with_auto_name(self):
        """
        目的：验证 register 不指定 name 时使用 cls.name()
        输入：注册 _DummyOperator 不指定 name
        预期：以 "dummy_op" 为 key 注册
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_DummyOperator)
        assert 'dummy_op' in registry._registry

    def test_register_with_explicit_name(self):
        """
        目的：验证 register 指定 name 时使用显式名称
        输入：注册 _DummyOperator 指定 name="custom_name"
        预期：以 "custom_name" 为 key 注册
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_DummyOperator, name="custom_name")
        assert 'custom_name' in registry._registry
        assert registry._registry['custom_name'] is _DummyOperator

    def test_register_no_name_method_raises(self):
        """
        目的：验证类没有 name 方法且未提供 name 参数时抛出 ValueError
        输入：注册一个没有 name 方法的普通类
        预期：抛出 ValueError
        """
        class _NoNameClass:
            pass

        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        with pytest.raises(ValueError, match="没有 name\\(\\) 方法"):
            registry.register(_NoNameClass)


class TestOperatorRegistryFilterFn:
    """测试 OperatorRegistry 的 filter_fn 过滤功能"""

    def test_filter_fn_excludes(self):
        """
        目的：验证 filter_fn 返回 False 时不注册
        输入：filter_fn 排除所有类
        预期：注册表为空
        """
        from bianque.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['bianque.engine.operator.feature.construction'],
            filter_fn=lambda cls: False,
        )
        registry.discover()
        assert len(registry.list_all()) == 0

    def test_filter_fn_includes_selectively(self):
        """
        目的：验证 filter_fn 可以选择性注册
        输入：filter_fn 只注册 name() 以 "square" 开头的算子
        预期：只注册 square_feature
        """
        from bianque.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['bianque.engine.operator.feature.construction'],
            filter_fn=lambda cls: cls.name().startswith('square'),
        )
        registry.discover()
        operators = registry.list_all()
        assert len(operators) == 1
        assert 'square_feature' in operators
