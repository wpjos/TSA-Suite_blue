# -*- coding: utf-8 -*-

"""
通用算子注册中心

提供自动发现和注册算子类的通用机制，一个类可实例化多次分别服务于
feature.construction、detection、evaluation 三个模块。

核心能力:
    - 自动扫描指定包下的所有模块，发现符合条件的算子类
    - 以算子的 ``name()`` 返回值为键进行注册
    - 支持通过名称查找算子类
    - 支持可选的过滤函数，控制注册哪些算子

使用示例::

    from bianque.engine.operator.cli.registry import OperatorRegistry
    from bianque.engine.operator.base import BaseOperator

    registry = OperatorRegistry(
        base_class=BaseOperator,
        scan_packages=['bianque.engine.operator.detection'],
        filter_fn=lambda cls: hasattr(cls, 'name'),
    )
    registry.discover()
    all_operators = registry.list_all()
"""

import importlib
import inspect
import pkgutil
from typing import Callable

from bianque.engine.operator.base import BaseOperator

__all__ = [
    'OperatorRegistry',
]


class OperatorRegistry:
    """
    通用算子注册中心

    通过自动扫描指定包发现并注册算子类，以 ``name()`` 返回值为键。
    同一个类可实例化多次，分别用于不同模块的算子注册。

    Attributes:
        _base_class (type): 算子基类，用于 ``issubclass`` 过滤
        _scan_packages (list[str]): 需要扫描的包路径列表（点分格式）
        _filter_fn (Callable[[type], bool] | None): 额外的过滤函数，
            返回 True 表示该类应被注册
        _registry (dict[str, type]): 已注册的算子 {name: class} 映射
        _discovered (bool): 是否已执行过 discover 扫描
    """

    def __init__(
        self,
        base_class: type,
        scan_packages: list[str],
        filter_fn: Callable[[type], bool] | None = None,
    ) -> None:
        """
        初始化注册中心

        Args:
            base_class (type): 算子基类，仅该类的非抽象子类会被注册
            scan_packages (list[str]): 需要扫描的包路径列表，如
                ``['bianque.engine.operator.detection']``
            filter_fn (Callable[[type], bool] | None): 额外的过滤函数。
                当提供时，只有 ``filter_fn(cls)`` 返回 True 的类才会被注册。
                默认为 None，表示不做额外过滤
        """
        self._base_class = base_class
        self._scan_packages = scan_packages
        self._filter_fn = filter_fn
        self._registry: dict[str, type] = {}
        self._discovered: bool = False

    def discover(self) -> None:
        """
        扫描指定包，自动发现并注册所有符合条件的算子类

        扫描规则:
            1. 递归遍历 ``_scan_packages`` 中所有包的子模块
            2. 对每个模块中的类，检查是否为 ``_base_class`` 的子类
            3. 排除抽象类（含未实现的抽象方法）
            4. 排除没有 ``name`` 类方法的类
            5. 如有 ``_filter_fn``，额外调用过滤
            6. 以 ``cls.name()`` 为键注册到 ``_registry``

        重复调用时会增量合并，不会清空已有注册。

        Raises:
            ImportError: 扫描的包路径不存在或无法导入时
        """
        for package_path in self._scan_packages:
            # 导入顶层包
            package = importlib.import_module(package_path)

            # 递归遍历所有子模块
            package_paths = getattr(package, '__path__', None)
            if package_paths is None:
                # 不是包（是普通模块），直接扫描其中的类
                self._scan_module(package)
                continue

            for _importer, modname, _ispkg in pkgutil.walk_packages(
                package_paths, prefix=package_path + '.'
            ):
                try:
                    module = importlib.import_module(modname)
                except ImportError:
                    # 跳过无法导入的模块（可能有缺失依赖）
                    continue
                self._scan_module(module)

        self._discovered = True

    def _scan_module(self, module) -> None:
        """
        扫描单个模块中的类并注册符合条件的算子

        Args:
            module: 已导入的 Python 模块对象
        """
        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            # 必须是 base_class 的子类（但不是 base_class 本身）
            if not issubclass(obj, self._base_class) or obj is self._base_class:
                continue

            # 排除抽象类
            if inspect.isabstract(obj):
                continue

            # 必须有 name 类方法
            if not hasattr(obj, 'name') or not callable(getattr(obj, 'name')):
                continue

            # 额外过滤
            if self._filter_fn is not None and not self._filter_fn(obj):
                continue

            # 以 name() 为键注册
            try:
                op_name = obj.name()
            except Exception:
                # name() 调用失败则跳过
                continue

            self._registry[op_name] = obj

    def get(self, name: str) -> type:
        """
        通过算子名称获取已注册的算子类

        Args:
            name (str): 算子名称，即 ``cls.name()`` 的返回值

        Returns:
            type: 对应的算子类

        Raises:
            KeyError: 指定名称的算子未注册时
        """
        if not self._discovered:
            self.discover()

        if name not in self._registry:
            available = ', '.join(sorted(self._registry.keys()))
            raise KeyError(
                f"未找到名为 '{name}' 的算子。可用算子: [{available}]"
            )
        return self._registry[name]

    def list_all(self) -> dict[str, type]:
        """
        返回所有已注册算子的名称到类的映射

        Returns:
            dict[str, type]: {算子名称: 算子类} 字典，按名称排序
        """
        if not self._discovered:
            self.discover()

        return dict(sorted(self._registry.items()))

    @property
    def discovered(self) -> bool:
        """
        是否已执行过 discover 扫描

        Returns:
            bool: True 表示已扫描
        """
        return self._discovered

    def register(self, cls: type, name: str | None = None) -> None:
        """
        手动注册一个算子类

        Args:
            cls (type): 要注册的算子类
            name (str | None): 注册名称。为 None 时使用 ``cls.name()``

        Raises:
            ValueError: 类没有 ``name`` 方法且未提供 name 参数时
        """
        if name is None:
            if not hasattr(cls, 'name') or not callable(getattr(cls, 'name')):
                raise ValueError(
                    f"类 {cls.__name__} 没有 name() 方法，请显式提供 name 参数"
                )
            name = cls.name()

        self._registry[name] = cls
