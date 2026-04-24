# -*- coding: utf-8 -*-

"""
算子基类定义模块

提供重新设计的算子基类 BaseOperator和可训练能力混入 LearnableOperatorMixin，采用三类参数分离原则和 Pydantic 参数验证机制。
另提供面向数值计算的中间层 NumericOperator，以及辅助类型 ArrayN、NumericData 和 DataFrameMeta。

核心组件:
- BaseOperator: 所有算子的基类，提供参数管理、名称标识、持久化接口和模板方法执行
- LearnableOperatorMixin: 可训练能力混入，为任何 BaseOperator 子类添加 fit 能力
- NumericOperator: 数值算子基类，接受 DataFrame 或 ndarray 输入，提供细粒度模板方法管线
- DataFrameMeta: DataFrame 元信息快照，用于 ndarray → DataFrame 的反向转换
- ArrayN: 数值 ndarray 类型别名
- NumericData: DataFrame | ArrayN 联合类型别名

参数管理遵循三类参数分离原则:
- 类型1（实例参数 Config）: 通过 __init__(config=, **kwargs) 传入，经 Pydantic 验证后存储，创建后不可变
- 类型2（训练参数 FitParams）: 通过 fit(x, params=, **kwargs) 传入，经 Pydantic 验证，每次训练独立
- 类型3（运行参数 RunParams）: 通过 run(x, params=, **kwargs) 传入，经 Pydantic 验证，每次运行独立

使用示例::

    from pydantic import BaseModel, Field

    class MyConfig(BaseModel):
        threshold: float = Field(default=0.5, gt=0)

    class MyRunParams(BaseModel):
        verbose: bool = False

    # 无需训练算子
    class MyOperator(BaseOperator[InputType, OutputType, MyConfig, MyRunParams]):
        def _run(self, x, params):
            threshold = self._resolve_param(params, 'threshold')
            ...

    # 可训练算子（LearnableOperatorMixin + BaseOperator 多重继承）
    class MyFitParams(BaseModel):
        epochs: int = 10

    class MyModel(
        LearnableOperatorMixin[InputType, TargetType, MyFitParams],
        BaseOperator[InputType, OutputType, MyConfig, MyRunParams]
    ):
        def _fit(self, x, y, params):
            ...
            self._fitted = True

        def _run(self, x, params):
            ...
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, Any, ClassVar, Self, Union, Annotated, get_origin, get_args

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from bianque.basic.util.random import generate_secure_upper_case_id

__all__ = [
    'BaseOperator',
    'LearnableOperatorMixin',
    'NumericOperator',
    'DataFrameMeta',
    'ArrayN',
    'NumericData',
    'I',
    'O',
    'C',
    'RP',
    'FP',
]

# ============================================================================
# 泛型类型变量
# ============================================================================

I = TypeVar("I")
"""输入数据类型泛型"""

O = TypeVar("O")
"""输出数据类型泛型"""

T = TypeVar("T")
"""训练目标数据类型泛型"""

C = TypeVar("C", bound=Union[BaseModel, None])
"""实例参数（Config）类型泛型，bound 为 Union[BaseModel, None]，None 表示该算子无实例参数"""

RP = TypeVar("RP", bound=Union[BaseModel, None])
"""运行参数（RunParams）类型泛型，bound 为 Union[BaseModel, None]，None 表示该算子无运行参数"""

FP = TypeVar("FP", bound=Union[BaseModel, None])
"""训练参数（FitParams）类型泛型，bound 为 Union[BaseModel, None]，None 表示该算子无训练参数"""

EO = TypeVar("EO", bound=Union[BaseModel, None])
"""附加输出（ExtraOutput）类型泛型，bound 为 Union[BaseModel, None]，None 表示该算子附加输出"""


class BaseOperator(Generic[I, O, C, RP], metaclass=ABCMeta):
    """
    算子基础类

    提供实例参数管理（Pydantic 验证）、名称标识、持久化接口和模板方法执行。
    ``run`` 方法采用模板方法模式：外层自动完成参数验证，子类实现 ``_run`` 接收验证后的强类型参数。

    泛型参数:
        - I: 输入数据类型
        - O: 输出数据类型
        - C: 实例参数类型（bound=Union[BaseModel, None]），通过 __init__(config=, **kwargs) 传入
        - RP: 运行参数类型（bound=Union[BaseModel, None]），通过 run(x, params=, **kwargs) 传入
    """

    _config_type: ClassVar[type[BaseModel] | None] = None
    """实例参数的 Pydantic 模型类型，由 __init_subclass__ 从泛型参数中自动提取"""

    _run_params_type: ClassVar[type[BaseModel] | None] = None
    """运行参数的 Pydantic 模型类型，由 __init_subclass__ 从泛型参数中自动提取"""

    _CONFIG_FILE_NAME: ClassVar[str] = "config.json"
    """持久化时实例参数的默认文件名"""

    _LAST_RUN_PARAMS_FILE_NAME: ClassVar[str] = "last_run_params.json"
    """持久化时最后一次运行参数的默认文件名"""

    @staticmethod
    def _extract_type_from_typevar(
            cls,
            base_class: type,
            typevar: TypeVar,
    ) -> type[BaseModel] | None:
        """从 ``__orig_bases__`` 中提取指定 TypeVar 对应的具体 BaseModel 子类型。

        遍历 ``cls.__orig_bases__``，找到以 ``base_class`` 为 origin 的泛型基类，
        将其 ``__parameters__`` 与实际 ``get_args`` 建立 TypeVar→type 映射，
        返回 ``typevar`` 对应的具体类型（必须是 BaseModel 的子类）。

        Args:
            cls: 被定义的子类（由 ``__init_subclass__`` 传入）
            base_class: 要匹配的泛型基类（如 BaseOperator、NumericOperator）
            typevar: 目标 TypeVar（如 C、RP、EO）

        Returns:
            type[BaseModel] | None: 提取到的具体类型，未找到或不是 BaseModel 子类时返回 None
        """
        for base in getattr(cls, '__orig_bases__', ()):
            origin = get_origin(base)
            if origin is None:
                continue
            if not (isinstance(origin, type) and issubclass(origin, base_class)):
                continue
            args = get_args(base)
            origin_params = getattr(origin, '__parameters__', ())
            if not args or not origin_params:
                continue
            type_map = dict(zip(origin_params, args))
            if typevar in type_map and isinstance(type_map[typevar], type) and issubclass(type_map[typevar], BaseModel):
                return type_map[typevar]
            break
        return None

    def __init_subclass__(cls, **kwargs):
        """
        子类定义时的钩子，自动从泛型参数中提取 Config(C) 和 RunParams(RP) 的实际类型。

        通过 ``_extract_type_from_typevar`` 静态方法从 ``__orig_bases__`` 中提取，
        支持直接继承和通过中间抽象类的多层继承场景。
        """
        super().__init_subclass__(**kwargs)
        if cls._config_type is None:
            cls._config_type = cls._extract_type_from_typevar(cls, BaseOperator, C)
        if cls._run_params_type is None:
            cls._run_params_type = cls._extract_type_from_typevar(cls, BaseOperator, RP)

    def __init__(self, *, oid: str | None = None, config: C | None = None, **kwargs):
        """
        初始化算子

        Args:
            oid(str, optional):
                算子实例唯一标识后缀，缺省自动生成（格式: {name()}$RANDOM_ID）
            config(C | None, optional):
                类型化实例参数，优先级高于键值对参数
            **kwargs:
                实例参数，将由对应的 Pydantic 模型进行类型校验和约束验证
        """
        self._oid = f"{self.name()}${oid if oid else generate_secure_upper_case_id(8)}"
        self._config: C | None = self._validated_config(config, **kwargs)
        self._last_run_params: RP | None = None

    @classmethod
    @abstractmethod
    def name(cls):
        ...

    @staticmethod
    def _validate_params(
            param_name: str,
            params_type: type[BaseModel] | None,
            params: BaseModel | None,
            **kwargs
    ) -> BaseModel | None:
        """
        通用参数校验方法

        对三类参数（Config / RunParams / FitParams）执行统一的校验逻辑：
        类型为 None 时直接返回 None；传入类型化实例时优先使用（kwargs 被忽略）；
        类型不匹配时 warning 并降级为 kwargs 构造。

        Args:
            param_name: 参数类别名称（用于日志提示）
            params_type: 期望的 Pydantic 模型类型，None 表示当前算子未定义该参数类型
            params: 类型化参数实例
            **kwargs: 参数键值对

        Returns:
            验证后的 Pydantic 模型实例，或 None

        Raises:
            pydantic.ValidationError: kwargs 构造时参数验证失败
        """
        if params_type is None:
            if params is not None:
                logger.warning(f"当前算子未定义 {param_name} 类型，{param_name} 参数将被忽略")
            if kwargs:
                logger.warning(f"当前算子未定义 {param_name} 类型，{param_name} 键值对参数将被忽略")
            return None
        if params is not None:
            if isinstance(params, params_type):
                if len(kwargs) > 0:
                    logger.warning(f"{param_name} 键值对与类型化 {param_name} 冲突，将被忽略")
                return params
            else:
                logger.warning(f"{param_name} 类型错误 type={type(params)}，将被忽略")
        return params_type(**kwargs)

    def _validated_config(self, config, **kwargs) -> C | None:
        """
        验证实例参数

        Args:
            config: 类型化实例参数
            **kwargs: 实例参数键值对

        Returns:
            验证后的实例参数，或 None

        Raises:
            pydantic.ValidationError: 参数验证失败时
        """
        return self._validate_params("config", self._config_type, config, **kwargs)

    def _validated_run_params(self, params: RP | None = None, **kwargs) -> RP | None:
        """
        验证运行参数

        Args:
            params: 类型化运行参数
            **kwargs: 运行参数键值对

        Returns:
            验证后的运行参数，或 None

        Raises:
            pydantic.ValidationError: 参数验证失败时
        """
        return self._validate_params("run_params", self._run_params_type, params, **kwargs)

    @property
    def oid(self) -> str:
        """
        获取算子名称

        Returns:
            str: 算子唯一标识名称
        """
        return self._oid

    @property
    def config(self) -> C | None:
        """
        获取实例参数

        Returns:
            实例参数模型，或 None（未配置实例参数类型时）
        """
        return self._config

    @property
    def last_run_params(self) -> RP | None:
        """
        获取最后一次运行使用的参数

        Returns:
            最后一次运行的验证后参数，或 None（未执行过 run 时）
        """
        return self._last_run_params

    def _resolve_param(self, params: Any, key: str, default: Any = None) -> Any:
        """
        按优先级解析参数值：运行参数 → 实例参数 → 默认值

        用于在子类的 ``_run`` 方法中，当运行参数需要覆盖实例参数时使用。
        值为 None 被视为"未指定"，将回退到下一优先级。

        Args:
            params:
                运行参数（Pydantic 模型实例、dict 或 None）
            key(str):
                参数键名
            default(Any, optional):
                所有来源均未找到时的默认值

        Returns:
            Any: 解析后的参数值
        """
        # 优先级1：运行参数
        if params is not None:
            if isinstance(params, BaseModel):
                value = getattr(params, key, None)
            elif isinstance(params, dict):
                value = params.get(key) if key in params else None
            else:
                value = None
            if value is not None:
                return value
        # 优先级2：实例参数
        if self._config is not None:
            value = getattr(self._config, key, None)
            if value is not None:
                return value
        # 优先级3：默认值
        return default

    def _can_run(self) -> None:
        """
        推理前置校验钩子

        默认不做任何校验（pass），子类可 override 以添加前置条件检查。
        由 ``run`` 模板方法在参数验证之后、调用 ``_run`` 之前自动调用。
        抛出异常时将中止 ``run`` 的执行。

        Raises:
            任意异常: 子类可根据需要抛出 RuntimeError 等异常以中止推理
        """
        pass

    def run(self, x: I, *, params: RP | None = None, **kwargs) -> O:
        """
        执行算子核心逻辑（模板方法）

        自动完成运行参数的 Pydantic 验证，然后调用子类实现的 ``_run`` 方法。
        支持两种传参方式：传入类型化 RunParams 实例，或传入关键字参数由 Pydantic 自动验证。

        Args:
            x(I): 输入数据
            params(RP | None, optional):
                类型化运行参数，优先级高于键值对参数
            **kwargs: 运行时参数键值对覆盖

        Returns:
            O: 处理后的输出结果
        """
        # 步骤1：验证并构建运行参数（Pydantic 校验）
        validated_params = self._validated_run_params(params, **kwargs)
        # 步骤2：执行前置校验（默认 pass，LearnableOperatorMixin 会检查训练状态）
        self._can_run()
        # 步骤3：调用子类实现的核心逻辑
        result = self._run(x, params=validated_params)
        # 步骤4：记录本次运行参数
        self._last_run_params = validated_params
        return result

    @abstractmethod
    def _run(self, x: I, *, params: RP | None) -> O:
        """
        子类实现的核心执行逻辑

        由 ``run`` 模板方法调用，接收已通过 Pydantic 验证的强类型参数。
        注意：``params`` 为 keyword-only 参数，确保调用时意图明确。

        Args:
            x(I): 输入数据
            params(RP | None): 验证后的运行参数，可能为 None（未定义 RunParams 类型时）

        Returns:
            O: 处理后的输出结果
        """
        ...

    def save(self, path: str | Path) -> None:
        """
        持久化算子到指定目录

        默认实现将实例参数（Config）和最后一次运行参数序列化为 JSON 文件。
        子类可 override 以保存额外状态（如模型权重），建议先调用 ``super().save(path)``。

        Args:
            path(str | Path):
                目标目录路径

        Raises:
            ValueError: 当 path 指向一个已存在的文件时
        """
        path = Path(path)
        if path.is_file():
            raise ValueError(f"期望目录路径，但 '{path}' 是一个已存在的文件")
        path.mkdir(parents=True, exist_ok=True)
        if self._config is not None:
            (path / self._CONFIG_FILE_NAME).write_text(
                self._config.model_dump_json(indent=2),
                encoding='utf-8'
            )
        if self._last_run_params is not None:
            (path / self._LAST_RUN_PARAMS_FILE_NAME).write_text(
                self._last_run_params.model_dump_json(indent=2),
                encoding='utf-8'
            )

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        """
        从指定目录加载算子

        默认实现从 JSON 文件恢复实例参数和最后一次运行参数并重建算子。
        子类可 override 以加载额外状态，建议先调用 ``instance = super().load(path, oid=oid)``。

        Args:
            path(str | Path):
                源目录路径
            oid(str | None, optional):
                算子实例唯一标识后缀，作用与 __init__ 中的 oid 参数一致（缺省自动生成）

        Returns:
            加载后的算子实例

        Raises:
            FileNotFoundError: 当目录或配置文件不存在时
        """
        path = Path(path)
        config_file = path / cls._CONFIG_FILE_NAME
        if cls._config_type is not None and config_file.exists():
            config_model = cls._config_type.model_validate_json(
                config_file.read_text(encoding='utf-8')
            )
            instance = cls(oid=oid, config=config_model)
        else:
            instance = cls(oid=oid)
        # 恢复最后一次运行参数
        run_params_file = path / cls._LAST_RUN_PARAMS_FILE_NAME
        if cls._run_params_type is not None and run_params_file.exists():
            instance._last_run_params = cls._run_params_type.model_validate_json(
                run_params_file.read_text(encoding='utf-8')
            )
        return instance


class LearnableOperatorMixin(Generic[I, T, FP], metaclass=ABCMeta):
    """
    可训练能力混入

    为任何 BaseOperator 子类添加 fit 能力。
    使用方式：多重继承 ``LearnableOperatorMixin[I, T, FP]`` + ``BaseOperator[I, O, C, RP]``。

    .. important::
        **继承顺序警告**：``LearnableOperatorMixin`` 必须放在 ``BaseOperator`` **前面**！

        正确示例（Mixin在前）::

            class MyModel(
                LearnableOperatorMixin[InputType, TargetType, MyFitParams],  # ← 必须在前
                BaseOperator[InputType, OutputType, MyConfig, MyRunParams]
            ):
                ...

        错误示例（Mixin在后）::

            class MyModel(
                BaseOperator[InputType, OutputType, MyConfig, MyRunParams],  # ← 错误位置
                LearnableOperatorMixin[InputType, TargetType, MyFitParams]
            ):
                ...

        原因：Python MRO（方法解析顺序）需要先解析 Mixin 的 ``_can_run`` 方法，
        否则无法正确拦截未训练算子的 ``run`` 调用。

    典型用法::

        class MyModel(
            LearnableOperatorMixin[InputType, TargetType, MyFitParams],
            BaseOperator[InputType, OutputType, MyConfig, MyRunParams]
        ):
            def _fit(self, x, y, params):
                ...
                self._fitted = True

            def _run(self, x, params):
                ...

    泛型参数:
        - I: 输入数据类型
        - T: 训练目标类型
        - FP: 训练参数类型（bound=Union[BaseModel, None]），通过 fit(x, params=, **kwargs) 传入
    """

    _fit_params_type: ClassVar[type[BaseModel] | None] = None
    """训练参数的 Pydantic 模型类型，由 __init_subclass__ 从泛型参数中自动提取"""

    _LAST_FIT_PARAMS_FILE_NAME: ClassVar[str] = "last_fit_params.json"
    """持久化时最后一次训练参数的默认文件名"""

    def __init_subclass__(cls, **kwargs):
        """
        子类定义时的钩子，自动从泛型参数中提取 FitParams(FP) 的实际类型。

        提取机制基于 ``__orig_bases__`` 和 ``__parameters__`` 的映射关系，
        与 BaseOperator 的类型提取机制一致。

        Raises:
            TypeError: 当继承顺序错误时（Mixin 必须放在 BaseOperator 前面）
        """
        super().__init_subclass__(**kwargs)

        # 检查 MRO 顺序：LearnableOperatorMixin 必须在 BaseOperator 前面
        # 这是为了确保 _can_run() 方法能够正确覆写并拦截未训练算子
        if issubclass(cls, BaseOperator):
            mro = cls.__mro__
            mixin_idx = -1
            base_idx = -1
            for i, c in enumerate(mro):
                if c is LearnableOperatorMixin:
                    mixin_idx = i
                if c is BaseOperator:
                    base_idx = i
            if mixin_idx > base_idx and base_idx != -1:
                raise TypeError(
                    f"类 {cls.__name__} 的继承顺序错误："
                    f"LearnableOperatorMixin 必须放在 BaseOperator 前面！\n"
                    f"当前顺序导致 _can_run() 无法正确拦截未训练算子。\n"
                    f"请改为：class {cls.__name__}(LearnableOperatorMixin[...], BaseOperator[...]): ..."
                )

        for base in getattr(cls, '__orig_bases__', ()):
            origin = get_origin(base)
            if origin is None:
                continue
            if not (isinstance(origin, type) and LearnableOperatorMixin in origin.__mro__):
                continue
            args = get_args(base)
            origin_params = getattr(origin, '__parameters__', ())
            if not args or not origin_params:
                continue
            # 建立 TypeVar → 实际类型 的映射
            type_map = dict(zip(origin_params, args))
            # 提取 FitParams 类型（对应泛型参数 FP）
            if FP in type_map and isinstance(type_map[FP], type) and issubclass(type_map[FP], BaseModel):
                cls._fit_params_type = type_map[FP]
            break

    def __init__(self, **kwargs):
        """
        初始化可训练能力

        通过 ``**kwargs`` 将参数透传给下一个基类（通常是 BaseOperator），
        然后初始化训练状态和训练参数记录。

        Args:
            **kwargs: 透传给 BaseOperator.__init__ 的参数（oid, config 等）
        """
        # 透传给 MRO 中的下一个基类（通常是 BaseOperator.__init__）
        super().__init__(**kwargs)
        self._fitted: bool = False
        self._last_fit_params: FP | None = None

    @property
    def is_fitted(self) -> bool:
        """
        是否已完成训练

        Returns:
            bool: 训练状态
        """
        return self._fitted

    @property
    def can_additional_fit(self) -> bool:
        """
        是否支持增训（在已完成训练的基础上继续训练）

        默认不支持，子类可 override 返回 True 以启用增训。

        Returns:
            bool: 是否支持增训
        """
        return False

    @property
    def last_fit_params(self) -> FP | None:
        """
        获取最后一次训练使用的参数

        Returns:
            最后一次训练的验证后参数，或 None（未执行过 fit 时）
        """
        return self._last_fit_params

    def _validated_fit_params(self, params: FP | None = None, **kwargs) -> FP | None:
        """
        验证训练参数

        委托 ``_validate_params`` 执行统一的校验逻辑（实现由 BaseOperator 提供）。

        Args:
            params: 类型化训练参数
            **kwargs: 训练参数键值对

        Returns:
            验证后的训练参数，或 None

        Raises:
            pydantic.ValidationError: 参数验证失败时
        """
        return self._validate_params("fit_params", self._fit_params_type, params, **kwargs)

    def fit(self, x: I, y: T, *, params: FP | None = None, **kwargs) -> Self:
        """
        训练算子（模板方法）

        自动完成前置校验（增训许可检查）和训练参数的 Pydantic 验证，
        然后调用子类实现的 ``_fit`` 方法。

        Args:
            x(I): 输入训练数据
            y(T): 训练目标数据
            params(FP | None, optional): 类型化训练参数，优先级高于键值对参数
            **kwargs: 训练参数键值对覆盖

        Returns:
            Self: 训练后的算子实例

        Raises:
            RuntimeError: 已训练且不支持增训时
        """
        # 前置校验：已训练且不支持增训时拒绝重复训练
        if self._fitted and not self.can_additional_fit:
            raise RuntimeError("训练已完成，且该算子不支持增训")
        # 验证并构建训练参数（Pydantic 校验）
        validated_params = self._validated_fit_params(params, **kwargs)
        # 调用子类实现的核心训练逻辑
        self._fit(x, y, params=validated_params)
        # 记录本次训练参数
        self._last_fit_params = validated_params
        return self

    @abstractmethod
    def _fit(self, x: I, y: T, *, params: FP | None) -> None:
        """
        子类实现的核心训练逻辑

        由 ``fit`` 模板方法调用，接收已通过 Pydantic 验证的强类型参数。
        子类**必须**在训练完成后设置 ``self._fitted = True``，否则后续 ``run`` 将抛出异常。
        注意：``params`` 为 keyword-only 参数，确保调用时意图明确。

        Args:
            x(I): 输入训练数据
            y(T): 训练目标数据（如标签或目标序列）
            params(FP | None): 验证后的训练参数，可能为 None（未定义 FitParams 类型时）
        """
        ...

    def _can_run(self) -> None:
        """
        校验训练状态，未训练时抛出 RuntimeError

        重写父类模板方法，确保推理前已完成训练。

        Raises:
            RuntimeError: 尚未完成训练时
        """
        super()._can_run()
        if not self._fitted:
            raise RuntimeError("训练尚未完成，无法执行推理")

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态到指定目录（最后一次训练参数）

        供子类在 save() 中调用，与 ``BaseOperator.save`` 协同工作。

        Args:
            path(Path): 目标目录路径
        """
        if self._last_fit_params is not None:
            (path / self._LAST_FIT_PARAMS_FILE_NAME).write_text(
                self._last_fit_params.model_dump_json(indent=2),
                encoding='utf-8'
            )

    def _load_fit_state(self, path: Path) -> None:
        """
        从指定目录恢复训练状态（最后一次训练参数）

        供子类在 load() 中调用，与 ``BaseOperator.load`` 协同工作。
        ``_fitted`` 状态不自动恢复，由子类在 override 中根据实际情况决定。

        Args:
            path(Path): 源目录路径
        """
        fit_params_file = path / self._LAST_FIT_PARAMS_FILE_NAME
        if self._fit_params_type is not None and fit_params_file.exists():
            self._last_fit_params = self._fit_params_type.model_validate_json(
                fit_params_file.read_text(encoding='utf-8')
            )

    def save(self, path: str | Path) -> None:
        """
        持久化可训练算子到指定目录

        在 BaseOperator.save 的基础上，额外保存最后一次训练参数。
        子类可 override 以保存模型权重等额外状态，建议先调用 ``super().save(path)``。

        Args:
            path(str | Path):
                目标目录路径

        Raises:
            ValueError: 当 path 指向一个已存在的文件时
        """
        super().save(path)
        self._save_fit_state(Path(path))

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        """
        从指定目录加载可训练算子

        在 BaseOperator.load 的基础上，额外恢复最后一次训练参数。
        ``_fitted`` 状态不自动恢复，由子类在 override 中根据实际情况决定。

        Args:
            path(str | Path):
                源目录路径
            oid(str | None, optional):
                算子实例唯一标识后缀，作用与 __init__ 中的 oid 参数一致（缺省自动生成）

        Returns:
            加载后的算子实例

        Raises:
            FileNotFoundError: 当目录或配置文件不存在时
        """
        instance = super().load(path, oid=oid)
        instance._load_fit_state(Path(path))
        return instance


# ============================================================================
# 支持DataFrame/ndarray双类型的基础算子类型
# ============================================================================

ArrayN = Annotated[np.ndarray[Any, np.dtype[np.integer | np.floating]], "ArrayN"]
"""数值 ndarray 类型别名，约束 dtype 为 integer 或 floating"""

NumericData = Annotated[pd.DataFrame | ArrayN, "NumericData"]
"""数值数据联合类型，支持 pd.DataFrame 和数值 ndarray 两种输入输出格式"""


class DataFrameMeta:
    """
    DataFrame 元信息快照

    保存 DataFrame 的列名、数据类型和索引，用于 ndarray → DataFrame 的反向转换。
    采用 ``__slots__`` 设计，轻量无额外开销。

    Attributes:
        column_names: 列名列表
        column_types: 列数据类型列表
        index: 行索引

    使用示例::

        meta = DataFrameMeta.from_dataframe(df)
        arr = df.values
        # ... 计算 ...
        result_df = pd.DataFrame(arr, columns=["score"], index=meta.index)
    """

    __slots__ = ('column_names', 'column_types', 'index')

    def __init__(self, column_names: list[str], column_types: list[np.dtype], index: pd.Index):
        """
        初始化 DataFrame 元信息

        Args:
            column_names: 列名列表
            column_types: 列数据类型列表
            index: 行索引
        """
        self.column_names: list[str] = column_names
        self.column_types: list[np.dtype] = column_types
        self.index = index

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """
        从 DataFrame 创建元信息快照

        Args:
            df: 源 DataFrame

        Returns:
            DataFrameMeta: 元信息快照实例
        """
        return cls(
            column_names=list(df.columns),
            column_types=list(df.dtypes),
            index=df.index
        )


class _NumericOperatorMixin(metaclass=ABCMeta):
    """数值算子共享 IO 处理混入（内部类，不对外暴露）

    提供 ``NumericOperator`` 和 ``BiNumericOperator`` 共享的数据 IO 处理方法，
    包括输入校验、筛选、解包、调整、打包和索引调整。

    该混入类不继承任何 Operator 基类，通过多重继承被 NumericOperator 和
    BiNumericOperator 组合使用。
    """

    def _validate_input(self, x: NumericData, params: RP | None) -> None:
        """
        验证输入数据类型

        根据输入类型分发到对应的校验钩子，非法类型直接抛出 TypeError。
        子类通常不需要覆写此方法，而是覆写 ``_validate_dataframe_input`` 或
        ``_validate_ndarray_input`` 以添加自定义校验逻辑。

        Args:
            x(NumericData): 输入数据
            params(RP | None): 运行参数

        Raises:
            TypeError: 输入既非 DataFrame 也非数值 ndarray 时
        """
        if isinstance(x, pd.DataFrame):
            self._validate_dataframe_input(x, params)
        elif isinstance(x, np.ndarray) and issubclass(x.dtype.type, (np.integer, np.floating)):
            self._validate_ndarray_input(x, params)
        else:
            raise TypeError(f"输入数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(x)}")

    def _validate_dataframe_input(self, x: pd.DataFrame, params: RP | None) -> None:
        """
        DataFrame 输入的自定义校验钩子

        默认无操作，子类可覆写以添加校验逻辑（如检查空 DataFrame、列名合法性等）。

        Args:
            x(pd.DataFrame): 输入 DataFrame
            params(RP | None): 运行参数
        """
        pass

    def _validate_ndarray_input(self, x: np.ndarray, params: RP | None) -> None:
        """
        ndarray 输入的自定义校验钩子

        默认无操作，子类可覆写以添加校验逻辑（如检查维度、dtype 等）。

        Args:
            x(np.ndarray): 输入 ndarray
            params(RP | None): 运行参数
        """
        pass

    def _filter_data(self, x: NumericData, params: RP | None) -> NumericData:
        """
        按需筛选或重排列

        默认原样返回，子类可覆写以实现列选择、列顺序调整等逻辑。

        Args:
            x(NumericData): 输入数据
            params(RP | None): 运行参数

        Returns:
            NumericData: 筛选后的数据
        """
        return x

    def _unwrap_data(self, x: NumericData, params: RP | None) -> tuple[DataFrameMeta | None, np.ndarray]:
        """
        将输入解包为元信息和 ndarray

        DataFrame 输入时创建 ``DataFrameMeta`` 快照并提取 ndarray 值；
        ndarray 输入时直接返回 (None, ndarray)。

        Args:
            x(NumericData): 输入数据
            params(RP | None): 运行参数

        Returns:
            tuple[DataFrameMeta | None, np.ndarray]: (元信息快照, ndarray 数据)，
                DataFrame 输入时 meta 非空，ndarray 输入时 meta 为 None

        Raises:
            TypeError: 输入类型不合法时（理论上不会触发，因为上游已校验）
        """
        if isinstance(x, pd.DataFrame):
            meta = DataFrameMeta.from_dataframe(x)
            return meta, x.to_numpy()
        elif isinstance(x, np.ndarray):
            return None, x
        else:
            raise TypeError(f"数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(x)}")

    def _adjust_data(self, x: np.ndarray, params: RP | None) -> np.ndarray:
        """
        按需调整 ndarray 数据

        默认原样返回，子类可覆写以实现标准化、归一化等预处理。

        Args:
            x(np.ndarray): 输入 ndarray
            params(RP | None): 运行参数

        Returns:
            np.ndarray: 调整后的 ndarray
        """
        return x

    def _wrap_data(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: RP | None) -> NumericData:
        """
        将计算结果打包为输出

        根据 meta 是否存在决定输出类型：
        - meta 存在（原始输入为 DataFrame）→ 重建 DataFrame（保留索引、自定义列名）
        - meta 为 None（原始输入为 ndarray）→ 直接返回 ndarray

        Args:
            output_data(np.ndarray): 计算结果 ndarray
            meta(DataFrameMeta | None): 输入阶段的元信息快照
            params(RP | None): 运行参数

        Returns:
            NumericData: 打包后的输出（DataFrame 或 ndarray）
        """
        if isinstance(meta, DataFrameMeta):
            output_index = self._adjust_index(output_data, meta, params)
            output_columns = self._name_output_columns(output_data, meta, params)
            return pd.DataFrame(output_data, columns=output_columns, index=output_index)
        else:
            return output_data

    def _adjust_index(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: RP | None) -> pd.Index:
        """
        确定输出 DataFrame 的行索引

        默认沿用输入 DataFrame 的原始索引，子类可覆写以自定义行索引（如截断、扩展等）。

        Args:
            output_data(np.ndarray): 输出 ndarray
            meta(DataFrameMeta | None): 输入元信息快照
            params(RP): 运行参数

        Returns:
            pd.Index: 输出 DataFrame 的行索引
        """
        return meta.index

    def _validate_and_wrap_output(
            self,
            output_data: np.ndarray | tuple[np.ndarray, BaseModel],
            meta: DataFrameMeta | None,
            params: RP | None,
    ) -> NumericData | tuple[NumericData, BaseModel]:
        """校验 ``_run_data`` 的输出并打包为 NumericData。

        根据 ``_eo_type`` 的值执行不同校验策略：

            - ``_eo_type is None``：输出必须是纯 ``np.ndarray``，打包后返回 ``NumericData``
            - ``_eo_type 非 None``：输出必须是 ``tuple[np.ndarray, EO]``，
              校验 EO 实例类型后打包返回 ``tuple[NumericData, EO]``

        Args:
            output_data: ``_run_data`` 的原始返回值
            meta: 输入元信息快照，用于回包构建 DataFrame
            params: 运行参数

        Returns:
            NumericData | tuple[NumericData, EO]: 打包后的输出

        Raises:
            TypeError: 输出不符合 EO 约束时
        """
        if self._eo_type is None:
            if not isinstance(output_data, np.ndarray):
                raise TypeError(
                    f"{type(self).__name__} 的 EO 为 None，"
                    f"_run_data 必须返回 np.ndarray，但当前是 {type(output_data)}"
                )
            return self._wrap_data(output_data, meta, params)
        else:
            if not (isinstance(output_data, tuple) and len(output_data) == 2):
                raise TypeError(
                    f"{type(self).__name__} 的 EO 为 {self._eo_type.__name__}，"
                    f"_run_data 必须返回 tuple[np.ndarray, EO]，但当前是 {type(output_data)}"
                )
            if not isinstance(output_data[1], self._eo_type):
                raise TypeError(
                    f"附加输出类型必须是 {self._eo_type.__name__}，"
                    f"但当前是 {type(output_data[1]).__name__}"
                )
            y = self._wrap_data(output_data[0], meta, params)
            return y, output_data[1]

    @classmethod
    def has_extra_output(cls) -> bool:
        """是否包含附加输出（EO 非 None 时返回 True）

        调用方可在实例化前预判 ``run()`` 的返回值形态：
            - ``False``: ``run()`` 返回 ``NumericData``
            - ``True``: ``run()`` 返回 ``tuple[NumericData, EO]``

        Returns:
            bool: 是否包含附加输出
        """
        return cls._eo_type is not None


class NumericOperator(_NumericOperatorMixin,
                      BaseOperator[NumericData, NumericData | tuple[NumericData, EO], C, RP],
                      Generic[EO, C, RP],
                      metaclass=ABCMeta):
    """数值算子基类

    支持 DataFrame 和 ndarray 双类型输入输出的算子基类。
    通过 ``_NumericOperatorMixin`` 提供统一的 IO 处理管线（验证 → 筛选 → 解包 →
    调整 → 计算 → 校验打包），子类仅需实现 ``_run_data`` 和 ``_name_output_columns``。

    当 EO（附加输出类型）非 None 时，``run()`` 返回 ``tuple[NumericData, EO]``；
    EO 为 None 时返回 ``NumericData``。可通过 ``has_extra_output()`` 预判返回形态。

    泛型参数:
        EO: 附加输出类型，bound 为 Union[BaseModel, None]，
            非 None 时 ``_run_data`` 需返回 ``tuple[np.ndarray, EO]``
        C: 实例参数类型，bound 为 Union[BaseModel, None]
        RP: 运行参数类型，bound 为 Union[BaseModel, None]
    """

    _eo_type: ClassVar[type[BaseModel] | None] = None
    """附加输出类型，由 __init_subclass__ 从 EO 泛型参数中自动提取"""

    def __init_subclass__(cls, **kwargs):
        """子类定义时的钩子，提取 EO 泛型参数对应的附加输出类型"""
        super().__init_subclass__(**kwargs)
        if cls._eo_type is None:
            cls._eo_type = cls._extract_type_from_typevar(cls, NumericOperator, EO)

    def _run(self, x: NumericData, *, params: RP | None) -> NumericData | tuple[NumericData, EO]:
        """
        数值算子模板方法管线

        按 ``_validate_input → _filter_data → _unwrap_data → _adjust_data
        → _run_data → _validate_and_wrap_output`` 的顺序编排，
        子类可通过覆写各步骤自定义行为。

        Args:
            x(NumericData): 输入数据（DataFrame 或 ndarray）
            params(RP | None): 验证后的运行参数

        Returns:
            NumericData | tuple[NumericData, EO]: 处理后的输出（与输入同类型），
                当 EO 非 None 时额外返回附加输出
        """
        # 步骤1：验证输入数据类型和合法性
        self._validate_input(x, params)
        # 步骤2：按需进行列名筛选和顺序调整
        data = self._filter_data(x, params)
        # 步骤3：按输入类型解包为 (元信息, ndarray)，DataFrame 输入时记录元信息用于后续回包
        meta, data = self._unwrap_data(data, params)
        # 步骤4：按需调整 ndarray 内容（如标准化、归一化等预处理）
        data = self._adjust_data(data, params)
        # 步骤5：执行子类实现的核心计算逻辑
        output_data = self._run_data(data, params)
        # 步骤6：根据 _eo_type 校验输出并打包
        return self._validate_and_wrap_output(output_data, meta, params)

    @abstractmethod
    def _run_data(self, x: np.ndarray, params: RP | None) -> np.ndarray | tuple[np.ndarray, EO]:
        """
        子类实现的核心计算逻辑

        接收预处理后的 ndarray，执行实际计算并返回结果 ndarray。

        Args:
            x(np.ndarray): 预处理后的输入数据
            params(RP | None): 运行参数，可能为 None

        Returns:
            np.ndarray: 计算结果
        """
        ...

    @abstractmethod
    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: RP | None) -> list[str]:
        """
        子类实现：确定输出 DataFrame 的列名

        当原始输入为 DataFrame 时，由子类决定输出列名（如特征名、分数名等）。

        Args:
            output_data(np.ndarray): 输出 ndarray
            meta(DataFrameMeta | None): 输入元信息快照，可用于获取原始列名
            params(RP): 运行参数

        Returns:
            list[str]: 输出列名列表
        """
        ...


class BiNumericOperator(_NumericOperatorMixin,
                        BaseOperator[tuple[NumericData, NumericData], NumericData | tuple[NumericData, EO], C, RP],
                        Generic[EO, C, RP],
                        metaclass=ABCMeta):
    """
    比较器基类 — 无状态纯函数

    度量 (y_pred, y_real) 之间的关系，输出任意方向的度量值。
    比较器不需要训练，是纯函数式的算子。

    特殊处理:
        - 输入为 ``tuple[x_real, x_pred]``
        - 以 x_pred 的 DataFrameMeta 作为输出参考（输出行数与 x_pred 一致）

    泛型参数:
        - EO: 附加输出类型
        - C: 实例参数类型
        - RP: 运行参数类型
    """

    _eo_type: ClassVar[type[BaseModel] | None] = None
    """附加输出类型，由 __init_subclass__ 从 EO 泛型参数中自动提取"""

    def __init_subclass__(cls, **kwargs):
        """子类定义时的钩子，提取 EO 泛型参数对应的附加输出类型"""
        super().__init_subclass__(**kwargs)
        if cls._eo_type is None:
            cls._eo_type = cls._extract_type_from_typevar(cls, BiNumericOperator, EO)

    def _run(self, x: tuple[NumericData, NumericData], *, params: RP | None) -> NumericData | tuple[NumericData, EO]:
        """双输入数值算子模板方法管线

        按 ``validate → filter → unwrap → adjust → run_data → validate_and_wrap``
        的顺序编排，对 x_real 和 x_pred 分别执行前四步预处理，
        然后调用子类的 ``_run_data`` 执行核心计算。

        输出以 x_pred 的 DataFrameMeta 为参考（行数与 x_pred 一致）。

        Args:
            x(tuple[NumericData, NumericData]): 输入元组 (x_real, x_pred)
            params(RP | None): 验证后的运行参数

        Returns:
            NumericData | tuple[NumericData, EO]: 处理后的输出（与 x_pred 同类型），
                当 EO 非 None 时额外返回附加输出
        """
        x_real, x_pred = x

        # 步骤1：验证输入数据类型和合法性
        self._validate_input(x_real, params)
        self._validate_input(x_pred, params)
        # 步骤2：按需进行列名筛选和顺序调整
        x_real = self._filter_data(x_real, params)
        x_pred = self._filter_data(x_pred, params)
        # 步骤3：按输入类型解包为 (元信息, ndarray)，DataFrame 输入时记录元信息用于后续回包
        real_meta, x_pred_meta = self._unwrap_data(x_real, params), self._unwrap_data(x_pred, params)
        real_meta, x_real = real_meta
        pred_meta, x_pred = x_pred_meta
        # 步骤4：按需调整 ndarray 内容（如标准化、归一化等预处理）
        x_real = self._adjust_data(x_real, params)
        x_pred = self._adjust_data(x_pred, params)
        # 步骤5：执行子类实现的核心计算逻辑
        output_data = self._run_data(x_real, x_pred, params)
        # 步骤6：根据 _eo_type 校验输出并打包
        return self._validate_and_wrap_output(output_data, pred_meta, params)

    @abstractmethod
    def _run_data(self, x_real: np.ndarray, x_pred: np.ndarray, params: RP | None) -> (np.ndarray |
                                                                                       tuple[np.ndarray, EO]):
        """子类实现的核心计算逻辑（双输入）

        接收预处理后的 x_real 和 x_pred ndarray，执行实际比较计算并返回结果。
        当 EO 非 None 时需返回 ``tuple[np.ndarray, EO]``，否则返回 ``np.ndarray``。

        Args:
            x_real(np.ndarray): 预处理后的真实值数据
            x_pred(np.ndarray): 预处理后的预测值数据
            params(RP | None): 运行参数，可能为 None

        Returns:
            np.ndarray | tuple[np.ndarray, EO]: 计算结果，
                EO 为 None 时返回 ndarray，否则返回 (ndarray, EO) 元组
        """
        ...

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: RP | None) -> list[str]:
        """确定输出 DataFrame 的列名

        默认沿用 x_pred 输入 DataFrame 的列名。子类可覆写以自定义列名策略。

        Args:
            output_data(np.ndarray): 输出 ndarray
            meta(DataFrameMeta | None): x_pred 输入元信息快照
            params(RP | None): 运行参数

        Returns:
            list[str]: 输出列名列表
        """
        return meta.column_names


class SupervisedNumericOperatorMixin(LearnableOperatorMixin[NumericData, NumericData, FP], Generic[FP],
                                     metaclass=ABCMeta):
    """有监督数值算子混入

    为 NumericOperator 的子类提供有监督训练能力。训练时同时接收输入数据 x 和
    标签数据 y，执行验证 → 筛选 → 解包 → 核心训练的模板管线。

    与 ``UnsupervisedNumericOperatorMixin`` 的区别在于训练时需要标签数据 y，
    ``_fit_data`` 接收 (x, y) 两个 ndarray 参数。

    泛型参数:
        FP: 训练参数类型，bound 为 Union[BaseModel, None]
    """

    def _fit(self, x: NumericData, y: NumericData, *, params: FP | None) -> None:
        """有监督训练模板方法管线

        按 ``_validate_fit_input → _filter_fit_data → _unwrap_fit_data
        → _fit_data`` 的顺序编排，子类可通过覆写各步骤自定义行为。

        Args:
            x(NumericData): 输入训练数据
            y(NumericData): 标签数据
            params(FP | None): 验证后的训练参数
        """
        # 步骤1：验证输入数据类型和合法性
        self._validate_fit_input(x, y, params=params)
        # 步骤2：按需进行列名筛选和顺序调整
        x_data, y_data = self._filter_fit_data(x, y, params=params)
        # 步骤3：按输入类型解包为 (元信息, ndarray)，DataFrame 输入时记录元信息用于后续回包
        x_meta, x_data, y_meta, y_data = self._unwrap_fit_data(x_data, y_data, params=params)
        # 步骤4：执行核心训练逻辑
        self._fit_data(x_data, y_data, params=params)
        # 步骤5：标记算子已完成训练
        self._fitted = True
        return

    def _validate_fit_input(self, x: NumericData, y: NumericData, params: FP | None) -> None:
        """验证训练输入数据类型

        检查 x 和 y 是否为合法的数值类型（DataFrame 或数值 ndarray），
        非法类型直接抛出 TypeError。

        Args:
            x(NumericData): 输入训练数据
            y(NumericData): 标签数据
            params(FP | None): 训练参数

        Raises:
            TypeError: 输入数据类型不合法时
        """
        # 验证 x
        if not (isinstance(x, pd.DataFrame) or (
                isinstance(x, np.ndarray) and issubclass(x.dtype.type, (np.integer, np.floating)))):
            raise TypeError(f"输入数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(x)}")
        # 验证 y
        if not (isinstance(y, pd.DataFrame) or (
                isinstance(y, np.ndarray) and issubclass(y.dtype.type, (np.integer, np.floating)))):
            raise TypeError(f"输入数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(y)}")

    def _unwrap_fit_data(self, x: NumericData, y: NumericData, params: FP | None) -> tuple[
        DataFrameMeta | None, np.ndarray, DataFrameMeta | None, np.ndarray]:
        """将训练输入数据解包为元信息和 ndarray

        分别对 x 和 y 进行解包：DataFrame 输入时创建 ``DataFrameMeta`` 快照并
        提取 ndarray 值；ndarray 输入时直接返回 (None, ndarray)。

        Args:
            x(NumericData): 输入训练数据
            y(NumericData): 标签数据
            params(FP | None): 训练参数

        Returns:
            tuple[DataFrameMeta | None, np.ndarray, DataFrameMeta | None, np.ndarray]:
                (x元信息, x的ndarray, y元信息, y的ndarray)

        Raises:
            TypeError: 输入类型不合法时
        """
        # 处理 x
        if isinstance(x, pd.DataFrame):
            x_meta = DataFrameMeta.from_dataframe(x)
            x_data = x.to_numpy()
        elif isinstance(x, np.ndarray):
            x_meta, x_data = None, x
        else:
            raise TypeError(f"数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(x)}")
        # 处理 y
        if isinstance(y, pd.DataFrame):
            y_meta = DataFrameMeta.from_dataframe(y)
            y_data = y.to_numpy()
        elif isinstance(y, np.ndarray):
            y_meta, y_data = None, y
        else:
            raise TypeError(f"数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(y)}")
        # 打包返回
        return x_meta, x_data, y_meta, y_data

    @abstractmethod
    def _fit_data(self, x: np.ndarray, y: np.ndarray, params: FP | None) -> None:
        """子类实现的核心训练逻辑（有监督）

        接收预处理后的 ndarray 数据 x 和标签 y，执行实际的模型训练。
        子类**无需**在此方法中设置 ``self._fitted = True``，模板方法会自动处理。

        Args:
            x(np.ndarray): 预处理后的输入训练数据
            y(np.ndarray): 预处理后的标签数据
            params(FP | None): 训练参数，可能为 None
        """
        ...


class UnsupervisedNumericOperatorMixin(LearnableOperatorMixin[NumericData, None, FP], Generic[FP],
                                       metaclass=ABCMeta):
    """无监督数值算子混入

    为 NumericOperator 的子类提供无监督训练能力。训练时仅接收输入数据 x，
    不需要标签数据 y，执行验证 → 筛选 → 解包 → 核心训练的模板管线。

    与 ``SupervisedNumericOperatorMixin`` 的区别在于训练时不需要标签，
    ``_fit_data`` 仅接收单个 ndarray 参数。

    泛型参数:
        FP: 训练参数类型，bound 为 Union[BaseModel, None]
    """

    def fit(self, x: NumericData, y: None = None, *, params: FP | None = None, **kwargs) -> Self:
        """无监督训练（适配封装，将标签参数固定为 None）

        Args:
            x(NumericData): 输入训练数据
            y(None): 标签数据（恒为 None，无监督训练不使用）
            params(FP | None): 训练参数
            **kwargs: 训练参数键值对

        Returns:
            Self: 训练后的算子实例
        """
        return super().fit(x, None, params=params, **kwargs)

    def _fit(self, x: NumericData, y: None = None, *, params: FP | None) -> None:
        """无监督训练模板方法管线

        按 ``_validate_fit_input → _filter_fit_data → _unwrap_fit_data
        → _fit_data`` 的顺序编排，子类可通过覆写各步骤自定义行为。

        Args:
            x(NumericData): 输入训练数据
            y(None): 标签数据（恒为 None）
            params(FP | None): 验证后的训练参数
        """
        # 步骤1：验证输入数据类型和合法性
        self._validate_fit_input(x, params=params)
        # 步骤2：按需进行列名筛选和顺序调整
        x_data = self._filter_fit_data(x, params=params)
        # 步骤3：按输入类型解包为 (元信息, ndarray)，DataFrame 输入时记录元信息用于后续回包
        x_meta, x_data = self._unwrap_fit_data(x_data, params=params)
        # 步骤4：执行核心训练逻辑
        self._fit_data(x_data, params=params)
        # 步骤5：标记算子已完成训练
        self._fitted = True
        return

    def _validate_fit_input(self, x: NumericData, params: FP | None) -> None:
        """验证训练输入数据类型

        检查 x 是否为合法的数值类型（DataFrame 或数值 ndarray），
        非法类型直接抛出 TypeError。

        Args:
            x(NumericData): 输入训练数据
            params(FP | None): 训练参数

        Raises:
            TypeError: 输入数据类型不合法时
        """
        if not (isinstance(x, pd.DataFrame) or (
                isinstance(x, np.ndarray) and issubclass(x.dtype.type, (np.integer, np.floating)))):
            raise TypeError(f"输入数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(x)}")

    def _filter_fit_data(self, x: NumericData, params: FP | None) -> NumericData:
        """按需筛选训练数据

        默认原样返回，子类可覆写以实现列选择、列顺序调整等逻辑。

        Args:
            x(NumericData): 输入数据
            params(FP | None): 训练参数

        Returns:
            NumericData: 筛选后的数据
        """
        return x

    def _unwrap_fit_data(self, x: NumericData, params: FP | None) -> tuple[DataFrameMeta | None, np.ndarray]:
        """将训练输入数据解包为元信息和 ndarray

        DataFrame 输入时创建 ``DataFrameMeta`` 快照并提取 ndarray 值；
        ndarray 输入时直接返回 (None, ndarray)。

        Args:
            x(NumericData): 输入数据
            params(FP | None): 训练参数

        Returns:
            tuple[DataFrameMeta | None, np.ndarray]: (元信息快照, ndarray 数据)

        Raises:
            TypeError: 输入类型不合法时
        """
        if isinstance(x, pd.DataFrame):
            x_meta = DataFrameMeta.from_dataframe(x)
            x_data = x.to_numpy()
        elif isinstance(x, np.ndarray):
            x_meta, x_data = None, x
        else:
            raise TypeError(f"数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(x)}")
        return x_meta, x_data

    @abstractmethod
    def _fit_data(self, x: np.ndarray, params: FP | None) -> None:
        """子类实现的核心训练逻辑（无监督）

        接收预处理后的 ndarray 数据，执行实际的模型训练。
        子类**无需**在此方法中设置 ``self._fitted = True``，模板方法会自动处理。

        Args:
            x(np.ndarray): 预处理后的输入训练数据
            params(FP | None): 训练参数，可能为 None
        """
        ...
