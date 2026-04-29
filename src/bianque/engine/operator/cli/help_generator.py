# -*- coding: utf-8 -*-

"""
算子帮助文档自动生成模块

从算子类的元信息（docstring、Config 类的 Pydantic Field 定义、Enum/Literal 类型注解等）
中自动提取并生成结构化 Markdown 格式的帮助文档。

核心能力:
    - 列表模式: 输出所有算子的名称和一行简介
    - 详情模式: 输出单个算子的完整参数表格、输入输出说明
    - 参数信息直接从 ``model_fields`` 中提取，与 HPO 模块共享同一数据源

提取来源:
    - ``cls.name()`` → 算子名称
    - ``cls.__doc__`` → 功能描述
    - ``cls._config_type.model_fields`` → 实例参数表格
    - ``field_info.annotation`` 中 Enum/Literal → 候选值
    - ``field_info.metadata`` 中 Ge/Le/Gt/Lt → 值域范围
    - ``field_info.description`` → 参数说明

输出格式面向 Agent Skill 友好，采用结构化 Markdown。

使用示例::

    from bianque.engine.operator.cli.help_generator import generate_list, generate_detail

    # 列表模式
    print(generate_list({"knn_scorer": KNNScorer, "zscore_detector": ZScoreDetector}))

    # 详情模式
    print(generate_detail(KNNScorer))
"""

import enum
import inspect
from typing import Literal, get_args, get_origin

from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

__all__ = [
    'generate_list',
    'generate_detail',
    'generate_config_params_table',
]


def generate_list(operators: dict[str, type]) -> str:
    """
    生成所有算子的列表概览（列表模式）

    输出格式为 Markdown 表格，包含算子名称和一行简介。
    简介取自类 docstring 的第一个非空行。

    Args:
        operators (dict[str, type]): {算子名称: 算子类} 映射，通常来自
            ``OperatorRegistry.list_all()``

    Returns:
        str: Markdown 格式的算子列表
    """
    lines = []
    lines.append("## 可用算子列表")
    lines.append("")
    lines.append("| 名称 | 简介 |")
    lines.append("|------|------|")

    for name, cls in sorted(operators.items()):
        summary = _extract_summary(cls)
        lines.append(f"| `{name}` | {summary} |")

    lines.append("")
    lines.append(f"共 {len(operators)} 个算子。使用 `help <算子名称>` 查看详细信息。")
    return '\n'.join(lines)


def generate_detail(cls: type) -> str:
    """
    生成单个算子的详细帮助文档（详情模式）

    包含算子名称、功能描述、实例参数表格、运行参数表格、
    附加输出结构说明等。

    Args:
        cls (type): 算子类

    Returns:
        str: Markdown 格式的详细帮助文档
    """
    lines = []

    # ---- 标题和名称 ----
    op_name = cls.name() if hasattr(cls, 'name') and callable(cls.name) else cls.__name__
    lines.append(f"## {op_name}")
    lines.append("")

    # ---- 功能描述 ----
    description = _extract_description(cls)
    if description:
        lines.append(description)
        lines.append("")

    # ---- 算子类型标签 ----
    tags = _extract_type_tags(cls)
    if tags:
        lines.append(f"**类型**: {', '.join(tags)}")
        lines.append("")

    # ---- 实例参数 (Config) ----
    config_type = getattr(cls, '_config_type', None)
    if config_type is not None and config_type is not type(None):
        lines.append(f"### 实例参数 ({config_type.__name__})")
        lines.append("")
        table = generate_config_params_table(config_type)
        lines.append(table)
        lines.append("")

    # ---- 运行参数 (RunParams) ----
    rp_type = getattr(cls, '_run_params_type', None)
    if rp_type is not None and rp_type is not type(None):
        lines.append(f"### 运行参数 ({rp_type.__name__})")
        lines.append("")
        table = generate_config_params_table(rp_type)
        lines.append(table)
        lines.append("")

    # ---- 附加输出 (ExtraOutput) ----
    eo_type = getattr(cls, '_eo_type', None)
    if eo_type is not None and eo_type is not type(None):
        lines.append(f"### 附加输出 ({eo_type.__name__})")
        lines.append("")
        eo_desc = _extract_model_fields_description(eo_type)
        if eo_desc:
            lines.append(eo_desc)
            lines.append("")

    # ---- 训练参数 (FitParams) ----
    fp_type = getattr(cls, '_fit_params_type', None)
    if fp_type is not None and fp_type is not type(None):
        lines.append(f"### 训练参数 ({fp_type.__name__})")
        lines.append("")
        table = generate_config_params_table(fp_type)
        lines.append(table)
        lines.append("")

    return '\n'.join(lines)


def generate_config_params_table(config_cls: type[BaseModel]) -> str:
    """
    从 Pydantic Config 类生成参数表格

    遍历 ``model_fields``，从 field_info 中提取类型、默认值、
    值域/候选值、说明等信息，格式化为 Markdown 表格。

    Args:
        config_cls (type[BaseModel]): Pydantic Config/Params 类

    Returns:
        str: Markdown 格式的参数表格
    """
    lines = []
    lines.append("| 参数名 | 类型 | 默认值 | 值域/候选 | 说明 |")
    lines.append("|--------|------|--------|-----------|------|")

    for field_name, field_info in config_cls.model_fields.items():
        # ---- 类型 ----
        type_str = _format_type(field_info.annotation)

        # ---- 默认值 ----
        default = field_info.default
        if default is PydanticUndefined or default is ...:
            default_str = "**必填**"
        elif default is None:
            default_str = "`None`"
        elif isinstance(default, enum.Enum):
            default_str = f"`{default.value}`"
        else:
            default_str = f"`{default}`"

        # ---- 值域/候选值 ----
        constraints = _extract_constraints(field_info)

        # ---- 说明 ----
        desc = field_info.description or ""

        lines.append(
            f"| `{field_name}` | {type_str} | {default_str} | {constraints} | {desc} |"
        )

    return '\n'.join(lines)


# ============================================================================
# 内部辅助函数
# ============================================================================

def _extract_summary(cls: type) -> str:
    """
    从类 docstring 中提取一行简介

    取 docstring 的第一个非空行，去除首尾空白。
    如果没有 docstring，返回 "(无描述)"。

    Args:
        cls (type): 目标类

    Returns:
        str: 一行简介文本
    """
    doc = cls.__doc__
    if not doc:
        return "(无描述)"

    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped

    return "(无描述)"


def _extract_description(cls: type) -> str:
    """
    从类 docstring 中提取完整功能描述

    提取 docstring 从开头到第一个空行之间的所有文本行，
    或者到 ``Attributes:`` / ``Args:`` 等 section 标记之前。

    Args:
        cls (type): 目标类

    Returns:
        str: 多行功能描述文本
    """
    doc = cls.__doc__
    if not doc:
        return ""

    lines = doc.strip().splitlines()
    desc_lines = []

    # 段落 section 标记
    section_markers = {'Attributes:', 'Args:', 'Returns:', 'Raises:', 'Note:', 'Notes:',
                       '示例', '泛型参数:', '校验规则:', '数据流:', '核心', '训练阶段:',
                       '推理阶段:', '输出:', '注意:'}

    for line in lines:
        stripped = line.strip()

        # 遇到 section 标记停止
        if any(stripped.startswith(marker) for marker in section_markers):
            break

        # 遇到空行时，如果已有内容则停止（只取第一段）
        if not stripped and desc_lines:
            break

        if stripped:
            desc_lines.append(stripped)

    return ' '.join(desc_lines)


def _extract_type_tags(cls: type) -> list[str]:
    """
    从类的 MRO 中提取类型标签

    检查常见的 Mixin 类型，生成人类可读的标签列表。

    Args:
        cls (type): 目标类

    Returns:
        list[str]: 类型标签列表，如 ``["Scorer", "可训练"]``
    """
    tags = []

    # 延迟导入避免循环依赖
    try:
        from bianque.engine.operator.detection.base import (
            BaseScorerMixin, BaseDeciderMixin, BaseDetector,
        )
        if issubclass(cls, BaseDetector):
            tags.append("Detector")
        elif issubclass(cls, BaseScorerMixin):
            tags.append("Scorer")
        elif issubclass(cls, BaseDeciderMixin):
            tags.append("Decider")
    except ImportError:
        pass

    try:
        from bianque.engine.operator.base import LearnableOperatorMixin
        if issubclass(cls, LearnableOperatorMixin):
            tags.append("可训练")
    except ImportError:
        pass

    try:
        from bianque.engine.operator.feature.construction.base import BaseFeatureMixin
        if issubclass(cls, BaseFeatureMixin):
            tags.append("特征算子")
    except ImportError:
        pass

    try:
        from bianque.engine.operator.evaluation.base import BaseMetricOperator
        if issubclass(cls, BaseMetricOperator):
            tags.append("评价指标")
    except ImportError:
        pass

    return tags


def _format_type(annotation) -> str:
    """
    格式化类型注解为人类可读字符串

    对 Enum 展示成员值列表，对 Literal 展示字面量。

    Args:
        annotation: 类型注解对象

    Returns:
        str: 格式化后的类型字符串
    """
    if annotation is None:
        return "Any"

    # Enum 类型
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        values = [str(e.value) for e in annotation]
        return f"enum({', '.join(values)})"

    # Literal 类型
    if get_origin(annotation) is Literal:
        args = get_args(annotation)
        values = [str(a) for a in args]
        return f"literal({', '.join(values)})"

    # 基本类型
    if annotation is int:
        return "int"
    elif annotation is float:
        return "float"
    elif annotation is str:
        return "str"
    elif annotation is bool:
        return "bool"

    # 其他复杂类型（list, dict, Optional 等）
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if args:
            args_str = ', '.join(_format_type(a) for a in args)
            origin_name = getattr(origin, '__name__', str(origin))
            return f"{origin_name}[{args_str}]"
        return str(origin)

    # 直接使用类名
    if hasattr(annotation, '__name__'):
        return annotation.__name__

    return str(annotation)


def _extract_constraints(field_info) -> str:
    """
    从 field_info 中提取值域和候选值约束

    检查 metadata 中的 Ge/Le/Gt/Lt 以及 annotation 中的 Enum/Literal。

    Args:
        field_info: Pydantic FieldInfo 对象

    Returns:
        str: 约束描述字符串，如 ``[1, 20]`` 或 ``euclidean, manhattan``
    """
    parts = []

    # ---- 从 metadata 提取数值边界 ----
    low = None
    high = None
    low_inclusive = True
    high_inclusive = True

    for meta in field_info.metadata:
        if isinstance(meta, Ge):
            low = meta.ge
            low_inclusive = True
        elif isinstance(meta, Gt):
            low = meta.gt
            low_inclusive = False
        elif isinstance(meta, Le):
            high = meta.le
            high_inclusive = True
        elif isinstance(meta, Lt):
            high = meta.lt
            high_inclusive = False

    if low is not None or high is not None:
        low_bracket = "[" if low_inclusive else "("
        high_bracket = "]" if high_inclusive else ")"
        low_str = str(low) if low is not None else "-∞"
        high_str = str(high) if high is not None else "+∞"
        parts.append(f"{low_bracket}{low_str}, {high_str}{high_bracket}")

    # ---- 从 annotation 提取候选值 ----
    annotation = field_info.annotation
    if annotation is not None:
        if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            values = [str(e.value) for e in annotation]
            parts.append(', '.join(values))
        elif get_origin(annotation) is Literal:
            values = [str(a) for a in get_args(annotation)]
            parts.append(', '.join(values))

    return ' | '.join(parts) if parts else "-"


def _extract_model_fields_description(model_cls: type[BaseModel]) -> str:
    """
    从 Pydantic BaseModel 的字段中生成简要描述表格

    用于展示附加输出（ExtraOutput）等结构的字段说明。

    Args:
        model_cls (type[BaseModel]): Pydantic BaseModel 子类

    Returns:
        str: Markdown 格式的字段描述表格
    """
    lines = []
    lines.append("| 字段名 | 类型 | 说明 |")
    lines.append("|--------|------|------|")

    for field_name, field_info in model_cls.model_fields.items():
        type_str = _format_type(field_info.annotation)
        desc = field_info.description or ""
        lines.append(f"| `{field_name}` | {type_str} | {desc} |")

    return '\n'.join(lines)
