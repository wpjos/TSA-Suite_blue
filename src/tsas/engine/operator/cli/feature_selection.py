# -*- coding: utf-8 -*-

"""特征选择器 CLI。

本模块提供 ``python -m tsas.engine.operator.cli feature_selection`` 子命令，支持单个
Selector 的帮助生成、训练、运行和模型加载。
"""

import argparse
from typing import cast

import pandas as pd
from pydantic import BaseModel

from tsas.engine.operator.base import BaseOperator, LearnableOperatorMixin, NumericData
from tsas.engine.operator.cli.common import (build_help_subparser, build_list_subparser, build_show_subparser,
                                             extract_encoding_arg, handle_help, handle_list, handle_show,
                                             instantiate_operator)
from tsas.engine.operator.cli.config_loader import load_config
from tsas.engine.operator.cli.io import ensure_encoding, load_data, save_data, save_json
from tsas.engine.operator.cli.registry import OperatorRegistry
from tsas.engine.operator.feature.selection.base import SupervisedFeatureSelector

__all__ = ['create_registry', 'main']


def create_registry() -> OperatorRegistry:
    """创建特征选择器注册中心。

    Returns:
        OperatorRegistry: 已完成发现流程的注册中心。
    """
    from tsas.engine.operator.feature.selection.base import BaseFeatureSelectorMixin

    registry = OperatorRegistry(
        base_class=BaseOperator,
        scan_packages=['tsas.engine.operator.feature.selection'],
        filter_fn=lambda cls: issubclass(cls, BaseFeatureSelectorMixin),
    )
    registry.discover()
    return registry


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。

    Returns:
        argparse.ArgumentParser: ``feature_selection`` 子命令解析器。
    """
    parser = argparse.ArgumentParser(prog='feature_selection', description='特征选择器 CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    build_list_subparser(subparsers)
    build_show_subparser(subparsers)
    build_help_subparser(subparsers)

    run_parser = subparsers.add_parser('run', help='运行特征选择器')
    run_parser.add_argument('--input', '-i', required=True, help='输入 CSV/TSV 文件')
    run_parser.add_argument('--config', '-c', required=True, help='选择器配置文件')
    run_parser.add_argument('--output', '-o', required=True, help='选择后特征输出文件')
    run_parser.add_argument('--eo-output', required=True, help='附加输出 JSON 文件')
    run_parser.add_argument('--load', default=None, help='已保存选择器模型目录')
    run_parser.add_argument('--label-column', default=None,
                            help='标签列名或列索引；提供时会从输入中剔除该列后再 run')

    fit_parser = subparsers.add_parser('fit', help='训练特征选择器')
    fit_parser.add_argument('--input', '-i', required=True, help='训练输入 CSV/TSV 文件')
    fit_parser.add_argument('--config', '-c', required=True, help='选择器配置文件')
    fit_parser.add_argument('--model-dir', '-m', required=True, help='模型保存目录')
    fit_parser.add_argument('--label-column', default=None,
                            help='标签列名或列索引；有监督选择器必须提供')
    return parser


def _load_operator(config_path: str, registry: OperatorRegistry, load_path: str | None = None) -> BaseOperator:
    """从配置文件或模型目录构造特征选择器。

    配置文件采用与 detection 模块一致的嵌套 dict 格式，``operator`` 字段
    为包含 ``name`` 和 ``config`` 的字典::

        {
            "operator": {
                "name": "column_selector",
                "config": {"input_columns": ["a", "c"]}
            }
        }

    Args:
        config_path (str): 配置文件路径。
        registry (OperatorRegistry): 特征选择器注册中心。
        load_path (str | None): 已保存模型目录。指定时从模型目录
            加载已训练的选择器，忽略配置中的实例参数。

    Returns:
        BaseOperator: 选择器实例。

    Raises:
        ValueError: 配置文件中缺少 ``operator`` 字段时。
    """
    config = load_config(config_path)
    # 提取 operator 嵌套字典（与 detection 模块格式一致）
    op_config = config.get('operator', {})
    if not op_config:
        raise ValueError("配置文件中缺少 'operator' 字段")

    if load_path:
        # 从配置中获取算子类名，通过注册中心查找类后调用 load
        op_name = op_config.get('name')
        if not op_name:
            raise ValueError("配置文件中 operator 字段缺少 'name' 子字段")
        op_cls = registry.get(op_name)
        return cast(type[BaseOperator], op_cls).load(load_path)

    # 委托公共函数完成核心实例化（查找类 → 构造 config → 创建实例）
    return instantiate_operator(op_config, registry)


def _resolve_label_column(label_arg: str | None, df: pd.DataFrame) -> str | int | None:
    """解析命令行传入的标签列参数。

    如果是字符串且匹配列名，则返回列名；否则尝试解析为整数列索引。

    Args:
        label_arg: 命令行传入的标签列名或索引字符串。
        df: 输入 DataFrame。

    Returns:
        str | int | None: 标签列名、列索引，或未指定时返回 None。

    Raises:
        ValueError: 列名不存在或索引越界时抛出。
    """
    if label_arg is None:
        return None
    if label_arg in df.columns:
        return label_arg
    try:
        idx = int(label_arg)
        if not (0 <= idx < df.shape[1]):
            raise ValueError(f"标签列索引 {label_arg} 越界")
        return idx
    except ValueError as exc:
        raise ValueError(f"输入数据缺少标签列 '{label_arg}'") from exc


def _split_data_by_label(df: pd.DataFrame, label_col: str | int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从输入 DataFrame 中分离特征和标签。

    Args:
        df: 包含标签列的输入数据。
        label_col: 标签列名或列索引。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (特征 DataFrame, 标签 DataFrame)。
    """
    if isinstance(label_col, str):
        y = df[[label_col]]
        x = df.drop(columns=[label_col])
    else:
        y = df.iloc[:, [label_col]]
        col_name = df.columns[label_col]
        x = df.drop(columns=[col_name])
    return x, y


def _handle_fit(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """处理选择器训练。

    Args:
        registry (OperatorRegistry): 特征选择器注册中心。
        args (argparse.Namespace): 命令行参数。

    Returns:
        None: 本函数将模型写入磁盘。

    Raises:
        TypeError: 非训练型选择器执行 ``fit`` 时抛出。
        ValueError: 有监督选择器未提供 ``--label-column`` 时抛出。
    """
    data = load_data(args.input)
    operator = _load_operator(args.config, registry)
    if not isinstance(operator, LearnableOperatorMixin):
        raise TypeError(f'{type(operator).__name__} 不支持 fit')

    label_col = _resolve_label_column(args.label_column, data)
    if label_col is not None:
        x, y = _split_data_by_label(data, label_col)
        if isinstance(operator, SupervisedFeatureSelector):
            operator.fit(x, y)
        else:
            operator.fit(x, None)
    else:
        if isinstance(operator, SupervisedFeatureSelector):
            raise ValueError(f'{type(operator).__name__} 是有监督选择器，必须提供 --label-column')
        operator.fit(data, None)

    operator.save(args.model_dir)


def _handle_run(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """处理选择器运行。

    Args:
        registry (OperatorRegistry): 特征选择器注册中心。
        args (argparse.Namespace): 命令行参数。

    Returns:
        None: 本函数写出主输出和 EO JSON。
    """
    data = load_data(args.input)
    operator = _load_operator(args.config, registry, load_path=args.load)

    label_col = _resolve_label_column(args.label_column, data)
    if label_col is not None:
        x, _ = _split_data_by_label(data, label_col)
    else:
        x = data

    output, eo = cast(tuple[NumericData, BaseModel], operator.run(x))
    save_data(cast(pd.DataFrame, output), args.output)
    save_json(eo.model_dump(mode='json'), args.eo_output)


def main(args: list[str] | None = None) -> None:
    """特征选择器 CLI 入口。

    Args:
        args (list[str] | None): 命令行参数列表。

    Returns:
        None: 本函数通过子命令完成实际操作。
    """
    encoding, remaining = extract_encoding_arg(args)
    ensure_encoding(encoding)
    parsed = _build_parser().parse_args(remaining)
    registry = create_registry()
    if parsed.command == 'list':
        handle_list(registry)
    elif parsed.command == 'show':
        handle_show(registry, parsed.operator_names)
    elif parsed.command == 'help':
        handle_help(registry, parsed.operator_names)
    elif parsed.command == 'fit':
        _handle_fit(registry, parsed)
    elif parsed.command == 'run':
        _handle_run(registry, parsed)


if __name__ == '__main__':
    main()
