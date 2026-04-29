# -*- coding: utf-8 -*-

"""
异常检测算子命令行入口

提供异常检测算子的命令行接口，支持 help、run、fit 三个子命令。
仅支持单算子模式，支持 Scorer 和 Detector（含 Composite）类型。

子命令:
    - ``help [算子名称]``: 查看所有算子列表或指定算子的详细帮助
    - ``run``: 执行异常检测
    - ``fit``: 训练检测器并可选保存模型

调用方式::

    python -m bianque.engine.operator.cli detection help
    python -m bianque.engine.operator.cli detection help knn_detector
    python -m bianque.engine.operator.cli detection fit --input train.csv --config detector.yaml --save model_dir/
    python -m bianque.engine.operator.cli detection run --input test.csv --config detector.yaml --load model_dir/ --output result.csv

配置文件示例 (YAML)::

    operator:
      name: "knn_detector"
      input_columns: ["sensor_1", "sensor_2", "sensor_3"]
      config:
        n_neighbors: 5
        percentile: 95.0
"""

import argparse
import sys

import numpy as np
import pandas as pd

from bianque.engine.operator.cli.config_loader import load_config
from bianque.engine.operator.cli.help_generator import generate_detail, generate_list
from bianque.engine.operator.cli.io import load_data, save_data
from bianque.engine.operator.cli.registry import OperatorRegistry
from bianque.engine.operator.base import BaseOperator, LearnableOperatorMixin

__all__ = [
    'main',
    'create_registry',
]


def create_registry() -> OperatorRegistry:
    """
    创建异常检测算子注册中心

    扫描 ``bianque.engine.operator.detection`` 包，
    注册所有 Scorer 和 Detector 类型的非抽象算子。

    Returns:
        OperatorRegistry: 已完成 discover 的注册中心实例
    """
    from bianque.engine.operator.detection.base import (
        BaseScorerMixin, BaseDeciderMixin, BaseDetector,
    )

    def _filter_scorer_or_detector(cls: type) -> bool:
        """只注册 Scorer 和 Detector（含 Decider）类型"""
        return issubclass(cls, (BaseScorerMixin, BaseDeciderMixin, BaseDetector))

    registry = OperatorRegistry(
        base_class=BaseOperator,
        scan_packages=['bianque.engine.operator.detection'],
        filter_fn=_filter_scorer_or_detector,
    )
    registry.discover()
    return registry


def _build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器

    Returns:
        argparse.ArgumentParser: 配置好子命令的解析器
    """
    parser = argparse.ArgumentParser(
        prog='python -m bianque.engine.operator.cli detection',
        description='异常检测算子命令行接口',
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # ---- help 子命令 ----
    help_parser = subparsers.add_parser('help', help='查看算子帮助信息')
    help_parser.add_argument(
        'operator_name', nargs='?', default=None,
        help='算子名称（不指定时列出所有可用算子）',
    )

    # ---- run 子命令 ----
    run_parser = subparsers.add_parser('run', help='执行异常检测')
    run_parser.add_argument('--input', '-i', required=True, help='输入数据文件路径')
    run_parser.add_argument('--output', '-o', required=True, help='输出数据文件路径')
    run_parser.add_argument('--config', '-c', required=True, help='算子配置文件路径')
    run_parser.add_argument('--load', default=None, help='加载已训练模型的目录路径')

    # ---- fit 子命令 ----
    fit_parser = subparsers.add_parser('fit', help='训练检测器')
    fit_parser.add_argument('--input', '-i', required=True, help='训练数据文件路径')
    fit_parser.add_argument('--config', '-c', required=True, help='算子配置文件路径')
    fit_parser.add_argument('--save', default=None, help='保存训练模型的目录路径')

    return parser


def _handle_help(registry: OperatorRegistry, operator_name: str | None) -> None:
    """
    处理 help 子命令

    Args:
        registry (OperatorRegistry): 算子注册中心
        operator_name (str | None): 算子名称，None 时列出全部
    """
    if operator_name is None:
        operators = registry.list_all()
        print(generate_list(operators))
    else:
        cls = registry.get(operator_name)
        print(generate_detail(cls))


def _instantiate_operator(
    config: dict, registry: OperatorRegistry
) -> tuple[BaseOperator, list[str] | None]:
    """
    根据配置文件实例化单个检测算子

    Args:
        config (dict): 解析后的配置字典
        registry (OperatorRegistry): 算子注册中心

    Returns:
        tuple[BaseOperator, list[str] | None]: (算子实例, 输入列名列表)
            输入列名为 None 时表示使用全部列

    Raises:
        ValueError: 配置格式不正确时
    """
    op_config = config.get('operator', {})
    if not op_config:
        raise ValueError("配置文件中缺少 'operator' 字段")

    op_name = op_config.get('name')
    if not op_name:
        raise ValueError("算子配置中缺少 'name' 字段")

    # 查找算子类
    op_cls = registry.get(op_name)

    # 实例化
    cls_config = op_config.get('config', {})
    if op_cls._config_type and cls_config:
        config_instance = op_cls._config_type(**cls_config)
        op_instance = op_cls(config=config_instance)
    else:
        op_instance = op_cls()

    # 输入列
    input_columns = op_config.get('input_columns', None)

    return op_instance, input_columns


def _select_columns(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    """
    根据列名列表选择 DataFrame 的子集

    Args:
        df (pd.DataFrame): 原始数据
        columns (list[str] | None): 列名列表，None 时返回全部列

    Returns:
        pd.DataFrame: 选择后的数据
    """
    if columns:
        return df[columns]
    return df


def _handle_run(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """
    处理 run 子命令

    加载数据和配置，实例化算子，执行检测并保存结果。

    Args:
        registry (OperatorRegistry): 算子注册中心
        args (argparse.Namespace): 解析后的命令行参数
    """
    # 加载数据和配置
    df = load_data(args.input)
    config = load_config(args.config)

    # 实例化算子
    op_instance, input_columns = _instantiate_operator(config, registry)

    # 加载已训练模型
    if args.load:
        from pathlib import Path
        op_instance = type(op_instance).load(Path(args.load))

    # 选择输入列并执行
    op_input = _select_columns(df, input_columns)
    output = op_instance.run(op_input)

    # 处理输出
    if isinstance(output, tuple):
        main_output = output[0]
    else:
        main_output = output

    # 转换为 DataFrame
    if isinstance(main_output, np.ndarray):
        if main_output.ndim == 1:
            main_output = pd.DataFrame({'result': main_output})
        else:
            main_output = pd.DataFrame(main_output)
    elif not isinstance(main_output, pd.DataFrame):
        main_output = pd.DataFrame({'result': [main_output]})

    # 合并原始数据和检测结果
    result = pd.concat([df, main_output], axis=1)

    # 保存
    save_data(result, args.output)
    print(f"异常检测完成，结果已保存至: {args.output}")


def _handle_fit(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """
    处理 fit 子命令

    加载数据和配置，实例化算子，执行训练。

    Args:
        registry (OperatorRegistry): 算子注册中心
        args (argparse.Namespace): 解析后的命令行参数
    """
    # 加载数据和配置
    df = load_data(args.input)
    config = load_config(args.config)

    # 实例化算子
    op_instance, input_columns = _instantiate_operator(config, registry)

    # 训练
    if not isinstance(op_instance, LearnableOperatorMixin):
        print(f"算子 '{op_instance.name()}' 不需要训练")
        return

    op_input = _select_columns(df, input_columns)
    op_instance.fit(op_input)
    print(f"算子 '{op_instance.name()}' 训练完成")

    # 保存模型
    if args.save:
        from pathlib import Path
        save_path = Path(args.save)
        op_instance.save(save_path)
        print(f"模型已保存至: {args.save}")


def main(args: list[str] | None = None) -> None:
    """
    异常检测 CLI 主函数

    Args:
        args (list[str] | None): 命令行参数列表。None 时使用 ``sys.argv[1:]``
    """
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        sys.exit(1)

    # 创建注册中心
    registry = create_registry()

    if parsed.command == 'help':
        _handle_help(registry, parsed.operator_name)
    elif parsed.command == 'run':
        _handle_run(registry, parsed)
    elif parsed.command == 'fit':
        _handle_fit(registry, parsed)


if __name__ == '__main__':
    main()
