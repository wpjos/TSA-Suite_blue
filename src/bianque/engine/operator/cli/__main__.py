# -*- coding: utf-8 -*-

"""
算子 CLI 统一分发入口

通过 ``python -m bianque.engine.operator.cli <模块> <子命令> [参数]`` 调用。

支持的模块:
    - ``feature_construction``: 特征构造算子
    - ``detection``: 异常检测算子
    - ``evaluation``: 评价指标算子

示例::

    python -m bianque.engine.operator.cli feature_construction help
    python -m bianque.engine.operator.cli detection run --input data.csv --config det.yaml
    python -m bianque.engine.operator.cli evaluation run --input data.csv --config eval.json
"""

import sys

# 模块名称到子模块的映射
_MODULE_MAP = {
    'feature_construction': 'bianque.engine.operator.cli.feature_construction',
    'detection': 'bianque.engine.operator.cli.detection',
    'evaluation': 'bianque.engine.operator.cli.evaluation',
}


def _print_usage() -> None:
    """输出统一入口的使用说明"""
    print("用法: python -m bianque.engine.operator.cli <模块> <子命令> [参数]")
    print()
    print("可用模块:")
    print("  feature_construction  特征构造算子")
    print("  detection             异常检测算子")
    print("  evaluation            评价指标算子")
    print()
    print("示例:")
    print("  python -m bianque.engine.operator.cli feature_construction help")
    print("  python -m bianque.engine.operator.cli detection run --input data.csv --config det.yaml")
    print("  python -m bianque.engine.operator.cli evaluation run --input data.csv --config eval.json")


def main(args: list[str] | None = None) -> None:
    """
    统一分发入口主函数

    解析第一个参数为模块名称，将剩余参数转发给对应子模块。

    Args:
        args (list[str] | None): 命令行参数列表。None 时使用 ``sys.argv[1:]``
    """
    if args is None:
        args = sys.argv[1:]

    if not args:
        _print_usage()
        sys.exit(1)

    module_name = args[0]

    if module_name in ('-h', '--help', 'help'):
        _print_usage()
        return

    if module_name not in _MODULE_MAP:
        print(f"错误: 未知模块 '{module_name}'")
        print(f"可用模块: {', '.join(sorted(_MODULE_MAP.keys()))}")
        sys.exit(1)

    # 动态导入子模块并调用其 main 函数
    import importlib
    sub_module = importlib.import_module(_MODULE_MAP[module_name])
    sub_module.main(args[1:])


if __name__ == '__main__':
    main()
