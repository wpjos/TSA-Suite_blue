# -*- coding: utf-8 -*-

"""
数据输入输出模块

提供统一的数据加载和保存接口，根据文件后缀自动选择格式。

表格数据支持格式（见 :func:`load_data` / :func:`save_data`）:
    - ``.csv``: CSV 逗号分隔值
    - ``.tsv``: TSV 制表符分隔值
    - ``.mat``: MATLAB MAT 文件（预留）
    - ``.h5`` / ``.hdf5``: HDF5 文件（预留）

结构化（字典）数据支持格式（见 :func:`save_structured`）:
    - ``.json``: JSON 格式
    - ``.yaml`` / ``.yml``: YAML 格式

使用示例::

    from tsas.engine.operator.cli.io import (
        load_data, save_data, save_json, save_structured,
    )

    df = load_data("input.csv")
    save_data(df, "output.csv")
    save_json({"f1": 0.85}, "result.json")
    save_structured({"f1": 0.85}, "result.yaml")
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

__all__ = [
    'load_data',
    'save_data',
    'save_json',
    'save_structured',
    'ensure_encoding',
]

# 支持的数据文件后缀 → 加载/保存函数的映射键
_SUPPORTED_EXTENSIONS = {'.csv', '.tsv'}
# 预留但尚未实现的格式
_RESERVED_EXTENSIONS = {'.mat', '.h5', '.hdf5'}


def load_data(path: str | Path) -> pd.DataFrame:
    """
    根据文件后缀自动加载数据为 DataFrame

    当前支持 CSV 格式。后缀名自动判断，大小写不敏感。

    Args:
        path (str | Path): 输入文件路径

    Returns:
        pd.DataFrame: 加载的数据，第一行作为表头

    Raises:
        FileNotFoundError: 文件不存在时
        ValueError: 文件后缀不支持或尚未实现时
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    suffix = path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(path)
    elif suffix == '.tsv':
        return pd.read_csv(path, sep='\t')
    elif suffix in _RESERVED_EXTENSIONS:
        raise ValueError(
            f"文件格式 '{suffix}' 尚未实现，预留扩展。"
            f"当前支持的格式: {sorted(_SUPPORTED_EXTENSIONS)}"
        )
    else:
        raise ValueError(
            f"不支持的文件格式 '{suffix}'。"
            f"当前支持的格式: {sorted(_SUPPORTED_EXTENSIONS)}"
        )


def save_data(df: pd.DataFrame, path: str | Path) -> None:
    """
    根据文件后缀自动保存 DataFrame 到文件

    当前支持 CSV 格式。自动创建目标目录（如不存在）。

    Args:
        df (pd.DataFrame): 要保存的数据
        path (str | Path): 输出文件路径

    Raises:
        ValueError: 文件后缀不支持或尚未实现时
    """
    path = Path(path)

    # 自动创建目标目录
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == '.csv':
        df.to_csv(path, index=False)
    elif suffix == '.tsv':
        df.to_csv(path, sep='\t', index=False)
    elif suffix in _RESERVED_EXTENSIONS:
        raise ValueError(
            f"文件格式 '{suffix}' 尚未实现，预留扩展。"
            f"当前支持的格式: {sorted(_SUPPORTED_EXTENSIONS)}"
        )
    else:
        raise ValueError(
            f"不支持的文件格式 '{suffix}'。"
            f"当前支持的格式: {sorted(_SUPPORTED_EXTENSIONS)}"
        )


def save_json(data: dict, path: str | Path) -> None:
    """
    保存字典数据为 JSON 文件

    使用 UTF-8 编码，2 空格缩进，保留非 ASCII 字符。
    自动创建目标目录（如不存在）。

    Args:
        data (dict): 要保存的字典数据
        path (str | Path): 输出文件路径
    """
    path = Path(path)

    # 自动创建目标目录
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_structured(data: dict, path: str | Path) -> None:
    """
    根据文件后缀自动保存结构化（字典）数据为 JSON 或 YAML 文件

    支持的后缀（大小写不敏感）：
        - ``.json``: JSON 格式（2 空格缩进，保留非 ASCII 字符）
        - ``.yaml`` / ``.yml``: YAML 格式（block 风格，保留非 ASCII 字符，保持键插入顺序）

    自动创建目标目录（如不存在）。YAML 输出前会先经 JSON 往返标准化，
    将 numpy/torch 等非原生类型统一转换为 JSON 兼容的 Python 原生类型，
    保证两种格式输出内容语义一致。

    Args:
        data (dict): 要保存的字典数据
        path (str | Path): 输出文件路径

    Raises:
        ValueError: 文件后缀既非 JSON 也非 YAML 时
    """
    path = Path(path)

    # 自动创建目标目录
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == '.json':
        # JSON：与 save_json 行为一致
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    elif suffix in ('.yaml', '.yml'):
        # YAML：先经 JSON 往返标准化，消除 numpy/torch 等非原生类型，
        # 避免 yaml.safe_dump 遇到非标准类型时抛出 RepresenterError
        normalized = json.loads(json.dumps(data, default=str))
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                normalized, f,
                allow_unicode=True,  # 保留中文等非 ASCII 字符
                default_flow_style=False,  # 使用 block 风格，更易读
                sort_keys=False,  # 保持字典插入顺序
                indent=2,
            )
    else:
        raise ValueError(
            f"不支持的结构化输出格式 '{suffix}'，仅支持 .json / .yaml / .yml"
        )


def ensure_encoding(encoding: str | None = None) -> None:
    """确保终端输出使用正确的编码

    处理流程：

    1. 确定目标编码（用户指定或自动探测为 ``utf-8``）
    2. 对 ``sys.stdout`` 和 ``sys.stderr`` 分别处理：
       - 检查当前编码是否已是目标编码 → 是则跳过
       - 尝试 ``reconfigure`` → 验证是否生效
       - 验证失败 → 用 ``io.TextIOWrapper`` 包装底层 buffer 做兜底替换
       - 兜底也失败 → 放弃（彻底无法修复）

    Windows 环境额外步骤：自动探测模式下执行 ``chcp 65001``
    将控制台代码页设为 UTF-8。

    Args:
        encoding (str | None): 用户指定的编码名称，如 ``'utf-8'``、``'gbk'``。
            传入 ``None`` 时启用自动探测模式，默认目标编码为 UTF-8。
    """
    import io

    target = encoding

    if target is None:
        # 自动探测模式：检查当前 stdout 编码是否已兼容 UTF-8
        enc = getattr(sys.stdout, 'encoding', '') or ''
        if 'utf' in enc.lower():
            return  # 已是 UTF-8，无需处理

        # 非 UTF-8 环境，目标设为 UTF-8
        target = 'utf-8'

        # Windows: 设置控制台代码页为 UTF-8 (65001)
        if sys.platform == 'win32':
            os.system('chcp 65001 >nul 2>&1')

    # 对 stdout 和 stderr 分别处理编码重配置
    for stream_name, stream in [('stdout', sys.stdout), ('stderr', sys.stderr)]:
        # 检查当前编码是否已是目标编码
        current_enc = getattr(stream, 'encoding', '') or ''
        if target.lower() in current_enc.lower():
            continue  # 已是目标编码

        # 第一步：尝试 reconfigure
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding=target, errors='replace')
            except (LookupError, UnicodeError, AttributeError):
                pass

        # 第二步：验证 reconfigure 是否真的生效
        current_enc = getattr(stream, 'encoding', '') or ''
        if target.lower() in current_enc.lower():
            continue  # reconfigure 成功

        # 第三步：reconfigure 未生效，用 TextIOWrapper 兜底替换
        try:
            buffer = getattr(stream, 'buffer', None)
            if buffer is not None:
                new_stream = io.TextIOWrapper(
                    buffer,
                    encoding=target,
                    errors='replace',
                    line_buffering=getattr(stream, 'line_buffering', False),
                )
                if stream_name == 'stdout':
                    sys.stdout = new_stream
                else:
                    sys.stderr = new_stream
        except Exception:
            pass  # 彻底无法修复，放弃
