"""内部本地库兼容层。

``cicada`` / ``bq_cicada`` 与 ``time_rcd`` / ``bq_rcd`` 都是同一份内部本地库
在不同项目里的传播名。算子层只引用 ``bq_xxx`` 优先、``xxx`` 兜底，避免在
pyproject.toml 里把传播名固化死。

调用方应 **按需调用** 本模块导出的函数（保持延迟导入语义），不要在模块顶层
调用——一来会触发 torch 等重型依赖，二来会让缺失本地库时连算子模块本身都
无法 ``import``。

这些库不在 PyPI 上，仅以 path 依赖方式安装。
"""

from __future__ import annotations

__all__ = [
    "import_cicada_class",
    "import_rcd_classes",
]


def import_cicada_class():
    """延迟导入 :class:`CICADA`，优先 ``bq_cicada``，回落到 ``cicada``。

    Returns:
        已加载的 ``CICADA`` 类。

    Raises:
        RuntimeError: ``bq_cicada`` 与 ``cicada`` 均不可用时抛出，
            给出 path 依赖安装指引。
    """
    try:
        from bq_cicada import CICADA  # type: ignore[import-not-found]
    except ImportError:
        try:
            from cicada import CICADA  # type: ignore[no-redef]
        except ImportError:
            raise RuntimeError(
                "未找到 CICADA 本地依赖（内部库，不在 PyPI 上）。"
                "请通过 `uv pip install -e /Users/chao/bq/bq_cicada` "
                "或在 pyproject.toml 中以 path 依赖安装 cicada / bq_cicada 后重试。"
            ) from None
    return CICADA


def import_rcd_classes():
    """延迟导入 Time-RCD 三件套，优先 ``bq_rcd``，回落到 ``time_rcd``。

    Returns:
        ``(TimeRCDPretrainTester, TimeRCDConfig, TimeSeriesConfig)`` 元组。

    Raises:
        RuntimeError: ``bq_rcd`` 与 ``time_rcd`` 均不可用时抛出，
            给出 path 依赖安装指引。
    """
    try:
        from bq_rcd.time_rcd import TimeRCDPretrainTester  # type: ignore[import-not-found]
        from bq_rcd.time_rcd.time_rcd_config import (  # type: ignore[import-not-found]
            TimeRCDConfig,
            TimeSeriesConfig,
        )
    except ImportError:
        try:
            from time_rcd import TimeRCDPretrainTester  # type: ignore[no-redef]
            from time_rcd.time_rcd_config import (  # type: ignore[no-redef]
                TimeRCDConfig,
                TimeSeriesConfig,
            )
        except ImportError:
            raise RuntimeError(
                "未找到 bq_rcd 本地依赖（内部库，不在 PyPI 上）。"
                "请通过 `uv pip install -e /Users/chao/bq/bq_rcd` "
                "或在 pyproject.toml 中以 path 依赖安装 bq_rcd / bq-rcd 后重试。"
            ) from None
    return TimeRCDPretrainTester, TimeRCDConfig, TimeSeriesConfig
