"""检测算子测试 conftest。

本机 venv 的 bq_cicada 在 py3.14 + macOS MPS 上崩溃（Metal SDK 断言），即使
我们想要 CPU 也会被 bq_cicada 自动选 MPS。这里把 CICADA.__init__ 包一层，
让 device=None 默认走 'cpu'，从而让本机测试套件可以完整跑过。

该 fixture 只在有 bq_cicada 的环境生效；缺包时直接 skip 即可。
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True, scope="session")
def _force_cicada_cpu():
    """把 bq_cicada.CICADA.__init__ 包一层，device=None 时默认走 'cpu'。

    触发条件：本地能 import bq_cicada（CICADA 算子依赖，否则无意义）。
    """
    try:
        from bq_cicada import CICADA  # type: ignore[import-not-found]
    except ImportError:
        yield
        return

    orig_init = CICADA.__init__

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("device") is None:
            kwargs["device"] = "cpu"
        return orig_init(self, *args, **kwargs)

    CICADA.__init__ = _patched_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        CICADA.__init__ = orig_init  # type: ignore[method-assign]
