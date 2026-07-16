"""
数组输出 helper 模块

为 transform 类 feature 算子（输出 1D ndarray 而非标量）提供共用工具。
与 signal_feature._apply_per_cell 的区别：后者返回 float ndarray（标量），
本模块的 _apply_per_cell_array 返回 object ndarray（每格是 1D 数组，长度可变）。

所有算法逻辑源自 bqlib ops.transform 系列，按 TSA-Suite 规范重写为
IndependentMapFeature 子类的 compute 方法调用形式。
"""

import numpy as np


def _apply_per_cell_array(x: np.ndarray, func) -> np.ndarray:
    """对 object ndarray 每格应用数组函数，返回 object ndarray。

    Args:
        x: 输入 ndarray，每格是 1D 信号段
        func: 数组函数，接收 1D float ndarray，返回 1D ndarray

    Returns:
        object ndarray，形状与 x 相同，每格是 func 的输出
    """
    result = np.empty(x.shape, dtype=object)
    for idx in np.ndindex(x.shape):
        sig = np.asarray(x[idx], dtype=float)
        result[idx] = func(sig)
    return result
