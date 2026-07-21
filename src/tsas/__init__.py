# -*- coding: utf-8 -*-

"""
Full-featured Time Series Analysis Suite

一个面向时间序列分析的全功能套件
"""

__version__ = '1.0.0'

# 兼容不同版本 sage-importance：旧版暴露 sage.SageValues，新版暴露 sage.Explanation。
# feature_selection_by_cwdw_sage.py 在模块顶层使用 sage.SageValues 做类型注解，
# 若不兼容则在注册表扫描时会直接抛出 AttributeError。
try:
    import sage
    if not hasattr(sage, 'SageValues') and hasattr(sage, 'Explanation'):
        sage.SageValues = sage.Explanation
except Exception:
    pass
