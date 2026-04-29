# -*- coding: utf-8 -*-

"""
帮助文档自动生成模块单元测试

对应源文件：
- cli/help_generator.py: generate_list, generate_detail, generate_config_params_table

测试范围：
- 列表模式生成
- 详情模式生成（含参数表格、类型标签、附加输出）
- 参数表格生成（各种类型：int/float/Enum/Literal/默认值/值域）
- 内部辅助函数（_extract_summary, _format_type, _extract_constraints）
"""

from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from bianque.engine.operator.cli.help_generator import (
    generate_list,
    generate_detail,
    generate_config_params_table,
    _extract_summary,
    _extract_description,
    _format_type,
    _extract_constraints,
)


# ============================================================================
# 辅助类
# ============================================================================

class _TestEnum(Enum):
    A = "alpha"
    B = "beta"


class _TestConfig(BaseModel):
    """测试用配置类"""
    n: int = Field(default=5, ge=1, le=20, description="数量参数")
    ratio: float = Field(default=0.5, gt=0, lt=1.0, description="比例参数")
    method: _TestEnum = Field(default=_TestEnum.A, description="方法选择")
    mode: Literal["fast", "slow"] = Field(default="fast", description="模式")
    name: str = Field(default="test", description="名称")
    required_param: int = Field(..., description="必填参数")


class _DocOperator:
    """
    测试算子

    这是一个用于测试的算子类。

    Attributes:
        config: 配置参数
    """
    _config_type = _TestConfig
    _run_params_type = None
    _extra_output_type = None
    _fit_params_type = None

    @classmethod
    def name(cls) -> str:
        return "test_operator"


class _NoDocOperator:
    """   """

    @classmethod
    def name(cls) -> str:
        return "no_doc_op"


class _NoneDocOperator:
    @classmethod
    def name(cls) -> str:
        return "none_doc_op"


# ============================================================================
# 测试类
# ============================================================================

class TestGenerateList:
    """测试 generate_list 列表模式"""

    def test_generate_list_basic(self):
        """
        目的：验证列表模式生成包含算子名称和简介
        输入：两个算子的映射
        预期：Markdown 表格包含两行算子数据
        """
        operators = {
            "test_operator": _DocOperator,
            "no_doc_op": _NoDocOperator,
        }
        result = generate_list(operators)

        assert "## 可用算子列表" in result
        assert "`test_operator`" in result
        assert "`no_doc_op`" in result
        assert "共 2 个算子" in result

    def test_generate_list_sorted(self):
        """
        目的：验证列表按名称排序
        输入：无序的算子映射
        预期：输出按字母序排列
        """
        operators = {
            "z_operator": _DocOperator,
            "a_operator": _NoDocOperator,
        }
        result = generate_list(operators)
        a_pos = result.index("a_operator")
        z_pos = result.index("z_operator")
        assert a_pos < z_pos

    def test_generate_list_empty(self):
        """
        目的：验证空算子列表的输出
        输入：空字典
        预期：包含表头但无数据行，显示共 0 个算子
        """
        result = generate_list({})
        assert "共 0 个算子" in result


class TestGenerateDetail:
    """测试 generate_detail 详情模式"""

    def test_generate_detail_basic(self):
        """
        目的：验证详情模式包含算子名称、描述和参数表格
        输入：带有 Config 的算子类
        预期：输出包含算子名称、描述和参数表格标题
        """
        result = generate_detail(_DocOperator)

        assert "## test_operator" in result
        assert "测试算子" in result
        assert "### 实例参数" in result
        assert "`n`" in result
        assert "`ratio`" in result

    def test_generate_detail_no_config(self):
        """
        目的：验证无 Config 的算子不输出参数表格
        输入：config_type 为 None 的算子
        预期：输出不包含"实例参数"标题
        """
        class _NoCfgOp:
            _config_type = None
            _run_params_type = None
            _extra_output_type = None
            _fit_params_type = None

            @classmethod
            def name(cls):
                return "no_cfg_op"

        result = generate_detail(_NoCfgOp)
        assert "### 实例参数" not in result


class TestGenerateConfigParamsTable:
    """测试 generate_config_params_table 参数表格生成"""

    def test_table_has_header(self):
        """
        目的：验证表格包含完整的 Markdown 表头
        输入：_TestConfig 类
        预期：包含参数名、类型、默认值、值域/候选、说明 5 列
        """
        result = generate_config_params_table(_TestConfig)
        assert "| 参数名 | 类型 | 默认值 | 值域/候选 | 说明 |" in result

    def test_int_field(self):
        """
        目的：验证 int 字段显示正确的类型、默认值和值域
        输入：n: int = Field(default=5, ge=1, le=20)
        预期：类型 int，默认值 5，值域 [1, 20]
        """
        result = generate_config_params_table(_TestConfig)
        assert "`n`" in result
        assert "`5`" in result
        assert "[1, 20]" in result

    def test_float_field_exclusive(self):
        """
        目的：验证 float 字段 gt/lt 显示开区间
        输入：ratio: float = Field(default=0.5, gt=0, lt=1.0)
        预期：值域 (0, 1.0)
        """
        result = generate_config_params_table(_TestConfig)
        assert "(0, 1.0)" in result

    def test_enum_field(self):
        """
        目的：验证 Enum 字段显示候选值
        输入：method: _TestEnum
        预期：类型包含 enum，候选值包含 alpha, beta
        """
        result = generate_config_params_table(_TestConfig)
        assert "alpha" in result
        assert "beta" in result

    def test_literal_field(self):
        """
        目的：验证 Literal 字段显示候选值
        输入：mode: Literal["fast", "slow"]
        预期：候选值包含 fast, slow
        """
        result = generate_config_params_table(_TestConfig)
        assert "fast" in result
        assert "slow" in result

    def test_required_field(self):
        """
        目的：验证必填字段显示为 **必填**
        输入：required_param: int = Field(...)
        预期：默认值列显示 **必填**
        """
        result = generate_config_params_table(_TestConfig)
        assert "**必填**" in result

    def test_description(self):
        """
        目的：验证 description 显示在说明列
        输入：各字段的 description
        预期：说明列包含描述文本
        """
        result = generate_config_params_table(_TestConfig)
        assert "数量参数" in result
        assert "比例参数" in result


class TestExtractSummary:
    """测试 _extract_summary 辅助函数"""

    def test_with_docstring(self):
        """
        目的：验证从 docstring 中提取第一行简介
        输入：有多行 docstring 的类
        预期：返回第一个非空行
        """
        result = _extract_summary(_DocOperator)
        assert result == "测试算子"

    def test_empty_docstring(self):
        """
        目的：验证空 docstring 返回 "(无描述)"
        输入：只有空白的 docstring
        预期：返回 "(无描述)"
        """
        result = _extract_summary(_NoDocOperator)
        assert result == "(无描述)"

    def test_none_docstring(self):
        """
        目的：验证无 docstring 返回 "(无描述)"
        输入：没有 docstring 的类
        预期：返回 "(无描述)"
        """
        result = _extract_summary(_NoneDocOperator)
        assert result == "(无描述)"


class TestFormatType:
    """测试 _format_type 辅助函数"""

    def test_int(self):
        """
        目的：验证 int 类型格式化
        预期：返回 "int"
        """
        assert _format_type(int) == "int"

    def test_float(self):
        """
        目的：验证 float 类型格式化
        预期：返回 "float"
        """
        assert _format_type(float) == "float"

    def test_str(self):
        """
        目的：验证 str 类型格式化
        预期：返回 "str"
        """
        assert _format_type(str) == "str"

    def test_bool(self):
        """
        目的：验证 bool 类型格式化
        预期：返回 "bool"
        """
        assert _format_type(bool) == "bool"

    def test_enum(self):
        """
        目的：验证 Enum 类型格式化
        预期：返回 "enum(alpha, beta)"
        """
        result = _format_type(_TestEnum)
        assert "enum" in result
        assert "alpha" in result

    def test_literal(self):
        """
        目的：验证 Literal 类型格式化
        预期：返回 "literal(fast, slow)"
        """
        result = _format_type(Literal["fast", "slow"])
        assert "literal" in result
        assert "fast" in result

    def test_none(self):
        """
        目的：验证 None 类型格式化
        预期：返回 "Any"
        """
        assert _format_type(None) == "Any"


class TestExtractDescription:
    """测试 _extract_description 辅助函数"""

    def test_with_docstring(self):
        """
        目的：验证从 docstring 中提取功能描述
        输入：有多段 docstring 的类
        预期：返回第一段文本
        """
        result = _extract_description(_DocOperator)
        assert "测试算子" in result

    def test_empty_docstring(self):
        """
        目的：验证空 docstring 返回空字符串
        输入：空白 docstring
        预期：返回 ""
        """
        result = _extract_description(_NoDocOperator)
        assert result == ""

    def test_none_docstring(self):
        """
        目的：验证无 docstring 返回空字符串
        输入：无 docstring 的类
        预期：返回 ""
        """
        result = _extract_description(_NoneDocOperator)
        assert result == ""


class TestGenerateDetailWithRealOperators:
    """测试 generate_detail 使用真实算子类"""

    def test_detection_scorer(self):
        """
        目的：验证检测算子的 detail 输出包含类型标签
        输入：KNNScorer 类
        预期：输出包含 "Scorer" 和 "可训练" 标签
        """
        from bianque.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "## knn_scorer" in result
        assert "Scorer" in result
        assert "可训练" in result
        assert "### 实例参数" in result

    def test_detection_detector(self):
        """
        目的：验证检测器类的 detail 输出
        输入：KNNDetector 类
        预期：输出包含 "Detector" 标签
        """
        from bianque.engine.operator.detection.knn import KNNDetector
        result = generate_detail(KNNDetector)
        assert "## knn_detector" in result
        assert "Detector" in result or "Decider" in result

    def test_evaluation_operator(self):
        """
        目的：验证评价算子的 detail 输出
        输入：BinaryClassificationMetric 类
        预期：输出包含 "评价指标" 标签
        """
        from bianque.engine.operator.evaluation.binary_classification import BinaryClassificationMetric
        result = generate_detail(BinaryClassificationMetric)
        assert "## binary_classification" in result
        assert "评价指标" in result

    def test_feature_operator(self):
        """
        目的：验证特征算子的 detail 输出
        输入：PCAFeature 类（可训练特征）
        预期：输出包含 "特征算子" 和 "可训练" 标签
        """
        from bianque.engine.operator.feature.construction.simple_feature import PCAFeature
        result = generate_detail(PCAFeature)
        assert "## pca_feature" in result
        assert "特征算子" in result
        assert "可训练" in result

    def test_operator_with_extra_output(self):
        """
        目的：验证带附加输出的算子显示 EO 信息
        输入：KNNScorer 类（有 KNNScorerExtraOutput）
        预期：输出包含 "### 附加输出" 标题
        """
        from bianque.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "### 附加输出" in result


class TestFormatTypeAdvanced:
    """测试 _format_type 对复杂类型的处理"""

    def test_list_type(self):
        """
        目的：验证 list[str] 类型格式化
        预期：返回包含 list 和 str 的字符串
        """
        result = _format_type(list[str])
        assert "list" in result
        assert "str" in result

    def test_class_with_name(self):
        """
        目的：验证自定义类名显示
        预期：返回类名
        """
        class MyCustomClass:
            pass
        result = _format_type(MyCustomClass)
        assert "MyCustomClass" in result


class TestExtractConstraints:
    """测试 _extract_constraints 辅助函数"""

    def test_ge_le_constraints(self):
        """
        目的：验证 ge/le 约束提取为闭区间
        输入：Field(ge=1, le=20)
        预期：返回 "[1, 20]"
        """
        field_info = _TestConfig.model_fields['n']
        result = _extract_constraints(field_info)
        assert "[1, 20]" in result

    def test_gt_lt_constraints(self):
        """
        目的：验证 gt/lt 约束提取为开区间
        输入：Field(gt=0, lt=1.0)
        预期：返回 "(0, 1.0)"
        """
        field_info = _TestConfig.model_fields['ratio']
        result = _extract_constraints(field_info)
        assert "(0, 1.0)" in result

    def test_no_constraints(self):
        """
        目的：验证无约束字段返回 "-"
        输入：普通 str 字段
        预期：返回 "-"
        """
        field_info = _TestConfig.model_fields['name']
        result = _extract_constraints(field_info)
        assert result == "-"

    def test_enum_constraints(self):
        """
        目的：验证 Enum 字段提取候选值
        输入：_TestEnum 类型字段
        预期：返回包含 "alpha, beta"
        """
        field_info = _TestConfig.model_fields['method']
        result = _extract_constraints(field_info)
        assert "alpha" in result
        assert "beta" in result
