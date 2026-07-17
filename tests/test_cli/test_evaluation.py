# -*- coding: utf-8 -*-

"""
评价指标算子 CLI 单元测试

对应源文件：
- cli/evaluation.py

测试范围：
- help 子命令（列表模式和详情模式）
- run 子命令（单算子、多算子、alias 去重）
- 配置校验错误场景
- _resolve_output_key 去重逻辑
- _result_to_dict 序列化逻辑
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import BaseModel

from tsas.engine.operator.cli.evaluation import (_resolve_output_key, _result_to_dict, create_registry, main)


# ============================================================================
# 公共 fixture
# ============================================================================

@pytest.fixture
def binary_csv(tmp_path):
    """创建二分类评价用 CSV（含 label 和 predict 列）"""
    df = pd.DataFrame({
        'label': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'predict': [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
    })
    path = tmp_path / "predictions.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def score_csv(tmp_path):
    """创建单列分数 CSV（供 self_evaluation 使用）"""
    np.random.seed(42)
    df = pd.DataFrame({'score': np.random.randn(50)})
    path = tmp_path / "scores.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def curve_csv(tmp_path):
    """创建二分类曲线评价用 CSV（含离散 label 列和连续 score 列）

    binary_classification_curve 要求 y_predict 为连续异常分数，
    因此单独构造 label + score 两列的测试数据。
    """
    df = pd.DataFrame({
        'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'score': [0.1, 0.05, 0.2, 0.15, 0.08, 0.12, 0.03, 0.25, 0.18, 0.06,
                  0.7, 0.85, 0.6, 0.9, 0.75, 0.65, 0.8, 0.95, 0.55, 0.72],
    })
    path = tmp_path / "curve_predictions.csv"
    df.to_csv(path, index=False)
    return path


# ============================================================================
# 测试类
# ============================================================================

class TestEvaluationHelp:
    """测试 evaluation help 子命令"""

    def test_help_list(self, capsys):
        """
        目的：验证 help 无参数时列出所有评价指标算子
        输入：['help']
        预期：输出包含 binary_classification 等算子名称
        """
        main(['help'])
        captured = capsys.readouterr()
        assert "binary_classification" in captured.out
        assert "self_evaluation" in captured.out

    def test_help_detail(self, capsys):
        """
        目的：验证 help 带算子名称时输出详情
        输入：['help', 'binary_classification']
        预期：输出包含算子描述和参数
        """
        main(['help', 'binary_classification'])
        captured = capsys.readouterr()
        assert "## binary_classification" in captured.out

    def test_help_unknown_operator(self):
        """
        目的：验证 help 查找不存在的算子时报错
        输入：['help', 'nonexistent']
        预期：抛出 KeyError
        """
        with pytest.raises(KeyError, match="未找到名为"):
            main(['help', 'nonexistent'])


class TestEvaluationRun:
    """测试 evaluation run 子命令"""

    def test_run_single_operator(self, binary_csv, tmp_path, capsys):
        """
        目的：验证单个评价算子的 run 流程
        输入：binary_classification_metric 配置 + 二分类数据
        预期：输出 JSON 包含 result 和 main_scores
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        main(['run', '--input', str(binary_csv), '--output', str(output_path),
              '--config', str(config_path)])

        captured = capsys.readouterr()
        assert "评价完成" in captured.out

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        assert "results" in result
        assert "binary_classification" in result["results"]
        assert "result" in result["results"]["binary_classification"]

    def test_run_multiple_operators(self, binary_csv, tmp_path, capsys):
        """
        目的：验证多个评价算子的 run 流程
        输入：两个不同算子的配置
        预期：输出 JSON 包含两个算子的结果
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                },
                {
                    "name": "point_adjust",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                },
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        main(['run', '--input', str(binary_csv), '--output', str(output_path),
              '--config', str(config_path)])

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        assert "binary_classification" in result["results"]
        assert "point_adjust" in result["results"]

    def test_run_with_alias(self, binary_csv, tmp_path, capsys):
        """
        目的：验证 alias 字段作为输出 key
        输入：配置中指定 alias
        预期：输出 JSON 使用 alias 作为 key
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "alias": "my_metric",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        main(['run', '--input', str(binary_csv), '--output', str(output_path),
              '--config', str(config_path)])

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        assert "my_metric" in result["results"]

    def test_run_single_input(self, score_csv, tmp_path, capsys):
        """
        目的：验证单输入算子（self_evaluation）的 run 流程
        输入：单列分数数据 + input_columns 配置
        预期：输出 JSON 包含 self_evaluation 结果
        """
        config = {
            "operators": [
                {
                    "name": "self_evaluation",
                    "input_columns": ["score"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        main(['run', '--input', str(score_csv), '--output', str(output_path),
              '--config', str(config_path)])

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        assert "self_evaluation" in result["results"]

    def test_run_empty_operators_raises(self, binary_csv, tmp_path):
        """
        目的：验证空算子列表时报错
        输入：operators 为空列表
        预期：抛出 ValueError
        """
        config = {"operators": []}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        with pytest.raises(ValueError, match="不能为空"):
            main(['run', '--input', str(binary_csv), '--output', str(output_path),
                  '--config', str(config_path)])

    def test_run_missing_name_raises(self, binary_csv, tmp_path):
        """
        目的：验证算子缺少 name 字段时报错
        输入：无 name 的算子配置
        预期：抛出 ValueError
        """
        config = {"operators": [{"truth_columns": ["label"]}]}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        with pytest.raises(ValueError, match="缺少 'name'"):
            main(['run', '--input', str(binary_csv), '--output', str(output_path),
                  '--config', str(config_path)])


class TestEvaluationRunOutputFormat:
    """测试 evaluation run 输出格式（--scalars-output 标量剥离、YAML 后缀输出）"""

    def test_scalars_output_only_json(self, binary_csv, tmp_path, capsys):
        """
        目的：验证仅指定 --scalars-output 时输出标量结果（剥离 list 类型字段）
        输入：binary_classification 配置，仅 --scalars-output（.json）
        预期：输出存在；result 中不含 list 字段（如 confusion_matrix），保留标量字段
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        scalars_path = tmp_path / "scalars.json"
        main(['run', '--input', str(binary_csv), '--scalars-output', str(scalars_path),
              '--config', str(config_path)])

        captured = capsys.readouterr()
        assert "标量结果" in captured.out

        with open(scalars_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        res = result["results"]["binary_classification"]["result"]
        # confusion_matrix 为 list，应被标量输出剥离
        assert "confusion_matrix" not in res
        # 标量字段保留
        assert "f1" in res

    def test_no_output_raises(self, binary_csv, tmp_path):
        """
        目的：验证既不指定 --output 也不指定 --scalars-output 时报错退出
        输入：无任何输出路径
        预期：SystemExit(1)
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        with pytest.raises(SystemExit):
            main(['run', '--input', str(binary_csv), '--config', str(config_path)])

    def test_output_yaml_format(self, binary_csv, tmp_path, capsys):
        """
        目的：验证 --output 使用 .yaml 后缀时输出 YAML 格式
        输入：binary_classification 配置 + .yaml 输出路径
        预期：输出为合法 YAML，yaml.safe_load 读回结构正确；完整结果保留 list 字段
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.yaml"
        main(['run', '--input', str(binary_csv), '--output', str(output_path),
              '--config', str(config_path)])

        content = output_path.read_text(encoding='utf-8')
        data = yaml.safe_load(content)
        assert "results" in data
        assert "binary_classification" in data["results"]
        # 完整结果应保留 confusion_matrix（list 字段）
        assert "confusion_matrix" in data["results"]["binary_classification"]["result"]

    def test_scalars_output_yaml_format(self, binary_csv, tmp_path, capsys):
        """
        目的：验证 --scalars-output 使用 .yml 后缀时输出 YAML 标量结果
        输入：binary_classification 配置 + .yml 标量输出路径
        预期：输出为合法 YAML；标量结果剥离了 list 字段（confusion_matrix）
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        scalars_path = tmp_path / "scalars.yml"
        main(['run', '--input', str(binary_csv), '--scalars-output', str(scalars_path),
              '--config', str(config_path)])

        data = yaml.safe_load(scalars_path.read_text(encoding='utf-8'))
        res = data["results"]["binary_classification"]["result"]
        # confusion_matrix（list）被剥离
        assert "confusion_matrix" not in res
        assert "f1" in res

    def test_both_outputs_yaml(self, binary_csv, tmp_path, capsys):
        """
        目的：验证 --output 与 --scalars-output 同时使用 .yaml 时均正确输出 YAML
        输入：两个 .yaml 输出路径
        预期：两个文件均为合法 YAML；完整版含 list 字段，标量版不含；同名标量值一致
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.yaml"
        scalars_path = tmp_path / "scalars.yaml"
        main(['run', '--input', str(binary_csv), '--output', str(output_path),
              '--scalars-output', str(scalars_path), '--config', str(config_path)])

        captured = capsys.readouterr()
        assert "完整结果" in captured.out
        assert "标量结果" in captured.out

        full = yaml.safe_load(output_path.read_text(encoding='utf-8'))
        scalars = yaml.safe_load(scalars_path.read_text(encoding='utf-8'))
        # 完整版含 confusion_matrix，标量版不含
        assert "confusion_matrix" in full["results"]["binary_classification"]["result"]
        assert "confusion_matrix" not in scalars["results"]["binary_classification"]["result"]
        # 两者同名标量字段一致
        assert (full["results"]["binary_classification"]["result"]["f1"]
                == scalars["results"]["binary_classification"]["result"]["f1"])

    def test_json_yaml_equivalence_full_output(self, binary_csv, tmp_path):
        """
        目的：验证同一 evaluation 结果以 .json 和 .yaml 输出后语义完全等价
        输入：binary_classification 配置 + 二分类数据；同一输入分别输出 .json / .yaml
        预期：json 读回 == yaml 读回（含标量、嵌套 dict、confusion_matrix 二维列表）
        说明：算子计算为确定性，相同输入产生相同结果
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        json_path = tmp_path / "result.json"
        yaml_path = tmp_path / "result.yaml"
        main(['run', '--input', str(binary_csv), '--output', str(json_path),
              '--config', str(config_path)])
        main(['run', '--input', str(binary_csv), '--output', str(yaml_path),
              '--config', str(config_path)])

        from_json = json.loads(json_path.read_text(encoding='utf-8'))
        from_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        assert from_json == from_yaml

    def test_json_yaml_equivalence_curve_arrays(self, curve_csv, tmp_path):
        """
        目的：验证含长曲线数组（thresholds/tpr/fpr 等）的 curve 结果 json/yaml 等价
        输入：binary_classification_curve 配置 + 连续分数数据
        预期：json 读回 == yaml 读回；两者均含 thresholds/tpr/fpr 列表且值完全一致
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification_curve",
                    "truth_columns": ["label"],
                    "predict_columns": ["score"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        json_path = tmp_path / "curve.json"
        yaml_path = tmp_path / "curve.yaml"
        main(['run', '--input', str(curve_csv), '--output', str(json_path),
              '--config', str(config_path)])
        main(['run', '--input', str(curve_csv), '--output', str(yaml_path),
              '--config', str(config_path)])

        from_json = json.loads(json_path.read_text(encoding='utf-8'))
        from_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        # 深度等价
        assert from_json == from_yaml
        # 确认含曲线数组字段且非空
        res_json = from_json["results"]["binary_classification_curve"]["result"]
        assert len(res_json["thresholds"]) > 0
        assert len(res_json["tpr"]) == len(res_json["thresholds"])
        assert len(res_json["fpr"]) == len(res_json["thresholds"])

    def test_json_yaml_equivalence_scalars_output(self, binary_csv, tmp_path):
        """
        目的：验证 --scalars-output 的标量结果在 json/yaml 两种格式下等价
        输入：binary_classification 配置；分别用 .json / .yaml 输出标量结果
        预期：json 读回 == yaml 读回；两者均不含 confusion_matrix（list 已剥离）
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        json_path = tmp_path / "scalars.json"
        yaml_path = tmp_path / "scalars.yaml"
        main(['run', '--input', str(binary_csv), '--scalars-output', str(json_path),
              '--config', str(config_path)])
        main(['run', '--input', str(binary_csv), '--scalars-output', str(yaml_path),
              '--config', str(config_path)])

        from_json = json.loads(json_path.read_text(encoding='utf-8'))
        from_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        assert from_json == from_yaml
        # 标量结果不含 list 字段
        assert "confusion_matrix" not in from_json["results"]["binary_classification"]["result"]

    def test_uppercase_yaml_suffix_e2e(self, binary_csv, tmp_path):
        """
        目的：验证 .YAML 大写后缀在端到端流程中正常输出 YAML
        输入：binary_classification 配置 + .YAML 输出路径
        预期：文件为合法 YAML，yaml.safe_load 读回结构正确
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.YAML"
        main(['run', '--input', str(binary_csv), '--output', str(output_path),
              '--config', str(config_path)])

        data = yaml.safe_load(output_path.read_text(encoding='utf-8'))
        assert "binary_classification" in data["results"]

    def test_json_yaml_cross_loader(self, binary_csv, tmp_path):
        """
        目的：验证 JSON 输出文件可被 yaml.safe_load 直接解析（跨解析器兼容）
        输入：evaluation 生成 .json 输出
        预期：yaml.safe_load 解析 .json 结果与 json.load 完全一致
        说明：JSON 是 YAML 1.2 子集，下游可统一用 PyYAML 加载两种格式
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        json_path = tmp_path / "result.json"
        main(['run', '--input', str(binary_csv), '--output', str(json_path),
              '--config', str(config_path)])

        via_json = json.loads(json_path.read_text(encoding='utf-8'))
        via_yaml = yaml.safe_load(json_path.read_text(encoding='utf-8'))
        assert via_json == via_yaml


class TestResolveOutputKey:
    """测试 _resolve_output_key 去重逻辑"""

    def test_no_alias_uses_name(self):
        """
        目的：验证无 alias 时使用算子名称
        输入：无 alias 的 spec
        预期：返回算子名称
        """
        used = set()
        key = _resolve_output_key({}, "my_op", used)
        assert key == "my_op"

    def test_alias_overrides_name(self):
        """
        目的：验证 alias 覆盖算子名称
        输入：spec 中有 alias
        预期：返回 alias
        """
        used = set()
        key = _resolve_output_key({"alias": "custom"}, "my_op", used)
        assert key == "custom"

    def test_auto_dedup(self):
        """
        目的：验证重复 key 自动追加后缀
        输入：已使用的 key 集合中已有 "my_op"
        预期：返回 "my_op_1"
        """
        used = {"my_op"}
        key = _resolve_output_key({}, "my_op", used)
        assert key == "my_op_1"

    def test_auto_dedup_multiple(self):
        """
        目的：验证多次重复时后缀递增
        输入：已使用 "my_op" 和 "my_op_1"
        预期：返回 "my_op_2"
        """
        used = {"my_op", "my_op_1"}
        key = _resolve_output_key({}, "my_op", used)
        assert key == "my_op_2"


class TestResultToDict:
    """测试 _result_to_dict 序列化逻辑"""

    def test_float(self):
        """
        目的：验证 float 直接返回
        输入：0.85
        预期：返回 0.85
        """
        assert _result_to_dict(0.85) == 0.85

    def test_int(self):
        """
        目的：验证 int 直接返回
        输入：42
        预期：返回 42
        """
        assert _result_to_dict(42) == 42

    def test_basemodel(self):
        """
        目的：验证 BaseModel 转为字典
        输入：Pydantic 模型实例
        预期：返回 model_dump() 结果
        """

        class _M(BaseModel):
            f1: float = 0.85

        result = _result_to_dict(_M())
        assert result == {"f1": 0.85}

    def test_ndarray(self):
        """
        目的：验证 ndarray 转为列表
        输入：numpy 数组
        预期：返回 tolist() 结果
        """
        arr = np.array([1.0, 2.0, 3.0])
        result = _result_to_dict(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_dict(self):
        """
        目的：验证嵌套字典递归转换
        输入：包含 ndarray 的字典
        预期：内部 ndarray 被转为列表
        """
        data = {"a": np.array([1, 2])}
        result = _result_to_dict(data)
        assert result == {"a": [1, 2]}

    def test_other_type(self):
        """
        目的：验证其他类型转为字符串
        输入：非标准类型对象
        预期：返回 str() 结果
        """
        result = _result_to_dict(object())
        assert isinstance(result, str)


class TestEvaluationNoCommand:
    """测试无子命令场景"""

    def test_no_command_exits(self):
        """
        目的：验证不提供子命令时 sys.exit(1)
        输入：空参数
        预期：SystemExit
        """
        with pytest.raises(SystemExit):
            main([])


class TestCreateEvaluationRegistry:
    """测试 create_registry 工厂函数"""

    def test_create_registry(self):
        """
        目的：验证工厂函数返回已 discover 的注册中心
        输入：无
        预期：返回包含评价指标算子的 OperatorRegistry
        """
        registry = create_registry()
        assert registry.discovered is True
        assert 'binary_classification' in registry.list_all()
        assert 'self_evaluation' in registry.list_all()
