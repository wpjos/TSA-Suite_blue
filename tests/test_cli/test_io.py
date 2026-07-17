# -*- coding: utf-8 -*-

"""
数据IO模块单元测试

对应源文件：
- cli/io.py: load_data, save_data, save_json, save_structured

测试范围：
- CSV 加载和保存
- TSV 加载和保存
- 预留格式报错（MAT、HDF5）
- 不支持的格式报错
- 文件不存在报错
- JSON 保存
- 自动创建目录
"""

import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from tsas.engine.operator.cli.io import (
    ensure_encoding,
    load_data,
    save_data,
    save_json,
    save_structured,
)


# ============================================================================
# 公共 fixture
# ============================================================================

@pytest.fixture
def sample_df():
    """测试用 DataFrame（3列4行）"""
    return pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0],
        'b': [5.0, 6.0, 7.0, 8.0],
        'c': [9.0, 10.0, 11.0, 12.0],
    })


# ============================================================================
# CSV 测试
# ============================================================================

class TestLoadDataCSV:
    """测试 CSV 格式的数据加载"""

    def test_load_csv(self, sample_df, tmp_path):
        """
        目的：验证 CSV 文件能正确加载为 DataFrame
        输入：包含 3 列 4 行数据的 CSV 文件
        预期：加载后 DataFrame 形状和值与原始一致
        """
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        result = load_data(csv_path)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_load_csv_string_path(self, sample_df, tmp_path):
        """
        目的：验证支持字符串路径
        输入：字符串格式的文件路径
        预期：正常加载
        """
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        result = load_data(str(csv_path))
        pd.testing.assert_frame_equal(result, sample_df)


class TestLoadDataErrors:
    """测试数据加载的错误场景"""

    def test_file_not_found(self):
        """
        目的：验证文件不存在时抛出 FileNotFoundError
        输入：一个不存在的文件路径
        预期：抛出 FileNotFoundError
        """
        with pytest.raises(FileNotFoundError, match="输入文件不存在"):
            load_data("/nonexistent/path/data.csv")

    def test_unsupported_format(self, tmp_path):
        """
        目的：验证不支持的文件后缀抛出 ValueError
        输入：后缀为 .xyz 的文件
        预期：抛出 ValueError，提示不支持
        """
        path = tmp_path / "data.xyz"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_data(path)

    def test_reserved_format_mat(self, tmp_path):
        """
        目的：验证预留格式（.mat）抛出 ValueError 并提示尚未实现
        输入：后缀为 .mat 的文件
        预期：抛出 ValueError，提示尚未实现
        """
        path = tmp_path / "data.mat"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="尚未实现"):
            load_data(path)

    def test_reserved_format_h5(self, tmp_path):
        """
        目的：验证预留格式（.h5）抛出 ValueError 并提示尚未实现
        输入：后缀为 .h5 的文件
        预期：抛出 ValueError，提示尚未实现
        """
        path = tmp_path / "data.h5"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="尚未实现"):
            load_data(path)


class TestSaveData:
    """测试数据保存"""

    def test_save_csv(self, sample_df, tmp_path):
        """
        目的：验证 DataFrame 能正确保存为 CSV
        输入：3列4行的 DataFrame
        预期：保存后重新加载，数据一致
        """
        csv_path = tmp_path / "output.csv"
        save_data(sample_df, csv_path)

        result = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_save_tsv(self, sample_df, tmp_path):
        """
        目的：验证 DataFrame 能正确保存为 TSV
        输入：3列4行的 DataFrame
        预期：保存后重新加载，数据一致
        """
        tsv_path = tmp_path / "output.tsv"
        save_data(sample_df, tsv_path)

        result = pd.read_csv(tsv_path, sep='\t')
        pd.testing.assert_frame_equal(result, sample_df)

    def test_save_creates_directory(self, sample_df, tmp_path):
        """
        目的：验证保存时自动创建不存在的目录
        输入：指定一个不存在的目录路径
        预期：目录自动创建，文件正常保存
        """
        csv_path = tmp_path / "subdir" / "nested" / "output.csv"
        save_data(sample_df, csv_path)
        assert csv_path.exists()

    def test_save_unsupported_format(self, sample_df, tmp_path):
        """
        目的：验证不支持的格式抛出 ValueError
        输入：后缀为 .xyz 的路径
        预期：抛出 ValueError
        """
        path = tmp_path / "output.xyz"
        with pytest.raises(ValueError, match="不支持的文件格式"):
            save_data(sample_df, path)

    def test_save_reserved_format(self, sample_df, tmp_path):
        """
        目的：验证预留格式抛出 ValueError
        输入：后缀为 .hdf5 的路径
        预期：抛出 ValueError，提示尚未实现
        """
        path = tmp_path / "output.hdf5"
        with pytest.raises(ValueError, match="尚未实现"):
            save_data(sample_df, path)


class TestLoadDataTSV:
    """测试 TSV 格式的数据加载"""

    def test_load_tsv(self, sample_df, tmp_path):
        """
        目的：验证 TSV 文件能正确加载
        输入：制表符分隔的数据文件
        预期：加载后数据一致
        """
        tsv_path = tmp_path / "test.tsv"
        sample_df.to_csv(tsv_path, sep='\t', index=False)

        result = load_data(tsv_path)
        pd.testing.assert_frame_equal(result, sample_df)


class TestSaveJson:
    """测试 JSON 保存"""

    def test_save_json_basic(self, tmp_path):
        """
        目的：验证字典能正确保存为 JSON
        输入：简单字典
        预期：文件内容为正确的 JSON
        """
        data = {"f1": 0.85, "far": 0.12}
        json_path = tmp_path / "result.json"

        save_json(data, json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_json_chinese(self, tmp_path):
        """
        目的：验证中文字符不被转义
        输入：包含中文的字典
        预期：文件中包含原始中文字符
        """
        data = {"名称": "测试"}
        json_path = tmp_path / "result.json"

        save_json(data, json_path)

        content = json_path.read_text(encoding='utf-8')
        assert "测试" in content
        assert "\\u" not in content

    def test_save_json_creates_directory(self, tmp_path):
        """
        目的：验证保存 JSON 时自动创建目录
        输入：指定不存在的目录路径
        预期：目录自动创建
        """
        json_path = tmp_path / "sub" / "result.json"
        save_json({"a": 1}, json_path)
        assert json_path.exists()


class TestSaveStructured:
    """测试 save_structured 按后缀分派 JSON / YAML"""

    def test_json_suffix_outputs_json(self, tmp_path):
        """
        目的：验证 .json 后缀输出合法 JSON，与 save_json 行为一致
        输入：简单字典 + .json 路径
        预期：json.loads 可解析，数据与原始一致
        """
        data = {"f1": 0.85, "far": 0.12}
        path = tmp_path / "result.json"
        save_structured(data, path)

        content = path.read_text(encoding='utf-8')
        assert json.loads(content) == data

    def test_yaml_suffix_outputs_yaml(self, tmp_path):
        """
        目的：验证 .yaml 后缀输出合法 YAML
        输入：含嵌套结构与二维列表的字典 + .yaml 路径
        预期：yaml.safe_load 可解析，数据与原始一致
        """
        data = {
            "results": {
                "binary_classification": {
                    "f1": 0.85,
                    "confusion_matrix": [[5, 0], [0, 5]],
                }
            }
        }
        path = tmp_path / "result.yaml"
        save_structured(data, path)

        content = path.read_text(encoding='utf-8')
        assert yaml.safe_load(content) == data

    def test_yml_suffix_outputs_yaml(self, tmp_path):
        """
        目的：验证 .yml 后缀同样输出 YAML
        输入：简单字典 + .yml 路径
        预期：yaml.safe_load 可解析，数据与原始一致
        """
        data = {"f1": 0.85}
        path = tmp_path / "result.yml"
        save_structured(data, path)

        content = path.read_text(encoding='utf-8')
        assert yaml.safe_load(content) == data

    def test_yaml_preserves_chinese(self, tmp_path):
        """
        目的：验证 YAML 输出保留中文字符（allow_unicode=True）
        输入：含中文键值的字典
        预期：文件中直接出现中文，无转义序列
        """
        data = {"名称": "测试", "指标": {"精确率": 0.9}}
        path = tmp_path / "result.yaml"
        save_structured(data, path)

        content = path.read_text(encoding='utf-8')
        assert "测试" in content
        assert "精确率" in content

    def test_json_preserves_chinese(self, tmp_path):
        """
        目的：验证 JSON 分支同样保留中文（与 save_json 行为一致）
        输入：含中文的字典 + .json 路径
        预期：文件中直接出现中文，不含 unicode 转义序列
        """
        data = {"名称": "测试"}
        path = tmp_path / "result.json"
        save_structured(data, path)

        content = path.read_text(encoding='utf-8')
        assert "测试" in content
        assert "\\u" not in content

    def test_creates_directory(self, tmp_path):
        """
        目的：验证保存时自动创建不存在的多级目录
        输入：嵌套不存在目录的 .yaml 路径
        预期：目录与文件均被创建
        """
        path = tmp_path / "sub" / "deep" / "result.yaml"
        save_structured({"a": 1}, path)
        assert path.exists()

    def test_unsupported_suffix_raises(self, tmp_path):
        """
        目的：验证既非 JSON 也非 YAML 的后缀抛出 ValueError
        输入：.xyz 后缀路径
        预期：抛出 ValueError，提示不支持的结构化输出格式
        """
        path = tmp_path / "result.xyz"
        with pytest.raises(ValueError, match="不支持的结构化输出格式"):
            save_structured({"a": 1}, path)

    def test_yaml_normalizes_numpy_types(self, tmp_path):
        """
        目的：验证 YAML 分支对 numpy 等非原生类型做标准化（JSON 往返），不抛出 RepresenterError
        输入：含 numpy ndarray 的字典（PyYAML 原生不支持 ndarray）
        预期：保存成功不报错；文件可被 yaml.safe_load 正常解析
        """
        data = {"arr": np.array([1.0, 2.0, 3.0])}
        path = tmp_path / "result.yaml"
        # 不应抛出 yaml.representer.RepresenterError
        save_structured(data, path)

        loaded = yaml.safe_load(path.read_text(encoding='utf-8'))
        # ndarray 经 default=str 转为字符串表示，键存在即证明成功落盘
        assert "arr" in loaded

    def test_yaml_block_style_and_key_order(self, tmp_path):
        """
        目的：验证 YAML 输出为 block 风格且保持键插入顺序
        输入：多键字典（故意非字母序排列）
        预期：文件不含 flow 风格的内联大括号；顶层键顺序与输入一致
        """
        data = {"zeta": 1, "alpha": 2, "beta": 3}
        path = tmp_path / "result.yaml"
        save_structured(data, path)

        content = path.read_text(encoding='utf-8')
        # block 风格：键逐行出现，不含内联大括号
        assert "{" not in content
        assert "}" not in content
        # 保持插入顺序：zeta 在 alpha 之前，alpha 在 beta 之前
        assert content.index("zeta") < content.index("alpha")
        assert content.index("alpha") < content.index("beta")

    def test_json_and_yaml_semantically_equal(self, tmp_path):
        """
        目的：验证同一字典分别输出 JSON 与 YAML 后语义完全等价
        输入：同一结构化字典
        预期：两种格式读回的数据相等，且均等于原始字典
        """
        data = {
            "results": {
                "op": {
                    "result": {"f1": 0.85},
                    "main_scores": {"f1": 0.85},
                }
            }
        }
        json_path = tmp_path / "result.json"
        yaml_path = tmp_path / "result.yaml"
        save_structured(data, json_path)
        save_structured(data, yaml_path)

        from_json = json.loads(json_path.read_text(encoding='utf-8'))
        from_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        assert from_json == from_yaml == data


# ============================================================================
# JSON / YAML 等价性专项测试
# ============================================================================
#
# 等价性测试数据集：覆盖 evaluation 结果中可能出现的全部数据形态。
# 每个用例都会被分别写入 .json 和 .yaml，再读回断言三者完全相等，
# 从而证明两种输出格式语义等价。
_EQUIV_CASES = [
    # (标识, 原始数据)
    ("scalar_int", {"n_samples": 100}),
    ("scalar_float", {"f1": 0.85, "far": 0.12}),
    ("negative_float", {"best_fpr": -0.001, "loss": -3.14}),
    ("float_precision", {"v": 0.1 + 0.2}),  # 0.30000000000000004
    ("bool", {"flag": True, "off": False}),
    ("none", {"main_scores": None}),
    ("string", {"op": "binary_classification"}),
    ("chinese", {"名称": "测试", "指标": {"精确率": 0.9}}),
    ("nested_dict", {"results": {"op": {"result": {"f1": 0.85}}}}),
    ("confusion_matrix", {"cm": [[5, 0], [0, 5]]}),
    ("curve_arrays", {
        "thresholds": [0.1, 0.5, 0.9],
        "tpr": [0.0, 0.5, 1.0],
        "fpr": [0.0, 0.1, 0.2],
    }),
    ("deep_nested", {"a": {"b": {"c": {"d": [1, 2, 3]}}}}),
    ("empty_dict", {}),
    ("empty_list", {"arr": []}),
    ("empty_string", {"s": ""}),
    ("mixed", {
        "int": 1, "float": 2.5, "bool": True,
        "none": None, "str": "x", "list": [1, "a", None, True],
    }),
    ("long_array", {"thresholds": [round(i * 0.01, 2) for i in range(50)]}),
    ("yaml_sensitive_chars", {"desc": "a: b#c", "path": "C:\\win\\path"}),
]


class TestJsonYamlEquivalence:
    """JSON 与 YAML 输出的语义等价性专项测试"""

    @pytest.mark.parametrize("case_id, data", _EQUIV_CASES)
    def test_json_yaml_roundtrip_equal(self, case_id, data, tmp_path):
        """
        目的：验证同一数据分别以 .json / .yaml 输出后，读回内容与原始数据完全相等
        输入：多种数据形态（标量、嵌套、列表、None、bool、中文、空容器等）
        预期：json 读回 == yaml 读回 == 原始 data
        说明：YAML 分支内部经 JSON 往返标准化，因此两者语义必然一致
        """
        json_path = tmp_path / f"{case_id}.json"
        yaml_path = tmp_path / f"{case_id}.yaml"
        save_structured(data, json_path)
        save_structured(data, yaml_path)

        from_json = json.loads(json_path.read_text(encoding='utf-8'))
        from_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        assert from_json == from_yaml == data

    @pytest.mark.parametrize("case_id, data", _EQUIV_CASES)
    def test_json_file_parseable_by_yaml(self, case_id, data, tmp_path):
        """
        目的：验证 .json 输出文件可被 YAML 解析器读取（JSON 是 YAML 1.2 的子集）
        输入：多种数据形态
        预期：yaml.safe_load 解析 .json 文件结果 == 原始 data
        说明：下游若用 PyYAML 统一加载，JSON 文件可直接复用，无需区分格式
        """
        json_path = tmp_path / f"{case_id}.json"
        save_structured(data, json_path)

        parsed = yaml.safe_load(json_path.read_text(encoding='utf-8'))
        assert parsed == data

    @pytest.mark.parametrize("upper_suffix", [".JSON", ".YAML", ".YML"])
    def test_uppercase_suffix_equivalent(self, upper_suffix, tmp_path):
        """
        目的：验证大写后缀（.JSON/.YAML/.YML）与小写行为一致（大小写不敏感）
        输入：同一数据 + 大写后缀路径
        预期：大写后缀输出与小写后缀语义等价，读回 == 原始 data
        """
        data = {"f1": 0.85, "confusion_matrix": [[5, 0], [0, 5]]}
        path = tmp_path / f"result{upper_suffix}"
        save_structured(data, path)

        content = path.read_text(encoding='utf-8')
        upper = upper_suffix.lower()
        if upper == '.json':
            assert json.loads(content) == data
        else:
            assert yaml.safe_load(content) == data

    def test_save_json_equals_save_structured_json(self, tmp_path):
        """
        目的：验证 save_structured(.json) 与 save_json 输出完全一致
        输入：同一字典，分别用两种函数保存为 .json
        预期：两文件字节内容完全相同（两者底层均调用 json.dump 同参数）
        """
        data = {"results": {"op": {"f1": 0.85, "arr": [1, 2, 3]}}}
        via_save_json = tmp_path / "a.json"
        via_save_structured = tmp_path / "b.json"
        save_json(data, via_save_json)
        save_structured(data, via_save_structured)

        assert via_save_json.read_bytes() == via_save_structured.read_bytes()

    def test_float_precision_preserved(self, tmp_path):
        """
        目的：验证浮点数精度在 JSON/YAML 两种格式间完全保持一致
        输入：含高精度浮点的字典（0.1+0.2 及 1/3 近似值）
        预期：两种格式读回的 float 值逐位相等（== 比较通过）
        """
        data = {"a": 0.1 + 0.2, "b": 1.0 / 3.0, "c": -2.718281828459045}
        json_path = tmp_path / "f.json"
        yaml_path = tmp_path / "f.yaml"
        save_structured(data, json_path)
        save_structured(data, yaml_path)

        from_json = json.loads(json_path.read_text(encoding='utf-8'))
        from_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        assert from_json["a"] == from_yaml["a"] == data["a"]
        assert from_json["b"] == from_yaml["b"] == data["b"]
        assert from_json["c"] == from_yaml["c"] == data["c"]

    def test_bool_none_semantics_preserved(self, tmp_path):
        """
        目的：验证 bool 与 None 在两种格式下的语义类型保持一致
        输入：含 True/False/None 的字典
        预期：读回后类型与值均与原始一致（bool 非 str，None 非 "null"）
        """
        data = {"t": True, "f": False, "n": None}
        json_path = tmp_path / "bn.json"
        yaml_path = tmp_path / "bn.yaml"
        save_structured(data, json_path)
        save_structured(data, yaml_path)

        from_json = json.loads(json_path.read_text(encoding='utf-8'))
        from_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        assert from_json["t"] is True and from_yaml["t"] is True
        assert from_json["f"] is False and from_yaml["f"] is False
        assert from_json["n"] is None and from_yaml["n"] is None

    def test_key_order_preserved_both_formats(self, tmp_path):
        """
        目的：验证 JSON 与 YAML 均保持字典键的插入顺序（非字母序）
        输入：故意非字母序排列的多键字典
        预期：两种格式读回的键顺序均与原始插入顺序一致
        """
        ordered_keys = ["zeta", "alpha", "middle", "beta"]
        data = {k: i for i, k in enumerate(ordered_keys)}
        json_path = tmp_path / "order.json"
        yaml_path = tmp_path / "order.yaml"
        save_structured(data, json_path)
        save_structured(data, yaml_path)

        from_json = json.loads(json_path.read_text(encoding='utf-8'))
        from_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        assert list(from_json.keys()) == ordered_keys
        assert list(from_yaml.keys()) == ordered_keys

    def test_chinese_not_escaped_both_formats(self, tmp_path):
        """
        目的：验证中文字符在 JSON 与 YAML 中均以原文存储，不产生转义序列
        输入：含中文键值的字典
        预期：两种文件均包含原文中文，且不含反斜杠 u 转义
        """
        data = {"算子": "二分类", "指标": {"精确率": 0.9}}
        json_path = tmp_path / "cn.json"
        yaml_path = tmp_path / "cn.yaml"
        save_structured(data, json_path)
        save_structured(data, yaml_path)

        json_text = json_path.read_text(encoding='utf-8')
        yaml_text = yaml_path.read_text(encoding='utf-8')
        assert "二分类" in json_text and "精确率" in json_text
        assert "二分类" in yaml_text and "精确率" in yaml_text
        assert "\\u" not in json_text
        assert "\\u" not in yaml_text

    def test_full_evaluation_result_equivalence(self, tmp_path):
        """
        目的：验证模拟的完整 evaluation 结果（含 main_scores、嵌套结构、曲线数组、混淆矩阵）
              在 JSON 与 YAML 两种格式下完全等价
        输入：构造一个贴近真实 binary_classification_curve 结果的字典
        预期：json 读回 == yaml 读回 == 原始 data（含 list 字段、标量字段、嵌套）
        """
        data = {
            "results": {
                "binary_classification_curve": {
                    "result": {
                        "n_samples": 100,
                        "auc_roc": 0.9823,
                        "auc_pr": 0.9561,
                        "best_f1": 0.9091,
                        "best_f1_threshold": 0.5,
                        "thresholds": [round(i * 0.1, 2) for i in range(11)],
                        "tpr": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        "fpr": [0.0, 0.05, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.4],
                    },
                    "main_scores": {"auc_roc": 0.9823, "best_f1": 0.9091},
                },
                "binary_classification": {
                    "result": {
                        "n_samples": 100,
                        "tp": 45, "fp": 5, "tn": 48, "fn": 2,
                        "f1": 0.9278, "far": 0.0943,
                        "confusion_matrix": [[48, 5], [2, 45]],
                    },
                    "main_scores": {"f1": 0.9278, "far": 0.0943},
                },
            }
        }
        json_path = tmp_path / "full.json"
        yaml_path = tmp_path / "full.yaml"
        save_structured(data, json_path)
        save_structured(data, yaml_path)

        from_json = json.loads(json_path.read_text(encoding='utf-8'))
        from_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        assert from_json == from_yaml == data


class TestEnsureEncoding:
    """测试 ensure_encoding 编码适配逻辑（覆盖自动探测、reconfigure、TextIOWrapper 兜底等分支）"""

    def test_auto_utf8_noop(self, monkeypatch):
        """
        目的：自动探测模式下 stdout 已兼容 UTF-8 时直接返回，不进入重配置流程
        输入：encoding=None，stdout.encoding='utf-8'
        预期：函数立即返回
        """

        class _Stream:
            encoding = 'utf-8'

        monkeypatch.setattr(sys, 'stdout', _Stream())
        # 不抛异常即代表提前返回
        ensure_encoding(None)

    def test_user_specified_already_match(self, monkeypatch):
        """
        目的：用户指定编码与当前编码一致时跳过重配置
        输入：encoding='utf-8'，stdout/stderr 均为 UTF-8
        预期：不调用 reconfigure
        """

        class _Stream:
            encoding = 'UTF-8'
            reconfigure_called = False

            def reconfigure(self, **kwargs):
                self.reconfigure_called = True

        out = _Stream()
        err = _Stream()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        ensure_encoding('utf-8')
        assert out.reconfigure_called is False
        assert err.reconfigure_called is False

    def test_reconfigure_effective(self, monkeypatch):
        """
        目的：reconfigure 成功改变编码后跳过 TextIOWrapper 兜底
        输入：encoding='gbk'，stdout 当前非 gbk，reconfigure 生效
        预期：调用 reconfigure 且 encoding 变为目标值
        """

        class _Stream:
            def __init__(self):
                self._enc = 'utf-8'
                self.reconfigure_called = False

            @property
            def encoding(self):
                return self._enc

            def reconfigure(self, encoding=None, errors=None):
                self.reconfigure_called = True
                self._enc = encoding

        out = _Stream()
        err = _Stream()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        ensure_encoding('gbk')
        assert out.reconfigure_called is True
        assert out._enc == 'gbk'

    def test_reconfigure_fallback_textiowrapper(self, monkeypatch):
        """
        目的：reconfigure 未生效时用 TextIOWrapper 兜底替换 sys.stdout / sys.stderr
        输入：reconfigure 不改变 encoding，stream 有 buffer
        预期：sys.stdout 与 sys.stderr 均被替换为 TextIOWrapper
        """

        class _Stream:
            encoding = 'latin-1'
            line_buffering = False
            buffer = io.BytesIO()

            def reconfigure(self, encoding=None, errors=None):
                pass  # 故意不生效

        out = _Stream()
        err = _Stream()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        ensure_encoding('gbk')
        assert isinstance(sys.stdout, io.TextIOWrapper)
        assert isinstance(sys.stderr, io.TextIOWrapper)

    def test_reconfigure_exception_caught(self, monkeypatch):
        """
        目的：reconfigure 抛 LookupError 时被捕获，继续走兜底替换
        输入：reconfigure 抛 LookupError，buffer 存在
        预期：不抛异常，sys.stdout 被兜底替换为 TextIOWrapper
        """

        class _Stream:
            encoding = 'latin-1'
            line_buffering = False
            buffer = io.BytesIO()

            def reconfigure(self, encoding=None, errors=None):
                raise LookupError("unknown encoding")

        out = _Stream()
        err = _Stream()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        ensure_encoding('gbk')
        assert isinstance(sys.stdout, io.TextIOWrapper)

    def test_no_reconfigure_method(self, monkeypatch):
        """
        目的：stream 无 reconfigure 方法时跳过该步骤，直接走兜底
        输入：stream 无 reconfigure，有 buffer
        预期：sys.stdout 被兜底替换为 TextIOWrapper
        """

        class _Stream:
            encoding = 'latin-1'
            line_buffering = False
            buffer = io.BytesIO()
            # 故意不定义 reconfigure

        out = _Stream()
        err = _Stream()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        ensure_encoding('gbk')
        assert isinstance(sys.stdout, io.TextIOWrapper)

    def test_no_buffer_skips_fallback(self, monkeypatch):
        """
        目的：stream 无 buffer（buffer=None）时跳过 TextIOWrapper 兜底
        输入：reconfigure 不生效，buffer=None
        预期：不替换 sys.stdout，仍为原 mock 对象
        """

        class _Stream:
            encoding = 'latin-1'
            buffer = None

            def reconfigure(self, encoding=None, errors=None):
                pass

        out = _Stream()
        err = _Stream()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        ensure_encoding('gbk')
        assert sys.stdout is out

    def test_fallback_exception_swallowed(self, monkeypatch):
        """
        目的：兜底阶段抛异常时被彻底吞掉，不向外传播
        输入：访问 buffer 抛 RuntimeError
        预期：不抛异常
        """

        class _Stream:
            encoding = 'latin-1'

            @property
            def buffer(self):
                raise RuntimeError("no buffer")

            def reconfigure(self, encoding=None, errors=None):
                pass

        out = _Stream()
        err = _Stream()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        # 不抛异常即通过
        ensure_encoding('gbk')

    def test_auto_non_utf8_non_win32(self, monkeypatch):
        """
        目的：自动探测模式下非 UTF-8 且非 Windows 时目标设为 utf-8 且不调用 chcp
        输入：encoding=None，stdout.encoding='gbk'，platform='linux'
        预期：reconfigure 被调用，目标为 utf-8
        """

        class _Stream:
            def __init__(self):
                self._enc = 'gbk'
                self.reconfigure_called = False

            @property
            def encoding(self):
                return self._enc

            def reconfigure(self, encoding=None, errors=None):
                self.reconfigure_called = True
                self._enc = encoding

        out = _Stream()
        err = _Stream()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        monkeypatch.setattr(sys, 'platform', 'linux')
        ensure_encoding(None)
        assert out.reconfigure_called is True
        assert out._enc == 'utf-8'

    def test_auto_non_utf8_win32_chcp(self, monkeypatch):
        """
        目的：自动探测模式下非 UTF-8 且 Windows 时执行 chcp 65001
        输入：encoding=None，stdout.encoding='gbk'，platform='win32'
        预期：os.system 被调用且命令含 chcp 65001；目标为 utf-8
        """

        class _Stream:
            def __init__(self):
                self._enc = 'gbk'

            @property
            def encoding(self):
                return self._enc

            def reconfigure(self, encoding=None, errors=None):
                self._enc = encoding

        out = _Stream()
        err = _Stream()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        monkeypatch.setattr(sys, 'platform', 'win32')
        calls = []
        monkeypatch.setattr(os, 'system', lambda cmd: calls.append(cmd))
        ensure_encoding(None)
        assert any('chcp 65001' in c for c in calls)
        assert out._enc == 'utf-8'
