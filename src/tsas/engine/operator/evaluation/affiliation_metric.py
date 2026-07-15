# -*- coding: utf-8 -*-

"""
Affiliation 评价指标算子（P/R/F1）

Paparrizos et al. affiliation-metrics 移植：每个真值段 J 定义"归属带"
E_J = (上一段尾中点, 下一段头中点)；在 E_J 内，预测点按距 J 边界的距离
衰减贡献（越近越满，越远越少）。Precision / Recall = 所有真值段积分
的归一化均值；F1 = 调和平均。

核心组件:
    - AffiliationResult: 隶属 P/R/F1 结果（Pydantic BaseModel）
    - AffiliationConfig: 配置类（继承 BaseMetricConfig）
    - AffiliationMetric: Affiliation P/R/F1 指标算子

使用示例::

    from tsas.engine.operator.evaluation import AffiliationMetric

    op = AffiliationMetric()
    result = op.run((y_truth, y_pred))
    print(result.f1)
"""

import math
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.evaluation._vus_utils import events_inclusive
from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'AffiliationResult',
    'AffiliationConfig',
    'AffiliationMetric',
]


class AffiliationResult(BaseModel):
    """Affiliation P/R/F1 评价指标结果

    Attributes:
        affiliation_precision (float): 隶属 Precision（归属带距离衰减得分）
        affiliation_recall (float): 隶属 Recall（归属带距离衰减得分）
        affiliation_f1 (float): 隶属 F1（precision/recall 的调和平均）
    """
    model_config = ConfigDict(frozen=True)

    affiliation_precision: float = Field(description="隶属 Precision（归属带距离衰减得分）")
    affiliation_recall: float = Field(description="隶属 Recall（归属带距离衰减得分）")
    affiliation_f1: float = Field(description="隶属 F1（precision/recall 的调和平均）")


class AffiliationConfig(BaseMetricConfig):
    """Affiliation 评价指标配置

    Attributes:
        pos_label (int): 正例标签值，默认 1
        main_scores (dict[str, str] | None): 主评分路径映射，默认
            ``{"affiliation_precision": "precision", "affiliation_recall": "recall",
            "affiliation_f1": "f1"}``
    """
    model_config = ConfigDict(frozen=True)

    pos_label: int = Field(default=1, description="正例标签值，默认 1")
    main_scores: dict[str, str] | None = Field(
        default={
            "affiliation_precision": "affiliation_precision",
            "affiliation_recall": "affiliation_recall",
            "affiliation_f1": "affiliation_f1",
        },
        description="主评分路径映射；float 字段使用对应属性名",
    )


class AffiliationMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        AffiliationResult,
        AffiliationConfig,
        None,
    ],
):
    """Affiliation P/R/F1 评价指标算子

    把 ``y_truth`` / ``y_pred`` 中的 ``pos_label`` 连续段视为事件，
    按 affiliation 公式计算软事件级 Precision / Recall / F1。
    与 event_based_f1 的区别：对每个真值段定义"归属带" E_J，
    在 E_J 内但落在 J 外的预测点按距 J 边界的距离衰减贡献，
    越靠近真值段权重越大。

    Input:
        y_truth: 真实标签数组（一维离散）
        y_pred: 预测标签数组，与 y_truth 等长

    Output:
        AffiliationResult: 隶属 Precision / Recall / F1

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_pred)
        MR: AffiliationResult — 隶属 P/R/F1 结果
        MC: AffiliationConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "affiliation_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> AffiliationResult:
        y_true, y_pred = x
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true 和 y_pred 长度不一致: {len(y_true)} vs {len(y_pred)}"
            )
        config = self.config
        pos_label = 1 if config is None else config.pos_label

        y_true_bin = (y_true == pos_label).astype(int)
        y_pred_bin = (y_pred == pos_label).astype(int)
        if int(np.sum(y_true_bin)) == 0:
            return AffiliationResult(
                affiliation_precision=0.0,
                affiliation_recall=0.0,
                affiliation_f1=0.0,
            )

        events_gt = events_inclusive(y_true_bin, pos_label=1)
        events_pred = events_inclusive(y_pred_bin, pos_label=1)
        if not events_gt or not events_pred:
            return AffiliationResult(
                affiliation_precision=0.0,
                affiliation_recall=0.0,
                affiliation_f1=0.0,
            )

        trange = (0, len(y_true_bin))
        e_gts = _all_e_gt(events_gt, trange)
        aff = _affiliation_partition(events_pred, e_gts)
        p_list = [
            _affiliation_precision_proba(aff[j], events_gt[j], e_gts[j])
            for j in range(len(events_gt))
        ]
        r_list = [
            _affiliation_recall_proba(aff[j], events_gt[j], e_gts[j])
            for j in range(len(events_gt))
        ]
        p_clean = [p for p in p_list if not math.isnan(p)]
        precision = float(sum(p_clean) / len(p_clean)) if p_clean else float("nan")
        recall = float(sum(r_list) / len(r_list)) if r_list else 0.0
        if math.isnan(precision):
            precision = 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return AffiliationResult(
            affiliation_precision=float(precision),
            affiliation_recall=float(recall),
            affiliation_f1=float(f1),
        )


# ─── affiliation zone 计算 ─────────────────────────────────


def _t_start(j: int, Js: list[tuple[int, int]], Trange: tuple[int, int]) -> int:
    """返回真值段 ``Js[j]`` 的起始索引（边界外推）。"""
    b = max(Trange)
    n = len(Js)
    if j == n:
        return 2 * b - _t_stop(n - 1, Js, Trange)
    return Js[j][0]


def _t_stop(j: int, Js: list[tuple[int, int]], Trange: tuple[int, int]) -> int:
    """返回真值段 ``Js[j]`` 的结束索引（边界外推）。"""
    if j == -1:
        return 2 * min(Trange) - _t_start(0, Js, Trange)
    return Js[j][1]


def _e_gt(
    j: int, Js: list[tuple[int, int]], Trange: tuple[int, int]
) -> tuple[float, float]:
    """计算真值段 ``Js[j]`` 的归属带左右端点。"""
    left = (_t_stop(j - 1, Js, Trange) + _t_start(j, Js, Trange)) / 2
    right = (_t_stop(j, Js, Trange) + _t_start(j + 1, Js, Trange)) / 2
    return (left, right)


def _all_e_gt(
    Js: list[tuple[int, int]], Trange: tuple[int, int]
) -> list[tuple[float, float]]:
    """计算所有真值段的归属带。"""
    return [_e_gt(j, Js, Trange) for j in range(len(Js))]


def _affiliation_partition(
    Is: list[tuple[int, int]], E_gts: list[tuple[float, float]]
) -> list[list[tuple[int, int] | None]]:
    """把每个 I 沿 E_gt 分段切割，返回 seg_idx → 各 I 与 E 的交集。"""
    out: list[list[tuple[int, int] | None]] = []
    for E in E_gts:
        cuts: list[tuple[int, int] | None] = []
        for I in Is:
            if I[1] < E[0] or I[0] > E[1]:
                cuts.append(None)
            else:
                a = max(I[0], E[0])
                b = min(I[1], E[1])
                if a < b:
                    cuts.append((a, b))
                else:
                    cuts.append(None)
        out.append(cuts)
    return out


# ─── interval helpers ─────────────────────────────────────


def _interval_length(J) -> int:
    return 0 if J is None else J[1] - J[0]


def _sum_lengths(Is) -> int:
    return sum(_interval_length(I) for I in Is)


def _intersection(I, J):
    if I is None or J is None:
        return None
    a = max(I[0], J[0])
    b = min(I[1], J[1])
    if a < b:
        return (a, b)
    return None


def _interval_subset(I, J) -> bool:
    return J[0] <= I[0] and I[1] <= J[1]


def _cut_into_three(I, J):
    if I is None:
        return (None, None, None)
    inter = _intersection(I, J)
    if inter is not None and inter[0] == I[0] and inter[1] == I[1]:
        return (None, I, None)
    if I[1] <= J[0]:
        return (I, None, None)
    if I[0] >= J[1]:
        return (None, None, I)
    if I[0] <= J[0] and I[1] >= J[1]:
        before = (I[0], inter[0]) if inter is not None and inter[0] > I[0] else None
        after = (inter[1], I[1]) if inter is not None and inter[1] < I[1] else None
        return (before, inter, after)
    if I[0] <= J[0]:
        before = (I[0], inter[0]) if inter is not None and inter[0] > I[0] else None
        middle = inter if inter is not None and inter[1] > inter[0] else None
        return (before, middle, None)
    if I[1] >= J[1]:
        middle = inter if inter is not None and inter[1] > inter[0] else None
        after = (inter[1], I[1]) if inter is not None and inter[1] < I[1] else None
        return (None, middle, after)
    return (None, inter, None)


# ─── CDF-style probability integrals ──────────────────────


def _integral_mini_interval_P_CDFmethod(I, J, E) -> float:
    assert _intersection(I, J) is None, "I and J should have void intersection"
    assert _interval_subset(J, E), "J ⊄ E"
    assert _interval_subset(I, E), "I ⊄ E"

    e_min, e_max = E[0], E[1]
    j_min, j_max = J[0], J[1]
    i_min, i_max = I[0], I[1]

    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    m = min(j_min - e_min, e_max - j_max)
    a = min(d_max, m) ** 2 - min(d_min, m) ** 2
    b = max(d_max, m) - max(d_min, m)
    return 0.5 * a + m * b


def _integral_mini_interval_Pprecision_CDFmethod(I, J, E) -> float:
    e_min, e_max = E[0], E[1]
    j_min, j_max = J[0], J[1]
    i_min, i_max = I[0], I[1]

    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    integral_min_piece = _integral_mini_interval_P_CDFmethod(I, J, E)
    integral_linear_piece = 0.5 * (d_max ** 2 - d_min ** 2)
    integral_remaining_piece = (j_max - j_min) * (i_max - i_min)
    delta_I = i_max - i_min
    delta_E = e_max - e_min
    return delta_I - (1 / delta_E) * (
        integral_min_piece + integral_linear_piece + integral_remaining_piece
    )


def _integral_interval_probaCDF_precision(I, J, E) -> float:
    if I is None:
        return 0
    left, middle, right = _cut_into_three(I, J)

    def _side(I_cut):
        return (
            _integral_mini_interval_Pprecision_CDFmethod(I_cut, J, E)
            if I_cut is not None
            else 0
        )

    def _middle(I_mid):
        if I_mid is None:
            return 0
        return max(I_mid) - min(I_mid)

    return _side(left) + _middle(middle) + _side(right)


def _integral_mini_interval_Precall_CDFmethod(I, J, E) -> float:
    if I[1] <= J[0]:
        i_pivot = I[1]
    elif I[0] >= J[1]:
        i_pivot = I[0]
    else:
        raise ValueError("I should be outside J")

    e_min, e_max = E[0], E[1]
    if i_pivot <= e_min or i_pivot >= e_max:
        return 0
    e_mean = (e_min + e_max) / 2

    def _split_J(J_, e_mean_):
        if J_ is None:
            return (None, None)
        if e_mean_ >= J_[1]:
            return (J_, None)
        if e_mean_ <= J_[0]:
            return (None, J_)
        return ((J_[0], e_mean_), (e_mean_, J_[1]))

    J_before, J_after = _split_J(J, e_mean)
    iemin_mean = (e_min + i_pivot) / 2
    Jb_closeE, Jb_closeI = _split_J(J_before, iemin_mean)
    iemax_mean = (e_max + i_pivot) / 2
    Ja_closeI, Ja_closeE = _split_J(J_after, iemax_mean)

    def _span(x):
        return (math.nan, math.nan) if x is None else (x[0], x[1])

    j_bb_min, j_bb_max = _span(Jb_closeE)
    j_ba_min, j_ba_max = _span(Jb_closeI)
    j_ab_min, j_ab_max = _span(Ja_closeI)
    j_aa_min, j_aa_max = _span(Ja_closeE)

    def _len_span(a, b):
        if math.isnan(a) or math.isnan(b):
            return 0
        return b - a

    if i_pivot >= J[1]:
        part1 = (i_pivot - e_min) * _len_span(j_bb_min, j_bb_max)
        part2 = 2 * i_pivot * _len_span(j_ba_min, j_ba_max) - (j_ba_max ** 2 - j_ba_min ** 2)
        part3 = 2 * i_pivot * _len_span(j_ab_min, j_ab_max) - (j_ab_max ** 2 - j_ab_min ** 2)
        part4 = (e_max + i_pivot) * _len_span(j_aa_min, j_aa_max) - (
            j_aa_max ** 2 - j_aa_min ** 2
        )
    else:
        part1 = (j_bb_max ** 2 - j_bb_min ** 2) - (
            e_min + i_pivot
        ) * _len_span(j_bb_min, j_bb_max)
        part2 = (j_ba_max ** 2 - j_ba_min ** 2) - 2 * i_pivot * _len_span(
            j_ba_min, j_ba_max
        )
        part3 = (j_ab_max ** 2 - j_ab_min ** 2) - 2 * i_pivot * _len_span(
            j_ab_min, j_ab_max
        )
        part4 = (e_max - i_pivot) * _len_span(j_aa_min, j_aa_max)
    parts = [part1, part2, part3, part4]
    valid = [p for p in parts if not math.isnan(p)]
    out = sum(valid)

    delta_J = J[1] - J[0]
    delta_E = e_max - e_min
    return delta_J - (1 / delta_E) * out


def _integral_interval_probaCDF_recall(I, J, E) -> float:
    if J is None:
        return 0
    left, middle, right = _cut_into_three(J, I)

    def _side(J_cut):
        return (
            _integral_mini_interval_Precall_CDFmethod(I, J_cut, E)
            if J_cut is not None
            else 0
        )

    def _middle(J_mid):
        if J_mid is None:
            return 0
        return max(J_mid) - min(J_mid)

    return _side(left) + _middle(middle) + _side(right)


# ─── per-event affiliation metrics ────────────────────────


def _affiliation_precision_proba(Is, J, E) -> float:
    """真值段 ``J`` 在归属带 ``E`` 内的 precision 概率（与 ``Is`` 中所有预测段整合）。"""
    if all(I is None for I in Is):
        return math.nan
    return sum(_integral_interval_probaCDF_precision(I, J, E) for I in Is) / _sum_lengths(
        [I for I in Is if I is not None]
    )


def _affiliation_recall_proba(Is, J, E) -> float:
    """真值段 ``J`` 在归属带 ``E`` 内的 recall 概率（按预测段 ``Is`` 整合）。"""
    Is_nn = [I for I in Is if I is not None]
    if not Is_nn:
        return 0
    if J[1] == J[0]:
        return 0
    E_gt_recall = _all_e_gt(Is_nn, E)
    Js = _affiliation_partition([J], E_gt_recall)
    return sum(_integral_interval_probaCDF_recall(I, J[0], E) for I, J in zip(Is_nn, Js)) / (
        J[1] - J[0]
    )