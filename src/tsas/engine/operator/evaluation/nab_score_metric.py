# -*- coding: utf-8 -*-

"""
NAB Score 评价指标算子（Numenta Anomaly Benchmark）

移植 NAB ``Sweeper`` 的打分逻辑：
- 每个标注异常点扩展为对称窗口（默认 ``window_length = 0.1 * N / n_anom``）
- 窗口内点用 sigmoid 加权 TP 分（越靠左分越高）
- 窗口外点用 sigmoid 加权 FP 分（距上一窗口越远惩罚越小）
- 前 ``probation_percent`` 比例（默认 15%）作 probaton 期，不参与打分

支持的 application profile：
- ``standard``（tp=1.0, fp=0.11, fn=1.0）：NAB 默认
- ``reward_low_FP_rate``（fp=0.22）：更重 FP 惩罚
- ``reward_low_FN_rate``（fn=2.0）：更重 FN 惩罚

核心组件:
    - NabScoreResult: NAB 打分结果（Pydantic BaseModel）
    - NabScoreConfig: 配置类（继承 BaseMetricConfig）
    - NabScoreMetric: NAB 打分指标算子

使用示例::

    from tsas.engine.operator.evaluation import NabScoreMetric

    op = NabScoreMetric(threshold=0.5)
    result = op.run((y_truth, y_score))
    print(result.score)
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
    'NabScoreResult',
    'NabScoreConfig',
    'NabScoreMetric',
]


NAB_PROFILES: dict[str, dict[str, float]] = {
    "standard": {"tp_weight": 1.0, "fp_weight": 0.11, "fn_weight": 1.0},
    "reward_low_FP_rate": {"tp_weight": 1.0, "fp_weight": 0.22, "fn_weight": 1.0},
    "reward_low_FN_rate": {"tp_weight": 1.0, "fp_weight": 0.11, "fn_weight": 2.0},
}

DEFAULT_WINDOW_SIZE_RATIO = 0.1
DEFAULT_PROBATION_PERCENT = 0.15


class NabScoreResult(BaseModel):
    """NAB 打分结果

    Attributes:
        nab_score (float): NAB score（无正负规范化，越大越好）
        profile (str): 使用的 NAB profile 名称
        threshold (float): 实际使用的决策阈值
    """
    model_config = ConfigDict(frozen=True)

    nab_score: float = Field(description="NAB score（无正负规范化，越大越好）")
    profile: str = Field(description="使用的 NAB profile 名称")
    threshold: float = Field(description="实际使用的决策阈值")


class NabScoreConfig(BaseMetricConfig):
    """NAB 打分评价指标配置

    Attributes:
        threshold (float | None): 二值化阈值；``None`` 时取 ``max(y_score)/2``
        window_length (int | None): 窗口半宽；``None`` 时按 NAB 默认推算
        probation_percent (float): probaton 期比例，默认 0.15
        profile (str): NAB profile 名称，默认 ``'standard'``
        main_scores (dict[str, str] | None): 主评分路径映射，默认 ``{"nab_score": "score"}``
    """
    model_config = ConfigDict(frozen=True)

    threshold: float | None = Field(
        default=None,
        description="二值化阈值；None 时取 max(y_score)/2",
    )
    window_length: int | None = Field(
        default=None,
        description="窗口半宽；None 时按 NAB 默认 0.1*N/n 推算",
    )
    probation_percent: float = Field(
        default=DEFAULT_PROBATION_PERCENT,
        description="probaton 期比例（默认 0.15）",
    )
    profile: str = Field(
        default="standard",
        description="NAB profile：standard / reward_low_FP_rate / reward_low_FN_rate",
    )
    main_scores: dict[str, str] | None = Field(
        default={"nab_score": "nab_score"},
        description="主评分路径映射；float 字段使用对应属性名",
    )


class NabScoreMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        NabScoreResult,
        NabScoreConfig,
        None,
    ],
):
    """NAB Score 评价指标算子

    Input:
        y_truth: 0/1 真值标签向量，shape (N,)
        y_score: 连续异常得分（越大越异常），shape (N,)

    Output:
        NabScoreResult: NAB 总分、使用的 profile、实际阈值

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_score)
        MR: NabScoreResult — NAB 结果
        MC: NabScoreConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "nab_score_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> NabScoreResult:
        y_true, y_score = x
        y_true = np.asarray(y_true).astype(int).ravel()
        y_score = np.asarray(y_score, dtype=float).ravel()
        if y_true.shape != y_score.shape:
            raise ValueError(
                f"y_true 和 y_score 形状不一致: {y_true.shape} vs {y_score.shape}"
            )

        config = self.config
        threshold = config.threshold if config is not None else None
        window_length = config.window_length if config is not None else None
        probation_percent = (
            config.probation_percent if config is not None else DEFAULT_PROBATION_PERCENT
        )
        profile = config.profile if config is not None else "standard"

        if profile not in NAB_PROFILES:
            raise ValueError(
                f"Unknown profile {profile!r}; choose from {list(NAB_PROFILES)}"
            )
        profile_weights = NAB_PROFILES[profile]

        n = len(y_true)
        if n == 0:
            return NabScoreResult(score=0.0, profile=profile, threshold=0.0)

        events = events_inclusive(y_true, pos_label=1)
        windows = _compute_windows(events, n, window_length=window_length)

        probation_length = min(
            int(math.floor(probation_percent * n)),
            int(probation_percent * 5000),
        )

        sweep, names = _sweep_scores(n, windows, profile_weights)
        names = _assign_scorable_names(names, windows, probation_length)

        if threshold is None:
            threshold = float(y_score.max() / 2.0) if y_score.size else 0.5
        curve = _score_curve(sweep, names, y_score, profile_weights)
        total, _ = _score_at_threshold(curve, threshold)

        return NabScoreResult(
            nab_score=float(total),
            profile=profile,
            threshold=float(threshold),
        )


def _sigmoid(x: float) -> float:
    """标准 sigmoid 函数。"""
    return 1.0 / (1.0 + math.exp(-x))


def _scaled_sigmoid(relative_position: float) -> float:
    """按 NAB 规则把相对位置映射到 ``[-1, 1]``。"""
    return 2.0 * _sigmoid(-5.0 * relative_position) - 1.0


def _max_tp_score() -> float:
    """最左端正样本（``relative_position = -1``）对应的 unweighted score。"""
    return _scaled_sigmoid(-1.0)


def _compute_windows(
    events: list[tuple[int, int]],
    length: int,
    window_length: int | None = None,
) -> list[tuple[int, int]]:
    """按 NAB ``applyWindows`` 规则为每个 anomaly event 构造对称窗口。"""
    n = len(events)
    if n == 0 or length == 0:
        return []
    if window_length is None:
        window_length = max(1, int(DEFAULT_WINDOW_SIZE_RATIO * length / n))
    out: list[tuple[int, int]] = []
    for start, end in events:
        center = (start + end) // 2
        half = max(window_length // 2, 1)
        left = max(center - half, 0)
        right = min(center + half, length - 1)
        out.append((int(left), int(right)))
    return out


def _resolve_window_at(
    i: int, windows: list[tuple[int, int]]
) -> tuple[int, tuple[int, int]] | None:
    """返回 ``(window_index, (left, right))`` 若 ``i`` 命中某窗口；否则 ``None``。"""
    for k, (left, right) in enumerate(windows):
        if left <= i <= right:
            return k, (left, right)
    return None


def _sweep_scores(
    length: int,
    windows: list[tuple[int, int]],
    profile: dict[str, float],
) -> tuple[np.ndarray, list[str | None]]:
    """计算每个点的 sweep score（含正负）和 ``windowName``。"""
    tp_weight = profile["tp_weight"]
    fp_weight = profile["fp_weight"]
    max_tp = _max_tp_score()

    scores = np.zeros(length)
    names: list[str | None] = [None] * length

    prev_right_index: int | None = None
    prev_width: int | None = None

    for i in range(length):
        in_window = _resolve_window_at(i, windows)
        if in_window is not None:
            _, (left, right) = in_window
            width = right - left + 1
            position_in_window = -(right - i + 1) / width
            unweighted = _scaled_sigmoid(position_in_window)
            scores[i] = unweighted * tp_weight / max_tp
            names[i] = f"window|{left}"
            prev_right_index = right
            prev_width = width
        else:
            if prev_right_index is None or prev_width is None:
                unweighted = -1.0
            else:
                numerator = abs(prev_right_index - i)
                denominator = float(prev_width - 1)
                position_past = numerator / denominator
                unweighted = _scaled_sigmoid(position_past)
            scores[i] = unweighted * fp_weight
            names[i] = None
    return scores, names


def _assign_scorable_names(
    names: list[str | None],
    windows: list[tuple[int, int]],
    probation_length: int,
) -> list[str | None]:
    """probaton 期内样本标记为 ``'probationary'``，NAB 排序时会跳过。"""
    out: list[str | None] = []
    for i, n in enumerate(names):
        if i < probation_length:
            out.append("probationary")
        else:
            out.append(n)
    return out


def _score_curve(
    sweep: np.ndarray,
    names: list[str | None],
    y_score: np.ndarray,
    profile_weights: dict[str, float],
) -> list[tuple[float | None, float, dict[str, float]]]:
    """计算"snapshot at threshold = T" 的 ``score_curve``。"""
    fn_weight = profile_weights["fn_weight"]
    parts: dict[str, float] = {"fp": 0.0}
    n = len(sweep)
    if n == 0:
        return [(None, 0.0, {"fp": 0.0})]

    for i in range(n):
        wn = names[i]
        if wn is not None and wn != "probationary" and wn not in parts:
            parts[wn] = -fn_weight
    init_score = sum(parts.values())
    curve: list[tuple[float | None, float, dict[str, float]]] = [
        (None, init_score, dict(parts))
    ]

    order = np.argsort(-y_score, kind="stable")
    for k, idx in enumerate(order):
        s = float(y_score[idx])
        wn = names[idx]
        if wn != "probationary":
            if wn is None:
                parts["fp"] += float(sweep[idx])
            else:
                parts[wn] = max(parts[wn], float(sweep[idx]))
        is_last_of_score = (k + 1 == n) or (
            float(y_score[order[k + 1]]) != s
        )
        if is_last_of_score:
            curve.append((s, sum(parts.values()), dict(parts)))
    return curve


def _score_at_threshold(
    curve: list[tuple[float | None, float, dict[str, float]]],
    threshold: float,
) -> tuple[float, dict[str, float]]:
    """``curve`` 中找阈值 ``threshold`` 对应的快照与 parts dict。"""
    if not curve:
        return 0.0, {"fp": 0.0}

    init_entry = curve[0] if curve[0][0] is None else None
    body = curve[1:] if init_entry is not None else curve
    prev = init_entry if init_entry is not None else body[0]

    for entry in body:
        t = entry[0]
        if t < threshold:
            return float(prev[1]), prev[2]
        prev = entry
    return float(prev[1]), prev[2] if len(prev) > 2 else {"fp": prev[1]}