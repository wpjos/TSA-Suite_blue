# -*- coding: utf-8 -*-

"""
VUS / Range-AUC 共享辅助函数

迁移自 bqlib ``scorers.anomaly.range_based`` 的内部辅助函数，
供 :mod:`vus_roc_metric` 与 :mod:`vus_pr_metric` 共享。
"""

import numpy as np


def events_inclusive(vec: np.ndarray, pos_label: int = 1) -> list[tuple[int, int]]:
    """0/1 向量 → 闭区间 ``[start, end]``（end 是最后一个正例的下标）。

    Args:
        vec: 标签数组
        pos_label: 正例标签值

    Returns:
        事件列表，每个元素 ``(start, end)`` 都是包含性索引。
    """
    arr = np.asarray(vec)
    events: list[tuple[int, int]] = []
    in_event = False
    start = 0

    for i, label in enumerate(arr):
        is_pos = label == pos_label
        if is_pos and not in_event:
            start = i
            in_event = True
        elif not is_pos and in_event:
            events.append((start, i - 1))
            in_event = False

    if in_event:
        events.append((start, len(arr) - 1))

    return events


def _extend_label_with_decay(
    label_bin: np.ndarray,
    window: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """对二值异常段做 sqrt 衰减扩展，并返回扩展后的连续区间列表。

    Args:
        label_bin: 0/1 数组，shape (N,)
        window: 扩展窗口大小（采样点）；0 表示不扩展

    Returns:
        (extended, segments)：
        - extended：原始标签 + 衰减填充，clip 到 [0, 1] 的 float 数组
        - segments：扩展后的合并区间 [(start, end), ...]
    """
    n = len(label_bin)
    extended = label_bin.astype(float)
    raw = events_inclusive(label_bin)

    if window <= 0 or not raw:
        return extended, raw

    half = window // 2
    for s, e in raw:
        for x in range(e + 1, min(e + half + 1, n)):
            extended[x] = max(extended[x], np.sqrt(1 - (x - e) / window))
        for x in range(max(s - half, 0), s):
            extended[x] = max(extended[x], np.sqrt(1 - (s - x) / window))
    np.minimum(extended, 1.0, out=extended)

    padded = [(max(0, s - half), min(n - 1, e + half)) for s, e in raw]
    merged: list[tuple[int, int]] = []
    for seg in padded:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)
    return extended, merged


def _curve_at_window(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    window: int,
    num_thresholds: int,
) -> tuple[float, float]:
    """在固定 window 下扫描阈值，返回 (ROC-AUC, PR-AUC)。

    Args:
        y_true_bin: 0/1 真实标签数组，shape (N,)
        y_score: 连续异常得分，shape (N,)，越高越异常
        window: 扩展窗口大小（采样点）；0 表示不扩展
        num_thresholds: 阈值采样点数

    Returns:
        (auc_roc, auc_pr)：固定窗口下的 ROC-AUC 与 PR-AUC
    """
    p_orig = int(np.sum(y_true_bin))
    n = len(y_true_bin)
    if p_orig == 0 or n == 0:
        return 0.0, 0.0

    extended, segments = _extend_label_with_decay(y_true_bin, window)
    p_new = (p_orig + float(np.sum(extended))) / 2.0
    n_new = n - p_new
    if n_new <= 0 or not segments:
        return 0.0, 0.0

    score_sorted = np.sort(y_score)[::-1]
    idx = np.unique(np.round(np.linspace(0, n - 1, num_thresholds)).astype(int))
    idx = idx[idx < n]

    k = len(idx)
    tf = np.zeros((k + 2, 2))
    prec = np.ones(k + 1)
    j = 0
    for i in idx:
        threshold = score_sorted[i]
        pred = y_score >= threshold
        product = extended * pred
        tp = float(np.sum(product))
        fp = float(np.sum(pred) - tp)

        existence = 0
        for seg in segments:
            if np.any(product[seg[0]: seg[1] + 1] > 0):
                existence += 1
        existence_ratio = existence / len(segments)

        recall = min(tp / p_new, 1.0) if p_new > 0 else 0.0
        tpr = recall * existence_ratio
        fpr = fp / n_new if n_new > 0 else 0.0
        precision = tp / float(np.sum(pred)) if np.any(pred) else 0.0

        j += 1
        tf[j] = (tpr, fpr)
        prec[j] = precision

    tf[j + 1] = (1.0, 1.0)

    width = tf[1:, 1] - tf[:-1, 1]
    height = (tf[1:, 0] + tf[:-1, 0]) / 2.0
    auc_roc = float(np.dot(width, height))

    width_pr = tf[1:-1, 0] - tf[:-2, 0]
    height_pr = (prec[1:] + prec[:-1]) / 2.0
    auc_pr = float(np.dot(width_pr, height_pr))
    return auc_roc, auc_pr