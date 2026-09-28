"""Metrics for landmark-arrival understanding.

These helpers intentionally avoid sklearn so they can run in the existing
CityNav environment. Inputs are one-dimensional probabilities and binary labels.
"""
from __future__ import annotations

import numpy as np


def _arrays(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    return p, y


def binary_auroc(probabilities, labels):
    p, y = _arrays(probabilities, labels)
    pos = y == 1
    neg = y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1, dtype=np.float64)
    # Tie-aware average ranks.
    values, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == idx
            ranks[mask] = ranks[mask].mean()
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def binary_auprc(probabilities, labels):
    p, y = _arrays(probabilities, labels)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[ys == 1].sum() / n_pos)


def threshold_metrics(probabilities, labels, threshold=0.5):
    p, y = _arrays(probabilities, labels)
    pred = p >= threshold
    pos = y == 1
    neg = ~pos
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    fn = int(np.sum((~pred) & pos))
    tn = int(np.sum((~pred) & neg))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y), 1)
    fpr = fp / max(fp + tn, 1)
    return dict(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fpr=float(fpr),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def expected_calibration_error(probabilities, labels, bins=10):
    p, y = _arrays(probabilities, labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if not mask.any():
            continue
        confidence = p[mask].mean()
        accuracy = y[mask].mean()
        ece += mask.mean() * abs(confidence - accuracy)
    return float(ece)


def summarize_arrival(probabilities, labels, threshold=0.5, bins=10):
    result = threshold_metrics(probabilities, labels, threshold=threshold)
    result["auroc"] = binary_auroc(probabilities, labels)
    result["auprc"] = binary_auprc(probabilities, labels)
    result["ece"] = expected_calibration_error(probabilities, labels, bins=bins)
    return result
