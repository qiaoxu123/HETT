"""Metrics for distance-conditioned scene matching."""
from __future__ import annotations

import numpy as np


DISTANCE_BUCKETS = ((0.0, 20.0, "0-20m"), (20.0, 40.0, "20-40m"),
                    (40.0, 80.0, "40-80m"), (80.0, float("inf"), ">80m"))


def distance_bucket(distance: float) -> str:
    for lower, upper, name in DISTANCE_BUCKETS:
        if lower <= distance < upper:
            return name
    raise AssertionError("unreachable")


def binary_auroc(positive, negative) -> float | None:
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        return None
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0).sum() + 0.5 * (delta == 0).sum()) / delta.size)


def paired_accuracy(positive, negative) -> float | None:
    positive, negative = np.asarray(positive), np.asarray(negative)
    return float(np.mean(positive > negative) + 0.5 * np.mean(positive == negative)) if len(positive) else None


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def spearman(x, y) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_pairs(rows: list[dict]) -> dict:
    def flat(selected):
        positive = [row["positive_score"] for row in selected]
        negative = [row["negative_score"] for row in selected]
        return {
            "samples": len(selected),
            "positive_mean": float(np.mean(positive)) if selected else None,
            "negative_mean": float(np.mean(negative)) if selected else None,
            "positive_negative_margin": float(np.mean(np.asarray(positive) - np.asarray(negative))) if selected else None,
            "auroc": binary_auroc(positive, negative),
            "same_map_hard_negative_accuracy": paired_accuracy(positive, negative),
            "spearman_score_vs_distance": spearman([row["distance_m"] for row in selected], positive),
        }
    positive = [row["positive_score"] for row in rows]
    negative = [row["negative_score"] for row in rows]
    result = flat(rows)
    result["distance_buckets"] = {}
    for _, _, name in DISTANCE_BUCKETS:
        selected = [row for row in rows if distance_bucket(row["distance_m"]) == name]
        result["distance_buckets"][name] = flat(selected)
    return result
