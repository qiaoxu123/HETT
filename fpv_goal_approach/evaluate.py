from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def binary_metrics(y_true, logits):
    y_pred = logits.argmax(1)
    probabilities = np.exp(logits - logits.max(1, keepdims=True))
    probabilities = probabilities[:, 1] / probabilities.sum(1)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    result["auroc"] = roc_auc_score(y_true, probabilities) if len(set(y_true)) == 2 else None
    return result


def multiclass_metrics(y_true, logits):
    y_pred = logits.argmax(1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_predictions(labels, logits, metadata):
    report = {
        "match": binary_metrics(labels["target_match"], logits["target_match"]),
        "arrival": binary_metrics(labels["arrival"], logits["arrival"]),
        "distance": multiclass_metrics(labels["distance_bin"], logits["distance_bin"]),
        "bearing": multiclass_metrics(labels["bearing_bin"], logits["bearing_bin"]),
        "phase": multiclass_metrics(labels["phase"], logits["phase"]),
    }
    arrival_pred = logits["arrival"].argmax(1)
    arrival_true = labels["arrival"]
    negatives = arrival_true == 0
    report["arrival"]["false_positive_rate"] = float((arrival_pred[negatives] == 1).mean()) if negatives.any() else 0.0
    buckets = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, float("inf")))
    report["distance_buckets"] = {}
    for low, high in buckets:
        indices = np.array([low <= row["distance_to_goal_m"] < high for row in metadata])
        if not indices.any():
            continue
        name = f"{low:g}-{high:g}m"
        bucket_arrival = arrival_true[indices]
        bucket_pred = arrival_pred[indices]
        neg = bucket_arrival == 0
        report["distance_buckets"][name] = {
            "count": int(indices.sum()),
            "arrival_accuracy": accuracy_score(bucket_arrival, bucket_pred),
            "arrival_fp_rate": float((bucket_pred[neg] == 1).mean()) if neg.any() else 0.0,
            "match_accuracy": accuracy_score(labels["target_match"][indices], logits["target_match"][indices].argmax(1)),
        }
    report["hard_negatives"] = {}
    for negative_type in ("near_goal", "wrong_view", "wrong_language"):
        indices = np.array([row["negative_type"] == negative_type for row in metadata])
        if indices.any():
            neg = arrival_true[indices] == 0
            report["hard_negatives"][negative_type] = {
                "count": int(indices.sum()),
                "match_accuracy": accuracy_score(labels["target_match"][indices], logits["target_match"][indices].argmax(1)),
                "arrival_false_positive_rate": float((arrival_pred[indices][neg] == 1).mean()) if neg.any() else 0.0,
            }
    return report


def save_report(report, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(report, handle, indent=2)
