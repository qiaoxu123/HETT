#!/usr/bin/env python3
"""Compare progress-stop controller arms and audit 20 m calibration."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


# Prediction files contain ``multiagent.space`` named tuples.  When this file is
# executed as ``scripts/analyze_progress_stop.py``, Python otherwise only adds
# ``scripts/`` to the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ARMS = {
    "control_095": "progress_stop_095_s0",
    "gated_080": "progress_stop_080_gated_s0",
    "ungated_080": "progress_stop_080_ungated_s0",
}
METRICS = ("sr", "spl", "ne", "oracle_sr", "lengths")


def xy(point):
    return np.asarray([float(point.x), float(point.y)])


def distances(prediction):
    goal = xy(prediction["goal"])
    return np.asarray([np.linalg.norm(xy(pose) - goal) for pose in prediction["trajectory"]])


def rank_auc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def classify(scores, labels, threshold):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    pred = scores >= threshold
    tp, fp = int((pred & labels).sum()), int((pred & ~labels).sum())
    tn, fn = int((~pred & ~labels).sum()), int((~pred & labels).sum())
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision_percent": 100 * tp / (tp + fp) if tp + fp else 0,
        "recall_percent": 100 * tp / (tp + fn) if tp + fn else 0,
        "specificity_percent": 100 * tn / (tn + fp) if tn + fp else 0,
        "accuracy_percent": 100 * (tp + tn) / len(labels),
    }


def summarize_predictions(predictions):
    rows = []
    for key, prediction in predictions.items():
        dist = distances(prediction)
        rows.append({
            "key": key,
            "final_distance_m": float(dist[-1]),
            "min_distance_m": float(dist.min()),
            "success": bool(dist[-1] <= 20),
            "ever_inside_20m": bool((dist <= 20).any()),
            "left_after_entering_20m": bool((dist <= 20).any() and dist[-1] > 20),
            "actions": len(prediction.get("actions", [])),
        })
    return rows


def aggregate(rows):
    n = len(rows)
    short = [r for r in rows if r["actions"] < 20]
    return {
        "episodes": n,
        "success_percent_recomputed": 100 * sum(r["success"] for r in rows) / n,
        "ever_inside_20m_percent": 100 * sum(r["ever_inside_20m"] for r in rows) / n,
        "left_after_entering_20m_percent": 100 * sum(r["left_after_entering_20m"] for r in rows) / n,
        "mean_actions": float(np.mean([r["actions"] for r in rows])),
        "median_final_distance_m": float(np.median([r["final_distance_m"] for r in rows])),
        "episodes_shorter_than_20": len(short),
        "shorter_than_20_percent": 100 * len(short) / n,
        "shorter_than_20_success_percent": (
            100 * sum(r["success"] for r in short) / len(short) if short else None
        ),
        "shorter_than_20_median_final_distance_m": (
            float(np.median([r["final_distance_m"] for r in short])) if short else None
        ),
    }


def paired(control, variant):
    base = {r["key"]: r for r in control}
    new = {r["key"]: r for r in variant}
    keys = sorted(base.keys() & new.keys())
    return {
        "episodes": len(keys),
        "changed_trajectory_length": sum(base[k]["actions"] != new[k]["actions"] for k in keys),
        "success_to_failure": sum(base[k]["success"] and not new[k]["success"] for k in keys),
        "failure_to_success": sum(not base[k]["success"] and new[k]["success"] for k in keys),
        "mean_action_change": float(np.mean([new[k]["actions"] - base[k]["actions"] for k in keys])),
        "mean_final_distance_change_m": float(np.mean([
            new[k]["final_distance_m"] - base[k]["final_distance_m"] for k in keys])),
    }


def calibration(predictions):
    scores, labels = [], []
    for prediction in predictions.values():
        progress = np.asarray(prediction.get("progress", []), dtype=float)
        dist = distances(prediction)
        count = min(len(progress), len(dist))
        scores.extend(progress[:count].tolist())
        labels.extend((dist[:count] <= 20).tolist())
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    return {
        "states": len(scores),
        "positive_percent": 100 * labels.mean(),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_below_0_percent": 100 * (scores < 0).mean(),
        "score_above_1_percent": 100 * (scores > 1).mean(),
        "roc_auc": rank_auc(scores, labels),
        "threshold_080": classify(scores, labels, 0.8),
        "threshold_095": classify(scores, labels, 0.95),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/progress_stop_comparison"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"arms": {}, "paired_vs_control": {}, "calibration_from_control": {}}
    rows_by_arm = {}
    predictions_by_arm = {}
    for arm, run in ARMS.items():
        result["arms"][arm] = {}
        rows_by_arm[arm] = {}
        predictions_by_arm[arm] = {}
        for split in ("val_seen", "val_unseen"):
            evaluation = args.runs / run / "evaluation"
            metrics = json.loads((evaluation / f"{split}_metrics.json").read_text())
            predictions = torch.load(evaluation / f"{split}_predictions.pt",
                                     map_location="cpu", weights_only=False)
            rows = summarize_predictions(predictions)
            predictions_by_arm[arm][split] = predictions
            rows_by_arm[arm][split] = rows
            result["arms"][arm][split] = {
                "metrics": {name: metrics[name] for name in METRICS},
                "trajectory": aggregate(rows),
            }
    for split in ("val_seen", "val_unseen"):
        result["calibration_from_control"][split] = calibration(
            predictions_by_arm["control_095"][split])
        result["paired_vs_control"][split] = {
            arm: paired(rows_by_arm["control_095"][split], rows_by_arm[arm][split])
            for arm in ("gated_080", "ungated_080")
        }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
