#!/usr/bin/env python3
"""Offline audit of landmark arrival, stage switching, and teacher trajectory tails."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_BASE = Path("/home/tenant2/Workspace/hett-crotonyl/runs/hett_baseline_fixed_20260911/evaluation")
DEFAULT_GEOMETRY = Path("/home/tenant2/Workspace/hett-experiments/09-contour-evidence/runs/landmark_target_distance/results.json")


def xy_array(poses):
    return np.asarray([[float(p.x), float(p.y)] for p in poses], dtype=np.float64)


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def polygon_union(contours):
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygon = Polygon(contour)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    return unary_union(polygons) if polygons else None


def distances(points, region):
    return np.asarray([Point(point).distance(region) for point in points], dtype=np.float64)


def first_true(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def rank_auc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    positives = labels.sum()
    negatives = (~labels).sum()
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average tied ranks.
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def classification(scores, labels, threshold):
    labels = np.asarray(labels, dtype=bool)
    pred = np.asarray(scores) >= threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    tn = int((~pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold), "n": len(labels), "positive_percent": float(100 * labels.mean()),
        "accuracy_percent": float(100 * (tp + tn) / len(labels)),
        "balanced_accuracy_percent": float(50 * (recall + specificity)),
        "precision_percent": float(100 * tp / (tp + fp)) if tp + fp else 0.0,
        "recall_percent": float(100 * recall), "specificity_percent": float(100 * specificity),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def fit_threshold(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 1001)))
    reports = [classification(scores, labels, threshold) for threshold in candidates]
    return max(reports, key=lambda report: (report["balanced_accuracy_percent"], report["accuracy_percent"]))["threshold"]


def summarize(rows, threshold_m):
    usable = [row for row in rows if row["has_landmark"]]
    switched = [row for row in usable if row["stage2_started"]]
    ever = [row for row in usable if row["pred_min_landmark_m"] <= threshold_m]
    teacher_reached = [row for row in usable if row["teacher_first_arrival_index"] is not None]
    teacher_started_far = [row for row in teacher_reached if row["teacher_start_landmark_m"] > threshold_m]
    return {
        "episodes": len(rows),
        "with_landmark": len(usable),
        "stage2_started_percent": 100 * len(switched) / len(usable),
        "stage_switch_near_landmark_percent": 100 * sum(row["switch_landmark_m"] <= threshold_m for row in switched) / len(switched) if switched else None,
        "stage_switch_distance_median_m": float(np.median([row["switch_landmark_m"] for row in switched])) if switched else None,
        "ever_reached_landmark_percent": 100 * len(ever) / len(usable),
        "final_near_landmark_percent": 100 * sum(row["pred_final_landmark_m"] <= threshold_m for row in usable) / len(usable),
        "left_after_reaching_percent": 100 * sum(row["pred_final_landmark_m"] > threshold_m for row in ever) / len(ever) if ever else None,
        "target_success_given_landmark_reached_percent": 100 * sum(row["target_success"] for row in ever) / len(ever) if ever else None,
        "teacher_reached_landmark_percent": 100 * len(teacher_reached) / len(usable),
        "teacher_reached_from_outside_percent": 100 * len(teacher_started_far) / sum(row["teacher_start_landmark_m"] > threshold_m for row in usable),
        "teacher_tail_length_m_median": float(np.median([row["teacher_tail_length_m"] for row in teacher_reached])),
        "teacher_tail_fraction_median": float(np.median([row["teacher_tail_fraction"] for row in teacher_reached])),
        "teacher_tail_excess_ratio_median": float(np.median([row["teacher_tail_excess_ratio"] for row in teacher_reached])),
        "teacher_target_distance_at_arrival_median_m": float(np.median([row["teacher_target_distance_at_arrival_m"] for row in teacher_reached])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/landmark_gate_audit")
    parser.add_argument("--threshold-m", type=float, default=10.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    geometry_data = json.loads(args.geometry.read_text())
    geometry = {(row["split"], row["map"], row["object_id"], row["description_id"]): row
                for row in geometry_data["rows"]}
    all_rows = {}
    progress_states = {}
    for split in ("val_seen", "val_unseen"):
        predictions = torch.load(args.base / f"{split}_predictions.pt", map_location="cpu", weights_only=False)
        rows = []
        state_scores, state_labels = [], []
        for key, prediction in predictions.items():
            geo = geometry[(split, *key)]
            region = polygon_union(geo["contours"])
            row = {"split": split, "id": list(key), "has_landmark": region is not None}
            if region is None:
                rows.append(row)
                continue

            goal = np.asarray(prediction["goal"], dtype=np.float64)[:2]
            pred_xy = xy_array(prediction["trajectory"])
            teacher_xy = xy_array(prediction["gt_trajectory"])
            pred_dist = distances(pred_xy, region)
            teacher_dist = distances(teacher_xy, region)
            arrival = first_true(teacher_dist <= args.threshold_m)
            stage2 = prediction.get("stage2_trajectory", [])
            switch_xy = xy_array(stage2)[0] if stage2 else pred_xy[-1]
            switch_distance = float(Point(switch_xy).distance(region))

            progress = np.asarray(prediction.get("progress", []), dtype=np.float64)
            aligned = min(len(progress), len(pred_dist))
            state_scores.extend(progress[:aligned].tolist())
            state_labels.extend((pred_dist[:aligned] <= args.threshold_m).tolist())

            if arrival is not None:
                tail = teacher_xy[arrival:]
                tail_length = path_length(tail)
                direct = float(np.linalg.norm(teacher_xy[-1] - teacher_xy[arrival]))
                total = path_length(teacher_xy)
                target_at_arrival = float(np.linalg.norm(goal - teacher_xy[arrival]))
            else:
                tail_length = direct = total = target_at_arrival = math.nan
            row.update({
                "stage2_started": bool(stage2), "switch_landmark_m": switch_distance,
                "pred_min_landmark_m": float(pred_dist.min()), "pred_final_landmark_m": float(pred_dist[-1]),
                "target_success": bool(np.linalg.norm(pred_xy[-1] - goal) <= 20),
                "teacher_start_landmark_m": float(teacher_dist[0]),
                "teacher_min_landmark_m": float(teacher_dist.min()),
                "teacher_first_arrival_index": arrival,
                "teacher_tail_length_m": tail_length,
                "teacher_tail_fraction": tail_length / total if total > 0 else 0.0,
                "teacher_tail_excess_ratio": tail_length / max(direct, 1.0),
                "teacher_target_distance_at_arrival_m": target_at_arrival,
            })
            rows.append(row)
        all_rows[split] = rows
        progress_states[split] = (state_scores, state_labels)

    learned_threshold = fit_threshold(*progress_states["val_seen"])
    summary = {split: summarize(rows, args.threshold_m) for split, rows in all_rows.items()}
    for split in ("val_seen", "val_unseen"):
        scores, labels = progress_states[split]
        summary[split]["progress_as_arrival"] = classification(scores, labels, learned_threshold)
        summary[split]["progress_as_arrival"]["roc_auc"] = rank_auc(scores, labels)
    output = {
        "definition": f"Arrived iff 2D pose distance to union of supplied landmark contours <= {args.threshold_m:g} m.",
        "progress_threshold_selected_on": "val_seen balanced accuracy",
        "summary": summary,
        "rows": all_rows,
    }
    (args.out / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
