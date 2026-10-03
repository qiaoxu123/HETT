from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from fpv_goal_approach.dataset import read_jsonl
from fpv_goal_approach.train import VARIANTS


def geometry_oracle(rows):
    truth = np.array([row["visual_confirmed_arrival"] for row in rows])
    pred = np.array([row["geometric_arrival"] for row in rows])
    negatives = truth == 0
    return {
        "accuracy": accuracy_score(truth, pred),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "precision": precision_score(truth, pred, zero_division=0),
        "recall": recall_score(truth, pred, zero_division=0),
        "f1": f1_score(truth, pred, zero_division=0),
        "false_positive_rate": float((pred[negatives] == 1).mean()),
        "note": "oracle analysis only; uses unavailable ground-truth distance",
    }


def plot_confusion(matrix, title, path):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(data: Path, training: Path):
    rows = []
    hard = {}
    reports = {}
    for variant in VARIANTS:
        directory = training / variant
        if not directory.exists():
            continue
        seen = json.loads((directory / "metrics_val_seen.json").read_text())
        unseen = json.loads((directory / "metrics_val_unseen.json").read_text())
        language_shuffle = json.loads((directory / "metrics_val_unseen_wrong_language_shuffle.json").read_text())
        frame_shuffle_path = directory / "metrics_val_unseen_frame_order_shuffle.json"
        frame_shuffle = json.loads(frame_shuffle_path.read_text()) if frame_shuffle_path.exists() else None
        reports[variant] = (seen, unseen)
        row = {
            "model": variant,
            "fpv_only": not VARIANTS[variant]["use_language"],
            "frames": VARIANTS[variant]["frames"],
            "language": VARIANTS[variant]["use_language"],
            "pose": VARIANTS[variant]["use_pose"],
            "seen_match_f1": seen["match"]["f1"],
            "unseen_match_f1": unseen["match"]["f1"],
            "seen_match_auroc": seen["match"]["auroc"],
            "unseen_match_auroc": unseen["match"]["auroc"],
            "seen_arrival_f1": seen["arrival"]["f1"],
            "unseen_arrival_f1": unseen["arrival"]["f1"],
            "seen_arrival_auroc": seen["arrival"]["auroc"],
            "unseen_arrival_auroc": unseen["arrival"]["auroc"],
            "unseen_arrival_fp_rate": unseen["arrival"]["false_positive_rate"],
            "unseen_bearing_accuracy": unseen["bearing"]["accuracy"],
            "unseen_phase_accuracy": unseen["phase"]["accuracy"],
            "language_shuffle_match_f1": language_shuffle["match"]["f1"],
            "language_shuffle_arrival_f1": language_shuffle["arrival"]["f1"],
            "frame_shuffle_match_f1": frame_shuffle["match"]["f1"] if frame_shuffle else "",
            "frame_shuffle_arrival_f1": frame_shuffle["arrival"]["f1"] if frame_shuffle else "",
        }
        rows.append(row)
        hard[variant] = {"val_seen": seen["hard_negatives"], "val_unseen": unseen["hard_negatives"]}
    output = training / "ablation_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    geometry = {
        split: geometry_oracle(read_jsonl(data / "metadata" / f"{split}.jsonl"))
        for split in ("val_seen", "val_unseen")
    }
    (training / "geometry_oracle.json").write_text(json.dumps(geometry, indent=2))
    (training / "hard_negative_metrics.json").write_text(json.dumps(hard, indent=2))
    best = max(reports, key=lambda name: reports[name][1]["arrival"]["auroc"])
    for split_index, split in enumerate(("seen", "unseen")):
        report = reports[best][split_index]
        plot_confusion(report["distance"]["confusion_matrix"], f"Distance {split} ({best})", training / f"confusion_distance_{split}.png")
        plot_confusion(report["bearing"]["confusion_matrix"], f"Bearing {split} ({best})", training / f"confusion_bearing_{split}.png")
    summary = {"best_unseen_arrival_auroc_variant": best, "geometry_oracle": geometry, "ablations": rows}
    (training / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize(args.data, args.training)
