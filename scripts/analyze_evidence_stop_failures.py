#!/usr/bin/env python3
"""Break down the frozen val_seen stop gate by metric-only target type."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.evidence_stop import EvidenceStopHead, first_stop_index
from scripts.audit_target_type_alignment import CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, default=Path(
        "runs/evidence_stop_head_belief/states.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "runs/evidence_stop_head_belief/model.pt"))
    parser.add_argument("--geometry-records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/records.pt"))
    parser.add_argument("--type-features", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/target_type_alignment/features.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/evidence_stop_head_belief/failure_by_type.json"))
    return parser.parse_args()


def rank_auc(labels, values):
    labels = np.asarray(labels, dtype=bool); values = np.asarray(values)
    positive, negative = values[labels], values[~labels]
    if not len(positive) or not len(negative):
        return None
    # Pairwise definition handles ties explicitly and avoids an extra runtime
    # dependency in the reproducible diagnostic.
    comparison = positive[:, None] - negative[None]
    return float((np.count_nonzero(comparison > 0)
                  + 0.5 * np.count_nonzero(comparison == 0)) / comparison.size)


def main():
    args = parse_args()
    saved = torch.load(args.states, map_location="cpu", weights_only=False)
    episodes = saved["val_seen"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = EvidenceStopHead(); model.load_state_dict(checkpoint["model"]); model.eval()
    features = torch.tensor(np.stack([
        state["features"] for episode in episodes for state in episode]))
    with torch.inference_mode():
        probability = model(
            (features - checkpoint["mean"]) / checkpoint["std"]
        ).sigmoid().numpy()

    records = torch.load(
        args.geometry_records, map_location="cpu", weights_only=False)["val_seen"]
    type_data = torch.load(
        args.type_features, map_location="cpu", weights_only=True)["text"]["val_seen"]
    labels_by_episode = np.asarray([
        int(type_data["labels"][int(record["index"])]) for record in records])
    if len(labels_by_episode) != len(episodes):
        raise ValueError("type labels and stop episodes are not aligned")

    cursor = 0; summaries = []
    for episode in episodes:
        count = len(episode)
        values = probability[cursor:cursor + count]
        chosen = first_stop_index(
            values, [state["eligible"] for state in episode],
            float(checkpoint["threshold"]))
        summaries.append({
            "baseline": bool(episode[-1]["label"]),
            "learned": bool(episode[chosen]["label"]),
            "oracle": any(state["label"] for state in episode),
            "states": episode,
        })
        cursor += count

    result = {"split": "val_seen", "groups": {}}
    for class_index, name in enumerate(CLASSES):
        selected = [summary for summary, label in zip(summaries, labels_by_episode)
                    if label == class_index]
        state_rows = [state for summary in selected for state in summary["states"]
                      if state["eligible"]]
        state_labels = np.asarray([row["label"] for row in state_rows], dtype=bool)
        center_soft = np.asarray([row["features"][6] for row in state_rows])
        belief_peak = np.asarray([row["features"][9] for row in state_rows])
        result["groups"][name] = {
            "episodes": len(selected),
            "baseline_sr": float(np.mean([x["baseline"] for x in selected]) * 100),
            "learned_sr": float(np.mean([x["learned"] for x in selected]) * 100),
            "trajectory_oracle_sr": float(np.mean([x["oracle"] for x in selected]) * 100),
            "recoverable_episodes": int(sum(
                x["oracle"] and not x["baseline"] for x in selected)),
            "center_semantic_state_auc": rank_auc(state_labels, center_soft),
            "local_belief_state_auc": rank_auc(state_labels, belief_peak),
        }
    result["interpretation"] = (
        "Target type is metric-only in this diagnostic and is never a policy input; "
        "the table identifies which coarse semantic classes fail to support stopping.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
