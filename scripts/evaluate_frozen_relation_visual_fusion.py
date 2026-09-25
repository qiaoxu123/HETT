#!/usr/bin/env python3
"""Fuse the frozen causal relation expert with the visual candidate verifier.

The fusion weight and score scales are selected only on val_seen.  The frozen
val_unseen split is then evaluated once with the same values.  Target locations
are used only for val_seen selection and final metrics, never by either model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.path_local_verifier import LandmarkAwareCandidateVerifier
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE
from scripts.train_landmark_visual_verifier import predict
from scripts.train_path_local_candidate_verifier import rows_for_records
from scripts.train_relation_geometry_ranker import (
    RelationGeometryRanker,
    corrected_landmark_geometry,
    matched_same_map_switch_indices,
    structured_relation_cues,
)
from scripts.train_rgb_candidate_ranker import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-checkpoint", type=Path, default=Path(
        "runs/landmark_visual_verifier/relative_margin_2048_fullval/model.pt"))
    parser.add_argument("--relation-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seen-features", type=Path, default=Path(
        "runs/landmark_visual_verifier/relative_margin_2048_fullval_features.pt"))
    parser.add_argument("--unseen-features", type=Path, default=Path(
        "runs/landmark_visual_verifier/frozen_val_unseen_features.pt"))
    parser.add_argument("--records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/expanded_train_records.pt"))
    parser.add_argument("--unseen-records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/val_unseen_records.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path(
        "runs/landmark_visual_verifier/frozen_relation_visual_fusion.json"))
    return parser.parse_args()


def load_records(path, split, count):
    records = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(records, dict):
        records = records[split]
    return records[:count]


@torch.inference_mode()
def relation_scores(models, data, rows, max_landmarks, device):
    geometry, valid = corrected_landmark_geometry(rows, max_landmarks)
    cues = structured_relation_cues(rows, max_landmarks)
    switched = matched_same_map_switch_indices(rows)
    wrong_geometry, wrong_valid = corrected_landmark_geometry(
        rows, max_landmarks, switched)
    output = {name: [] for name in ("correct", "wrong_relation", "wrong_geometry")}
    for model in models:
        output["correct"].append(model(
            data["relation"].to(device), geometry.to(device), valid.to(device),
            cues.to(device)).cpu())
        output["wrong_relation"].append(model(
            data["relation"][switched].to(device), geometry.to(device),
            valid.to(device), cues[switched].to(device)).cpu())
        output["wrong_geometry"].append(model(
            data["relation"].to(device), wrong_geometry.to(device),
            wrong_valid.to(device), cues.to(device)).cpu())
    return {name: torch.stack(value).mean(0) for name, value in output.items()}


def evaluate(distance, visual, relation, visual_scale, relation_scale, alpha):
    result = {"visual_only": metrics(distance, visual.argmax(-1))}
    for name, score in relation.items():
        fused = visual / visual_scale + alpha * score / relation_scale
        result[f"fused_{name}"] = metrics(distance, fused.argmax(-1))
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seen_all = torch.load(args.seen_features, map_location="cpu", weights_only=False)
    seen = seen_all["val_seen"]
    unseen = torch.load(args.unseen_features, map_location="cpu", weights_only=False)
    seen_records = load_records(args.records, "val_seen", len(seen["distance"]))
    unseen_records = load_records(
        args.unseen_records, "val_unseen", len(unseen["distance"]))
    rows = {
        "val_seen": rows_for_records(args.candidate_dir, "val_seen", seen_records),
        "val_unseen": rows_for_records(
            args.candidate_dir, "val_unseen", unseen_records),
    }

    visual_state = torch.load(
        args.visual_checkpoint, map_location="cpu", weights_only=True)
    visual_model = LandmarkAwareCandidateVerifier(
        int(visual_state["vision_dim"]),
        relation_dim=int(visual_state.get("relation_dim", 768))).to(device)
    visual_model.load_state_dict(visual_state["model"])
    visual_model.eval()

    relation_models = []
    for seed in (int(value) for value in args.seeds.split(",") if value):
        model = RelationGeometryRanker(cue_dim=12).to(device)
        model.load_state_dict(torch.load(
            args.relation_checkpoint_dir / f"model_seed{seed}.pt",
            map_location="cpu", weights_only=True))
        relation_models.append(model.eval())

    visual = {
        "val_seen": predict(visual_model, seen, device),
        "val_unseen": predict(visual_model, unseen, device),
    }
    relation = {
        "val_seen": relation_scores(
            relation_models, seen, rows["val_seen"], args.max_landmarks, device),
        "val_unseen": relation_scores(
            relation_models, unseen, rows["val_unseen"], args.max_landmarks, device),
    }
    visual_scale = (visual["val_seen"][:, 0] - visual["val_seen"][:, 1]).std()
    relation_scale = (
        relation["val_seen"]["correct"][:, 0]
        - relation["val_seen"]["correct"][:, 1]).std()

    best = None
    for alpha_step in range(41):
        alpha = alpha_step / 20.0
        fused = (visual["val_seen"] / visual_scale
                 + alpha * relation["val_seen"]["correct"] / relation_scale)
        value = metrics(seen["distance"], fused.argmax(-1))
        key = value["hit20"], -value["mean_ne"], -alpha
        if best is None or key > best[0]:
            best = key, alpha
    alpha = best[1]
    result = {
        "experiment": "frozen causal relation expert plus visual verifier",
        "selection": {
            "split": "val_seen", "alpha": alpha,
            "visual_logit_difference_std": float(visual_scale),
            "relation_logit_difference_std": float(relation_scale),
        },
        "val_seen": evaluate(
            seen["distance"], visual["val_seen"], relation["val_seen"],
            visual_scale, relation_scale, alpha),
        "val_unseen": evaluate(
            unseen["distance"], visual["val_unseen"], relation["val_unseen"],
            visual_scale, relation_scale, alpha),
        "anti_leakage": {
            "relation_model": "frozen before this experiment",
            "visual_model": "frozen before this experiment",
            "alpha_and_scales_selected_on": "val_seen only",
            "target_in_model_forward": False,
            "target_usage": "val_seen alpha selection and final metrics only",
            "test_unseen_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
