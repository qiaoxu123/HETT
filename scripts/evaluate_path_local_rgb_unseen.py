#!/usr/bin/env python3
"""One-shot frozen val_unseen evaluation of path-local RGB evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import belief_distinct_topk_xy
from independent_nav.semantic import ProjectSemanticNet
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.observation import cropclient
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, load_rows, predict_beliefs,
)
from scripts.evaluate_path_local_pointcloud_teacher import (
    first_joint_path_observation, metrics, mode_prior, normalized_to_world,
    ranks, shuffled_records,
)
from scripts.evaluate_path_local_rgb_evidence import collect_rgb_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--type-features", type=Path, default=DEFAULT_SOURCE /
                        "target_type_alignment/features.pt")
    parser.add_argument("--semantic-checkpoint", type=Path, default=Path(
        "runs/independent_semantic_trainseen_valseen_select/model.pt"))
    parser.add_argument("--frozen-results", type=Path, default=Path(
        "runs/path_local_rgb_evidence/results.json"))
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/path_local_rgb_evidence/val_unseen_records.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/path_local_rgb_evidence/val_unseen_results.json"))
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--evidence-radius-m", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def geometry_records(rows, beliefs, split="val_unseen", source_indices=None):
    trajectories = load_mturk_trajectories(split, "all", 50.0)
    if source_indices is None:
        source_indices = list(range(len(rows)))
    if len(rows) != len(source_indices):
        raise ValueError("rows and source indices must align")
    records = []
    for row, logits, source_index in zip(rows, beliefs, source_indices):
        trajectory = trajectories[int(source_index)]
        modes = belief_distinct_topk_xy(logits, 2)
        modes_world = np.stack([
            normalized_to_world(mode, row["map_name"]) for mode in modes])
        _, action_step, joint_visible = first_joint_path_observation(
            trajectory.start_pose, modes_world[0], modes_world[1])
        records.append({
            "index": int(source_index), "map_name": row["map_name"],
            "target": np.asarray(row["target"]),
            "modes": np.asarray(modes), "prior": mode_prior(logits, modes),
            "observation_action_step": action_step,
            "joint_visible": joint_visible,
        })
    return records


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = json.loads(args.frozen_results.read_text())
    beta = float(frozen["selected_on_train_seen"]["rgb_weight"])
    if args.cache.exists():
        rgb_records = torch.load(
            args.cache, map_location="cpu", weights_only=False)
    else:
        rows = load_rows(args.candidate_dir / "val_unseen.jsonl")
        relation = torch.load(
            args.relation_cache, map_location="cpu", weights_only=True)
        data = BeliefInferenceDataset(rows, relation["val_unseen"])
        beliefs, _ = predict_beliefs(args.checkpoint, data, device, 64)
        base = geometry_records(rows, beliefs)
        features = torch.load(
            args.type_features, map_location="cpu", weights_only=True)
        type_model = train_probe(
            features["text"]["train_seen"], features["text"]["val_seen"],
            args.seed, 40, device)
        with torch.inference_mode():
            predicted_type = type_model(
                features["text"]["val_unseen"]["features"].float().to(device)
            ).argmax(-1).cpu().numpy()
        semantic = ProjectSemanticNet().to(device)
        semantic.load_state_dict(torch.load(
            args.semantic_checkpoint, map_location="cpu", weights_only=True))
        semantic.eval(); cropclient.load_image_cache()
        rgb_records = collect_rgb_records(
            args, base, "val_unseen", predicted_type, semantic, device)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rgb_records, args.cache)

    static = np.zeros(len(rgb_records), dtype=np.int64)
    rgb_rank = ranks(rgb_records, beta)
    wrong_rank = ranks(rgb_records, beta, "wrong_type_evidence")
    shuffled = shuffled_records(rgb_records, args.seed)
    shuffled_rank = ranks(shuffled, beta, "shuffled_evidence")
    oracle = np.asarray([
        np.linalg.norm(row["modes"] - row["target"][None], axis=-1).argmin()
        for row in rgb_records])
    result = {
        "stage": "frozen one-shot path-local RGB val_unseen",
        "selected_before_val_unseen": {"rgb_weight": beta},
        "val_unseen": {
            "static_top1": metrics(rgb_records, static),
            "online_rgb_evidence": metrics(rgb_records, rgb_rank),
            "wrong_target_type_control": metrics(rgb_records, wrong_rank),
            "shuffled_episode_evidence_control": metrics(
                shuffled, shuffled_rank),
            "two_mode_privileged_oracle": metrics(rgb_records, oracle),
            "rgb_changed_top1_percent": float(np.mean(rgb_rank != 0) * 100),
        },
        "anti_leakage": {
            "model_and_hyperparameters_frozen_before_load": True,
            "val_unseen_used_for_selection": False,
            "pointcloud_used": False,
            "target_use": "final metrics only",
            "test_unseen_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
