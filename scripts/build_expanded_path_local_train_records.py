#!/usr/bin/env python3
"""Build a larger target-free train_seen path-observation record set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.semantic import ProjectSemanticNet
from multiagent.observation import cropclient
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, load_rows, predict_beliefs,
)
from scripts.evaluate_path_local_pointcloud_teacher import balanced_indices
from scripts.evaluate_path_local_rgb_evidence import collect_rgb_records
from scripts.evaluate_path_local_rgb_unseen import geometry_records


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
    parser.add_argument("--base-records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/records.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/path_local_rgb_evidence/expanded_train_records.pt"))
    parser.add_argument("--train-count", type=int, default=8192)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--evidence-radius-m", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = load_rows(args.candidate_dir / "train_seen.jsonl")
    indices = balanced_indices(all_rows, args.train_count, args.seed)
    rows = [all_rows[index] for index in indices]
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)["train_seen"]
    data = BeliefInferenceDataset(rows, relation[indices])
    beliefs, _ = predict_beliefs(args.checkpoint, data, device, 64)
    base = geometry_records(
        rows, beliefs, split="train_seen", source_indices=indices)

    features = torch.load(
        args.type_features, map_location="cpu", weights_only=True)
    type_model = train_probe(
        features["text"]["train_seen"], features["text"]["val_seen"],
        args.seed, 40, device)
    with torch.inference_mode():
        predicted = type_model(
            features["text"]["train_seen"]["features"].float().to(device)
        ).argmax(-1).cpu().numpy()
    semantic = ProjectSemanticNet().to(device)
    semantic.load_state_dict(torch.load(
        args.semantic_checkpoint, map_location="cpu", weights_only=True))
    semantic.eval()
    cropclient.load_image_cache()
    train_records = collect_rgb_records(
        args, base, "train_seen", predicted, semantic, device)
    base_records = torch.load(
        args.base_records, map_location="cpu", weights_only=False)
    saved = {
        "train_seen": train_records,
        "val_seen": base_records["val_seen"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.output)
    print({
        "output": str(args.output), "train_seen": len(train_records),
        "val_seen": len(saved["val_seen"]), "val_unseen_loaded": False,
        "test_unseen_loaded": False,
    })


if __name__ == "__main__":
    main()
