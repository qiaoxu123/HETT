#!/usr/bin/env python3
"""One-shot frozen val_unseen candidate-verifier evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.path_local_verifier import PathLocalCandidateVerifier
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE
from scripts.train_path_local_candidate_verifier import (
    build_split, encode_path_patches, predict, rows_for_records,
)
from scripts.train_rgb_candidate_ranker import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "runs/path_local_candidate_verifier/train8192/model.pt"))
    parser.add_argument("--records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/val_unseen_records.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--text-features", type=Path, default=DEFAULT_SOURCE /
                        "target_type_alignment/features.pt")
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/path_local_candidate_verifier/val_unseen_features.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/path_local_candidate_verifier/frozen_val_unseen.json"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    return parser.parse_args()


def load_or_encode(args, device):
    if args.cache.exists():
        return torch.load(args.cache, map_location="cpu", weights_only=False)
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    if isinstance(records, dict):
        records = records["val_unseen"]
    rows = rows_for_records(args.candidate_dir, "val_unseen", records)
    indices = [int(record["index"]) for record in records]
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)["val_unseen"]
    text = torch.load(
        args.text_features, map_location="cpu", weights_only=True)["text"][
            "val_unseen"]["features"]
    processor = AutoImageProcessor.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    image = encode_path_patches(
        records, "val_unseen", encoder, processor, args, device)
    data = build_split(
        records, rows, image, text[indices], relation[indices])
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.cache)
    return data


def main():
    args = parse_args(); device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    data = load_or_encode(args, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PathLocalCandidateVerifier(int(checkpoint["vision_dim"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    score = predict(model, data, device)
    distance = data["distance"]
    semantic_rank = (data["prior"] + 0.5 * data["semantic"]).argmax(-1)
    result = {
        "experiment": "one-shot frozen path-local verifier val_unseen",
        "count": len(distance),
        "val_unseen": {
            "static_top1": metrics(
                distance, torch.zeros(len(distance), dtype=torch.long)),
            "coarse_semantic_fusion": metrics(distance, semantic_rank),
            "frozen_instance_relation_verifier": metrics(
                distance, score.argmax(-1)),
            "swap_candidate_rgb_control": metrics(
                distance, predict(model, data, device,
                                  "swap_candidate_rgb").argmax(-1)),
            "same_map_wrong_instruction_control": metrics(
                distance, predict(model, data, device,
                                  "same_map_wrong_instruction").argmax(-1)),
            "two_mode_oracle": metrics(distance, distance.argmin(-1)),
        },
        "anti_leakage": {
            "checkpoint_and_hyperparameters_frozen_before_load": True,
            "val_unseen_used_for_selection": False,
            "target_use": "final metrics only",
            "pointcloud_used": False, "test_unseen_loaded": False,
        },
        "decision_rule": (
            "run strict unseen rollout only if frozen verifier retains a "
            "clear advantage over coarse semantic fusion"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
