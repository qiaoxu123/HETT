#!/usr/bin/env python3
"""One-shot frozen unseen evaluation for per-landmark visual verification."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.path_local_verifier import LandmarkAwareCandidateVerifier
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE
from scripts.train_landmark_visual_verifier import (
    encode_landmark_patches,
    predict,
)
from scripts.train_path_local_candidate_verifier import rows_for_records
from scripts.train_relation_geometry_ranker import (
    corrected_landmark_geometry,
    matched_same_map_switch_indices,
    structured_relation_cues,
)
from scripts.train_rgb_candidate_ranker import landmark_geometry, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "runs/landmark_visual_verifier/relative_margin_2048_fullval/model.pt"))
    parser.add_argument("--base-features", type=Path, default=Path(
        "runs/path_local_candidate_verifier/val_unseen_features.pt"))
    parser.add_argument("--records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/val_unseen_records.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/landmark_visual_verifier/frozen_val_unseen_features.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/landmark_visual_verifier/frozen_val_unseen.json"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument(
        "--geometry-frame", choices=("raw", "audited"), default="raw")
    parser.add_argument("--explicit-relation-cues", action="store_true")
    return parser.parse_args()


def load_or_encode(args, device):
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    if isinstance(records, dict):
        records = records["val_unseen"]
    rows = rows_for_records(args.candidate_dir, "val_unseen", records)
    switched = matched_same_map_switch_indices(rows)
    if args.cache.exists():
        data = torch.load(args.cache, map_location="cpu", weights_only=False)
        if args.geometry_frame == "audited":
            geometry, valid = corrected_landmark_geometry(
                rows, args.max_landmarks)
            wrong_geometry, wrong_valid = corrected_landmark_geometry(
                rows, args.max_landmarks, switched)
            data["geometry"] = geometry.float()
            data["valid"] = valid
            data["wrong_geometry"] = wrong_geometry.float()
            data["wrong_valid"] = wrong_valid
        if args.explicit_relation_cues:
            cues = structured_relation_cues(rows, args.max_landmarks)
            data["relation"] = torch.cat(
                (data["relation"][..., :768].float(), cues), dim=-1)
            data["wrong_relation"] = torch.cat((
                data["wrong_relation"][..., :768].float(), cues[switched]), dim=-1)
        return data
    data = copy.copy(torch.load(
        args.base_features, map_location="cpu", weights_only=False))
    wrong_rows = [rows[index] for index in switched.tolist()]
    processor = AutoImageProcessor.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    landmark, visual_valid, wrong_landmark, wrong_visual_valid = (
        encode_landmark_patches(
            records, rows, wrong_rows, "val_unseen", data["image"].shape[-1],
            encoder, processor, args, device))
    wrong_geometry, wrong_valid = landmark_geometry(rows, switched)
    data.update({
        "landmark_image": landmark,
        "landmark_visual_valid": visual_valid,
        "wrong_landmark_image": wrong_landmark,
        "wrong_landmark_visual_valid": wrong_visual_valid,
        "wrong_text": data["text"][switched],
        "wrong_relation": data["relation"][switched],
        "wrong_geometry": wrong_geometry,
        "wrong_valid": wrong_valid.bool(),
    })
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.cache)
    if args.geometry_frame == "audited":
        geometry, valid = corrected_landmark_geometry(
            rows, args.max_landmarks)
        wrong_geometry, wrong_valid = corrected_landmark_geometry(
            rows, args.max_landmarks, switched)
        data["geometry"] = geometry.float()
        data["valid"] = valid
        data["wrong_geometry"] = wrong_geometry.float()
        data["wrong_valid"] = wrong_valid
    if args.explicit_relation_cues:
        cues = structured_relation_cues(rows, args.max_landmarks)
        data["relation"] = torch.cat(
            (data["relation"][..., :768].float(), cues), dim=-1)
        data["wrong_relation"] = torch.cat((
            data["wrong_relation"][..., :768].float(), cues[switched]), dim=-1)
    return data


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_or_encode(args, device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True)
    model = LandmarkAwareCandidateVerifier(
        int(checkpoint["vision_dim"]),
        relation_dim=int(checkpoint.get("relation_dim", 768))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    distance = data["distance"]
    names = (
        "normal", "swap_candidate_rgb", "zero_landmark_rgb",
        "wrong_landmark_rgb", "wrong_text_only", "wrong_relation_only",
        "wrong_geometry_only", "same_map_wrong_instruction")
    values = {name: metrics(
        distance, predict(model, data, device, name).argmax(-1))
              for name in names}
    values.update({
        "coarse_semantic_fusion": metrics(
            distance,
            (data["prior"] + 0.5 * data["semantic"]).argmax(-1)),
        "two_mode_oracle": metrics(distance, distance.argmin(-1)),
    })
    result = {
        "experiment": "one-shot frozen per-landmark visual verifier unseen",
        "geometry_frame": args.geometry_frame,
        "explicit_relation_cues": args.explicit_relation_cues,
        "count": len(distance),
        "val_unseen": values,
        "language_control_gap_pp": (
            values["normal"]["hit20"]
            - values["same_map_wrong_instruction"]["hit20"]),
        "visibility": {
            "visible_landmark_percent": float(
                data["landmark_visual_valid"].sum()
                / data["valid"].sum().clamp_min(1) * 100),
            "any_visible_episode_percent": float(
                data["landmark_visual_valid"].any(-1).float().mean() * 100),
        },
        "anti_leakage": {
            "checkpoint_and_hyperparameters_frozen_before_val_unseen": True,
            "target_in_model_forward": False,
            "target_usage": "final distance metrics only",
            "pointcloud_at_runtime": False,
            "test_unseen_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
