#!/usr/bin/env python3
"""Replace the path-local point-cloud teacher with online RGB prediction."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import move_toward
from independent_nav.projection import transform_points, view_corners, world_to_image_homography
from independent_nav.semantic import ProjectSemanticNet
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.observation import cropclient
from multiagent.space import Point2D
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_path_local_pointcloud_teacher import (
    metrics, normalized_to_world, ranks, shuffled_records, tune,
)
from scripts.evaluate_temporal_candidate_reranking import local_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-records", type=Path, default=Path(
        "runs/path_local_pointcloud_teacher/records.pt"))
    parser.add_argument("--semantic-checkpoint", type=Path, default=Path(
        "runs/independent_semantic_trainseen_valseen_select/model.pt"))
    parser.add_argument("--type-features", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/target_type_alignment/features.pt"))
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/path_local_rgb_evidence/records.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/path_local_rgb_evidence/results.json"))
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--evidence-radius-m", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def reconstruct_observation_pose(start_pose, modes_world, action_step):
    """Replay the target-free midpoint progress actions from the teacher gate."""
    midpoint = Point2D(*np.asarray(modes_world).mean(0))
    pose = start_pose
    for _ in range(int(action_step)):
        pose = move_toward(pose, midpoint, 10)
    return pose


def candidate_patches(rgb, pose, modes_world, view_size, patch_size, patch_side_m):
    homography = world_to_image_homography(
        view_corners(pose.x, pose.y, pose.yaw, 50.0), view_size)
    pixels = transform_points(modes_world, homography)
    side_pixels = max(4, int(round(patch_side_m / 100.0 * view_size)))
    patches = []
    for pixel in pixels:
        patch = cv2.getRectSubPix(
            rgb, (side_pixels, side_pixels), (float(pixel[0]), float(pixel[1])))
        patches.append(cv2.resize(
            patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA))
    return patches


def semantic_evidence(probability, target_class, radius_pixels):
    class_probability = probability[target_class]
    center = (class_probability.shape[1] / 2, class_probability.shape[0] / 2)
    soft = local_score(class_probability, center, radius_pixels)
    prediction = probability.argmax(0)
    yy, xx = np.ogrid[:prediction.shape[0], :prediction.shape[1]]
    local = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2
             <= radius_pixels ** 2)
    hard_fraction = float(np.mean(prediction[local] == target_class))
    return 0.5 * soft + 0.5 * hard_fraction


@torch.inference_mode()
def collect_rgb_records(args, source, split, predicted_type, model, device):
    trajectories = load_mturk_trajectories(split, "all", 50.0)
    output, pending, metadata = [], [], []
    radius_pixels = args.evidence_radius_m / args.patch_side_m * args.patch_size

    def flush():
        if not pending:
            return
        tensor = torch.from_numpy(np.stack(pending)).permute(0, 3, 1, 2)
        probability = model(tensor.float().to(device).div(255.0)).softmax(1).cpu().numpy()
        for offset in range(0, len(metadata), 2):
            record, type_index = metadata[offset]
            evidence, wrong = [], []
            for candidate in range(2):
                value = probability[offset + candidate]
                evidence.append(semantic_evidence(
                    value, type_index + 1, radius_pixels))
                wrong.append(semantic_evidence(
                    value, ((type_index + 1) % 4) + 1, radius_pixels))
            output.append({
                "index": record["index"], "map_name": record["map_name"],
                "target": np.asarray(record["target"]),
                "modes": np.asarray(record["modes"]),
                "prior": np.asarray(record["prior"]),
                "evidence": np.asarray(evidence),
                "wrong_type_evidence": np.asarray(wrong),
                "observation_action_step": record["observation_action_step"],
            })
        pending.clear(); metadata.clear()

    for record in source:
        index = int(record["index"])
        trajectory = trajectories[index]
        modes_world = np.stack([
            normalized_to_world(mode, record["map_name"])
            for mode in record["modes"]])
        pose = reconstruct_observation_pose(
            trajectory.start_pose, modes_world,
            record["observation_action_step"])
        rgb = cropclient.crop_image(
            record["map_name"], pose, (args.view_size, args.view_size), "rgb")
        patches = candidate_patches(
            rgb, pose, modes_world, args.view_size, args.patch_size,
            args.patch_side_m)
        pending.extend(patches)
        metadata.extend([(record, int(predicted_type[index]))] * 2)
        if len(pending) >= args.batch_size:
            flush()
    flush()
    return sorted(output, key=lambda record: record["index"])


def load_or_collect(args, device):
    if args.cache.exists():
        return torch.load(args.cache, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.teacher_records, map_location="cpu", weights_only=False)
    features = torch.load(args.type_features, map_location="cpu", weights_only=True)
    type_model = train_probe(
        features["text"]["train_seen"], features["text"]["val_seen"],
        args.seed, 40, device)
    semantic = ProjectSemanticNet().to(device)
    semantic.load_state_dict(torch.load(
        args.semantic_checkpoint, map_location="cpu", weights_only=True))
    semantic.eval(); cropclient.load_image_cache()
    saved = {}
    for split in ("train_seen", "val_seen"):
        with torch.inference_mode():
            predicted = type_model(
                features["text"][split]["features"].float().to(device)
            ).argmax(-1).cpu().numpy()
        saved[split] = collect_rgb_records(
            args, teacher[split], split, predicted, semantic, device)
        print(split, len(saved[split]), flush=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.cache)
    return saved


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = load_or_collect(args, device)
    train_records, val_records = saved["train_seen"], saved["val_seen"]
    beta = tune(train_records)
    static = np.zeros(len(val_records), dtype=np.int64)
    rgb_rank = ranks(val_records, beta)
    wrong_rank = ranks(val_records, beta, "wrong_type_evidence")
    shuffled = shuffled_records(val_records, args.seed)
    shuffled_rank = ranks(shuffled, beta, "shuffled_evidence")
    oracle_rank = np.asarray([
        np.linalg.norm(record["modes"] - record["target"][None], axis=-1).argmin()
        for record in val_records])
    result = {
        "stage": "path-local online RGB replaces point-cloud teacher",
        "counts": {"train_seen": len(train_records), "val_seen": len(val_records)},
        "selected_on_train_seen": {"rgb_weight": beta},
        "val_seen": {
            "static_top1": metrics(val_records, static),
            "online_rgb_evidence": metrics(val_records, rgb_rank),
            "wrong_target_type_control": metrics(val_records, wrong_rank),
            "shuffled_episode_evidence_control": metrics(
                shuffled, shuffled_rank),
            "two_mode_privileged_oracle": metrics(val_records, oracle_rank),
            "rgb_changed_top1_percent": float(np.mean(rgb_rank != 0) * 100.0),
        },
        "config": {
            "semantic_checkpoint": str(args.semantic_checkpoint.resolve()),
            "official_view_side_m": 100.0,
            "candidate_patch_side_m": args.patch_side_m,
            "candidate_evidence_radius_m": args.evidence_radius_m,
            "altitude_m": 50.0,
        },
        "anti_leakage": {
            "runtime_inputs": (
                "online RGB + pose + predicted modes + predicted language type"),
            "pointcloud_evidence_used": False,
            "observation_pose": "target-free cached path step",
            "fusion_weight_selection": "train_seen only",
            "val_seen_target_use": "final metrics only",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
        "scope": "RGB candidate reranking gate; closed-loop evaluation follows only if positive",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

