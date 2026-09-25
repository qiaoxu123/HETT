#!/usr/bin/env python3
"""Can a 50 m path-local point-cloud teacher disambiguate two belief modes?

The observation pose follows the official HETT movement toward the midpoint of
Top-1 and Top-2.  It is the first budget-feasible pose where both hypotheses are
inside the official 100 x 100 m top-down view.  Pose selection has no target
input.  Point-cloud semantics are a training teacher, never a deployment input.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import (
    OfficialProtocol, belief_distinct_topk_xy, canonical_instruction,
    move_toward, optimistic_translation_steps,
)
from independent_nav.pointcloud import read_ply
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.mapdata import MAP_BOUNDS
from multiagent.space import Point2D
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, load_rows, predict_beliefs,
)
from scripts.evaluate_pointcloud_active_view_oracle import (
    NATIVE_TARGET_CLASSES, ply_for_map,
)


PLY_ROOT = Path(
    "/home/rental/20260922_1/Workspace/DATA/SensatUrban_Dataset/"
    "SensatUrban_Dataset/ply")


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
    parser.add_argument("--ply-root", type=Path, default=PLY_ROOT)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=0,
                        help="0 evaluates all val_seen rows")
    parser.add_argument("--point-stride", type=int, default=16)
    parser.add_argument("--candidate-radius", type=float, default=10.0)
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/path_local_pointcloud_teacher/records.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/path_local_pointcloud_teacher/results.json"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def normalized_to_world(xy, map_name):
    bounds = MAP_BOUNDS[map_name]
    return np.asarray((bounds.x_min + float(xy[0]) * 410.0,
                       bounds.y_max - float(xy[1]) * 410.0))


def world_to_normalized(xy, map_name):
    bounds = MAP_BOUNDS[map_name]
    return ((float(xy[0]) - bounds.x_min) / 410.0,
            (bounds.y_max - float(xy[1])) / 410.0)


def visible_in_official_crop(pose, point, altitude_m=50.0):
    """Official crop is a yaw-rotated square with half-side = altitude."""
    delta = np.asarray(point, dtype=np.float64) - np.asarray(pose.xy)
    forward = np.asarray((np.cos(pose.yaw), np.sin(pose.yaw)))
    left = np.asarray((-np.sin(pose.yaw), np.cos(pose.yaw)))
    return bool(abs(delta @ forward) <= altitude_m
                and abs(delta @ left) <= altitude_m)


def first_joint_path_observation(start_pose, top1_world, top2_world,
                                 protocol=OfficialProtocol()):
    """Select a target-free observation pose on the midpoint progress path."""
    midpoint = Point2D(*(0.5 * (np.asarray(top1_world) + np.asarray(top2_world))))
    pose = start_pose
    fallback = (pose, 0)
    for action_count in range(1, protocol.max_action_len + 1):
        pose = move_toward(pose, midpoint, protocol.move_iteration)
        remaining = protocol.max_action_len - action_count
        top1_normalized = np.asarray(top1_world) / protocol.map_size_m
        pose_normalized = np.asarray(pose.xy) / protocol.map_size_m
        if optimistic_translation_steps(
                pose_normalized, top1_normalized, protocol) > remaining:
            break
        fallback = (pose, action_count)
        if (visible_in_official_crop(pose, top1_world, protocol.altitude_m)
                and visible_in_official_crop(
                    pose, top2_world, protocol.altitude_m)):
            return pose, action_count, True
        if pose.xy.dist_to(midpoint) <= 5.0:
            break
    return fallback[0], fallback[1], False


def balanced_indices(rows, count, seed):
    if count <= 0 or count >= len(rows):
        return list(range(len(rows)))
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["map_name"]].append(index)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    names = sorted(groups)
    while len(selected) < count:
        changed = False
        for name in names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
                changed = True
        if not changed:
            break
    return sorted(selected)


def semantic_support(tree, native, candidate_world, expected, radius):
    indices = tree.query_ball_point(candidate_world, radius)
    if not indices:
        return 0.0, 0.0, 0
    labels = native[np.asarray(indices, dtype=np.int64)]
    matches = int(np.isin(labels, expected).sum())
    fraction = matches / len(labels)
    quality = min(matches / 12.0, 1.0)
    return fraction, quality, len(labels)


def mode_prior(logits, modes):
    size = logits.shape[-1]
    values = []
    for x, y in modes:
        row = min(size - 1, int(y * size))
        column = min(size - 1, int(x * size))
        values.append(float(logits[row, column]))
    return torch.tensor(values).softmax(0).numpy()


def collect_split(args, split, rows, beliefs, predicted_types, true_types,
                  selected_indices):
    trajectories = load_mturk_trajectories(split, "all", 50.0)
    if len(rows) != len(trajectories):
        raise ValueError(f"row/trajectory length mismatch for {split}")
    grouped = defaultdict(list)
    for index in selected_indices:
        grouped[rows[index]["map_name"]].append(index)
    records = []
    for map_name, indices in sorted(grouped.items()):
        xyz, _, _, native = read_ply(
            ply_for_map(args.ply_root, map_name), args.point_stride,
            return_native_semantic=True)
        tree = cKDTree(xyz[:, :2])
        for index in indices:
            row, trajectory = rows[index], trajectories[index]
            if (row["map_name"] != trajectory.map_name
                    or canonical_instruction(row["description"])
                    != canonical_instruction(trajectory.descriptions[0])):
                raise ValueError(f"alignment mismatch: {split}/{index}")
            modes = belief_distinct_topk_xy(beliefs[index], 2)
            modes_world = np.stack([
                normalized_to_world(mode, map_name) for mode in modes])
            pose, action_step, joint_visible = first_joint_path_observation(
                trajectory.start_pose, modes_world[0], modes_world[1])
            visibility = np.asarray([
                visible_in_official_crop(pose, point) for point in modes_world])

            evidence, oracle_type_evidence, wrong_evidence, point_counts = [], [], [], []
            for visible, point in zip(visibility, modes_world):
                values = []
                for target_type in (
                        int(predicted_types[index]), int(true_types[index]),
                        (int(predicted_types[index]) + 1) % 4):
                    fraction, quality, count = semantic_support(
                        tree, native, point, NATIVE_TARGET_CLASSES[target_type],
                        args.candidate_radius)
                    values.append((0.5 * fraction + 0.5 * quality) if visible else 0.0)
                evidence.append(values[0]); oracle_type_evidence.append(values[1])
                wrong_evidence.append(values[2]); point_counts.append(count)
            records.append({
                "index": index, "map_name": map_name,
                "target": np.asarray(row["target"], dtype=np.float64),
                "modes": np.asarray(modes, dtype=np.float64),
                "prior": mode_prior(beliefs[index], modes),
                "evidence": np.asarray(evidence),
                "oracle_type_evidence": np.asarray(oracle_type_evidence),
                "wrong_type_evidence": np.asarray(wrong_evidence),
                "visible": visibility, "joint_visible": joint_visible,
                "observation_action_step": action_step,
                "point_counts": np.asarray(point_counts),
            })
        print(split, map_name, len(indices), flush=True)
        del tree, xyz, native
    return sorted(records, key=lambda record: record["index"])


def distances(records):
    return np.asarray([
        np.linalg.norm(row["modes"] - row["target"][None], axis=-1) * 410.0
        for row in records])


def ranks(records, beta, field="evidence"):
    score = np.stack([row["prior"] + beta * row[field] for row in records])
    return score.argmax(-1)


def metrics(records, selected):
    distance = distances(records)
    chosen = distance[np.arange(len(records)), selected]
    return {
        "count": len(records),
        "hit20": float(np.mean(chosen <= 20.0) * 100.0),
        "mean_ne_m": float(chosen.mean()),
        "median_ne_m": float(np.median(chosen)),
    }


def tune(train_records):
    best = None
    for beta in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0):
        value = metrics(train_records, ranks(train_records, beta))
        key = value["hit20"], -value["mean_ne_m"], -beta
        if best is None or key > best[0]:
            best = key, beta
    return best[1]


def shuffled_records(records, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    output = copy.deepcopy(records)
    for destination, source in zip(output, order):
        destination["shuffled_evidence"] = records[int(source)]["evidence"].copy()
    return output


def load_or_collect(args, device):
    if args.cache.exists():
        return torch.load(args.cache, map_location="cpu", weights_only=False)
    relation = torch.load(args.relation_cache, map_location="cpu", weights_only=True)
    type_features = torch.load(args.type_features, map_location="cpu", weights_only=True)
    type_model = train_probe(
        type_features["text"]["train_seen"],
        type_features["text"]["val_seen"], args.seed, 40, device)
    saved = {}
    for split, sample_count in (
            ("train_seen", args.train_samples), ("val_seen", args.val_samples)):
        rows = load_rows(args.candidate_dir / f"{split}.jsonl")
        data = BeliefInferenceDataset(rows, relation[split])
        beliefs, _ = predict_beliefs(args.checkpoint, data, device, 64)
        with torch.inference_mode():
            predicted = type_model(
                type_features["text"][split]["features"].float().to(device)
            ).argmax(-1).cpu().numpy()
        true_type = type_features["text"][split]["labels"].numpy()
        indices = balanced_indices(rows, sample_count, args.seed)
        saved[split] = collect_split(
            args, split, rows, beliefs, predicted, true_type, indices)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.cache)
    return saved


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = load_or_collect(args, device)
    train_records, val_records = saved["train_seen"], saved["val_seen"]
    beta = tune(train_records)
    static_rank = np.zeros(len(val_records), dtype=np.int64)
    pointcloud_rank = ranks(val_records, beta)
    oracle_type_rank = ranks(val_records, beta, "oracle_type_evidence")
    wrong_type_rank = ranks(val_records, beta, "wrong_type_evidence")
    shuffled = shuffled_records(val_records, args.seed)
    shuffled_rank = ranks(shuffled, beta, "shuffled_evidence")
    two_mode_oracle = distances(val_records).argmin(-1)
    result = {
        "stage": "fixed-50m path-local Top-1/Top-2 point-cloud teacher gate",
        "counts": {"train_seen": len(train_records), "val_seen": len(val_records)},
        "selected_on_train_seen": {"pointcloud_weight": beta},
        "val_seen": {
            "static_top1": metrics(val_records, static_rank),
            "predicted_type_pointcloud_teacher": metrics(
                val_records, pointcloud_rank),
            "oracle_language_type_pointcloud_teacher": metrics(
                val_records, oracle_type_rank),
            "wrong_type_control": metrics(val_records, wrong_type_rank),
            "shuffled_episode_evidence_control": metrics(
                shuffled, shuffled_rank),
            "two_mode_privileged_oracle": metrics(
                val_records, two_mode_oracle),
            "teacher_changed_top1_percent": float(
                np.mean(pointcloud_rank != 0) * 100.0),
            "joint_visibility_percent": float(np.mean([
                row["joint_visible"] for row in val_records]) * 100.0),
            "mean_observation_action_step": float(np.mean([
                row["observation_action_step"] for row in val_records])),
            "zero_point_candidate_percent": float(np.mean([
                np.any(row["point_counts"] == 0) for row in val_records]) * 100.0),
        },
        "anti_leakage": {
            "observation_pose_inputs": "start pose + predicted Top-1/Top-2 only",
            "observation_pose_has_target_argument": (
                "target" in inspect.signature(first_joint_path_observation).parameters),
            "pointcloud_role": "training teacher only",
            "fusion_weight_selection": "train_seen only",
            "val_seen_target_use": "final metrics only",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
        "scope": (
            "semantic teacher reranking at official path-local observations; "
            "not RGB prediction or deployable navigation"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

