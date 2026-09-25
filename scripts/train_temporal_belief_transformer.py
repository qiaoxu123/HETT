#!/usr/bin/env python3
"""Train the RGB temporal evidence head on top of the frozen stage-1 model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.belief_transformer import (
    BeliefTransformer, TemporalBeliefTransformer, belief_loss, grid_centers,
)
from independent_nav.projection import transform_points, view_corners, world_to_image_homography
from independent_nav.semantic import ProjectSemanticNet
from multiagent.mapdata import MAP_BOUNDS
from scripts.train_belief_transformer import (
    BeliefDataset, load_rows, predict_metrics, to_model_args,
)
from scripts.audit_target_type_alignment import CLASS_TO_INDEX


DEFAULT_SOURCE = Path(
    "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/runs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("runs/temporal_belief_rgb_gate1"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--semantic-checkpoint", type=Path,
                        default=Path("runs/pointcloud_semantic_mixed_gate2/model.pt"))
    parser.add_argument("--static-checkpoint", type=Path,
                        default=Path("runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/temporal_belief_transformer_gate1"))
    parser.add_argument("--feature-cache", type=Path,
                        default=Path("runs/temporal_belief_rgb_gate1/semantic_bev.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.1)
    parser.add_argument("--evidence-radius-m", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def read_manifest(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def normalized_cells_world(map_name, grid_size):
    cells = grid_centers(grid_size).reshape(-1, 2).numpy()
    bounds = MAP_BOUNDS[map_name]
    return np.column_stack((
        bounds.x_min + cells[:, 0] * 410.0,
        bounds.y_max - cells[:, 1] * 410.0,
    ))


@torch.inference_mode()
def project_episode(model, rows, dataset_root, grid_size, evidence_radius_m, device):
    images = []
    for frame in rows["frames"]:
        source = cv2.imread(str(dataset_root / frame["file"]), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(dataset_root / frame["file"])
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
    image = torch.stack(images).float().to(device).div(255.0)
    probability = model(image).softmax(1)
    height, width = image.shape[-2:]
    world = normalized_cells_world(rows["map_name"], grid_size)
    projected, visible, support = [], [], []
    for frame_index, frame in enumerate(rows["frames"]):
        corners = view_corners(
            frame["x"], frame["y"], frame["yaw_rad"], frame["altitude_m"])
        homography = world_to_image_homography(corners, width)
        pixel = transform_points(world, homography)
        valid = ((pixel[:, 0] >= 0) & (pixel[:, 0] <= width - 1)
                 & (pixel[:, 1] >= 0) & (pixel[:, 1] <= height - 1))
        grid = torch.from_numpy(np.column_stack((
            pixel[:, 0] / max(width - 1, 1) * 2.0 - 1.0,
            pixel[:, 1] / max(height - 1, 1) * 2.0 - 1.0,
        )).astype(np.float32)).reshape(1, grid_size, grid_size, 2).to(device)
        sampled = F.grid_sample(
            probability[frame_index:frame_index + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        projected.append(sampled.cpu().half())
        visible.append(torch.from_numpy(valid.reshape(grid_size, grid_size)))
        local = np.linalg.norm(
            world - np.asarray([frame["x"], frame["y"]]), axis=-1
        ) <= evidence_radius_m
        support.append(torch.from_numpy(
            (valid & local).reshape(grid_size, grid_size)))
    return torch.stack(projected), torch.stack(visible), torch.stack(support)


def build_feature_cache(args, device):
    if args.feature_cache.exists():
        cached = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
        if (cached.get("metadata", {}).get("evidence_radius_m")
                == args.evidence_radius_m):
            return cached
    model = ProjectSemanticNet().to(device)
    model.load_state_dict(torch.load(
        args.semantic_checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    cache = {"splits": {}, "metadata": {
        "semantic_checkpoint": str(args.semantic_checkpoint.resolve()),
        "semantic_checkpoint_sha256": digest(args.semantic_checkpoint),
        "grid_size": args.grid_size,
        "evidence_radius_m": args.evidence_radius_m,
    }}
    for split in ("train_seen", "val_seen"):
        manifest_path = args.dataset / f"{split}.jsonl"
        manifest = read_manifest(manifest_path)
        maximum_frames = max(len(row["frames"]) for row in manifest)
        semantics, visibility, evidence_support = [], [], []
        for index, row in enumerate(manifest):
            semantic, visible, support = project_episode(
                model, row, args.dataset, args.grid_size,
                args.evidence_radius_m, device)
            if len(semantic) < maximum_frames:
                missing = maximum_frames - len(semantic)
                semantic = torch.cat((semantic, torch.zeros(
                    missing, semantic.shape[1], args.grid_size, args.grid_size,
                    dtype=semantic.dtype)), dim=0)
                visible = torch.cat((visible, torch.zeros(
                    missing, args.grid_size, args.grid_size,
                    dtype=torch.bool)), dim=0)
                support = torch.cat((support, torch.zeros(
                    missing, args.grid_size, args.grid_size,
                    dtype=torch.bool)), dim=0)
            semantics.append(semantic); visibility.append(visible)
            evidence_support.append(support)
            if (index + 1) % 64 == 0:
                print(f"project {split} {index + 1}/{len(manifest)}", flush=True)
        cache["splits"][split] = {
            "source_rows": torch.tensor([row["source_row"] for row in manifest]),
            "semantic": torch.stack(semantics),
            "visibility": torch.stack(visibility),
            "evidence_support": torch.stack(evidence_support),
            "map_names": [row["map_name"] for row in manifest],
            "frame_counts": torch.tensor([len(row["frames"]) for row in manifest]),
            "manifest_sha256": digest(manifest_path),
        }
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.feature_cache)
    return cache


def same_map_shuffle(map_names):
    groups = defaultdict(list)
    for index, name in enumerate(map_names):
        groups[name].append(index)
    result = list(range(len(map_names)))
    for values in groups.values():
        if len(values) > 1:
            for offset, index in enumerate(values):
                result[index] = values[(offset + 1) % len(values)]
    return torch.tensor(result)


class TemporalTrainingDataset(Dataset):
    def __init__(self, base, features, target_types):
        self.base = base
        self.semantic = features["semantic"]
        self.visibility = features["visibility"]
        self.evidence_support = features["evidence_support"]
        self.shuffle = same_map_shuffle(features["map_names"])
        self.target_types = target_types

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        shuffled = int(self.shuffle[index])
        item.update({
            "semantic_bev": self.semantic[index],
            "visibility": self.visibility[index],
            "evidence_support": self.evidence_support[index],
            "shuffled_semantic_bev": self.semantic[shuffled],
            "shuffled_visibility": self.visibility[shuffled],
            "shuffled_evidence_support": self.evidence_support[shuffled],
            "shuffle_changed": shuffled != index,
            "target_type": self.target_types[index],
        })
        return item


def aligned_dataset(args, split, features, relation):
    rows = load_rows(args.candidate_dir / f"{split}.jsonl")
    source = features["source_rows"].tolist()
    selected_rows = [rows[index] for index in source]
    selected_relation = relation[split][source]
    # Source-row alignment is semantic, not merely positional.
    trajectories = json.loads((args.data_root / "processed_citynav" /
                               f"citynav_{split}.json").read_text())
    objects = json.loads((args.data_root / "cityrefer" / "objects.json").read_text())
    target_types = []
    for row, source_index in zip(selected_rows, source):
        if row["description"] != trajectories[source_index]["descriptions"][0]:
            raise ValueError(f"candidate/trajectory mismatch at {split}:{source_index}")
        trajectory = trajectories[source_index]
        name = f'{trajectory["area"]}_block_{trajectory["block"]}'
        target = objects[name][str(trajectory["object_ids"][0])]
        target_types.append(CLASS_TO_INDEX[target["object_type"]])
    base = BeliefDataset(selected_rows, selected_relation, args.grid_size, 9)
    return TemporalTrainingDataset(
        base, features, torch.tensor(target_types, dtype=torch.long))


def evidence_loss(raw, labels, visibility):
    target = labels[:, None].expand_as(raw).float()
    selected = visibility.bool()
    if not selected.any():
        return raw.sum() * 0.0
    selected_target = target[selected]
    positives = selected_target.sum()
    negatives = selected_target.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).detach().clamp(max=30.0)
    return F.binary_cross_entropy_with_logits(
        raw[selected], selected_target, pos_weight=pos_weight)


def temporal_loss(output, labels, visibility, target_type, evidence_weight):
    beliefs = output["belief_logits"]
    weights = torch.linspace(0.5, 1.0, beliefs.shape[1], device=beliefs.device)
    goal = sum(weight * belief_loss(beliefs[:, index], labels)
               for index, weight in enumerate(weights)) / weights.sum()
    type_loss = F.cross_entropy(output["target_type_logits"], target_type)
    return (goal + evidence_weight * evidence_loss(
        output["raw_evidence"], labels, visibility) + 0.2 * type_loss)


def temporal_args(batch, device):
    return to_model_args(batch, device)


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size):
    model.eval()
    initial, steps, shuffled, shuffled_rgb_fixed_pose, wrong_type = [], [], [], [], []
    pose_only, unmasked, targets, changed = [], [], [], []
    type_correct, type_total = 0, 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        args = temporal_args(batch, device)
        semantic = batch["semantic_bev"].to(device)
        visibility = batch["evidence_support"].to(device)
        normal = model(args, semantic, visibility)
        control = model(
            args, batch["shuffled_semantic_bev"].to(device),
            batch["shuffled_evidence_support"].to(device))
        fixed_pose_control = model(
            args, batch["shuffled_semantic_bev"].to(device), visibility)
        pose_control = model(args, torch.zeros_like(semantic), visibility)
        wrong_type_control = model(args, semantic, visibility, type_roll=1)
        invalid = model(args, semantic, torch.ones_like(visibility))
        initial.append(normal["initial_logits"].cpu())
        steps.append(normal["belief_logits"].cpu())
        shuffled.append(control["belief_logits"].cpu())
        shuffled_rgb_fixed_pose.append(
            fixed_pose_control["belief_logits"].cpu())
        pose_only.append(pose_control["belief_logits"].cpu())
        wrong_type.append(wrong_type_control["belief_logits"].cpu())
        unmasked.append(invalid["belief_logits"].cpu())
        targets.append(batch["target"])
        changed.append(batch["shuffle_changed"])
        type_correct += int((normal["target_type_logits"].argmax(-1).cpu()
                             == batch["target_type"]).sum())
        type_total += len(batch["target_type"])
    initial, steps = torch.cat(initial), torch.cat(steps)
    targets = torch.cat(targets)
    shuffled = torch.cat(shuffled)
    shuffled_rgb_fixed_pose = torch.cat(shuffled_rgb_fixed_pose)
    pose_only = torch.cat(pose_only)
    wrong_type = torch.cat(wrong_type)
    unmasked = torch.cat(unmasked)
    def per_frame(values):
        return [predict_metrics(values[:, index], targets,
                                model.static_model.grid_size)
                for index in range(values.shape[1])]
    result = {
        "static_initial": predict_metrics(initial, targets, model.static_model.grid_size),
        "after_frames": [predict_metrics(steps[:, index], targets,
                                          model.static_model.grid_size)
                         for index in range(steps.shape[1])],
        "shuffled_same_map_rgb": per_frame(shuffled),
        "shuffled_rgb_original_visibility": per_frame(shuffled_rgb_fixed_pose),
        "visibility_only_no_rgb": per_frame(pose_only),
        "wrong_target_type": per_frame(wrong_type),
        "visibility_mask_removed": per_frame(unmasked),
        "shuffle_changed_percent": torch.cat(changed).float().mean().mul(100).item(),
        "target_type_accuracy": 100.0 * type_correct / max(type_total, 1),
    }
    return result


def train(model, train_data, val_data, args, device):
    for parameter in model.static_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=args.learning_rate, weight_decay=1e-3)
    best, best_state, best_epoch, best_frames, history = None, None, None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        model.static_model.eval()
        for batch in DataLoader(train_data, batch_size=args.batch_size, shuffle=True):
            labels = batch["labels"].to(device)
            visibility = batch["evidence_support"].to(device)
            output = model(
                temporal_args(batch, device),
                batch["semantic_bev"].to(device), visibility)
            loss = temporal_loss(
                output, labels, visibility, batch["target_type"].to(device),
                args.evidence_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        value = evaluate(model, val_data, device, args.batch_size)
        choices = [(metric["hit20"], -metric["mean_ne"], -index, index, metric)
                   for index, metric in enumerate(value["after_frames"], 1)]
        epoch_choice = max(choices)
        key = epoch_choice[:3]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "val_seen": value})
        print(f"epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} "
              f"initial={value['static_initial']['hit20']:.2f} "
              f"temporal={epoch_choice[4]['hit20']:.2f} "
              f"frames={epoch_choice[3]}", flush=True)
        if best is None or key > best:
            best, best_epoch, best_frames = key, epoch, epoch_choice[3]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, best_frames, history


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = build_feature_cache(args, device)
    if args.cache_only:
        print(json.dumps({
            "feature_cache": str(args.feature_cache),
            "grid_size": features["metadata"]["grid_size"],
            "evidence_radius_m": features["metadata"]["evidence_radius_m"],
            "counts": {split: len(value["source_rows"])
                       for split, value in features["splits"].items()},
        }, indent=2))
        return
    relation = torch.load(args.relation_cache, map_location="cpu", weights_only=True)
    train_data = aligned_dataset(
        args, "train_seen", features["splits"]["train_seen"], relation)
    val_data = aligned_dataset(
        args, "val_seen", features["splits"]["val_seen"], relation)
    static = BeliefTransformer(
        grid_size=args.grid_size, hidden=args.hidden, layers=args.layers)
    checkpoint = torch.load(args.static_checkpoint, map_location="cpu", weights_only=True)
    static.load_state_dict(checkpoint["model"])
    model = TemporalBeliefTransformer(
        static, semantic_channels=5, hidden=args.temporal_hidden).to(device)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        model.eval()
        best_epoch = saved.get("selected_epoch", 0)
        best_frames = saved.get("selected_frames", 0)
        history = []
    else:
        model, best_epoch, best_frames, history = train(
            model, train_data, val_data, args, device)
    result = {
        "stage": "frozen RGB semantic projection + signed temporal belief update",
        "selected_epoch_on_val_seen": best_epoch,
        "selected_frames_on_val_seen": best_frames,
        "train_seen": evaluate(model, train_data, device, args.batch_size),
        "val_seen": evaluate(model, val_data, device, args.batch_size),
        "history": history,
        "anti_leakage": {
            "temporal_head_training": "train_seen only",
            "checkpoint_selection": "val_seen only",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
            "target_coordinates_in_model_forward": False,
            "target_usage": "belief labels and final distance metrics only",
            "frame_selection": "trajectory order only; no target coordinate/distance/identity",
            "static_model": "frozen stage-1 checkpoint",
            "semantic_model": "frozen train_seen/val_seen-selected checkpoint",
            "scope_limit": "human-trajectory perception replay, not autonomous view selection",
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint:
        torch.save({"model": model.state_dict(), "config": result["config"],
                    "selected_epoch": best_epoch,
                    "selected_frames": best_frames}, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["val_seen"], indent=2))


if __name__ == "__main__":
    main()
