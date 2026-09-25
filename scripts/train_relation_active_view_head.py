#!/usr/bin/env python3
"""Train a runtime view-value head from point-cloud teacher utilities."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_pointcloud_active_view_oracle import (
    balanced_subset, collect, eligible, load_episode_metadata, navigation_metrics,
    posterior_rank, prior_scores, privileged_view_oracle, select_view,
)
from scripts.evaluate_temporal_candidate_reranking import episode_key
from scripts.train_relation_geometry_ranker import opposite_direction_cues
from scripts.train_relation_geometry_ranker import relation_cue_vector
from scripts.validate_candidate_geometry import mask_landmark_span


DEFAULT_RUNS = Path(
    "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/runs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_RUNS /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--type-features", type=Path,
                        default=DEFAULT_RUNS / "target_type_alignment/features.pt")
    parser.add_argument("--ply-root", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/DATA/SensatUrban_Dataset/"
        "SensatUrban_Dataset/ply"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--record-cache", type=Path,
                        default=Path("runs/relation_active_view_gate10/records.pt"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/relation_active_view_gate10"))
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--val-count", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def prepare_splits(args, device):
    features = torch.load(args.type_features, map_location="cpu", weights_only=True)
    type_model = train_probe(
        features["text"]["train_seen"], features["text"]["val_seen"],
        seed=args.seed, epochs=40, device=device)
    output = {}
    for split in ("train_seen", "val_seen"):
        trajectories = json.loads((args.data_root / "processed_citynav" /
                                   f"citynav_{split}.json").read_text())
        prediction = type_model(
            features["text"][split]["features"].float().to(device)
        ).argmax(-1).cpu()
        candidates = [json.loads(line) for line in
                      (args.candidate_root / f"{split}.jsonl").read_text().splitlines()
                      if line.strip()]
        if len(candidates) != len(trajectories):
            raise ValueError(f"candidate alignment mismatch for {split}")
        values = {}
        for index, (trajectory, candidate) in enumerate(zip(trajectories, candidates)):
            key = episode_key(trajectory)
            if candidate["map_name"] != key[0] or candidate["description"] != trajectory["descriptions"][0]:
                raise ValueError(f"semantic mismatch for {key}")
            values[key] = {
                "episode_key": key,
                "candidates": np.asarray(candidate["candidates"][:5], dtype=np.float64),
                "target": np.asarray(candidate["target"], dtype=np.float64),
                "predicted_type": int(prediction[index]),
            }
        output[split] = values
    return output


def teacher_utility(record, config=(0.5, 0.0, 0.1), risk_weight=0.25):
    count = len(record["candidates"])
    belief = np.exp(prior_scores(count)); belief /= belief.sum()
    pair_distance = np.linalg.norm(
        record["candidates_world"][:, None] - record["candidates_world"][None], axis=-1)
    values = []
    for view, signatures in enumerate(record["signatures"]):
        discrimination = 0.0
        for left in range(count):
            for right in range(left + 1, count):
                relevance = min(pair_distance[left, right] / 20.0, 1.0)
                delta = float(np.abs(signatures[left] - signatures[right]).mean())
                discrimination += belief[left] * belief[right] * relevance * delta
        joint = float(np.sum(belief * signatures[:, 5]))
        cost = np.linalg.norm(record["poses"][view, :2] - record["start"]) / 100.0
        risk = 1.0 - record["valid_fraction"][view]
        values.append(config[0] * discrimination + config[1] * joint
                      - config[2] * cost - risk_weight * risk)
    return np.asarray(values, dtype=np.float32)


def local_coordinates(points, camera, center):
    forward = center - camera
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.asarray([-forward[1], forward[0]])
    delta = points - camera[None]
    return np.column_stack((delta @ forward, delta @ right)) / 100.0


def view_features(record, mode="normal"):
    candidates = record["candidates_world"]
    center = candidates.mean(0)
    landmarks = record["landmark_centers_world"]
    landmark_cues = torch.zeros(9, 12)
    available_cues = torch.tensor(record["landmark_relation_cues"])
    cue_count = min(len(available_cues), 9)
    if cue_count:
        landmark_cues[:cue_count] = available_cues[:cue_count]
    if mode == "opposite_relation":
        landmark_cues = opposite_direction_cues(landmark_cues)
    values = []
    for view, (pose, descriptor) in enumerate(zip(
            record["poses"], record["view_descriptors"])):
        camera = pose[:2]
        candidate_local = local_coordinates(candidates, camera, center).reshape(-1)
        landmark_local = np.zeros((9, 2), dtype=np.float32)
        landmark_valid = np.zeros(9, dtype=np.float32)
        if mode != "no_landmarks":
            count = min(len(landmarks), 9)
            if count:
                landmark_local[:count] = local_coordinates(
                    landmarks[:count], camera, center)
                landmark_valid[:count] = 1.0
        else:
            landmark_cues.zero_()
        direction = 2.0 * math.pi * descriptor[2] / 8.0
        cost = np.linalg.norm(camera - record["start"]) / 100.0
        values.append(np.concatenate((
            [descriptor[1] / 50.0, math.sin(direction), math.cos(direction),
             cost, float(record["road_required"])],
            candidate_local.astype(np.float32), landmark_local.reshape(-1),
            landmark_valid, landmark_cues.numpy().reshape(-1).astype(np.float32),
        )))
    return np.stack(values).astype(np.float32)


def attach_per_landmark_cues(records, metadata):
    for record in records:
        if "landmark_relation_cues" in record:
            continue
        episode = metadata[record["episode_key"]]
        cues = []
        for query in episode["landmark_names"][:record["landmark_count_resolved"]]:
            prompt, _ = mask_landmark_span(episode["description"], query, query)
            cues.append(relation_cue_vector(prompt).numpy())
        record["landmark_relation_cues"] = np.asarray(
            cues, dtype=np.float32).reshape(-1, 12)


class ViewValueHead(nn.Module):
    def __init__(self, dimension, hidden=96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def tensors(records, mode="normal"):
    return (torch.tensor(np.stack([view_features(row, mode) for row in records])),
            torch.tensor(np.stack([teacher_utility(row) for row in records])))


def train_head(train_records, val_records, epochs, seed, device):
    train_x, train_y = tensors(train_records)
    val_x, val_y = tensors(val_records)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = ViewValueHead(train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    dataset = TensorDataset(train_x, train_y)
    best, state, best_epoch = None, None, None
    for epoch in range(1, epochs + 1):
        model.train()
        for features, utility in DataLoader(dataset, batch_size=32, shuffle=True):
            score = model(features.to(device))
            target = F.softmax(utility.to(device) / 0.05, dim=-1)
            loss = -(target * F.log_softmax(score, dim=-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = model(val_x.to(device)).cpu()
        selected = score.argmax(-1)
        row = torch.arange(len(selected))
        accuracy = (selected == val_y.argmax(-1)).float().mean().item()
        regret = (val_y.max(-1).values - val_y[row, selected]).mean().item()
        key = accuracy, -regret
        if best is None or key > best:
            best, best_epoch = key, epoch
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model.eval(), best_epoch


@torch.inference_mode()
def select_learned(model, records, device, mode="normal"):
    features, _ = tensors(records, mode)
    return model(features.to(device)).argmax(-1).cpu().numpy()


def ranks_for_views(records, views, config=(0.5, 0.0, 0.1, 2.0, 1.0)):
    return np.asarray([posterior_rank(record, int(view), *config[3:])
                       for record, view in zip(records, views)])


def view_navigation(records, views):
    return navigation_metrics(records, ranks_for_views(records, views))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.record_cache.exists():
        saved = torch.load(args.record_cache, map_location="cpu", weights_only=False)
        train_records, val_records = saved["train_seen"], saved["val_seen"]
        population = saved["population"]
    else:
        episodes = prepare_splits(args, device)
        population = {split: len(eligible(values)) for split, values in episodes.items()}
        sampled = {
            "train_seen": balanced_subset(eligible(episodes["train_seen"]), args.train_count),
            "val_seen": balanced_subset(eligible(episodes["val_seen"]), args.val_count),
        }
        objects = json.loads((args.data_root / "cityrefer" / "objects.json").read_text())
        render_args = SimpleNamespace(
            ply_root=args.ply_root, data_root=args.data_root,
            point_stride=24, heights=(20.0, 35.0, 50.0), directions=8,
            orbit_radius=30.0, view_mode="cluster", max_range=110.0,
            resolution=48, hfov=90.0, candidate_radius=10.0)
        metadata = {split: load_episode_metadata(args.data_root, split)
                    for split in ("train_seen", "val_seen")}
        train_records = collect(
            render_args, "train_seen", sampled["train_seen"],
            metadata["train_seen"], objects)
        val_records = collect(
            render_args, "val_seen", sampled["val_seen"],
            metadata["val_seen"], objects)
        args.record_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"train_seen": train_records, "val_seen": val_records,
                    "population": population}, args.record_cache)

    metadata = {split: load_episode_metadata(args.data_root, split)
                for split in ("train_seen", "val_seen")}
    attach_per_landmark_cues(train_records, metadata["train_seen"])
    attach_per_landmark_cues(val_records, metadata["val_seen"])

    model, epoch = train_head(
        train_records, val_records, args.epochs, args.seed, device)
    learned_views = select_learned(model, val_records, device)
    teacher_views = np.asarray([select_view(row, 0.5, 0.0, 0.1)
                                for row in val_records])
    fixed_views = np.asarray([next(
        index for index, descriptor in enumerate(row["view_descriptors"])
        if descriptor[1] == 35.0 and descriptor[2] == 0)
        for row in val_records])
    rng = np.random.default_rng(args.seed)
    random_views = np.asarray([rng.integers(len(row["poses"])) for row in val_records])
    wrong_relation_views = select_learned(model, val_records, device, "opposite_relation")
    no_landmark_views = select_learned(model, val_records, device, "no_landmarks")
    utility = np.stack([teacher_utility(row) for row in val_records])
    result = {
        "counts": {"train_seen": len(train_records), "val_seen": len(val_records)},
        "diagnostic_population": population,
        "selected_epoch_on_val_seen_teacher_utility": epoch,
        "view_selection": {
            "teacher_top1_match_percent": float(
                (learned_views == utility.argmax(-1)).mean() * 100.0),
            "mean_teacher_utility_regret": float(
                np.mean(utility.max(-1) - utility[np.arange(len(utility)), learned_views])),
        },
        "val_seen_diagnostic": {
            "learned_view_head": view_navigation(val_records, learned_views),
            "teacher_heuristic": view_navigation(val_records, teacher_views),
            "fixed_35m_direction0": view_navigation(val_records, fixed_views),
            "random_view": view_navigation(val_records, random_views),
            "opposite_relation_control": view_navigation(
                val_records, wrong_relation_views),
            "no_landmark_geometry_control": view_navigation(
                val_records, no_landmark_views),
            "privileged_best_view": privileged_view_oracle(
                val_records, (0.5, 0.0, 0.1, 2.0, 1.0)),
        },
        "anti_leakage": {
            "view_head_training": "train_seen only",
            "view_head_selection": "val_seen teacher utility only, not target success",
            "view_head_runtime_inputs": (
                "candidate geometry, named-landmark geometry, relation cues, pose, height"),
            "pointcloud": "training/evaluation observation teacher only",
            "target_coordinates": "diagnostic subset and final metrics only",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
        "scope": "view-selection teacher gate; observation update still uses teacher signatures",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "selected_epoch": epoch,
                "input_dimension": model.network[0].in_features},
               args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
