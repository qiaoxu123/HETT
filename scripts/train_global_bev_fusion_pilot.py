#!/usr/bin/env python3
"""Overfit gate for recurrent 64x64 RGB/language global-BEV fusion."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.belief_transformer import rasterize_landmarks
from independent_nav.global_bev_fusion import (
    GlobalBEVFusion, decode_exact_coordinates, exact_location_loss,
)
from independent_nav.online_fusion_sample import candidate_prior
from independent_nav.projection import (
    transform_points, view_corners, world_to_image_homography,
)
from multiagent.mapdata import MAP_BOUNDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("runs/online_fusion_pilot_step1")
    )
    parser.add_argument(
        "--candidate-dir", type=Path, default=Path(
            "../45-active-altitude/runs/candidate_rgb_data/"
            "geometry_top5_oof_grouped_enriched"
        )
    )
    parser.add_argument(
        "--relation-cache", type=Path, default=Path(
            "../45-active-altitude/runs/relation_geometry_ranker_contrastive/"
            "relation_features.pt"
        )
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/global_bev_fusion_pilot")
    )
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def world_grid(map_name, grid_size):
    bounds = MAP_BOUNDS[map_name]
    axis = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((
        bounds.x_min + xx.reshape(-1) * 410.0,
        bounds.y_max - yy.reshape(-1) * 410.0,
    ))


def project_rgb(image, frame, map_name, grid_size):
    height, width = image.shape[:2]
    pose = frame["pose"]
    corners = view_corners(pose["x"], pose["y"], pose["yaw"], 50.0)
    homography = world_to_image_homography(corners, width)
    pixel = transform_points(world_grid(map_name, grid_size), homography)
    map_x = pixel[:, 0].reshape(grid_size, grid_size).astype(np.float32)
    map_y = pixel[:, 1].reshape(grid_size, grid_size).astype(np.float32)
    valid = ((map_x >= 0) & (map_x <= width - 1)
             & (map_y >= 0) & (map_y <= height - 1))
    projected = cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    projected[~valid] = 0
    return projected, valid


def landmark_union(episode, grid_size, max_landmarks=9):
    count = min(len(episode["landmark_boundaries"]), max_landmarks)
    boundaries = torch.zeros(1, max_landmarks, 16, 2)
    valid = torch.zeros(1, max_landmarks, dtype=torch.bool)
    if count:
        boundaries[0, :count] = torch.tensor(
            episode["landmark_boundaries"][:count]
        )
        valid[0, :count] = True
    return rasterize_landmarks(boundaries, valid, grid_size)[0].any(0).float()


def prepare_split(args, split, relation_cache):
    episodes = load_jsonl(args.dataset / f"{split}.jsonl")
    candidates = load_jsonl(args.candidate_dir / f"{split}.jsonl")
    maximum = max(len(row["frames"]) for row in episodes)
    rgb, visibility, frame_valid = [], [], []
    priors, landmarks, relations, targets = [], [], [], []
    for episode in episodes:
        source = int(episode["source_row"])
        row = candidates[source]
        if row["description"] != episode["instruction"]:
            raise ValueError(f"semantic alignment mismatch at {split}:{source}")
        images, masks = [], []
        for frame in episode["frames"]:
            image = cv2.imread(str(args.dataset / frame["rgb_file"]), cv2.IMREAD_COLOR)
            projected, visible = project_rgb(
                image, frame, episode["map_name"], args.grid_size
            )
            images.append(torch.from_numpy(projected.copy()).permute(2, 0, 1))
            masks.append(torch.from_numpy(visible))
        valid_steps = [True] * len(images)
        while len(images) < maximum:
            images.append(torch.zeros_like(images[0]))
            masks.append(torch.zeros_like(masks[0]))
            valid_steps.append(False)
        rgb.append(torch.stack(images).float().div(255.0))
        visibility.append(torch.stack(masks))
        frame_valid.append(torch.tensor(valid_steps))
        priors.append(torch.from_numpy(candidate_prior(
            episode["ranked_candidates"], args.grid_size
        )))
        landmarks.append(landmark_union(episode, args.grid_size))
        count = max(1, min(len(episode["landmark_names"]), 9))
        relations.append(relation_cache[split][source, :count].float().mean(0))
        # Labels are joined here, after all inference-visible tensors exist.
        targets.append(torch.tensor(row["target"], dtype=torch.float32))
    return {
        "rgb": torch.stack(rgb), "visibility": torch.stack(visibility),
        "frame_valid": torch.stack(frame_valid), "initial_belief": torch.stack(priors),
        "landmark_bev": torch.stack(landmarks), "relation": torch.stack(relations),
        "target": torch.stack(targets),
    }


def to_device(data, device):
    return {key: value.to(device) for key, value in data.items()}


def forward(model, data, rgb=None):
    return model(
        data["initial_belief"], data["landmark_bev"], data["relation"],
        data["rgb"] if rgb is None else rgb, data["visibility"], data["frame_valid"],
    )


@torch.inference_mode()
def metrics(model, data):
    normal = forward(model, data)
    coordinate = decode_exact_coordinates(
        normal["belief_logits"], normal["offsets"], data["frame_valid"]
    )
    distance = torch.linalg.vector_norm(coordinate - data["target"], dim=-1) * 410.0
    zero = forward(model, data, torch.zeros_like(data["rgb"]))
    shuffled = forward(model, data, torch.roll(data["rgb"], 1, 0))
    normal_final = normal["belief_logits"][
        torch.arange(len(distance), device=distance.device),
        data["frame_valid"].long().sum(1) - 1,
    ]
    zero_final = zero["belief_logits"][
        torch.arange(len(distance), device=distance.device),
        data["frame_valid"].long().sum(1) - 1,
    ]
    shuffled_final = shuffled["belief_logits"][
        torch.arange(len(distance), device=distance.device),
        data["frame_valid"].long().sum(1) - 1,
    ]
    return {
        "count": len(distance),
        "mean_ne_m": float(distance.mean()),
        "median_ne_m": float(distance.median()),
        "hit5": float(distance.le(5).float().mean() * 100),
        "hit10": float(distance.le(10).float().mean() * 100),
        "hit20": float(distance.le(20).float().mean() * 100),
        "zero_rgb_mean_absolute_logit_change": float(
            (normal_final - zero_final).abs().mean()
        ),
        "shuffled_rgb_mean_absolute_logit_change": float(
            (normal_final - shuffled_final).abs().mean()
        ),
        "predicted_xy": coordinate.cpu().tolist(),
    }


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation_cache = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True
    )
    train = to_device(prepare_split(args, "train_seen", relation_cache), device)
    val = to_device(prepare_split(args, "val_seen", relation_cache), device)
    model = GlobalBEVFusion(args.grid_size, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best, best_state, history = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        output = forward(model, train)
        loss, pieces = exact_location_loss(
            output["belief_logits"], output["offsets"], train["target"],
            train["frame_valid"],
        )
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        value = float(loss.detach())
        if value < best:
            best = value
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            train_metrics = metrics(model.eval(), train)
            history.append({
                "epoch": epoch, "loss": value,
                "classification": float(pieces["classification"]),
                "offset": float(pieces["offset"]),
                "train_mean_ne_m": train_metrics["mean_ne_m"],
                "train_hit20": train_metrics["hit20"],
            })
            print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state); model.eval()
    train_metrics, val_metrics = metrics(model, train), metrics(model, val)
    # Changing labels after forward cannot change prediction because target is
    # not accepted by the model interface.
    before = forward(model, val)["belief_logits"].detach().clone()
    val["target"] = torch.roll(val["target"], 1, 0)
    after = forward(model, val)["belief_logits"].detach()
    target_independence = float((before - after).abs().max())
    gates = {
        "train_overfit_hit20_at_least_90": train_metrics["hit20"] >= 90.0,
        "train_overfit_mean_ne_at_most_10m": train_metrics["mean_ne_m"] <= 10.0,
        "rgb_changes_output": train_metrics["zero_rgb_mean_absolute_logit_change"] > 1e-4,
        "target_label_not_in_forward": target_independence == 0.0,
    }
    result = {
        "step": "Online Todo 2 minimal recurrent global BEV overfit gate",
        "model": {
            "grid": f"{args.grid_size}x{args.grid_size}",
            "cell_size_m": 410.0 / args.grid_size,
            "state": "ConvGRU-style persistent BEV",
            "output": "cell logits + continuous within-cell offset",
        },
        "train_seen_overfit": train_metrics,
        "val_seen_diagnostic_only": val_metrics,
        "history": history,
        "target_label_forward_max_difference": target_independence,
        "gates": gates,
        "passed": all(gates.values()),
        "anti_leakage": {
            "raw_pose_in_learned_model": False,
            "pose_use": "fixed RGB-to-BEV projection only",
            "target_coordinate_use": "exact location loss and metrics only",
            "target_used_for_frame_or_route_selection": False,
            "twenty_meter_radius": "evaluation only",
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
        "config": vars(args) | {"dataset": str(args.dataset),
                                "candidate_dir": str(args.candidate_dir),
                                "relation_cache": str(args.relation_cache),
                                "output_dir": str(args.output_dir)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": result["config"]},
               args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"train": train_metrics, "val": val_metrics,
                      "gates": gates, "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
