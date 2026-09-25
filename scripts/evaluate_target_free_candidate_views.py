#!/usr/bin/env python3
"""Evaluate fixed-height RGB views chosen only from predicted candidates.

Unlike the human-trajectory replay, neither view position nor view order uses
the ground-truth target.  The first K geometric candidates are visited at 50 m
and their RGB semantic evidence is accumulated before reranking all candidates.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.projection import transform_points, view_corners, world_to_image_homography
from independent_nav.semantic import ProjectSemanticNet
from scripts.audit_target_type_alignment import train_probe
from scripts.build_altitude_view_benchmark import render_crop
from scripts.evaluate_temporal_candidate_reranking import (
    add_shuffled_control,
    episode_key,
    fused_rank,
    local_score,
    navigation_metrics,
    noisy_or,
    prepare_episode_data,
)


def render_observation(rgb, raster, center, altitude, image_size):
    corners = view_corners(float(center[0]), float(center[1]), 0.0, altitude)
    image = render_crop(rgb, raster, corners, image_size)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), corners


@torch.inference_mode()
def collect(args, split, episode_data, model, device):
    grouped = defaultdict(list)
    for item in episode_data.values():
        grouped[item["episode_key"][0]].append(item)

    output = []
    for map_name, items in sorted(grouped.items()):
        rgb = cv2.imread(str(args.data_root / "rgbd" / f"{map_name}.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(map_name)
        with rasterio.open(args.data_root / "rgbd" / f"{map_name}.tif") as raster:
            x_min, y_max = raster.bounds.left, raster.bounds.top
            for item in items:
                candidates_world = np.column_stack((
                    x_min + item["candidates"][:, 0] * 410.0,
                    y_max - item["candidates"][:, 1] * 410.0,
                ))
                correct_frames, wrong_frames = [], []
                semantic_class = item["predicted_type"] + 1
                wrong_class = ((item["predicted_type"] + 1) % 4) + 1
                for visit_rank in range(min(args.views, len(candidates_world))):
                    rgb_view, corners = render_observation(
                        rgb, raster, candidates_world[visit_rank], args.altitude, args.render_size)
                    resized = cv2.resize(rgb_view, (args.image_size, args.image_size),
                                         interpolation=cv2.INTER_AREA)
                    tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1)[None]
                    probabilities = model(tensor.float().to(device).div(255.0)).softmax(1)[0].cpu().numpy()
                    homography = world_to_image_homography(corners, args.image_size)
                    pixels = transform_points(candidates_world, homography)
                    radius_pixels = args.radius_m / (2.0 * args.altitude) * args.image_size
                    correct_frames.append(np.asarray([
                        local_score(probabilities[semantic_class], point, radius_pixels)
                        for point in pixels]))
                    wrong_frames.append(np.asarray([
                        local_score(probabilities[wrong_class], point, radius_pixels)
                        for point in pixels]))

                row = dict(item)
                for count in range(1, len(correct_frames) + 1):
                    row[f"views_{count}"] = noisy_or(np.stack(correct_frames[:count]))
                    row[f"wrong_views_{count}"] = noisy_or(np.stack(wrong_frames[:count]))
                row["temporal"] = row[f"views_{len(correct_frames)}"]
                output.append(row)
    return output


def tune(sequences, views):
    best = None
    for count in range(1, views + 1):
        field = f"views_{count}"
        for beta in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
            value = navigation_metrics(sequences, fused_rank(sequences, field, beta))
            key = (value["hit20"], -value["mean_ne_m"], -count, -beta)
            if best is None or key > best[0]:
                best = key, count, beta
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--type-features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--radius-m", type=float, default=10.0)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectSemanticNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    episodes = prepare_episode_data(
        args.data_root, args.candidate_root, args.type_features, device)
    seen = collect(args, "val_seen", episodes["val_seen"], model, device)
    unseen = collect(args, "val_unseen", episodes["val_unseen"], model, device)
    selected_views, beta = tune(seen, args.views)
    field = f"views_{selected_views}"
    wrong_field = f"wrong_views_{selected_views}"
    shuffled, changed = add_shuffled_control([
        {**row, "temporal": row[field]} for row in unseen])

    zero = np.zeros(len(unseen), dtype=np.int64)
    per_view = {
        str(count): navigation_metrics(
            unseen, fused_rank(unseen, f"views_{count}", beta))
        for count in range(1, args.views + 1)
    }
    result = {
        "config": {
            "altitude_m": args.altitude,
            "candidate_radius_m": args.radius_m,
            "maximum_views": args.views,
            "views_selected_on_val_seen": selected_views,
            "prior_weight_selected_on_val_seen": beta,
            "view_order": "geometric candidate ranks 1..K",
        },
        "val_seen_episodes": len(seen),
        "val_unseen": {
            "episodes": len(unseen),
            "geometry_generator_top1": navigation_metrics(unseen, zero),
            "target_free_rgb_by_view_count": per_view,
            "selected_target_free_rgb_fusion": navigation_metrics(
                unseen, fused_rank(unseen, field, beta)),
            "wrong_target_type_control": navigation_metrics(
                unseen, fused_rank(unseen, wrong_field, beta)),
            "shuffled_episode_evidence_control": navigation_metrics(
                shuffled, fused_rank(shuffled, "shuffled", beta)),
            "shuffled_evidence_changed_percent": changed * 100.0,
            "top5_oracle": navigation_metrics(unseen, np.asarray([
                np.linalg.norm(row["candidates"] - row["target"], axis=-1).argmin()
                for row in unseen])),
        },
        "anti_leakage": {
            "view_position": "predicted candidate coordinates only",
            "view_order": "candidate prior rank only",
            "altitude": "fixed 50 m by default",
            "target_coordinates": "val_seen hyperparameter selection and metrics only",
            "target_mask_or_object_identity": False,
            "val_unseen_used_for_selection": False,
        },
        "scope": "target-free offline active-view candidate reranking; simulator dynamics excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
