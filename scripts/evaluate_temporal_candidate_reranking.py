#!/usr/bin/env python3
"""Replay CityNav RGB observations to rerank independent geometric candidates."""

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


def episode_key(row):
    return (f'{row["area"]}_block_{row["block"]}',
            str(row["object_ids"][0]), int(row["ann_ids"][0]))


def candidate_key(row):
    return row["episode_key"]


def noisy_or(values):
    return 1.0 - np.prod(1.0 - np.clip(values, 0.0, 1.0), axis=0)


def local_score(probability, point, radius):
    x, y = float(point[0]), float(point[1])
    height, width = probability.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return 0.0
    yy, xx = np.ogrid[:height, :width]
    selected = probability[(xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2]
    if not len(selected):
        return 0.0
    count = max(4, int(np.ceil(len(selected) * 0.1)))
    return float(np.partition(selected, -count)[-count:].mean())


def navigation_metrics(sequences, rank):
    distances = np.asarray([
        np.linalg.norm(row["candidates"] - row["target"], axis=-1) * 410.0
        for row in sequences])
    selected = distances[np.arange(len(rank)), rank]
    return {
        "count": len(rank),
        "hit20": float((selected <= 20.0).mean() * 100),
        "mean_ne_m": float(selected.mean()),
        "median_ne_m": float(np.median(selected)),
    }


def fused_rank(sequences, field, beta):
    visual = np.asarray([row[field] for row in sequences])
    prior = -np.arange(visual.shape[1], dtype=np.float64) / max(visual.shape[1] - 1, 1)
    return np.argmax(visual + beta * prior[None], axis=1)


def tune_beta(sequences):
    best = None
    for beta in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        value = navigation_metrics(sequences, fused_rank(sequences, "temporal", beta))
        key = (value["hit20"], -value["mean_ne_m"], -beta)
        if best is None or key > best[0]:
            best = key, beta
    return best[1]


def prepare_episode_data(data_root, candidate_root, type_features, device):
    features = torch.load(type_features, map_location="cpu", weights_only=True)
    text_model = train_probe(
        features["text"]["train_seen"], features["text"]["val_seen"],
        seed=0, epochs=40, device=device)
    prepared = {}
    for split in ("val_seen", "val_unseen"):
        trajectories = json.load(open(
            data_root / "processed_citynav" / f"citynav_{split}.json"))
        predictions = text_model(
            features["text"][split]["features"].float().to(device)).argmax(-1).cpu()
        predicted_type = {episode_key(row): int(predictions[index])
                          for index, row in enumerate(trajectories)}
        filtered = [row for row in trajectories
                    if row.get("dist_marker_to_target", 0.0) <= 30.0]
        candidates = [json.loads(line) for line in
                      (candidate_root / f"{split}.jsonl").read_text().splitlines()]
        if len(filtered) != len(candidates):
            raise ValueError(f"candidate alignment mismatch for {split}")
        values = {}
        for trajectory, candidate in zip(filtered, candidates):
            key = episode_key(trajectory)
            if (candidate["map_name"] != key[0]
                    or candidate["description"] != trajectory["descriptions"][0]):
                raise ValueError(f"candidate semantic mismatch for {key}")
            values[key] = {
                "episode_key": key,
                "candidates": np.asarray(candidate["candidates"][:5], dtype=np.float64),
                "target": np.asarray(candidate["target"], dtype=np.float64),
                "predicted_type": predicted_type[key],
            }
        prepared[split] = values
    return prepared


@torch.inference_mode()
def collect(args, split, episode_data, model, device):
    rows = [json.loads(line) for line in
            (args.benchmark / "manifest.jsonl").read_text().splitlines()]
    rows = [row for row in rows
            if row["split"] == split and float(row["altitude_m"]) == args.altitude]
    bounds = {}
    for map_name in {row["map_name"] for row in rows}:
        with rasterio.open(args.data_root / "rgbd" / f"{map_name}.tif") as raster:
            bounds[map_name] = (raster.bounds.left, raster.bounds.top)
    sequences = defaultdict(list)
    for row in rows:
        key = (row["map_name"], str(row["object_id"]), int(row["ann_id"]))
        if key not in episode_data:
            continue
        item = episode_data[key]
        source = cv2.imread(str(args.benchmark / "images" / row["file_name"]))
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (args.image_size, args.image_size),
                         interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None]
        probabilities = model(image.float().to(device).div(255.0)).softmax(1)[0].cpu().numpy()
        semantic_class = item["predicted_type"] + 1
        wrong_class = ((item["predicted_type"] + 1) % 4) + 1
        x_min, y_max = bounds[row["map_name"]]
        world = np.column_stack((
            x_min + item["candidates"][:, 0] * 410.0,
            y_max - item["candidates"][:, 1] * 410.0))
        homography = world_to_image_homography(view_corners(
            row["x"], row["y"], row["yaw_rad"], row["altitude_m"]), args.image_size)
        pixels = transform_points(world, homography)
        radius_pixels = args.radius_m / (2.0 * args.altitude) * args.image_size
        correct = np.asarray([
            local_score(probabilities[semantic_class], point, radius_pixels)
            for point in pixels])
        wrong = np.asarray([
            local_score(probabilities[wrong_class], point, radius_pixels)
            for point in pixels])
        sequences[key].append((int(row["pose_index"]), correct, wrong))

    result = []
    for key, observations in sequences.items():
        observations.sort(key=lambda value: value[0])
        item = episode_data[key]
        result.append({
            **item,
            "first": observations[0][1],
            "temporal": noisy_or(np.stack([value[1] for value in observations])),
            "wrong_type": noisy_or(np.stack([value[2] for value in observations])),
            "observations": len(observations),
        })
    return result


def add_shuffled_control(sequences):
    groups = defaultdict(list)
    for index, row in enumerate(sequences):
        groups[(row["episode_key"][0], row["predicted_type"])].append(index)
    shuffled, changed = [], 0
    for values in groups.values():
        for offset, index in enumerate(values):
            replacement = values[(offset + 1) % len(values)]
            copy = dict(sequences[index])
            copy["shuffled"] = sequences[replacement]["temporal"]
            changed += int(replacement != index)
            shuffled.append(copy)
    return shuffled, changed / max(len(sequences), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--type-features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--radius-m", type=float, default=10.0)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectSemanticNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    episode_data = prepare_episode_data(
        args.data_root, args.candidate_root, args.type_features, device)
    seen = collect(args, "val_seen", episode_data["val_seen"], model, device)
    unseen = collect(args, "val_unseen", episode_data["val_unseen"], model, device)
    beta = tune_beta(seen)
    shuffled, changed = add_shuffled_control(unseen)

    zero = np.zeros(len(unseen), dtype=np.int64)
    result = {
        "config": {"altitude_m": args.altitude, "candidate_radius_m": args.radius_m,
                   "prior_weight_selected_on_val_seen": beta},
        "val_seen_replay_count": len(seen),
        "val_unseen": {
            "episodes": len(unseen),
            "mean_observations": float(np.mean([row["observations"] for row in unseen])),
            "geometry_generator_top1": navigation_metrics(unseen, zero),
            "single_frame_visual_fusion": navigation_metrics(
                unseen, fused_rank(unseen, "first", beta)),
            "temporal_visual_fusion": navigation_metrics(
                unseen, fused_rank(unseen, "temporal", beta)),
            "wrong_target_type_control": navigation_metrics(
                unseen, fused_rank(unseen, "wrong_type", beta)),
            "shuffled_episode_evidence_control": navigation_metrics(
                shuffled, fused_rank(shuffled, "shuffled", beta)),
            "shuffled_evidence_changed_percent": changed * 100.0,
            "top5_oracle": navigation_metrics(unseen, np.asarray([
                np.linalg.norm(row["candidates"] - row["target"], axis=-1).argmin()
                for row in unseen])),
        },
        "anti_leakage": {
            "runtime_candidate_inputs": "language, named-landmark geometry, RGB and pose",
            "target_coordinates": "val_seen weight selection and reported metrics only",
            "target_mask_or_object_identity": False,
            "instruction_type": "predicted by train_seen/val_seen-selected text probe",
            "val_unseen_used_for_selection": False,
            "important_limit": "teacher-forced human trajectory replay; benchmark poses were selected by target-distance bands",
        },
        "scope": "offline candidate reranking feasibility; not autonomous SR/SPL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
