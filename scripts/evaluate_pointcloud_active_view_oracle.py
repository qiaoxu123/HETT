#!/usr/bin/env python3
"""Point-cloud oracle gate for relation-conditioned candidate disambiguation.

The evaluated subset contains episodes whose Top-5 includes a successful point
while rank 1 is wrong.  Target coordinates are used only to form that diagnostic
subset and compute metrics.  View generation, selection and posterior updates
use predicted candidates, instruction landmarks/type, start pose and teacher
point-cloud visibility/semantics.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
from shapely import contains_xy
from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.pointcloud import PerspectivePointRenderer, read_ply, unproject_depth
from scripts.build_altitude_view_benchmark import resolve_landmarks
from scripts.evaluate_temporal_candidate_reranking import (
    episode_key,
    navigation_metrics,
    prepare_episode_data,
)


NATIVE_TARGET_CLASSES = {
    0: (2,),       # Building
    1: (9,),       # Car
    2: (0, 10),    # Ground / footpath
    3: (5,),       # Parking
}
ROAD_CLASSES = (7, 10)


def ply_for_map(root, map_name):
    matches = sorted(set(root.glob(f"*/{map_name}.ply")))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one PLY for {map_name}, found {matches}")
    return matches[0]


def balanced_subset(items, count):
    by_map = defaultdict(list)
    for item in items:
        by_map[item["episode_key"][0]].append(item)
    for values in by_map.values():
        values.sort(key=lambda row: (row["episode_key"][1], row["episode_key"][2]))
    output, maps = [], sorted(by_map)
    while len(output) < min(count, len(items)):
        changed = False
        for name in maps:
            if by_map[name] and len(output) < count:
                output.append(by_map[name].pop(0))
                changed = True
        if not changed:
            break
    return output


def eligible(items):
    output = []
    for item in items.values():
        distance = np.linalg.norm(item["candidates"] - item["target"], axis=-1) * 410.0
        if distance[0] > 20.0 and distance.min() <= 20.0:
            output.append(item)
    return output


def load_episode_metadata(data_root, split):
    trajectories = json.load(open(
        data_root / "processed_citynav" / f"citynav_{split}.json"))
    processed = json.load(open(data_root / "cityrefer" / "processed_descriptions.json"))
    metadata = {}
    for row in trajectories:
        key = episode_key(row)
        object_id, ann_id = key[1], key[2]
        annotation = processed.get(key[0], {}).get(object_id, [])
        annotation = annotation[ann_id] if ann_id < len(annotation) else {}
        metadata[key] = {
            "start": np.asarray(row["trajectory"][0][:2], dtype=np.float64),
            "landmark_names": annotation.get("landmarks", []),
            "description": row["descriptions"][0],
        }
    return metadata


def ground_height(xyz, native, xy, radius=10.0):
    distance = np.linalg.norm(xyz[:, :2] - np.asarray(xy)[None], axis=1)
    local = distance <= radius
    ground = local & np.isin(native, (0, 5, 7, 10))
    if ground.any():
        return float(np.median(xyz[ground, 2]))
    if local.any():
        return float(np.percentile(xyz[local, 2], 10))
    return float(np.percentile(xyz[:, 2], 10))


def generate_views(candidates, xyz, native, heights, directions, radius,
                   view_mode="cluster"):
    poses, descriptors = [], []
    anchors = ([(index, candidate) for index, candidate in enumerate(candidates)]
               if view_mode == "candidate_orbits"
               else [(-1, candidates.mean(0))])
    for candidate_index, candidate in anchors:
        target_z = ground_height(xyz, native, candidate) + 2.0
        for height in heights:
            for direction_index in range(directions):
                angle = 2.0 * math.pi * direction_index / directions
                camera_xy = candidate + radius * np.asarray([math.cos(angle), math.sin(angle)])
                camera_z = ground_height(xyz, native, camera_xy) + height
                camera = np.asarray([camera_xy[0], camera_xy[1], camera_z])
                target = np.asarray([candidate[0], candidate[1], target_z])
                look = target - camera
                look /= max(np.linalg.norm(look), 1e-9)
                poses.append(np.concatenate((camera, look)))
                descriptors.append((candidate_index, float(height), direction_index))
    return np.stack(poses), descriptors


def polygon_mask(valid, world, polygon):
    mask = np.zeros(valid.shape, dtype=bool)
    if valid.any() and not polygon.is_empty:
        mask[valid] = contains_xy(polygon, world[..., 0][valid], world[..., 1][valid])
    return mask


def observation_features(rendered, pose, candidates, predicted_type,
                         landmark_polygons, road_required, radius_m, hfov):
    valid = rendered["valid"]
    semantic = rendered["semantic"]
    depth = rendered["depth"]
    world = unproject_depth(depth, pose, hfov)
    landmark_visibility = []
    for polygon in landmark_polygons:
        inside = polygon_mask(valid, world, polygon)
        landmark_visibility.append(min(float(inside.sum()) / 20.0, 1.0))
    landmark_visibility = np.asarray(landmark_visibility, dtype=np.float64)
    landmark_joint = float(landmark_visibility.mean()) if len(landmark_visibility) else 1.0

    expected = NATIVE_TARGET_CLASSES[predicted_type]
    signatures = []
    for candidate in candidates:
        distance = np.linalg.norm(world[..., :2] - candidate[None, None], axis=-1)
        local = valid & (distance <= radius_m)
        target = local & np.isin(semantic, expected)
        visible_pixels = int(local.sum())
        target_pixels = int(target.sum())
        target_fraction = target_pixels / max(visible_pixels, 1)
        target_quality = min(target_pixels / 12.0, 1.0)
        visible_quality = min(visible_pixels / 20.0, 1.0)

        road_score = 1.0
        if road_required and landmark_polygons:
            values = []
            for polygon in landmark_polygons:
                center = np.asarray([polygon.centroid.x, polygon.centroid.y])
                corridor = LineString([candidate, center]).buffer(3.0)
                corridor_mask = polygon_mask(valid, world, corridor)
                values.append(min(float((corridor_mask & np.isin(
                    semantic, ROAD_CLASSES)).sum()) / 10.0, 1.0))
            road_score = max(values) if values else 0.0
        joint = target_quality * landmark_joint * road_score
        signatures.append((target_fraction, target_quality, visible_quality,
                           landmark_joint, road_score, joint))
    return np.asarray(signatures, dtype=np.float64), float(valid.mean())


def prior_scores(count, weight=0.3):
    return -weight * np.arange(count, dtype=np.float64) / max(count - 1, 1)


def select_view(record, discrimination_weight, joint_weight, cost_weight,
                risk_weight=0.25, field="signatures"):
    count = len(record["candidates"])
    belief = np.exp(prior_scores(count))
    belief /= belief.sum()
    pair_distance = np.linalg.norm(
        record["candidates_world"][:, None] - record["candidates_world"][None], axis=-1)
    best = None
    for index, signatures in enumerate(record[field]):
        discrimination = 0.0
        for left in range(count):
            for right in range(left + 1, count):
                # Candidates closer than the success radius do not need active separation.
                relevance = min(pair_distance[left, right] / 20.0, 1.0)
                delta = float(np.abs(signatures[left] - signatures[right]).mean())
                discrimination += belief[left] * belief[right] * relevance * delta
        joint = float(np.sum(belief * signatures[:, 5]))
        cost = np.linalg.norm(record["poses"][index, :2] - record["start"]) / 100.0
        risk = 1.0 - record["valid_fraction"][index]
        utility = (discrimination_weight * discrimination + joint_weight * joint
                   - cost_weight * cost - risk_weight * risk)
        key = (utility, -cost, -index)
        if best is None or key > best[0]:
            best = key, index
    return best[1]


def posterior_rank(record, view, target_weight, joint_weight, field="signatures"):
    signatures = record[field][view]
    score = (prior_scores(len(signatures))
             + target_weight * (0.5 * signatures[:, 0] + 0.5 * signatures[:, 1])
             + joint_weight * signatures[:, 5])
    return int(np.argmax(score))


def evaluate(records, config, field="signatures"):
    ranks, views = [], []
    for record in records:
        view = select_view(record, *config[:3], field=field)
        rank = posterior_rank(record, view, *config[3:], field=field)
        ranks.append(rank)
        views.append(view)
    return navigation_metrics(records, np.asarray(ranks)), np.asarray(ranks), views


def tune(records):
    best = None
    for discrimination in (0.5, 1.0, 2.0):
        for joint_view in (0.0, 0.5, 1.0):
            for cost in (0.0, 0.05, 0.1):
                for target in (0.5, 1.0, 2.0):
                    for joint_update in (0.0, 0.5, 1.0):
                        config = (discrimination, joint_view, cost, target, joint_update)
                        metrics, _, _ = evaluate(records, config)
                        key = (metrics["hit20"], -metrics["mean_ne_m"], tuple(-x for x in config))
                        if best is None or key > best[0]:
                            best = key, config
    return best[1]


def privileged_view_oracle(records, config):
    ranks = []
    for record in records:
        best = None
        for view in range(len(record["poses"])):
            rank = posterior_rank(record, view, *config[3:])
            distance = np.linalg.norm(record["candidates"][rank] - record["target"]) * 410.0
            key = (-float(distance <= 20.0), distance, view)
            if best is None or key < best[0]:
                best = key, rank
        ranks.append(best[1])
    return navigation_metrics(records, np.asarray(ranks))


def collect(args, split, items, metadata, objects):
    by_map = defaultdict(list)
    for item in items:
        by_map[item["episode_key"][0]].append(item)
    records = []
    for map_name, map_items in sorted(by_map.items()):
        xyz, rgb, _, native = read_ply(
            ply_for_map(args.ply_root, map_name), args.point_stride,
            return_native_semantic=True)
        with rasterio.open(args.data_root / "rgbd" / f"{map_name}.tif") as raster:
            x_min, y_max = raster.bounds.left, raster.bounds.top
        for item in map_items:
            candidates_world = np.column_stack((
                x_min + item["candidates"][:, 0] * 410.0,
                y_max - item["candidates"][:, 1] * 410.0))
            poses, descriptors = generate_views(
                candidates_world, xyz, native, args.heights,
                args.directions, args.orbit_radius, args.view_mode)
            lower = poses[:, :2].min(0) - args.max_range
            upper = poses[:, :2].max(0) + args.max_range
            local = ((xyz[:, 0] >= lower[0]) & (xyz[:, 0] <= upper[0])
                     & (xyz[:, 1] >= lower[1]) & (xyz[:, 1] <= upper[1]))
            renderer = PerspectivePointRenderer(xyz[local], rgb[local], native[local])
            episode = metadata[item["episode_key"]]
            landmarks, _ = resolve_landmarks(
                objects[map_name], episode["landmark_names"])
            landmark_polygons = []
            for landmark in landmarks:
                polygon = Polygon(landmark["contour"])
                landmark_polygons.append(polygon if polygon.is_valid else polygon.buffer(0))
            description = episode["description"].lower()
            road_required = any(word in description for word in (
                "across", "road", "street", "corner", "junction", "intersection"))
            signatures, wrong_signatures, validity = [], [], []
            for pose in poses:
                rendered = renderer.render(
                    pose, args.resolution, args.resolution, args.hfov, args.max_range)
                value, valid_fraction = observation_features(
                    rendered, pose, candidates_world, item["predicted_type"],
                    landmark_polygons, road_required, args.candidate_radius, args.hfov)
                signatures.append(value)
                wrong_value, _ = observation_features(
                    rendered, pose, candidates_world,
                    (item["predicted_type"] + 1) % 4,
                    landmark_polygons, road_required, args.candidate_radius, args.hfov)
                wrong_signatures.append(wrong_value)
                validity.append(valid_fraction)
            records.append({
                **item,
                "start": episode["start"],
                "candidates_world": candidates_world,
                "poses": poses,
                "view_descriptors": descriptors,
                "signatures": np.stack(signatures),
                "wrong_signatures": np.stack(wrong_signatures),
                "valid_fraction": np.asarray(validity),
                "road_required": road_required,
                "landmark_count_resolved": len(landmark_polygons),
            })
            del renderer
        print(split, map_name, len(map_items), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--type-features", type=Path, required=True)
    parser.add_argument("--ply-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=64)
    parser.add_argument("--heights", nargs="+", type=float, default=(20.0, 35.0, 50.0))
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--orbit-radius", type=float, default=30.0)
    parser.add_argument("--view-mode", choices=("cluster", "candidate_orbits"),
                        default="cluster")
    parser.add_argument("--candidate-radius", type=float, default=10.0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--max-range", type=float, default=110.0)
    parser.add_argument("--point-stride", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = prepare_episode_data(
        args.data_root, args.candidate_root, args.type_features, device)
    objects = json.load(open(args.data_root / "cityrefer" / "objects.json"))
    sampled, metadata = {}, {}
    population = {}
    for split in ("val_seen", "val_unseen"):
        values = eligible(episodes[split])
        population[split] = len(values)
        sampled[split] = balanced_subset(values, args.samples_per_split)
        metadata[split] = load_episode_metadata(args.data_root, split)
    seen = collect(args, "val_seen", sampled["val_seen"], metadata["val_seen"], objects)
    config = tune(seen)
    unseen = collect(args, "val_unseen", sampled["val_unseen"], metadata["val_unseen"], objects)
    result_metrics, ranks, views = evaluate(unseen, config)
    wrong_type_metrics, _, _ = evaluate(unseen, config, field="wrong_signatures")
    distances = np.asarray([
        np.linalg.norm(row["candidates"] - row["target"], axis=-1) * 410.0
        for row in unseen])
    oracle_rank = distances.argmin(-1)
    view_heights = [row["view_descriptors"][view][1]
                    for row, view in zip(unseen, views)]
    result = {
        "config": {
            "samples_per_split": args.samples_per_split,
            "heights_m": args.heights,
            "directions_per_candidate": args.directions,
            "orbit_radius_m": args.orbit_radius,
            "view_mode": args.view_mode,
            "candidate_evidence_radius_m": args.candidate_radius,
            "resolution": args.resolution,
            "point_stride": args.point_stride,
            "weights_selected_on_val_seen": {
                "discrimination": config[0], "joint_view": config[1],
                "travel_cost": config[2], "target_update": config[3],
                "joint_update": config[4],
            },
        },
        "diagnostic_population": population,
        "val_unseen_top5_contains_target_top1_wrong": {
            "sample_count": len(unseen),
            "baseline_top1_hit20": 0.0,
            "relation_conditioned_pointcloud_oracle": result_metrics,
            "wrong_target_type_control": wrong_type_metrics,
            "privileged_best_rendered_view": privileged_view_oracle(unseen, config),
            "top5_oracle": navigation_metrics(unseen, oracle_rank),
            "selected_height_percent": {
                str(height): 100.0 * view_heights.count(height) / max(len(view_heights), 1)
                for height in args.heights
            },
            "road_relation_percent": 100.0 * np.mean([row["road_required"] for row in unseen]),
            "mean_resolved_landmarks": float(np.mean([
                row["landmark_count_resolved"] for row in unseen])),
        },
        "anti_leakage": {
            "target_use": "diagnostic-subset filtering and final metrics only",
            "view_generation": "Top-5 candidates, fixed orbit, fixed heights and directions",
            "view_selection": "candidate priors, teacher semantic visibility, named landmarks, roads, travel and validity",
            "target_coordinate_or_identity_in_view_utility": False,
            "weights": "selected on val_seen before val_unseen evaluation",
            "near_candidate_pairs": "pairwise discrimination weight suppressed inside 20 m",
        },
        "scope": "sampled point-cloud teacher upper-bound gate; not RGB or closed-loop navigation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
