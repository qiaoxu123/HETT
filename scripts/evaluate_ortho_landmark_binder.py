#!/usr/bin/env python3
"""Evaluate map-conditioned landmark binding on real orthomosaic RGB crops."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.projection import rasterize_polygon, view_corners, world_to_image_homography
from independent_nav.semantic import ProjectSemanticDepthNet, ProjectSemanticNet
from scripts.validate_perspective_landmark_matching import build_catalog, resolve


def rotated_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["map_name"]].append(row)
    result = {}
    for values in grouped.values():
        offset = max(1, len(values) // 2)
        for index, row in enumerate(values):
            result[row["image_id"]] = values[(index + offset) % len(values)]
    return result


def landmark_mask(landmark, row, image_size):
    corners = view_corners(
        float(row["x"]), float(row["y"]), float(row["yaw_rad"]),
        float(row["altitude_m"]))
    transform = world_to_image_homography(corners, image_size)
    return rasterize_polygon(landmark["polygon"], transform, image_size).astype(bool)


def summarize(records, threshold):
    groups = {"all": records}
    for altitude in sorted({row["altitude_m"] for row in records}):
        groups[str(altitude)] = [
            row for row in records if row["altitude_m"] == altitude]
    output = {}
    for name, values in groups.items():
        output[name] = {
            "queries": len(values),
            "correct_name_pose": float(np.mean([
                row["coverage"] >= threshold for row in values]) * 100),
            "wrong_pose_control": float(np.mean([
                row["wrong_pose_coverage"] >= threshold for row in values]) * 100),
            "nearest_wrong_name_leakage": float(np.mean([
                row["wrong_name_leakage"] >= threshold for row in values]) * 100),
            "median_building_coverage": float(np.median([
                row["coverage"] for row in values])),
        }
    return output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--min-mask-pixels", type=int, default=8)
    parser.add_argument("--min-visible-fraction", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--max-visualizations", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = (ProjectSemanticDepthNet() if "depth_output.weight" in state
             else ProjectSemanticNet()).to(device)
    model.load_state_dict(state)
    model.eval()
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    catalogs, groups = build_catalog(objects)
    rows = [json.loads(line) for line in
            (args.benchmark / "manifest.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["split"] == args.split]
    wrong_rows = rotated_rows(rows)
    records, visualization_count = [], 0

    for row in rows:
        source = cv2.imread(str(args.benchmark / "images" / row["file_name"]))
        if source is None:
            continue
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (args.image_size, args.image_size),
                         interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None]
        semantic = model(image.float().to(device).div(255.0))
        predicted_building = semantic.argmax(1)[0].cpu().numpy() == 1
        names = row.get("processed_description", {}).get("landmarks", [])
        visible_fractions = row.get("landmark_visible_fractions", [])
        for query_index, query_name in enumerate(names):
            if (query_index >= len(visible_fractions)
                    or visible_fractions[query_index] < args.min_visible_fraction):
                continue
            landmark = resolve(query_name, groups[row["map_name"]])
            if landmark is None:
                continue
            mask = landmark_mask(landmark, row, args.image_size)
            pixels = int(mask.sum())
            if pixels < args.min_mask_pixels:
                continue
            wrong_pose_mask = landmark_mask(
                landmark, wrong_rows[row["image_id"]], args.image_size)
            alternatives = [item for item in catalogs[row["map_name"]]
                            if item["id"] != landmark["id"]]
            wrong_name_mask = np.zeros_like(mask)
            if alternatives:
                wrong_landmark = min(
                    alternatives,
                    key=lambda item: item["polygon"].distance(landmark["polygon"]))
                wrong_name_mask = landmark_mask(wrong_landmark, row, args.image_size)
            record = {
                "altitude_m": float(row["altitude_m"]),
                "coverage": float((predicted_building & mask).sum() / pixels),
                "wrong_pose_coverage": float(
                    (predicted_building & mask & wrong_pose_mask).sum() / pixels),
                "wrong_name_leakage": float(
                    (predicted_building & mask & wrong_name_mask).sum() / pixels),
            }
            records.append(record)
            if (args.visualization_dir is not None
                    and visualization_count < args.max_visualizations):
                args.visualization_dir.mkdir(parents=True, exist_ok=True)
                overlay = rgb.copy()
                overlay[predicted_building] = (
                    0.55 * overlay[predicted_building]
                    + 0.45 * np.asarray([255, 220, 0])).astype(np.uint8)
                edge_kernel = np.ones((3, 3), np.uint8)
                edge = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_GRADIENT, edge_kernel) > 0
                wrong_edge = cv2.morphologyEx(
                    wrong_name_mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                    edge_kernel) > 0
                overlay[wrong_edge] = [40, 120, 255]
                overlay[edge] = [30, 255, 30]
                panel = np.concatenate((rgb, overlay), axis=1)
                output_path = args.visualization_dir / (
                    f'{visualization_count:02d}_{row["image_id"]}_{landmark["id"]}.png')
                cv2.imwrite(str(output_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                visualization_count += 1

    result = {
        "split": args.split,
        "source_images": len(rows),
        "visible_named_landmark_queries": len(records),
        "metrics_by_altitude_m": summarize(records, args.threshold),
        "threshold_sweep": {
            str(threshold): {
                "correct_name_pose": float(np.mean([
                    row["coverage"] >= threshold for row in records]) * 100),
                "wrong_pose_control": float(np.mean([
                    row["wrong_pose_coverage"] >= threshold for row in records]) * 100),
                "nearest_wrong_name_leakage": float(np.mean([
                    row["wrong_name_leakage"] >= threshold for row in records]) * 100),
            }
            for threshold in (0.1, 0.25, 0.5, 0.75)
        },
        "anti_leakage": {
            "goal_target_coordinates_or_identity": False,
            "landmark_geometry": "named map polygon from the instruction",
            "model_input": "orthomosaic RGB only",
            "visibility": "benchmark metadata used only to select evaluable queries",
            "wrong_pose": "another view from the same map",
        },
        "scope": "orthomosaic RGB transfer gate; not AirSim perspective RGB",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
