#!/usr/bin/env python3
"""Test direct map-prism-to-RGB landmark binding without predicted depth."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely import contains_xy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.landmark import project_landmark_prism
from independent_nav.pointcloud import unproject_depth
from independent_nav.semantic import ProjectSemanticDepthNet
from scripts.evaluate_pointcloud_oracle_ladder import rotated_rows
from scripts.validate_perspective_landmark_matching import build_catalog, resolve


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--min-visible-pixels", type=int, default=8)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--max-visualizations", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectSemanticDepthNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    processed = json.load(open(args.data_root / "cityrefer/processed_descriptions.json"))
    catalogs, groups = build_catalog(objects)
    trajectories = json.load(open(
        args.data_root / "processed_citynav" / f"citynav_{args.split}.json"))
    by_map = defaultdict(list)
    for episode in trajectories:
        by_map[f'{episode["area"]}_block_{episode["block"]}'].append(episode)
    manifest = [json.loads(line) for line in
                (args.dataset / "manifest.jsonl").read_text().splitlines()]
    manifest = [row for row in manifest if row["split"] == args.split]
    wrong_rows = rotated_rows(manifest)

    records = []
    visualization_count = 0
    for row in manifest:
        values = np.load(args.dataset / row["file"])
        image = torch.from_numpy(values["rgb"].copy()).permute(2, 0, 1)[None]
        semantic, _ = model.predict_all(image.float().to(device).div(255.0))
        predicted_building = semantic.argmax(1)[0].cpu().numpy() == 1
        valid = values["valid"].astype(bool)
        world = unproject_depth(values["depth"].astype(np.float32), row["pose"], args.hfov)
        episode = by_map[row["map_name"]][row["episode_index_within_map"]]
        object_id, ann_id = str(episode["object_ids"][0]), int(episode["ann_ids"][0])
        annotations = processed.get(row["map_name"], {}).get(object_id, [])
        queries = annotations[ann_id].get("landmarks", []) if ann_id < len(annotations) else []
        for query_name in queries:
            landmark = resolve(query_name, groups[row["map_name"]])
            if landmark is None:
                continue
            inside = np.zeros(valid.shape, dtype=bool)
            inside[valid] = contains_xy(
                landmark["polygon"], world[..., 0][valid], world[..., 1][valid])
            true_pixels = inside & (values["semantic"] == 1)
            visible = int(true_pixels.sum())
            if visible < args.min_visible_pixels:
                continue
            obj = objects[row["map_name"]][str(landmark["id"])]
            mask = project_landmark_prism(
                obj["contour"], obj["position"], obj["dimension"], row["pose"],
                valid.shape, args.hfov)
            wrong_pose_mask = project_landmark_prism(
                obj["contour"], obj["position"], obj["dimension"],
                wrong_rows[row["frame_id"]]["pose"], valid.shape, args.hfov)
            wrong_pose_available = not np.allclose(
                row["pose"], wrong_rows[row["frame_id"]]["pose"])
            alternatives = [item for item in catalogs[row["map_name"]]
                            if item["id"] != landmark["id"]]
            wrong_name_available = bool(alternatives)
            if alternatives:
                wrong_landmark = min(
                    alternatives,
                    key=lambda item: item["polygon"].distance(landmark["polygon"]))
                wrong_obj = objects[row["map_name"]][str(wrong_landmark["id"])]
                wrong_name_mask = project_landmark_prism(
                    wrong_obj["contour"], wrong_obj["position"], wrong_obj["dimension"],
                    row["pose"], valid.shape, args.hfov)
            else:
                wrong_name_mask = np.zeros(valid.shape, dtype=bool)
            if (args.visualization_dir is not None
                    and visualization_count < args.max_visualizations):
                args.visualization_dir.mkdir(parents=True, exist_ok=True)
                overlay = values["rgb"].copy()
                overlay[predicted_building & valid] = (
                    0.55 * overlay[predicted_building & valid]
                    + 0.45 * np.asarray([255, 220, 0])).astype(np.uint8)
                edge_kernel = np.ones((3, 3), np.uint8)
                correct_edge = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_GRADIENT, edge_kernel) > 0
                wrong_edge = cv2.morphologyEx(
                    wrong_name_mask.astype(np.uint8), cv2.MORPH_GRADIENT, edge_kernel) > 0
                overlay[wrong_edge] = [40, 120, 255]
                overlay[correct_edge] = [30, 255, 30]
                overlay[true_pixels] = (
                    0.5 * overlay[true_pixels]
                    + 0.5 * np.asarray([255, 40, 40])).astype(np.uint8)
                panel = np.concatenate((values["rgb"], overlay), axis=1)
                cv2.putText(panel, "RGB", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, "yellow=building green=map red=teacher blue=wrong",
                            (valid.shape[1] + 3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.28, (255, 255, 255), 1, cv2.LINE_AA)
                path = args.visualization_dir / (
                    f'{visualization_count:02d}_{row["frame_id"]}_{landmark["id"]}.png')
                cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                visualization_count += 1
            result = {"visible_pixels": visible,
                      "wrong_pose_available": wrong_pose_available,
                      "wrong_name_available": wrong_name_available}
            for dilation in (0, 2, 4, 8):
                kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)

                def dilate(value):
                    return (cv2.dilate(value.astype(np.uint8), kernel) > 0
                            if dilation else value)

                for name, candidate in (("correct", mask), ("wrong_pose", wrong_pose_mask),
                                        ("wrong_name", wrong_name_mask)):
                    candidate = dilate(candidate)
                    result[f"{name}_{dilation}"] = float(
                        (predicted_building & valid & candidate & true_pixels).sum() / visible)
                result[f"geometry_recall_{dilation}"] = float(
                    (dilate(mask) & true_pixels).sum() / visible)
            records.append(result)

    output = {"split": args.split, "frames": len(manifest),
              "visible_named_landmark_queries": len(records), "threshold": 0.25,
              "dilation_sweep": {},
              "anti_leakage": {
                  "target_coordinates_or_identity": False,
                  "teacher_depth_and_semantic": "visibility and metrics only",
                  "runtime_inputs": "rendered RGB, camera pose, named map landmark prism",
                  "training": "none; deterministic feasibility gate",
              }}
    for dilation in (0, 2, 4, 8):
        wrong_pose_records = [row for row in records if row["wrong_pose_available"]]
        wrong_name_records = [row for row in records if row["wrong_name_available"]]
        output["dilation_sweep"][str(dilation)] = {
            "geometry_projection_recall": float(np.mean([
                row[f"geometry_recall_{dilation}"] >= 0.25 for row in records]) * 100),
            "rgb_semantic_correct_name_pose": float(np.mean([
                row[f"correct_{dilation}"] >= 0.25 for row in records]) * 100),
            "wrong_pose_control": float(np.mean([
                row[f"wrong_pose_{dilation}"] >= 0.25
                for row in wrong_pose_records]) * 100),
            "wrong_pose_queries": len(wrong_pose_records),
            "nearest_wrong_name_control": float(np.mean([
                row[f"wrong_name_{dilation}"] >= 0.25
                for row in wrong_name_records]) * 100),
            "wrong_name_queries": len(wrong_name_records),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
