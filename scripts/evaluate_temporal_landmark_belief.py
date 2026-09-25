#!/usr/bin/env python3
"""Replay CityNav RGB views and evaluate persistent named-landmark evidence."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.belief import PositiveEvidenceBelief
from independent_nav.semantic import ProjectSemanticNet
from scripts.evaluate_ortho_landmark_binder import landmark_mask, rotated_rows
from scripts.validate_perspective_landmark_matching import build_catalog, resolve


def select_threshold(positive, negative):
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        tpr = float((positive >= threshold).mean())
        fpr = float((negative >= threshold).mean())
        key = (tpr - fpr, tpr, -fpr, threshold)
        if best is None or key > best[0]:
            best = key, float(threshold)
    return best[1]


def accumulate(values, floor):
    belief = PositiveEvidenceBelief(evidence_floor=floor)
    for value in values:
        belief.update(value)
    return belief.confidence


def summarize(sequences, threshold, floor):
    first = np.asarray([accumulate([row["values"][0]["correct"]], floor)
                        for row in sequences])
    temporal = np.asarray([accumulate(
        [value["correct"] for value in row["values"]], floor) for row in sequences])
    wrong_pose = np.asarray([accumulate(
        [value["wrong_pose"] for value in row["values"]], floor) for row in sequences])
    wrong_name = np.asarray([accumulate(
        [value["wrong_name"] for value in row["values"]], floor) for row in sequences])
    stable = np.asarray([sum(value["correct"] >= 0.25 for value in row["values"]) >= 2
                         for row in sequences])
    return {
        "sequences": len(sequences),
        "mean_observations": float(np.mean([len(row["values"]) for row in sequences])),
        "first_observation_confirmation": float((first >= threshold).mean() * 100),
        "temporal_confirmation": float((temporal >= threshold).mean() * 100),
        "temporal_delta_pp": float(((temporal >= threshold).mean()
                                    - (first >= threshold).mean()) * 100),
        "confirmed_in_at_least_two_frames": float(stable.mean() * 100),
        "wrong_pose_acceptance": float((wrong_pose >= threshold).mean() * 100),
        "nearest_wrong_name_acceptance": float((wrong_name >= threshold).mean() * 100),
        "confidence_median": float(np.median(temporal)),
    }, temporal, wrong_pose, wrong_name


@torch.inference_mode()
def collect(args, split, model, device, objects, catalogs, groups):
    rows = [json.loads(line) for line in
            (args.benchmark / "manifest.jsonl").read_text().splitlines()]
    rows = [row for row in rows
            if row["split"] == split and float(row["altitude_m"]) == args.altitude]
    wrong = rotated_rows(rows)
    sequences = defaultdict(list)
    visibility = defaultdict(bool)
    for row in rows:
        source = cv2.imread(str(args.benchmark / "images" / row["file_name"]))
        if source is None:
            continue
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (args.image_size, args.image_size),
                         interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None]
        prediction = model(image.float().to(device).div(255.0)).argmax(1)[0].cpu().numpy()
        predicted_building = prediction == 1
        names = row.get("processed_description", {}).get("landmarks", [])
        fractions = row.get("landmark_visible_fractions", [])
        for index, query_name in enumerate(names):
            landmark = resolve(query_name, groups[row["map_name"]])
            if landmark is None:
                continue
            mask = landmark_mask(landmark, row, args.image_size)
            pixels = int(mask.sum())
            if pixels < args.min_mask_pixels:
                continue
            wrong_pose_mask = landmark_mask(
                landmark, wrong[row["image_id"]], args.image_size)
            alternatives = [item for item in catalogs[row["map_name"]]
                            if item["id"] != landmark["id"]]
            wrong_name_mask = np.zeros_like(mask)
            if alternatives:
                nearest = min(
                    alternatives,
                    key=lambda item: item["polygon"].distance(landmark["polygon"]))
                wrong_name_mask = landmark_mask(nearest, row, args.image_size)
            key = (int(row["source_row"]), row["map_name"], landmark["id"])
            sequences[key].append({
                "pose_index": int(row["pose_index"]),
                "correct": float((predicted_building & mask).sum() / pixels),
                "wrong_pose": float(
                    (predicted_building & mask & wrong_pose_mask).sum() / pixels),
                "wrong_name": float(
                    (predicted_building & mask & wrong_name_mask).sum() / pixels),
            })
            if index < len(fractions) and fractions[index] >= args.min_visible_fraction:
                visibility[key] = True
    result = []
    for key, values in sequences.items():
        if visibility[key]:
            result.append({"key": key, "values": sorted(
                values, key=lambda value: value["pose_index"])})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--min-mask-pixels", type=int, default=8)
    parser.add_argument("--min-visible-fraction", type=float, default=0.1)
    parser.add_argument("--evidence-floor", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectSemanticNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    catalogs, groups = build_catalog(objects)
    seen = collect(args, "val_seen", model, device, objects, catalogs, groups)
    unseen = collect(args, "val_unseen", model, device, objects, catalogs, groups)

    _, seen_positive, seen_wrong_pose, seen_wrong_name = summarize(
        seen, 0.5, args.evidence_floor)
    threshold = select_threshold(
        seen_positive, np.concatenate((seen_wrong_pose, seen_wrong_name)))
    seen_result, _, _, _ = summarize(seen, threshold, args.evidence_floor)
    unseen_result, _, _, _ = summarize(unseen, threshold, args.evidence_floor)
    result = {
        "config": {"altitude_m": args.altitude, "evidence_floor": args.evidence_floor,
                   "threshold_selected_on_val_seen": threshold},
        "val_seen": seen_result,
        "val_unseen": unseen_result,
        "anti_leakage": {
            "goal_target_coordinates_or_identity": False,
            "threshold_selection": "val_seen only",
            "val_unseen_used_for_selection": False,
            "runtime_inputs": "RGB, trajectory pose, instruction landmark, map polygon",
            "visibility_metadata": "positive sequence eligibility for metrics only",
            "negative_evidence": "disabled because occlusion is unknown",
            "temporal_update": "positive noisy-OR; order invariant",
        },
        "scope": "named-landmark temporal confirmation, not target localization or SR/SPL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
