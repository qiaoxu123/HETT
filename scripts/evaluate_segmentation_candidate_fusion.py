#!/usr/bin/env python3
"""Convert RGB segmentation into candidate semantic scores and fuse them."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_semantic_component_fusion import (
    evaluate, normalized, semantic_component_score, tune,
)
from scripts.fuse_relation_type_scores import type_compatibility
from scripts.train_candidate_score_calibrator import direct_similarity
from scripts.train_lightweight_semantic_segmentation import TinyUNet, crop_at
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, RelationGeometryRanker, build_split, load_rows,
    scores as relation_scores,
)
from scripts.train_rgb_candidate_ranker import metrics, normalized_to_world


ROOT = Path("runs")
DATA = Path("data")
SPLITS = ("val_unseen", "test_unseen")
SEGMENTATION_DIR = ROOT / "lightweight_semantic_segmentation_40m"
SEGMENTATION_SIDE_M = 40.0


@torch.inference_mode()
def load_or_predict(rows, device, side_m=SEGMENTATION_SIDE_M,
                    resolution=128, batch_size=32):
    cache_path = ROOT / "test_unseen_fusion/segmentation40_candidate_components.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cached[split].shape == (len(rows[split]), 5, 4) for split in SPLITS):
            return cached
        raise ValueError("segmentation candidate cache shape mismatch")
    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(
        SEGMENTATION_DIR / "model.pt",
        map_location="cpu", weights_only=True))
    model.eval()
    def component_proximity(prediction):
        values = torch.full((len(prediction), 4), 1e-6)
        center = np.asarray([resolution / 2, resolution / 2])
        meters_per_pixel = side_m / resolution
        for batch_index, semantic_mask in enumerate(prediction.cpu().numpy()):
            for class_index in range(1, 5):
                binary = (semantic_mask == class_index).astype(np.uint8)
                count, _, stats, centroids = cv2.connectedComponentsWithStats(
                    binary, connectivity=8)
                if count <= 1:
                    continue
                valid = stats[1:, cv2.CC_STAT_AREA] >= 4
                if not valid.any():
                    continue
                distance_m = np.linalg.norm(centroids[1:][valid] - center, axis=-1).min()
                distance_m *= meters_per_pixel
                values[batch_index, class_index - 1] = float(np.exp(-distance_m / 20.0))
        return values
    output = {}
    for split in SPLITS:
        grouped = defaultdict(list)
        for row_index, row in enumerate(rows[split]):
            for candidate_index, candidate in enumerate(row["candidates"]):
                grouped[row["map_name"]].append((row_index, candidate_index, candidate))
        result = torch.empty(len(rows[split]), 5, 4)
        completed = 0
        for map_name, entries in sorted(grouped.items()):
            rgb = cv2.imread(str(DATA / "rgbd" / f"{map_name}.png"), cv2.IMREAD_COLOR)
            with rasterio.open(DATA / "rgbd" / f"{map_name}.tif") as raster:
                transform = raster.transform
            side_pixels = max(8, int(round(side_m / abs(transform.a))))
            patches, slots = [], []
            for row_index, candidate_index, candidate in entries:
                x, y = normalized_to_world(candidate, map_name)
                pixel_row, pixel_col = rasterio.transform.rowcol(transform, x, y)
                patch = crop_at(rgb, (float(pixel_col), float(pixel_row)),
                                side_pixels, resolution, cv2.INTER_AREA)
                patches.append(torch.from_numpy(patch[..., ::-1].copy()).permute(2, 0, 1))
                slots.append((row_index, candidate_index))
                if len(patches) == batch_size:
                    batch = torch.stack(patches).float().to(device).div(255.0)
                    proximity = component_proximity(model(batch).argmax(1))
                    for slot, value in zip(slots, proximity):
                        result[slot] = value
                    completed += len(patches)
                    patches, slots = [], []
            if patches:
                batch = torch.stack(patches).float().to(device).div(255.0)
                proximity = component_proximity(model(batch).argmax(1))
                for slot, value in zip(slots, proximity):
                    result[slot] = value
                completed += len(patches)
            print(f"segmentation {split} {map_name}: {completed}/{len(rows[split]) * 5}",
                  flush=True)
        output[split] = result.half()
    torch.save(output, cache_path)
    return output


@torch.inference_mode()
def load_or_predict_car_centers(rows, device, side_m=40.0,
                                resolution=128, batch_size=32):
    cache_path = ROOT / "test_unseen_fusion/car_center_candidate_proximity.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cached[split].shape == (len(rows[split]), 5) for split in SPLITS):
            return cached
        raise ValueError("car center candidate cache shape mismatch")
    model = TinyUNet(classes=4).to(device)
    model.load_state_dict(torch.load(
        ROOT / "object_center_heatmap_40m/model.pt",
        map_location="cpu", weights_only=True))
    model.eval()
    axis = torch.arange(resolution, device=device) + 0.5 - resolution / 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    distance = torch.sqrt(xx.square() + yy.square()) * side_m / resolution
    decay = torch.exp(-distance / 20.0)[None]
    output = {}
    for split in SPLITS:
        grouped = defaultdict(list)
        for row_index, row in enumerate(rows[split]):
            for candidate_index, candidate in enumerate(row["candidates"]):
                grouped[row["map_name"]].append((row_index, candidate_index, candidate))
        result = torch.empty(len(rows[split]), 5)
        completed = 0
        for map_name, entries in sorted(grouped.items()):
            rgb = cv2.imread(str(DATA / "rgbd" / f"{map_name}.png"), cv2.IMREAD_COLOR)
            with rasterio.open(DATA / "rgbd" / f"{map_name}.tif") as raster:
                transform = raster.transform
            side_pixels = max(8, int(round(side_m / abs(transform.a))))
            patches, slots = [], []
            for row_index, candidate_index, candidate in entries:
                x, y = normalized_to_world(candidate, map_name)
                pixel_row, pixel_col = rasterio.transform.rowcol(transform, x, y)
                patch = crop_at(rgb, (float(pixel_col), float(pixel_row)),
                                side_pixels, resolution, cv2.INTER_AREA)
                patches.append(torch.from_numpy(patch[..., ::-1].copy()).permute(2, 0, 1))
                slots.append((row_index, candidate_index))
                if len(patches) == batch_size:
                    batch = torch.stack(patches).float().to(device).div(255.0)
                    # Channel 1 is Car in the center-heatmap model.
                    proximity = (model(batch).sigmoid()[:, 1] * decay).amax((-1, -2)).cpu()
                    for slot, value in zip(slots, proximity):
                        result[slot] = value
                    completed += len(patches)
                    patches, slots = [], []
            if patches:
                batch = torch.stack(patches).float().to(device).div(255.0)
                proximity = (model(batch).sigmoid()[:, 1] * decay).amax((-1, -2)).cpu()
                for slot, value in zip(slots, proximity):
                    result[slot] = value
                completed += len(patches)
            print(f"car centers {split} {map_name}: {completed}/{len(rows[split]) * 5}",
                  flush=True)
        output[split] = result.half()
    torch.save(output, cache_path)
    return output


def make_data(rows):
    relation_feature = torch.load(
        ROOT / "relation_geometry_ranker_contrastive/relation_features.pt",
        map_location="cpu", weights_only=True)["val_unseen"]
    test_relation = torch.load(
        ROOT / "test_unseen_fusion/relation_features.pt",
        map_location="cpu", weights_only=True)
    return {
        "val_unseen": build_split(rows["val_unseen"], relation_feature, 9),
        "test_unseen": build_split(rows["test_unseen"], test_relation, 9),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    rows = {
        "val_unseen": load_rows(candidate_dir / "val_unseen.jsonl", 5),
        "test_unseen": load_rows(ROOT / "test_unseen_fusion/test_unseen.jsonl", 5),
    }
    proximity = load_or_predict(rows, device)
    car_center = load_or_predict_car_centers(rows, device)
    for split in SPLITS:
        proximity[split] = proximity[split].clone()
        proximity[split][..., 1] = car_center[split]
    data = make_data(rows)
    type_feature = torch.load(
        ROOT / "target_type_alignment/features.pt", map_location="cpu", weights_only=True)
    scale40 = torch.load(
        ROOT / "target_type_alignment/multiscale/scale_40.pt",
        map_location="cpu", weights_only=True)
    test40 = torch.load(
        ROOT / "test_unseen_fusion/siglip40_features.pt",
        map_location="cpu", weights_only=True)
    candidate40 = {
        "val_unseen": torch.load(
            ROOT / "relation_type_fusion/scale40_candidate_features.pt",
            map_location="cpu", weights_only=True)["val_unseen"],
        "test_unseen": test40["candidate"],
    }
    text = {"val_unseen": type_feature["text"]["val_unseen"]["features"],
            "test_unseen": test40["text"]}
    collected = {split: {"relation": [], "type": [], "text_probability": []}
                 for split in SPLITS}
    for seed in (0, 1, 2):
        text_model = train_probe(type_feature["text"]["train_seen"],
                                 type_feature["text"]["val_seen"], seed, 40, device)
        image_model = train_probe(scale40["target"]["train_seen"],
                                  scale40["target"]["val_seen"], seed, 40, device)
        relation_model = RelationGeometryRanker(cue_dim=len(CUE_NAMES)).to(device)
        relation_model.load_state_dict(torch.load(
            ROOT / f"relation_geometry_ranker_contrastive/model_seed{seed}.pt",
            map_location="cpu", weights_only=True))
        relation_model.eval()
        for split in SPLITS:
            collected[split]["relation"].append(normalized(
                relation_scores(relation_model, data[split], device)))
            type_score, text_logits, _ = type_compatibility(
                text_model, image_model, text[split], candidate40[split], device)
            collected[split]["type"].append(normalized(type_score))
            collected[split]["text_probability"].append(F.softmax(text_logits, -1))
        del text_model, image_model, relation_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    objects = json.load(open(DATA / "cityrefer/objects.json"))
    parts, proxy, semantic_diagnostics = {}, {}, {}
    for split in SPLITS:
        text_probability = torch.stack(collected[split]["text_probability"]).mean(0)
        semantic = normalized(torch.logsumexp(
            text_probability.clamp_min(1e-6).log()[:, None]
            + proximity[split].float().clamp_min(1e-6).log(), dim=-1))
        prior = normalized(-torch.arange(5, dtype=torch.float32)[None].expand(
            len(rows[split]), 5))
        parts[split] = {
            "relation": torch.stack(collected[split]["relation"]).mean(0),
            "type": torch.stack(collected[split]["type"]).mean(0),
            "appearance40": direct_similarity(text[split], candidate40[split]),
            "prior": prior,
            "semantic": semantic,
        }
        proxy[split] = {"distance": data[split]["distance"],
                        "landmark_count": data[split]["landmark_count"]}
        ideal_semantic = normalized(semantic_component_score(
            rows[split], text_probability, objects))
        centered_predicted = semantic - semantic.mean(-1, keepdim=True)
        centered_ideal = ideal_semantic - ideal_semantic.mean(-1, keepdim=True)
        correlation = (centered_predicted * centered_ideal).sum(-1) / (
            centered_predicted.square().sum(-1).sqrt()
            * centered_ideal.square().sum(-1).sqrt()).clamp_min(1e-6)
        semantic_diagnostics[split] = {
            "predicted_semantic_only": metrics(
                data[split]["distance"], semantic.argmax(-1)),
            "ideal_semantic_only": metrics(
                data[split]["distance"], ideal_semantic.argmax(-1)),
            "predicted_vs_ideal_score_correlation": correlation.mean().item(),
            "predicted_vs_ideal_top1_agreement_percent": 100.0 * semantic.argmax(-1).eq(
                ideal_semantic.argmax(-1)).float().mean().item(),
        }
    all_val = torch.ones(len(rows["val_unseen"]), dtype=torch.bool)
    all_test = torch.ones(len(rows["test_unseen"]), dtype=torch.bool)
    base_weights = tune(parts["val_unseen"], proxy["val_unseen"]["distance"],
                        proxy["val_unseen"]["landmark_count"], all_val, False)
    semantic_weights = tune(parts["val_unseen"], proxy["val_unseen"]["distance"],
                            proxy["val_unseen"]["landmark_count"], all_val, True)
    result = {
        "segmentation_checkpoint": "train_seen maps only; selected on val_unseen mIoU",
        "fusion_weights": "selected on val_unseen, fixed on test_unseen",
        "val_unseen": {
            "base40": evaluate(parts["val_unseen"], base_weights,
                               proxy["val_unseen"], all_val),
            "segmentation_fusion": evaluate(parts["val_unseen"], semantic_weights,
                                            proxy["val_unseen"], all_val),
        },
        "test_unseen": {
            "base40": evaluate(parts["test_unseen"], base_weights,
                               proxy["test_unseen"], all_test),
            "segmentation_fusion": evaluate(parts["test_unseen"], semantic_weights,
                                            proxy["test_unseen"], all_test),
        },
        "semantic_diagnostics": semantic_diagnostics,
        "weights": {"base": {key: list(value) for key, value in base_weights.items()},
                    "segmentation": {key: list(value)
                                     for key, value in semantic_weights.items()}},
        "anti_leakage": {
            "segmentation_training": "train_seen RGB and all-object class masks",
            "fusion_selection": "val_unseen only",
            "test_target_used_for_training_or_selection": False,
        },
    }
    result["semantic_input"] = (
        "40 m segmentation components; validated car channel replaced by center heatmap")
    path = ROOT / "test_unseen_fusion/segmentation40_car_center_fusion_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
