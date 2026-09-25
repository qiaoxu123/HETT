#!/usr/bin/env python3
"""Evaluate basic instruction/RGB type matching across fine-view scales."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import rasterio
import torch
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import (
    CLASSES,
    encode_object_crops,
    evaluate_probe,
    first_type_match,
    load_examples,
    summarize_runs,
    train_probe,
)
from scripts.train_rgb_candidate_ranker import (
    encode_images,
    extract_crop,
    metrics as navigation_metrics,
    normalized_to_world,
    prepare_rows,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--base-cache", type=Path,
                        default=Path("runs/target_type_alignment/features.pt"))
    parser.add_argument("--candidate-cache-160", type=Path,
                        default=Path("runs/rgb_candidate_ranker_oof_grouped/features.pt"))
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("runs/candidate_rgb_data/geometry_top5_oof_grouped_enriched"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("runs/target_type_alignment/multiscale"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/target_type_alignment/multiscale_results.json"))
    parser.add_argument("--scales", default="40,80")
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path,
                        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", default="0,1,2")
    return parser.parse_args()


@torch.inference_mode()
def encode_candidate_crops(rows, model, processor, data_root, device,
                           crop_side_m, batch_size, top_k=5):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["map_name"]].append((index, row))
    result = None
    completed = 0
    for map_name, entries in sorted(grouped.items()):
        rgb = cv2.imread(str(data_root / "rgbd" / f"{map_name}.png"),
                         cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(map_name)
        with rasterio.open(data_root / "rgbd" / f"{map_name}.tif") as raster:
            transform = raster.transform
        images, slots = [], []
        for row_index, row in entries:
            for rank, candidate in enumerate(row["candidates"][:top_k]):
                images.append(extract_crop(
                    rgb, transform, normalized_to_world(candidate, map_name),
                    crop_side_m))
                slots.append((row_index, rank))
                if len(images) == batch_size:
                    vectors = encode_images(model, processor, images, device)
                    if result is None:
                        result = torch.empty(len(rows), top_k, vectors.shape[-1])
                    for slot, vector in zip(slots, vectors):
                        result[slot] = vector
                    completed += len(images)
                    if completed % (batch_size * 25) == 0:
                        print(f"scale={crop_side_m:g}m candidates "
                              f"{completed}/{len(rows) * top_k}", flush=True)
                    images, slots = [], []
        if images:
            vectors = encode_images(model, processor, images, device)
            if result is None:
                result = torch.empty(len(rows), top_k, vectors.shape[-1])
            for slot, vector in zip(slots, vectors):
                result[slot] = vector
            completed += len(images)
    print(f"scale={crop_side_m:g}m candidates complete: {completed}", flush=True)
    return result.half()


def concatenate_modalities(items):
    result = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        keys = items[0][split]["keys"]
        if any(item[split]["keys"] != keys for item in items[1:]):
            raise ValueError("object feature caches have different row order")
        result[split] = {
            "features": torch.cat([item[split]["features"] for item in items], dim=-1),
            "labels": items[0][split]["labels"],
            "keys": keys,
        }
    return result


def type_agreement(text_prediction, image_prediction, truth):
    agreement = text_prediction.eq(image_prediction)
    return {
        "prediction_agreement": 100.0 * agreement.float().mean().item(),
        "both_correct": 100.0 * (
            text_prediction.eq(truth) & image_prediction.eq(truth)).float().mean().item(),
        "agreement_and_correct": 100.0 * (
            agreement & text_prediction.eq(truth)).float().mean().item(),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, examples = load_examples(args.data_root)
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True)
    candidate_160 = torch.load(
        args.candidate_cache_160, map_location="cpu", weights_only=True)
    rows, _ = prepare_rows(
        args.candidate_dir / "val_unseen.jsonl",
        len(candidate_160["val_unseen"]["distance"]), 0, 5, False)
    expected_distance = torch.tensor([
        [((candidate_xy[0] - row["target"][0]) ** 2
          + (candidate_xy[1] - row["target"][1]) ** 2) ** 0.5 * 410.0
         for candidate_xy in row["candidates"][:5]]
        for row in rows])
    if not torch.allclose(expected_distance,
                          candidate_160["val_unseen"]["distance"],
                          atol=1e-4, rtol=1e-5):
        raise ValueError("candidate feature cache and rows are not aligned")
    scales = [int(value) for value in args.scales.split(",") if value]

    processor = encoder = None
    image_by_scale = {160: base["image"]}
    candidate_by_scale = {160: candidate_160["val_unseen"]["image"]}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for scale in scales:
        cache_path = args.cache_dir / f"scale_{scale}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            if encoder is None:
                processor = AutoProcessor.from_pretrained(
                    args.model, cache_dir=str(args.hf_cache), local_files_only=True)
                encoder = AutoModel.from_pretrained(
                    args.model, cache_dir=str(args.hf_cache), local_files_only=True,
                    torch_dtype=torch.bfloat16).to(device).eval()
            print(f"encoding scale={scale}m target objects", flush=True)
            target = encode_object_crops(
                examples, encoder, processor, args.data_root, device,
                scale, args.feature_batch_size)
            print(f"encoding scale={scale}m candidate crops", flush=True)
            candidate = encode_candidate_crops(
                rows, encoder, processor, args.data_root, device,
                scale, args.feature_batch_size)
            cached = {"target": target, "candidate": candidate}
            torch.save(cached, cache_path)
        image_by_scale[scale] = cached["target"]
        candidate_by_scale[scale] = cached["candidate"]
    if encoder is not None:
        del encoder
        torch.cuda.empty_cache()

    scale_names = [str(scale) for scale in sorted(image_by_scale)]
    modalities = {name: image_by_scale[int(name)] for name in scale_names}
    modalities["40+80+160"] = concatenate_modalities(
        [image_by_scale[scale] for scale in (40, 80, 160)])
    candidate_modalities = {
        name: candidate_by_scale[int(name)] for name in scale_names}
    candidate_modalities["40+80+160"] = torch.cat(
        [candidate_by_scale[scale] for scale in (40, 80, 160)], dim=-1)

    seeds = [int(value) for value in args.seeds.split(",") if value]
    result = {"config": {"scales_m": scale_names, "classes": CLASSES,
                         "seeds": seeds, "selection": "first predicted type match"},
              "runs": {name: [] for name in modalities}}
    text_truth = base["text"]["val_unseen"]["labels"]
    object_keys = examples["val_unseen"]["text_object_keys"]
    for seed in seeds:
        text_model = train_probe(base["text"]["train_seen"],
                                 base["text"]["val_seen"], seed,
                                 args.epochs, device)
        _, text_prediction = evaluate_probe(
            text_model, base["text"]["val_unseen"], device)
        candidate_text = text_model(
            candidate_160["val_unseen"]["text"].float().to(device)
        ).argmax(-1).cpu()
        for name, image_data in modalities.items():
            image_model = train_probe(image_data["train_seen"],
                                      image_data["val_seen"], seed,
                                      args.epochs, device)
            image_metrics, image_prediction = evaluate_probe(
                image_model, image_data["val_unseen"], device)
            by_key = dict(zip(image_data["val_unseen"]["keys"],
                              image_prediction.tolist()))
            paired_image = torch.tensor([by_key[key] for key in object_keys])
            agreement = type_agreement(text_prediction, paired_image, text_truth)

            candidate_image_features = candidate_modalities[name]
            shape = candidate_image_features.shape
            candidate_image = image_model(
                candidate_image_features.float().reshape(-1, shape[-1]).to(device)
            ).argmax(-1).cpu().reshape(shape[:2])
            rank, has_match = first_type_match(candidate_text, candidate_image)
            navigation = navigation_metrics(
                candidate_160["val_unseen"]["distance"], rank)
            navigation["rows_with_type_match"] = (
                100.0 * has_match.float().mean().item())
            navigation["selection_changed"] = (
                100.0 * rank.ne(0).float().mean().item())
            result["runs"][name].append({
                "seed": seed, "image": image_metrics,
                "paired_text_image": agreement, "navigation": navigation,
                "candidate_image_distribution": dict(Counter(
                    CLASSES[index] for index in candidate_image.flatten().tolist())),
            })
            del image_model
        del text_model
        torch.cuda.empty_cache()

    result["summary"] = {}
    for name, runs in result["runs"].items():
        result["summary"][name] = {
            "image": summarize_runs([run["image"] for run in runs]),
            "paired_prediction_agreement": sum(
                run["paired_text_image"]["prediction_agreement"] for run in runs
            ) / len(runs),
            "paired_both_correct": sum(
                run["paired_text_image"]["both_correct"] for run in runs
            ) / len(runs),
            "navigation_mean": {
                key: sum(run["navigation"][key] for run in runs) / len(runs)
                for key in ("mean_ne", "hit20", "hit50", "selection_changed")
            },
        }
    result["baseline_navigation"] = navigation_metrics(
        candidate_160["val_unseen"]["distance"],
        torch.zeros(len(rows), dtype=torch.long))
    result["anti_leakage"] = {
        "training": "train_seen only",
        "model_selection": "val_seen only",
        "final_test": "map-disjoint val_unseen",
        "target_coordinates_used_for": "navigation metrics only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"baseline": result["baseline_navigation"],
                      "summary": result["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
