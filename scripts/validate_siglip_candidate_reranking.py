#!/usr/bin/env python3
"""Test whether frozen SigLIP2 can rerank spatial goal candidates from RGB.

The evaluation uses a deterministic map-balanced subset of val_unseen.  Each
candidate is represented by an equal-sized orthomosaic crop.  A shuffled-text
control distinguishes genuine language grounding from image-only preferences.
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.mapdata import MAP_BOUNDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path,
                        default=Path("runs/named_landmark_candidates/val_unseen.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/siglip_candidate_reranking/results.json"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path,
                        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crop-side-m", type=float, default=160.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def balanced_sample(rows, count, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["map_name"]].append(row)
    for values in grouped.values():
        rng.shuffle(values)
    selected = []
    names = sorted(grouped)
    while len(selected) < min(count, len(rows)):
        added = False
        for name in names:
            if grouped[name] and len(selected) < count:
                selected.append(grouped[name].pop())
                added = True
        if not added:
            break
    return selected


def normalized_to_world(xy, map_name):
    bounds = MAP_BOUNDS[map_name]
    return bounds.x_min + xy[0] * 410.0, bounds.y_max - xy[1] * 410.0


def extract_crop(rgb, transform, center, side_m, out_size=512):
    cx, cy = center
    half = side_m / 2.0
    left, bottom, right, top = cx - half, cy - half, cx + half, cy + half
    inv = ~transform
    src_left, src_top = inv * (left, top)
    src_right, src_bottom = inv * (right, bottom)
    x0, y0 = math.floor(src_left), math.floor(src_top)
    x1, y1 = math.ceil(src_right), math.ceil(src_bottom)
    canvas = np.zeros((max(1, y1 - y0), max(1, x1 - x0), 3), dtype=np.uint8)
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    if ix1 > ix0 and iy1 > iy0:
        canvas[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = rgb[iy0:iy1, ix0:ix1]
    return cv2.cvtColor(cv2.resize(canvas, (out_size, out_size),
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def build_crops(rows, data_root, top_k, side_m):
    images = [None] * (len(rows) * top_k)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["map_name"]].append((index, row))
    for map_name, entries in sorted(grouped.items()):
        rgb_path = data_root / "rgbd" / f"{map_name}.png"
        tif_path = data_root / "rgbd" / f"{map_name}.tif"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        with rasterio.open(tif_path) as raster:
            transform = raster.transform
        for row_index, row in entries:
            for rank, candidate in enumerate(row["candidates"][:top_k]):
                center = normalized_to_world(candidate, map_name)
                images[row_index * top_k + rank] = extract_crop(
                    rgb, transform, center, side_m)
    return images


@torch.inference_mode()
def encode_images(model, processor, images, device, batch_size):
    vectors = []
    for offset in range(0, len(images), batch_size):
        inputs = processor(images=images[offset:offset + batch_size],
                           return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        features = model.get_image_features(pixel_values=pixel_values)
        vectors.append(F.normalize(features.float(), dim=-1).cpu())
        print(f"images {min(offset + batch_size, len(images))}/{len(images)}", flush=True)
    return torch.cat(vectors)


@torch.inference_mode()
def encode_text(model, processor, texts, device, batch_size):
    vectors = []
    for offset in range(0, len(texts), batch_size):
        inputs = processor(text=texts[offset:offset + batch_size], padding="max_length",
                           max_length=64, truncation=True, return_tensors="pt")
        model_inputs = {"input_ids": inputs["input_ids"].to(device)}
        if "attention_mask" in inputs:
            model_inputs["attention_mask"] = inputs["attention_mask"].to(device)
        features = model.get_text_features(**model_inputs)
        vectors.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(vectors)


def summarize(distances, selected_rank):
    chosen = distances[torch.arange(len(distances)), selected_rank]
    return {
        "mean_ne": chosen.mean().item(),
        "median_ne": chosen.median().item(),
        "hit20": (chosen <= 20).float().mean().mul(100).item(),
        "hit50": (chosen <= 50).float().mean().mul(100).item(),
    }


def main():
    args = parse_args()
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines()]
    rows = balanced_sample(rows, args.sample_count, args.seed)
    images = build_crops(rows, args.data_root, args.top_k, args.crop_side_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    image_vectors = encode_images(model, processor, images, device, args.batch_size)
    text_vectors = encode_text(model, processor, [row["description"] for row in rows],
                               device, args.batch_size)
    image_vectors = image_vectors.reshape(len(rows), args.top_k, -1)
    true_scores = torch.einsum("nkd,nd->nk", image_vectors, text_vectors)
    shuffled_scores = torch.einsum("nkd,nd->nk", image_vectors, text_vectors.roll(1, 0))

    candidates = torch.tensor([row["candidates"][:args.top_k] for row in rows])
    targets = torch.tensor([row["target"] for row in rows])
    distances = torch.linalg.vector_norm(candidates - targets[:, None], dim=-1) * 410.0
    spatial_rank = torch.zeros(len(rows), dtype=torch.long)
    oracle_rank = distances.argmin(-1)
    true_rank = true_scores.argmax(-1)
    shuffled_rank = shuffled_scores.argmax(-1)
    results = {
        "config": {
            **vars(args),
            "candidates": str(args.candidates),
            "data_root": str(args.data_root),
            "output": str(args.output),
            "hf_cache": str(args.hf_cache),
        },
        "count": len(rows),
        "maps": len({row["map_name"] for row in rows}),
        "spatial_top1": summarize(distances, spatial_rank),
        "siglip_instruction": summarize(distances, true_rank),
        "siglip_shuffled_instruction": summarize(distances, shuffled_rank),
        "oracle_top_k": summarize(distances, oracle_rank),
        "rank_changed_percent": (true_rank != 0).float().mean().mul(100).item(),
        "true_vs_shuffled_same_rank_percent": (
            true_rank == shuffled_rank).float().mean().mul(100).item(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
