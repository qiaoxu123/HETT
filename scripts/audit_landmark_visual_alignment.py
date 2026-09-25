#!/usr/bin/env python3
"""Audit symbolic and frozen-vision alignment for named CityRefer landmarks."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import rasterio
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_rgb_candidate_ranker import encode_images, extract_crop
from scripts.validate_candidate_geometry import (
    normalize_name,
    polygon_centroid,
    resolve_landmark,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--crop-side-m", type=float, default=160.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path,
                        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/landmark_visual_alignment/results.json"))
    return parser.parse_args()


def retrieval_metrics(similarity, image_labels, image_maps, query_labels, query_maps):
    ranks, correct, margins, chance = [], [], [], []
    for index, (label, map_name) in enumerate(zip(query_labels, query_maps)):
        available = [i for i, value in enumerate(image_maps) if value == map_name]
        scores = similarity[index, available]
        order = scores.argsort(descending=True)
        labels = [image_labels[available[i]] for i in order.tolist()]
        valid_ranks = [rank + 1 for rank, value in enumerate(labels) if value == label]
        if not valid_ranks:
            continue
        rank = valid_ranks[0]
        ranks.append(rank)
        correct.append(rank == 1)
        positive = max(scores[j].item() for j, i in enumerate(available)
                       if image_labels[i] == label)
        negatives = [scores[j].item() for j, i in enumerate(available)
                     if image_labels[i] != label]
        margins.append(positive - max(negatives) if negatives else 0.0)
        class_count = len({image_labels[i] for i in available})
        chance.append(1.0 / max(class_count, 1))
    count = len(ranks)
    return {
        "queries": count,
        "top1": 100.0 * sum(correct) / max(count, 1),
        "mrr": 100.0 * sum(1.0 / rank for rank in ranks) / max(count, 1),
        "chance_top1": 100.0 * sum(chance) / max(count, 1),
        "mean_positive_hard_negative_margin": sum(margins) / max(count, 1),
        "positive_margin_rate": 100.0 * sum(value > 0 for value in margins) / max(count, 1),
    }


@torch.inference_mode()
def encode_text(model, processor, texts, device, batch_size):
    result = []
    for offset in range(0, len(texts), batch_size):
        values = texts[offset:offset + batch_size]
        inputs = processor(text=values, padding="max_length", max_length=64,
                           truncation=True, return_tensors="pt")
        kwargs = {"input_ids": inputs["input_ids"].to(device)}
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"].to(device)
        result.append(F.normalize(model.get_text_features(**kwargs).float(), dim=-1).cpu())
    return torch.cat(result)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    processed = json.load(open(args.data_root / "cityrefer/processed_descriptions.json"))
    trajectories = json.load(open(
        args.data_root / f"processed_citynav/citynav_{args.split}.json"))

    referenced = defaultdict(set)
    mentions = 0
    exact_mentions = 0
    ambiguous_mentions = 0
    for item in trajectories:
        if item.get("dist_marker_to_target", 0.0) > 30.0:
            continue
        map_name = f'{item["area"]}_block_{item["block"]}'
        object_id = str(item["object_ids"][0])
        ann_id = int(item["ann_ids"][0])
        annotations = processed.get(map_name, {}).get(object_id, [])
        queries = annotations[ann_id].get("landmarks", []) if ann_id < len(annotations) else []
        candidates = [obj for obj in objects[map_name].values() if obj.get("name")]
        by_name = defaultdict(list)
        for obj in candidates:
            by_name[normalize_name(obj["name"])].append(obj)
        exact = {key: values[-1] for key, values in by_name.items()}
        for query in queries[:9]:
            mentions += 1
            key = normalize_name(query)
            exact_mentions += int(key in exact)
            ambiguous_mentions += int(len(by_name.get(key, [])) > 1)
            resolved = resolve_landmark(query, candidates, exact)
            referenced[map_name].add(normalize_name(resolved["name"]))

    images, image_labels, image_maps = [], [], []
    image_instance_labels, appearance_query_labels = [], []
    appearance_query_maps, appearance_query_texts = [], []
    in_bounds, object_count = 0, 0
    for map_name, labels in sorted(referenced.items()):
        rgb = cv2.imread(str(args.data_root / "rgbd" / f"{map_name}.png"),
                         cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(map_name)
        with rasterio.open(args.data_root / "rgbd" / f"{map_name}.tif") as raster:
            transform = raster.transform
        inv = ~transform
        for object_id, obj in objects[map_name].items():
            label = normalize_name(obj.get("name", ""))
            if not label or label not in labels:
                continue
            center = polygon_centroid(obj["contour"])
            px, py = inv * tuple(center)
            in_bounds += int(0 <= px < rgb.shape[1] and 0 <= py < rgb.shape[0])
            object_count += 1
            images.append(extract_crop(rgb, transform, center, args.crop_side_m))
            image_labels.append(label)
            image_maps.append(map_name)
            instance_label = f"{map_name}:{object_id}"
            image_instance_labels.append(instance_label)
            descriptions = obj.get("descriptions", [])
            if descriptions:
                appearance_query_labels.append(instance_label)
                appearance_query_maps.append(map_name)
                appearance_query_texts.append(descriptions[0])

    query_labels, query_maps, query_texts = [], [], []
    for map_name, labels in sorted(referenced.items()):
        display_names = {}
        for obj in objects[map_name].values():
            key = normalize_name(obj.get("name", ""))
            if key in labels:
                display_names[key] = obj["name"]
        for label in sorted(labels):
            query_labels.append(label)
            query_maps.append(map_name)
            query_texts.append(display_names[label])

    processor = AutoProcessor.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    image_features = []
    for offset in range(0, len(images), args.batch_size):
        image_features.append(encode_images(
            model, processor, images[offset:offset + args.batch_size], device))
    image_features = torch.cat(image_features)
    text_features = encode_text(model, processor, query_texts, device, args.batch_size)
    similarity = text_features @ image_features.T
    appearance_features = encode_text(
        model, processor, appearance_query_texts, device, args.batch_size)
    appearance_similarity = appearance_features @ image_features.T
    result = {
        "config": {"split": args.split, "crop_side_m": args.crop_side_m,
                   "model": args.model},
        "symbolic": {
            "mentions": mentions,
            "exact_name_rate": 100.0 * exact_mentions / max(mentions, 1),
            "exact_ambiguous_rate": 100.0 * ambiguous_mentions / max(mentions, 1),
            "unique_name_queries": len(query_labels),
            "landmark_instances": len(image_labels),
            "crop_centers_in_raster_rate": 100.0 * in_bounds / max(object_count, 1),
        },
        "visual_retrieval": retrieval_metrics(
            similarity, image_labels, image_maps, query_labels, query_maps),
        "appearance_oracle_retrieval": retrieval_metrics(
            appearance_similarity, image_instance_labels, image_maps,
            appearance_query_labels, appearance_query_maps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
