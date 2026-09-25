#!/usr/bin/env python3
"""Test whether frozen text/RGB features encode CityRefer target types."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_rgb_candidate_ranker import (
    encode_images,
    encode_texts,
    extract_crop,
    metrics as navigation_metrics,
    prepare_rows,
)
from scripts.validate_candidate_geometry import polygon_centroid


CLASSES = ("Building", "Car", "Ground", "Parking")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("runs/candidate_rgb_data/geometry_top5_oof_grouped_enriched"))
    parser.add_argument("--candidate-cache", type=Path,
                        default=Path("runs/rgb_candidate_ranker_oof_grouped/features.pt"))
    parser.add_argument("--feature-cache", type=Path,
                        default=Path("runs/target_type_alignment/features.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/target_type_alignment/results.json"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path,
                        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--crop-side-m", type=float, default=160.0)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", default="0,1,2")
    return parser.parse_args()


def load_examples(data_root):
    objects = json.load(open(data_root / "cityrefer/objects.json"))
    examples = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        trajectories = json.load(open(
            data_root / f"processed_citynav/citynav_{split}.json"))
        text, labels, text_object_keys, target_objects = [], [], [], {}
        for row in trajectories:
            map_name = f'{row["area"]}_block_{row["block"]}'
            object_id = str(row["object_ids"][0])
            obj = objects[map_name][object_id]
            label = CLASS_TO_INDEX[obj["object_type"]]
            text.append(row["descriptions"][0])
            labels.append(label)
            text_object_keys.append(f"{map_name}:{object_id}")
            target_objects[(map_name, object_id)] = obj
        examples[split] = {
            "text": text,
            "text_labels": torch.tensor(labels, dtype=torch.long),
            "text_object_keys": text_object_keys,
            "objects": target_objects,
        }
    return objects, examples


@torch.inference_mode()
def encode_object_crops(examples, model, processor, data_root, device,
                        crop_side_m, batch_size):
    result = {}
    for split, split_data in examples.items():
        grouped = defaultdict(list)
        for key, obj in split_data["objects"].items():
            grouped[key[0]].append((key, obj))
        vectors, labels, keys = [], [], []
        for map_name, entries in sorted(grouped.items()):
            rgb = cv2.imread(str(data_root / "rgbd" / f"{map_name}.png"),
                             cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(map_name)
            with rasterio.open(data_root / "rgbd" / f"{map_name}.tif") as raster:
                transform = raster.transform
            for offset in range(0, len(entries), batch_size):
                batch = entries[offset:offset + batch_size]
                images = [extract_crop(rgb, transform, polygon_centroid(obj["contour"]),
                                       crop_side_m) for _, obj in batch]
                vectors.append(encode_images(model, processor, images, device))
                labels.extend(CLASS_TO_INDEX[obj["object_type"]] for _, obj in batch)
                keys.extend(f"{key[0]}:{key[1]}" for key, _ in batch)
            print(f"{split}: encoded {len(keys)}/{len(split_data['objects'])} object crops",
                  flush=True)
        result[split] = {
            "features": torch.cat(vectors).half(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "keys": keys,
        }
    return result


def classification_metrics(labels, predictions, class_count=4):
    labels = labels.long().cpu()
    predictions = predictions.long().cpu()
    confusion = torch.zeros(class_count, class_count, dtype=torch.long)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[truth, prediction] += 1
    recalls, f1s = [], []
    for index in range(class_count):
        tp = confusion[index, index].item()
        fn = confusion[index].sum().item() - tp
        fp = confusion[:, index].sum().item() - tp
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "accuracy": 100.0 * confusion.diag().sum().item() / max(len(labels), 1),
        "balanced_accuracy": 100.0 * sum(recalls) / class_count,
        "macro_f1": 100.0 * sum(f1s) / class_count,
        "per_class_recall": {name: 100.0 * recalls[index]
                             for index, name in enumerate(CLASSES)},
        "confusion": confusion.tolist(),
    }


def first_type_match(text_prediction, image_predictions):
    """Preserve geometry order and select its first predicted-type match."""
    matches = image_predictions.eq(text_prediction[:, None])
    has_match = matches.any(-1)
    rank = matches.float().argmax(-1)
    rank = torch.where(has_match, rank, torch.zeros_like(rank))
    return rank, has_match


def train_probe(train, val, seed, epochs, device):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    feature_dim = train["features"].shape[-1]
    model = nn.Linear(feature_dim, len(CLASSES)).to(device)
    counts = torch.bincount(train["labels"], minlength=len(CLASSES)).float()
    weights = counts.sum() / counts.clamp_min(1)
    weights = (weights / weights.mean()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    dataset = TensorDataset(train["features"].float(), train["labels"])
    best_state, best_score = None, -1.0
    for _ in range(epochs):
        model.train()
        for features, labels in DataLoader(dataset, batch_size=512, shuffle=True):
            loss = F.cross_entropy(model(features.to(device)), labels.to(device),
                                   weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(val["features"].float().to(device)).argmax(-1).cpu()
        score = classification_metrics(val["labels"], prediction)["macro_f1"]
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


@torch.inference_mode()
def evaluate_probe(model, data, device):
    prediction = model(data["features"].float().to(device)).argmax(-1).cpu()
    return classification_metrics(data["labels"], prediction), prediction


def summarize_runs(values):
    keys = ("accuracy", "balanced_accuracy", "macro_f1")
    return {key: {
        "mean": sum(run[key] for run in values) / len(values),
        "min": min(run[key] for run in values),
        "max": max(run[key] for run in values),
    } for key in keys}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, examples = load_examples(args.data_root)
    if args.feature_cache.exists():
        features = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    else:
        processor = AutoProcessor.from_pretrained(
            args.model, cache_dir=str(args.hf_cache), local_files_only=True)
        encoder = AutoModel.from_pretrained(
            args.model, cache_dir=str(args.hf_cache), local_files_only=True,
            torch_dtype=torch.bfloat16).to(device).eval()
        text_features = {}
        for split, data in examples.items():
            print(f"{split}: encoding {len(data['text'])} instructions", flush=True)
            text_features[split] = {
                "features": encode_texts(encoder, processor, data["text"], device,
                                          args.feature_batch_size).half(),
                "labels": data["text_labels"],
            }
        image_features = encode_object_crops(
            examples, encoder, processor, args.data_root, device,
            args.crop_side_m, args.feature_batch_size)
        features = {"text": text_features, "image": image_features}
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, args.feature_cache)
        del encoder
        torch.cuda.empty_cache()

    candidate = torch.load(args.candidate_cache, map_location="cpu", weights_only=True)
    candidate_rows, _ = prepare_rows(
        args.candidate_dir / "val_unseen.jsonl",
        len(candidate["val_unseen"]["distance"]), seed=0, top_k=5,
        inject_positive=False)
    expected_descriptions = [row["description"] for row in candidate_rows]
    # Cached text embeddings cannot identify their source by themselves. Distance
    # alignment guarantees the same balanced-sample row order and candidate order.
    expected_distance = torch.tensor([
        [((candidate_xy[0] - row["target"][0]) ** 2
          + (candidate_xy[1] - row["target"][1]) ** 2) ** 0.5 * 410.0
         for candidate_xy in row["candidates"][:5]]
        for row in candidate_rows])
    if not torch.allclose(expected_distance, candidate["val_unseen"]["distance"],
                          atol=1e-4, rtol=1e-5):
        raise ValueError("candidate feature cache and rows are not aligned")

    seeds = [int(value) for value in args.seeds.split(",") if value]
    result = {
        "config": {"classes": CLASSES, "crop_side_m": args.crop_side_m,
                   "model": args.model, "seeds": seeds,
                   "split_rule": "train_seen -> val_seen selection -> val_unseen test"},
        "counts": {}, "runs": [],
        "baselines": {"random_balanced_accuracy": 25.0,
                      "random_macro_f1_approximately": 25.0},
    }
    for modality in ("text", "image"):
        result["counts"][modality] = {
            split: dict(Counter(CLASSES[index] for index in data["labels"].tolist()))
            for split, data in features[modality].items()}

    for seed in seeds:
        text_model = train_probe(features["text"]["train_seen"],
                                 features["text"]["val_seen"], seed,
                                 args.epochs, device)
        image_model = train_probe(features["image"]["train_seen"],
                                  features["image"]["val_seen"], seed,
                                  args.epochs, device)
        text_metrics, _ = evaluate_probe(
            text_model, features["text"]["val_unseen"], device)
        image_metrics, _ = evaluate_probe(
            image_model, features["image"]["val_unseen"], device)

        candidate_text = text_model(
            candidate["val_unseen"]["text"].float().to(device)).argmax(-1).cpu()
        shape = candidate["val_unseen"]["image"].shape
        candidate_image = image_model(
            candidate["val_unseen"]["image"].float().reshape(-1, shape[-1]).to(device)
        ).argmax(-1).cpu().reshape(shape[:2])
        rank, has_match = first_type_match(candidate_text, candidate_image)
        distance = candidate["val_unseen"]["distance"]
        chosen = navigation_metrics(distance, rank)
        chosen["rows_with_type_match"] = 100.0 * has_match.float().mean().item()
        chosen["selection_changed"] = 100.0 * rank.ne(0).float().mean().item()
        chosen["predicted_text_distribution"] = dict(Counter(
            CLASSES[index] for index in candidate_text.tolist()))
        result["runs"].append({"seed": seed, "text": text_metrics,
                               "image": image_metrics,
                               "type_match_navigation": chosen})
        del text_model, image_model
        torch.cuda.empty_cache()

    result["summary"] = {
        "text": summarize_runs([run["text"] for run in result["runs"]]),
        "image": summarize_runs([run["image"] for run in result["runs"]]),
        "navigation": {
            "spatial_top1": navigation_metrics(
                candidate["val_unseen"]["distance"],
                torch.zeros(len(candidate_rows), dtype=torch.long)),
            "type_match_runs": [run["type_match_navigation"] for run in result["runs"]],
            "oracle_top5": navigation_metrics(
                candidate["val_unseen"]["distance"],
                candidate["val_unseen"]["distance"].argmin(-1)),
        },
    }
    # Retained solely to make the anti-leakage contract explicit in the artifact.
    result["anti_leakage"] = {
        "unseen_candidate_rows": len(expected_descriptions),
        "target_coordinates_used_for": "evaluation distance only",
        "candidate_choice_inputs": "predicted instruction type, predicted RGB type, geometry rank",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
