#!/usr/bin/env python3
"""Pilot candidate-level RGB/language ranker with frozen SigLIP2 features."""

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.mapdata import MAP_BOUNDS
from scripts.validate_candidate_geometry import mask_landmark_span


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("runs/candidate_rgb_data/geometry_top5"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/rgb_candidate_ranker/results.json"))
    parser.add_argument("--cache", type=Path,
                        default=Path("runs/rgb_candidate_ranker/features.pt"))
    parser.add_argument("--relation-cache", type=Path,
                        default=Path("runs/rgb_candidate_ranker/relation_features.pt"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path,
                        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--train-count", type=int, default=4096)
    parser.add_argument("--val-count", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crop-side-m", type=float, default=160.0)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--switch-loss-weight", type=float, default=0.2)
    parser.add_argument("--switch-margin", type=float, default=0.5)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def balanced_sample(rows, count, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["map_name"]].append(row)
    for values in grouped.values():
        rng.shuffle(values)
    selected, names = [], sorted(grouped)
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
    inv = ~transform
    x0f, y0f = inv * (cx - half, cy + half)
    x1f, y1f = inv * (cx + half, cy - half)
    x0, y0, x1, y1 = math.floor(x0f), math.floor(y0f), math.ceil(x1f), math.ceil(y1f)
    canvas = np.zeros((max(1, y1 - y0), max(1, x1 - x0), 3), dtype=np.uint8)
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    if ix1 > ix0 and iy1 > iy0:
        canvas[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = rgb[iy0:iy1, ix0:ix1]
    resized = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def prepare_rows(path, count, seed, top_k, inject_positive):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = balanced_sample(rows, count, seed)
    rng = random.Random(seed + 9173)
    injected = 0
    for row in rows:
        candidates = np.asarray(row["candidates"][:top_k], dtype=np.float32)
        target = np.asarray(row["target"], dtype=np.float32)
        distance = np.linalg.norm(candidates - target[None], axis=-1) * 410.0
        if inject_positive and distance.min() > 20.0:
            candidates[-1] = target
            injected += 1
        if inject_positive:
            order = list(range(len(candidates)))
            rng.shuffle(order)
            candidates = candidates[order]
        row["candidates"] = candidates.tolist()
    return rows, injected


def same_map_switch_indices(rows):
    """Choose a different instruction from the same map for every row."""
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["map_name"]].append(index)
    switched = []
    for index, row in enumerate(rows):
        group = grouped[row["map_name"]]
        start = group.index(index)
        replacement = index
        for offset in range(1, len(group) + 1):
            candidate = group[(start + offset) % len(group)]
            if rows[candidate]["description"] != row["description"]:
                replacement = candidate
                break
        switched.append(replacement)
    return torch.tensor(switched, dtype=torch.long)


def attach_switched_text(features, rows):
    indices = same_map_switch_indices(rows)
    features["switched_text"] = features["text"][indices]
    features["switch_changed"] = indices.ne(torch.arange(len(rows)))


def landmark_geometry(rows, reference_indices=None, max_landmarks=9):
    """Candidate-to-landmark centroid/boundary geometry, without target labels."""
    if reference_indices is None:
        reference_indices = torch.arange(len(rows))
    candidates = torch.tensor([row["candidates"] for row in rows], dtype=torch.float32)
    centers = torch.zeros(len(rows), max_landmarks, 2)
    boundaries = torch.zeros(len(rows), max_landmarks, 16, 2)
    valid = torch.zeros(len(rows), max_landmarks)
    for row_index, reference_index in enumerate(reference_indices.tolist()):
        reference = rows[reference_index]
        values = reference.get("landmark_centers", [])[:max_landmarks]
        contours = reference.get("landmark_boundaries", [])[:max_landmarks]
        if values:
            count = len(values)
            centers[row_index, :count] = torch.tensor(values)
            boundaries[row_index, :count] = torch.tensor(contours)
            valid[row_index, :count] = 1.0

    delta = candidates[:, :, None] - centers[:, None]
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    centroid = torch.cat((
        delta,
        (torch.log1p(distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (delta[..., 1] / distance).unsqueeze(-1),
        (delta[..., 0] / distance).unsqueeze(-1),
    ), dim=-1)
    boundary_delta = candidates[:, :, None, None] - boundaries[:, None]
    boundary_distance = torch.linalg.vector_norm(boundary_delta, dim=-1)
    nearest = boundary_distance.argmin(-1)
    gather = nearest[..., None, None].expand(-1, -1, -1, 1, 2)
    nearest_delta = boundary_delta.gather(3, gather).squeeze(3)
    nearest_distance = torch.linalg.vector_norm(nearest_delta, dim=-1).clamp_min(1e-6)
    boundary = torch.cat((
        nearest_delta,
        (torch.log1p(nearest_distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (nearest_delta[..., 1] / nearest_distance).unsqueeze(-1),
        (nearest_delta[..., 0] / nearest_distance).unsqueeze(-1),
    ), dim=-1)
    return torch.cat((centroid, boundary), dim=-1), valid


def attach_landmark_geometry(features, rows):
    switched = same_map_switch_indices(rows)
    features["geometry"], features["landmark_valid"] = landmark_geometry(rows)
    features["switched_geometry"], features["switched_landmark_valid"] = landmark_geometry(
        rows, switched)


@torch.inference_mode()
def encode_landmark_relations(rows, device, batch_size=256, max_landmarks=9):
    """Encode one landmark-masked relation context per landmark, without labels."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    bert = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True).to(device).eval()
    prompts, slots = [], []
    for row_index, row in enumerate(rows):
        for landmark_index, (query, name) in enumerate(zip(
                row.get("relation_queries", [])[:max_landmarks],
                row.get("landmark_names", [])[:max_landmarks])):
            prompt, _ = mask_landmark_span(row["description"], query, name)
            prompts.append(prompt)
            slots.append((row_index, landmark_index))
    features = torch.zeros(len(rows), max_landmarks, 768)
    for offset in range(0, len(prompts), batch_size):
        values = prompts[offset:offset + batch_size]
        tokens = tokenizer(values, padding=True, truncation=True, max_length=128,
                           return_tensors="pt").to(device)
        hidden = bert(**tokens).last_hidden_state
        is_mask = tokens.input_ids == tokenizer.mask_token_id
        weights = tokens.attention_mask.unsqueeze(-1)
        mean = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        mask_index = is_mask.float().argmax(-1)
        row_index = torch.arange(len(values), device=device)
        vectors = torch.where(is_mask.any(-1, keepdim=True),
                              hidden[row_index, mask_index], mean).float().cpu()
        for slot, vector in zip(slots[offset:offset + batch_size], vectors):
            features[slot] = vector
    del bert
    torch.cuda.empty_cache()
    return features.half()


def attach_relation_features(features, relation_features, rows):
    switched = same_map_switch_indices(rows)
    features["relation"] = relation_features
    features["switched_relation"] = relation_features[switched]


def validate_feature_alignment(features, rows):
    """Reject a reused feature cache if its candidates no longer match the rows."""
    candidates = torch.tensor([row["candidates"] for row in rows])
    target = torch.tensor([row["target"] for row in rows])
    expected = torch.linalg.vector_norm(candidates - target[:, None], dim=-1) * 410.0
    if features["distance"].shape != expected.shape or not torch.allclose(
            features["distance"], expected, atol=1e-4, rtol=1e-5):
        raise ValueError("feature cache is not aligned with candidate rows")


@torch.inference_mode()
def encode_texts(model, processor, texts, device, batch_size):
    vectors = []
    for offset in range(0, len(texts), batch_size):
        inputs = processor(text=texts[offset:offset + batch_size], padding="max_length",
                           max_length=64, truncation=True, return_tensors="pt")
        kwargs = {"input_ids": inputs["input_ids"].to(device)}
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"].to(device)
        vectors.append(F.normalize(model.get_text_features(**kwargs).float(), dim=-1).cpu())
    return torch.cat(vectors)


@torch.inference_mode()
def encode_images(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    pixels = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
    return F.normalize(model.get_image_features(pixel_values=pixels).float(), dim=-1).cpu()


def encode_split(rows, model, processor, data_root, device, top_k, side_m, batch_size):
    text = encode_texts(model, processor, [row["description"] for row in rows],
                        device, batch_size)
    image = None
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["map_name"]].append((index, row))
    processed = 0
    for map_name, entries in sorted(grouped.items()):
        rgb = cv2.imread(str(data_root / "rgbd" / f"{map_name}.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(map_name)
        with rasterio.open(data_root / "rgbd" / f"{map_name}.tif") as raster:
            transform = raster.transform
        pending_images, pending_slots = [], []
        for row_index, row in entries:
            for rank, candidate in enumerate(row["candidates"][:top_k]):
                pending_images.append(extract_crop(
                    rgb, transform, normalized_to_world(candidate, map_name), side_m))
                pending_slots.append((row_index, rank))
                if len(pending_images) == batch_size:
                    vectors = encode_images(model, processor, pending_images, device)
                    if image is None:
                        image = torch.empty(len(rows), top_k, vectors.shape[-1])
                    for slot, vector in zip(pending_slots, vectors):
                        image[slot] = vector
                    processed += len(pending_images)
                    print(f"RGB features {processed}/{len(rows) * top_k}", flush=True)
                    pending_images, pending_slots = [], []
        if pending_images:
            vectors = encode_images(model, processor, pending_images, device)
            if image is None:
                image = torch.empty(len(rows), top_k, vectors.shape[-1])
            for slot, vector in zip(pending_slots, vectors):
                image[slot] = vector
            processed += len(pending_images)
            print(f"RGB features {processed}/{len(rows) * top_k}", flush=True)

    candidates = torch.tensor([row["candidates"][:top_k] for row in rows])
    target = torch.tensor([row["target"] for row in rows])
    distance = torch.linalg.vector_norm(candidates - target[:, None], dim=-1) * 410.0
    return {"image": image.half(), "text": text.half(), "distance": distance}


class CandidateRanker(nn.Module):
    def __init__(self, feature_dim, top_k, relation_dim=768):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.text = nn.Sequential(nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.1),
                                   nn.Linear(256, 1))
        self.relation_text = nn.Sequential(nn.Linear(relation_dim, 128), nn.SiLU(),
                                           nn.Linear(128, 64))
        self.geometry = nn.Sequential(nn.Linear(10, 64), nn.LayerNorm(64), nn.SiLU())
        self.relation_score = nn.Sequential(nn.Linear(128, 64), nn.SiLU(),
                                            nn.Linear(64, 1))
        self.relation_scale = nn.Parameter(torch.tensor(0.1))
        self.rank_bias = nn.Parameter(torch.zeros(top_k))

    def forward(self, image, text, geometry=None, landmark_valid=None, relation=None):
        image = self.image(image.float())
        projected_text = self.text(text.float())[:, None].expand_as(image)
        value = torch.cat((image * projected_text,
                           torch.abs(image - projected_text)), dim=-1)
        score = self.score(value).squeeze(-1) + self.rank_bias
        if geometry is not None and relation is not None:
            geometry = self.geometry(geometry.float())
            relation = self.relation_text(relation.float())[:, None].expand_as(geometry)
            relation_value = torch.cat((geometry * relation,
                                        torch.abs(geometry - relation)), dim=-1)
            relation_score = self.relation_score(relation_value).squeeze(-1)
            valid = landmark_valid[:, None].float()
            relation_score = (relation_score * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)
            score = score + self.relation_scale * relation_score
        return score


def metrics(distance, rank):
    chosen = distance[torch.arange(len(distance)), rank]
    return {
        "mean_ne": chosen.mean().item(),
        "median_ne": chosen.median().item(),
        "hit20": (chosen <= 20).float().mean().mul(100).item(),
        "hit50": (chosen <= 50).float().mean().mul(100).item(),
    }


@torch.inference_mode()
def tune_spatial_weight(model, data, device):
    score = model(data["image"].to(device), data["text"].to(device),
                  data["geometry"].to(device),
                  data["landmark_valid"].to(device),
                  data["relation"].to(device)).cpu()
    prior = -torch.arange(score.shape[-1], dtype=score.dtype)[None]
    best = None
    for weight in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        rank = (score + weight * prior).argmax(-1)
        result = metrics(data["distance"], rank)
        key = (result["hit20"], -result["mean_ne"])
        if best is None or key > best[0]:
            best = key, weight
    return best[1]


@torch.inference_mode()
def evaluate(model, data, device, spatial_weight):
    score = model(data["image"].to(device), data["text"].to(device),
                  data["geometry"].to(device),
                  data["landmark_valid"].to(device),
                  data["relation"].to(device)).cpu()
    prior = -torch.arange(score.shape[-1], dtype=score.dtype)[None]
    true_rank = score.argmax(-1)
    hybrid_rank = (score + spatial_weight * prior).argmax(-1)
    shuffled_rank = model(data["image"].to(device), data["text"].roll(1, 0).to(device),
                          data["geometry"].to(device),
                          data["landmark_valid"].to(device),
                          data["relation"].roll(1, 0).to(device)).cpu().argmax(-1)
    switched_score = model(data["image"].to(device),
                           data["switched_text"].to(device),
                           data["switched_geometry"].to(device),
                           data["switched_landmark_valid"].to(device),
                           data["switched_relation"].to(device)).cpu()
    switched_rank = switched_score.argmax(-1)
    distance = data["distance"]
    nearest = distance.argmin(-1)
    row = torch.arange(len(distance))
    return {
        "rgb_text": metrics(distance, true_rank),
        "geometry_rgb_text": metrics(distance, hybrid_rank),
        "shuffled_text": metrics(distance, shuffled_rank),
        "same_map_switched_text": metrics(distance, switched_rank),
        "same_map_switch_target_margin": (
            score[row, nearest] - switched_score[row, nearest]).mean().item(),
        "same_map_switch_coverage": data["switch_changed"].float().mean().item(),
        "spatial_top1": metrics(distance, torch.zeros(len(distance), dtype=torch.long)),
        "oracle_top5": metrics(distance, distance.argmin(-1)),
    }


def train_one(train, val_seen, feature_dim, top_k, seed, epochs, device,
              switch_loss_weight, switch_margin):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = CandidateRanker(feature_dim, top_k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    dataset = TensorDataset(
        train["image"], train["text"], train["switched_text"], train["distance"],
        train["geometry"], train["landmark_valid"],
        train["switched_geometry"], train["switched_landmark_valid"],
        train["relation"], train["switched_relation"])
    best_state, best_loss = None, float("inf")
    for epoch in range(epochs):
        model.train()
        for (image, text, switched_text, distance, geometry, landmark_valid,
             switched_geometry, switched_landmark_valid, relation,
             switched_relation) in DataLoader(
                dataset, batch_size=256, shuffle=True):
            score = model(image.to(device), text.to(device), geometry.to(device),
                          landmark_valid.to(device), relation.to(device))
            switched_score = model(
                image.to(device), switched_text.to(device), switched_geometry.to(device),
                switched_landmark_valid.to(device), switched_relation.to(device))
            target = F.softmax(-distance.to(device) / 20.0, dim=-1)
            ranking_loss = -(target * F.log_softmax(score, dim=-1)).sum(-1).mean()
            nearest = distance.argmin(-1).to(device)
            positive = score.gather(1, nearest[:, None]).squeeze(1)
            switched = switched_score.gather(1, nearest[:, None]).squeeze(1)
            switch_loss = F.softplus(switch_margin - positive + switched).mean()
            loss = ranking_loss + switch_loss_weight * switch_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = model(val_seen["image"].to(device), val_seen["text"].to(device),
                          val_seen["geometry"].to(device),
                          val_seen["landmark_valid"].to(device),
                          val_seen["relation"].to(device))
            target = F.softmax(-val_seen["distance"].to(device) / 20.0, dim=-1)
            val_loss = -(target * F.log_softmax(score, dim=-1)).sum(-1).mean().item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"seed={seed} epoch={epoch + 1}/{epochs} val_loss={val_loss:.5f}", flush=True)
    model.load_state_dict(best_state)
    return model.eval(), best_loss


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, injected = prepare_rows(args.candidate_dir / "train_seen.jsonl",
                                        args.train_count, args.seed, args.top_k, True)
    seen_rows, _ = prepare_rows(args.candidate_dir / "val_seen.jsonl",
                                args.val_count, args.seed, args.top_k, False)
    unseen_rows, _ = prepare_rows(args.candidate_dir / "val_unseen.jsonl",
                                  args.val_count, args.seed, args.top_k, False)
    if args.cache.exists():
        features = torch.load(args.cache, map_location="cpu", weights_only=True)
    else:
        processor = AutoProcessor.from_pretrained(
            args.model, cache_dir=str(args.hf_cache), local_files_only=True)
        encoder = AutoModel.from_pretrained(
            args.model, cache_dir=str(args.hf_cache), local_files_only=True,
            torch_dtype=torch.bfloat16).to(device).eval()
        features = {
            "train_seen": encode_split(train_rows, encoder, processor, args.data_root,
                                       device, args.top_k, args.crop_side_m,
                                       args.feature_batch_size),
            "val_seen": encode_split(seen_rows, encoder, processor, args.data_root,
                                     device, args.top_k, args.crop_side_m,
                                     args.feature_batch_size),
            "val_unseen": encode_split(unseen_rows, encoder, processor, args.data_root,
                                       device, args.top_k, args.crop_side_m,
                                       args.feature_batch_size),
        }
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, args.cache)
        del encoder
        torch.cuda.empty_cache()

    if args.relation_cache.exists():
        relation_features = torch.load(
            args.relation_cache, map_location="cpu", weights_only=True)
    else:
        relation_features = {
            "train_seen": encode_landmark_relations(train_rows, device),
            "val_seen": encode_landmark_relations(seen_rows, device),
            "val_unseen": encode_landmark_relations(unseen_rows, device),
        }
        args.relation_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(relation_features, args.relation_cache)

    validate_feature_alignment(features["train_seen"], train_rows)
    validate_feature_alignment(features["val_seen"], seen_rows)
    validate_feature_alignment(features["val_unseen"], unseen_rows)

    attach_switched_text(features["train_seen"], train_rows)
    attach_switched_text(features["val_seen"], seen_rows)
    attach_switched_text(features["val_unseen"], unseen_rows)
    attach_landmark_geometry(features["train_seen"], train_rows)
    attach_landmark_geometry(features["val_seen"], seen_rows)
    attach_landmark_geometry(features["val_unseen"], unseen_rows)
    attach_relation_features(features["train_seen"], relation_features["train_seen"],
                             train_rows)
    attach_relation_features(features["val_seen"], relation_features["val_seen"],
                             seen_rows)
    attach_relation_features(features["val_unseen"], relation_features["val_unseen"],
                             unseen_rows)

    results = {
        "config": {**vars(args), "candidate_dir": str(args.candidate_dir),
                   "data_root": str(args.data_root), "output": str(args.output),
                   "cache": str(args.cache), "relation_cache": str(args.relation_cache),
                   "hf_cache": str(args.hf_cache)},
        "counts": {"train_seen": len(train_rows), "val_seen": len(seen_rows),
                   "val_unseen": len(unseen_rows), "injected_train_positive": injected},
        "runs": [],
    }
    feature_dim = features["train_seen"]["image"].shape[-1]
    for seed in [int(value) for value in args.seeds.split(",") if value]:
        model, best_loss = train_one(features["train_seen"], features["val_seen"],
                                     feature_dim, args.top_k, seed, args.epochs, device,
                                     args.switch_loss_weight, args.switch_margin)
        spatial_weight = tune_spatial_weight(model, features["val_seen"], device)
        results["runs"].append({
            "seed": seed, "best_val_loss": best_loss,
            "spatial_weight": spatial_weight,
            "val_seen": evaluate(model, features["val_seen"], device, spatial_weight),
            "val_unseen": evaluate(model, features["val_unseen"], device, spatial_weight),
        })
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
