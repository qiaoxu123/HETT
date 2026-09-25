#!/usr/bin/env python3
"""Ablate union versus identity-preserving landmark maps for goal grounding.

Both variants use the same frozen BERT vectors, positional channels, start
marker, convolutional decoder, optimizer and examples.  The only difference is
whether landmark polygons are collapsed into one binary plane or retain a
learned, language-aligned feature vector derived from each landmark's name.
Only CityNav development splits are evaluated.
"""

import argparse
import json
import math
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.mapdata import MAP_BOUNDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/named_landmark_ablation/results.json"))
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path) as stream:
        return json.load(stream)


def normalized_xy(x, y, map_name, map_meters=410.0):
    bounds = MAP_BOUNDS[map_name]
    return np.array([(x - bounds.x_min) / map_meters,
                     (bounds.y_max - y) / map_meters], dtype=np.float32)


def normalized_name(value):
    return value.lower().replace(" ", "")


def closest_landmark(query, candidates):
    wanted = normalized_name(query)
    return max(candidates, key=lambda obj: SequenceMatcher(
        None, wanted, normalized_name(obj["name"])).ratio())


def object_mask(obj, map_name, size):
    mask = np.zeros((size, size), dtype=np.uint8)
    points = []
    for x, y in obj["contour"]:
        nx, ny = normalized_xy(x, y, map_name)
        points.append([int(np.clip(nx * size, 0, size - 1)),
                       int(np.clip(ny * size, 0, size - 1))])
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    return mask


def build_split(split, trajectories, objects, processed, grid_size, max_landmarks):
    rows = []
    candidate_cache = {}
    match_cache = {}
    mask_cache = {}
    for item in trajectories:
        if item.get("dist_marker_to_target", 0.0) > 30.0:
            continue
        map_name = f'{item["area"]}_block_{item["block"]}'
        object_id = str(item["object_ids"][0])
        ann_id = int(item["ann_ids"][0])
        descriptions = processed.get(map_name, {}).get(object_id, [])
        queries = descriptions[ann_id].get("landmarks", []) if ann_id < len(descriptions) else []
        if map_name not in candidate_cache:
            candidates = [obj for obj in objects[map_name].values() if obj.get("name")]
            exact = {normalized_name(obj["name"]): obj for obj in candidates}
            candidate_cache[map_name] = (candidates, exact)
        candidates, exact = candidate_cache[map_name]
        landmark_names, masks = [], []
        for query in queries[:max_landmarks]:
            if not candidates:
                break
            key = (map_name, normalized_name(query))
            if key not in match_cache:
                match_cache[key] = exact.get(key[1]) or closest_landmark(query, candidates)
            obj = match_cache[key]
            landmark_names.append(obj["name"])
            mask_key = (map_name, obj["name"])
            if mask_key not in mask_cache:
                mask_cache[mask_key] = object_mask(obj, map_name, grid_size)
            masks.append(mask_cache[mask_key])
        rows.append({
            "split": split,
            "map_name": map_name,
            "description": item["descriptions"][0],
            "start": normalized_xy(*item["trajectory"][0][:2], map_name),
            "target": normalized_xy(*item["target_positions"][-1][:2], map_name),
            "names": landmark_names,
            "masks": masks,
        })
    return rows


@torch.inference_mode()
def encode_language(rows_by_split, device, batch_size=256):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    bert = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True).to(device).eval()
    descriptions = sorted({row["description"] for rows in rows_by_split.values() for row in rows})
    names = sorted({name for rows in rows_by_split.values() for row in rows for name in row["names"]})

    def encode(values):
        result = {}
        for offset in range(0, len(values), batch_size):
            texts = values[offset:offset + batch_size]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=128,
                               return_tensors="pt").to(device)
            hidden = bert(**tokens).last_hidden_state
            weights = tokens.attention_mask.unsqueeze(-1)
            vectors = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            result.update(zip(texts, vectors.float().cpu()))
        return result

    text_vectors = encode(descriptions)
    encoded_names = encode(names)
    name_vectors = torch.stack([encoded_names[name] for name in names]) if names else torch.empty(0, 768)
    name_to_id = {name: index + 1 for index, name in enumerate(names)}
    # ID zero is padding and deliberately has an all-zero vector.
    name_vectors = torch.cat((torch.zeros(1, 768), name_vectors), dim=0)
    del bert
    torch.cuda.empty_cache()
    return text_vectors, name_vectors, name_to_id


def make_tensors(rows, text_vectors, name_to_id, grid_size, max_landmarks):
    count = len(rows)
    masks = torch.zeros(count, max_landmarks, grid_size, grid_size, dtype=torch.uint8)
    ids = torch.zeros(count, max_landmarks, dtype=torch.long)
    valid = torch.zeros(count, max_landmarks, dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for landmark_index, (name, mask) in enumerate(zip(row["names"], row["masks"])):
            ids[row_index, landmark_index] = name_to_id[name]
            masks[row_index, landmark_index] = torch.from_numpy(mask)
            valid[row_index, landmark_index] = 1.0
    text = torch.stack([text_vectors[row["description"]] for row in rows])
    start = torch.from_numpy(np.stack([row["start"] for row in rows]))
    target = torch.from_numpy(np.stack([row["target"] for row in rows]))
    cells = torch.clamp((target * grid_size).long(), 0, grid_size - 1)
    labels = cells[:, 1] * grid_size + cells[:, 0]
    return text, masks, ids, valid, start, target, labels


class FilmBlock(nn.Module):
    def __init__(self, channels, text_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film = nn.Linear(text_dim, channels * 2)

    def forward(self, value, text):
        scale, shift = self.film(text).chunk(2, dim=-1)
        hidden = self.norm1(self.conv1(value))
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = F.silu(hidden)
        return F.silu(value + self.norm2(self.conv2(hidden)))


class SpatialGoalModel(nn.Module):
    def __init__(self, name_vectors, grid_size, feature_dim=32, named=True):
        super().__init__()
        self.named = named
        self.grid_size = grid_size
        self.register_buffer("name_vectors", name_vectors)
        self.name_projection = nn.Sequential(
            nn.Linear(768, 128), nn.SiLU(), nn.Linear(128, feature_dim))
        self.union_projection = nn.Parameter(torch.randn(feature_dim) * 0.02)
        self.text = nn.Sequential(nn.Linear(768, 192), nn.LayerNorm(192), nn.SiLU())
        self.stem = nn.Conv2d(feature_dim + 4, 64, 3, padding=1)
        self.blocks = nn.ModuleList([FilmBlock(64, 192) for _ in range(4)])
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(32, 1, 1))

        axis = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("xx", xx[None, None])
        self.register_buffer("yy", yy[None, None])

    def forward(self, text, masks, name_ids, valid, start):
        masks = masks.float()
        if self.named:
            vectors = self.name_projection(self.name_vectors[name_ids])
            vectors = vectors * valid.unsqueeze(-1)
            spatial = torch.einsum("bkf,bkhw->bfhw", vectors, masks)
            divisor = masks.sum(1, keepdim=True).clamp_min(1.0)
            spatial = spatial / divisor
        else:
            union = masks.amax(1, keepdim=True)
            spatial = union * self.union_projection[None, :, None, None]

        batch = len(text)
        xx = self.xx.expand(batch, -1, -1, -1)
        yy = self.yy.expand(batch, -1, -1, -1)
        sigma = 1.25 / self.grid_size
        start_x = start[:, 0].view(-1, 1, 1, 1)
        start_y = start[:, 1].view(-1, 1, 1, 1)
        start_marker = torch.exp(-((xx - start_x) ** 2 + (yy - start_y) ** 2) /
                                 (2.0 * sigma ** 2))
        union = masks.amax(1, keepdim=True)
        value = torch.cat((spatial, union, start_marker, xx, yy), dim=1)
        condition = self.text(text)
        value = F.silu(self.stem(value))
        for block in self.blocks:
            value = block(value, condition)
        return self.head(value).flatten(1)


def make_loader(data, batch_size, shuffle):
    return DataLoader(TensorDataset(*data), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def train(model, data, device, epochs, batch_size, label):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for text, masks, ids, valid, start, _, labels in make_loader(data, batch_size, True):
            logits = model(text.to(device), masks.to(device), ids.to(device),
                           valid.to(device), start.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(text)
        print(f"{label} epoch {epoch + 1}/{epochs} loss={total / len(data[0]):.5f}", flush=True)
    return model.eval()


@torch.inference_mode()
def evaluate(model, data, device, batch_size, grid_size):
    all_logits, all_targets, all_valid = [], [], []
    for text, masks, ids, valid, start, target, _ in make_loader(data, batch_size, False):
        logits = model(text.to(device), masks.to(device), ids.to(device),
                       valid.to(device), start.to(device))
        all_logits.append(logits.cpu())
        all_targets.append(target)
        all_valid.append(valid)
    logits, target, valid = torch.cat(all_logits), torch.cat(all_targets), torch.cat(all_valid)
    max_k = min(20, logits.shape[-1])
    indices = logits.topk(max_k, dim=-1).indices
    candidates = torch.stack(((indices % grid_size + 0.5) / grid_size,
                              (indices // grid_size + 0.5) / grid_size), dim=-1)
    distances = torch.linalg.vector_norm(candidates - target[:, None], dim=-1) * 410.0
    def summarize(selected):
        chosen = distances[selected]
        top1 = chosen[:, 0]
        summary = {
            "count": int(selected.sum()),
            "mean_ne": top1.mean().item(),
            "median_ne": top1.median().item(),
            "p90_ne": torch.quantile(top1, 0.9).item(),
        }
        for k in (1, 5, 10, 20):
            nearest = chosen[:, :k].amin(-1)
            summary[f"top{k}_oracle_ne"] = nearest.mean().item()
            summary[f"top{k}_hit20"] = (nearest <= 20).float().mean().mul(100).item()
            summary[f"top{k}_hit50"] = (nearest <= 50).float().mean().mul(100).item()
        return summary

    landmark_count = valid.sum(-1)
    result = summarize(torch.ones(len(target), dtype=torch.bool))
    result["by_landmark_count"] = {}
    groups = {
        "0": landmark_count == 0,
        "1": landmark_count == 1,
        "2": landmark_count == 2,
        "3_plus": landmark_count >= 3,
    }
    for label, selected in groups.items():
        if selected.any():
            result["by_landmark_count"][label] = summarize(selected)
    return result, candidates


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = load_json(args.data_root / "cityrefer/objects.json")
    processed = load_json(args.data_root / "cityrefer/processed_descriptions.json")
    rows = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        trajectories = load_json(args.data_root / f"processed_citynav/citynav_{split}.json")
        rows[split] = build_split(split, trajectories, objects, processed,
                                  args.grid_size, args.max_landmarks)
        print(split, len(rows[split]), flush=True)

    text_vectors, name_vectors, name_to_id = encode_language(rows, device)
    data = {split: make_tensors(values, text_vectors, name_to_id,
                                args.grid_size, args.max_landmarks)
            for split, values in rows.items()}

    config = {**vars(args), "data_root": str(args.data_root), "output": str(args.output)}
    config["candidate_output"] = str(args.candidate_output) if args.candidate_output else None
    results = {
        "config": config,
        "counts": {split: len(values) for split, values in rows.items()},
        "unique_landmark_names": len(name_to_id),
        "splits": {},
    }
    for named, label in ((False, "union"), (True, "named")):
        seed_everything(args.seed)
        model = SpatialGoalModel(name_vectors, args.grid_size, args.feature_dim, named=named)
        model = train(model, data["train_seen"], device, args.epochs, args.batch_size, label)
        for split in ("val_seen", "val_unseen"):
            evaluation, candidates = evaluate(
                model, data[split], device, args.batch_size, args.grid_size)
            results["splits"].setdefault(split, {})[label] = evaluation
            print(split, label, json.dumps(results["splits"][split][label]), flush=True)
            if named and split == "val_unseen" and args.candidate_output:
                args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.candidate_output, "w") as stream:
                    for row, row_candidates in zip(rows[split], candidates):
                        stream.write(json.dumps({
                            "map_name": row["map_name"],
                            "description": row["description"],
                            "target": row["target"].tolist(),
                            "landmark_count": len(row["names"]),
                            "candidates": row_candidates.tolist(),
                        }) + "\n")
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()
