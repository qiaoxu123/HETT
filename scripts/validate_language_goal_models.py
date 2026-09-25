#!/usr/bin/env python3
"""Pre-integration language-conditioned goal-localization comparison.

This script deliberately stays outside the HETT rollout.  It compares a point
regressor, a spatial heatmap classifier and a conditional coordinate diffusion
model using the same frozen BERT, landmark raster and start-position inputs.
Only development splits are evaluated.
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
                        default=Path("runs/preintegration_goal_models/results.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--diffusion-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=20)
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


def named_objects(objects, map_name):
    return [obj for obj in objects[map_name].values() if obj.get("name")]


def closest_landmark(query, candidates):
    # The production map uses Levenshtein matching.  Exact normalized matching
    # covers nearly every processed name; difflib is a dependency-free fallback.
    from difflib import SequenceMatcher
    query = query.lower().replace(" ", "")
    return max(candidates, key=lambda obj: SequenceMatcher(
        None, query, obj["name"].lower().replace(" ", "")).ratio())


def landmark_raster(map_name, queries, objects, size=20):
    mask = np.zeros((size, size), dtype=np.float32)
    candidates = named_objects(objects, map_name)
    if not candidates:
        return mask
    for query in queries:
        obj = closest_landmark(query, candidates)
        points = []
        for x, y in obj["contour"]:
            nx, ny = normalized_xy(x, y, map_name)
            points.append([int(np.clip(nx * size, 0, size - 1)),
                           int(np.clip(ny * size, 0, size - 1))])
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1.0)
    return mask


def build_split(split, trajectories, objects, processed, raster_size=20):
    rows = []
    for item in trajectories:
        if item.get("dist_marker_to_target", 0.0) > 30.0:
            continue
        map_name = f'{item["area"]}_block_{item["block"]}'
        object_id = str(item["object_ids"][0])
        ann_id = int(item["ann_ids"][0])
        description = item["descriptions"][0]
        descriptions = processed.get(map_name, {}).get(object_id, [])
        queries = descriptions[ann_id].get("landmarks", []) if ann_id < len(descriptions) else []
        target = normalized_xy(*item["target_positions"][-1][:2], map_name)
        start = normalized_xy(*item["trajectory"][0][:2], map_name)
        rows.append({
            "split": split,
            "map_name": map_name,
            "description": description,
            "start": start,
            "target": target,
            "landmark": landmark_raster(map_name, queries, objects, raster_size),
        })
    return rows


@torch.inference_mode()
def encode_text(rows_by_split, device, batch_size=128):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    bert = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True).to(device).eval()
    descriptions = sorted({row["description"] for rows in rows_by_split.values() for row in rows})
    encoded = {}
    for offset in range(0, len(descriptions), batch_size):
        texts = descriptions[offset:offset + batch_size]
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=128,
                           return_tensors="pt").to(device)
        vectors = bert(**tokens).pooler_output.float().cpu()
        encoded.update(zip(texts, vectors))
    del bert
    torch.cuda.empty_cache()
    return encoded


def tensors(rows, text_vectors, grid_size):
    text = torch.stack([text_vectors[row["description"]] for row in rows])
    landmark = torch.from_numpy(np.stack([row["landmark"] for row in rows])).unsqueeze(1)
    start = torch.from_numpy(np.stack([row["start"] for row in rows]))
    target = torch.from_numpy(np.stack([row["target"] for row in rows]))
    cell_xy = torch.clamp((target * grid_size).long(), 0, grid_size - 1)
    cell = cell_xy[:, 1] * grid_size + cell_xy[:, 0]
    return text, landmark, start, target, cell


class ConditionEncoder(nn.Module):
    def __init__(self, raster_size):
        super().__init__()
        self.text = nn.Sequential(nn.Linear(768, 192), nn.LayerNorm(192), nn.SiLU())
        self.map = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
        )
        map_dim = 16 * math.ceil(raster_size / 2) ** 2
        self.fuse = nn.Sequential(
            nn.Linear(192 + map_dim + 2, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(256, 192), nn.SiLU(),
        )

    def forward(self, text, landmark, start):
        return self.fuse(torch.cat((self.text(text), self.map(landmark), start), dim=-1))


class GoalModel(nn.Module):
    def __init__(self, raster_size, output_dim, sigmoid=False):
        super().__init__()
        self.condition = ConditionEncoder(raster_size)
        self.head = nn.Sequential(nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, output_dim))
        self.sigmoid = sigmoid

    def forward(self, text, landmark, start):
        value = self.head(self.condition(text, landmark, start))
        return value.sigmoid() if self.sigmoid else value


class DiffusionGoal(nn.Module):
    def __init__(self, raster_size, steps):
        super().__init__()
        self.condition = ConditionEncoder(raster_size)
        self.time = nn.Embedding(steps, 48)
        self.net = nn.Sequential(
            nn.Linear(192 + 48 + 2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2),
        )

    def forward(self, noisy_xy, timestep, text, landmark, start):
        cond = self.condition(text, landmark, start)
        return self.net(torch.cat((noisy_xy, self.time(timestep), cond), dim=-1))


def loader(data, batch_size, shuffle):
    return DataLoader(TensorDataset(*data), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def train_goal_model(model, train_data, device, epochs, batch_size, heatmap=False):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for text, landmark, start, target, cell in loader(train_data, batch_size, True):
            text, landmark, start = text.to(device), landmark.to(device), start.to(device)
            target, cell = target.to(device), cell.to(device)
            output = model(text, landmark, start)
            loss = F.cross_entropy(output, cell) if heatmap else F.smooth_l1_loss(output, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(text)
        print(f'{"heatmap" if heatmap else "point"} epoch {epoch + 1}/{epochs} loss={total / len(train_data[0]):.5f}')
    return model.eval()


def cosine_schedule(steps, device):
    x = torch.linspace(0, steps, steps + 1, device=device)
    alpha_bar = torch.cos(((x / steps + 0.008) / 1.008) * math.pi * 0.5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-4, 0.999)


def train_diffusion(model, train_data, device, epochs, batch_size, steps):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    betas = cosine_schedule(steps, device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for text, landmark, start, target, _ in loader(train_data, batch_size, True):
            text, landmark, start = text.to(device), landmark.to(device), start.to(device)
            clean = target.to(device) * 2.0 - 1.0
            timestep = torch.randint(0, steps, (len(text),), device=device)
            noise = torch.randn_like(clean)
            abar = alpha_bar[timestep].unsqueeze(-1)
            noisy = abar.sqrt() * clean + (1.0 - abar).sqrt() * noise
            prediction = model(noisy, timestep, text, landmark, start)
            loss = F.mse_loss(prediction, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(text)
        print(f'diffusion epoch {epoch + 1}/{epochs} loss={total / len(train_data[0]):.5f}')
    return model.eval(), betas


@torch.inference_mode()
def predict_model(model, data, device, batch_size, grid_size=None):
    predictions, targets = [], []
    for text, landmark, start, target, _ in loader(data, batch_size, False):
        output = model(text.to(device), landmark.to(device), start.to(device))
        if grid_size:
            index = output.argmax(-1)
            x = (index % grid_size + 0.5) / grid_size
            y = (index // grid_size + 0.5) / grid_size
            output = torch.stack((x, y), dim=-1)
        predictions.append(output.cpu())
        targets.append(target)
    return torch.cat(predictions), torch.cat(targets)


@torch.inference_mode()
def predict_logits(model, data, device, batch_size):
    outputs = []
    for text, landmark, start, _, _ in loader(data, batch_size, False):
        outputs.append(model(
            text.to(device), landmark.to(device), start.to(device)).cpu())
    return torch.cat(outputs)


@torch.inference_mode()
def sample_diffusion(model, data, device, batch_size, betas, sample_count):
    steps = len(betas)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    all_samples, all_targets = [], []
    for text, landmark, start, target, _ in loader(data, batch_size, False):
        n = len(text)
        text = text.to(device).repeat_interleave(sample_count, 0)
        landmark = landmark.to(device).repeat_interleave(sample_count, 0)
        start = start.to(device).repeat_interleave(sample_count, 0)
        value = torch.randn(n * sample_count, 2, device=device)
        for step in reversed(range(steps)):
            timestep = torch.full((len(value),), step, dtype=torch.long, device=device)
            noise = model(value, timestep, text, landmark, start)
            alpha, abar, beta = alphas[step], alpha_bar[step], betas[step]
            mean = (value - beta / torch.sqrt(1.0 - abar) * noise) / torch.sqrt(alpha)
            if step:
                previous_abar = alpha_bar[step - 1]
                variance = beta * (1.0 - previous_abar) / (1.0 - abar)
                value = mean + variance.sqrt() * torch.randn_like(value)
            else:
                value = mean
        value = ((value.clamp(-1, 1) + 1.0) * 0.5).reshape(n, sample_count, 2)
        all_samples.append(value.cpu())
        all_targets.append(target)
    return torch.cat(all_samples), torch.cat(all_targets)


def diffusion_medoid(samples):
    distances = torch.cdist(samples, samples)
    return samples[torch.arange(len(samples)), distances.mean(-1).argmin(-1)]


def standardized(score):
    return (score - score.mean(-1, keepdim=True)) / score.std(
        -1, keepdim=True, unbiased=False).clamp_min(1e-6)


def candidate_scores(samples, point_prediction, heatmap_logits, grid_size):
    point_score = -torch.linalg.vector_norm(
        samples - point_prediction.unsqueeze(1), dim=-1)
    cell_xy = torch.clamp((samples * grid_size).long(), 0, grid_size - 1)
    cell = cell_xy[..., 1] * grid_size + cell_xy[..., 0]
    heatmap_score = torch.log_softmax(heatmap_logits, dim=-1).gather(1, cell)
    density_score = -torch.cdist(samples, samples).mean(-1)
    return {
        "point": standardized(point_score),
        "heatmap": standardized(heatmap_score),
        "density": standardized(density_score),
    }


def select_candidates(samples, scores, weights):
    combined = sum(weights[name] * scores[name] for name in weights)
    index = combined.argmax(-1)
    return samples[torch.arange(len(samples)), index]


def choose_seen_weights(samples, target, scores):
    """Small deterministic search on val_seen; val_unseen remains untouched."""
    candidates = []
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    for point_weight in values:
        for heatmap_weight in values:
            density_weight = 1.0 - point_weight - heatmap_weight
            if density_weight < 0:
                continue
            weights = {
                "point": point_weight,
                "heatmap": heatmap_weight,
                "density": density_weight,
            }
            prediction = select_candidates(samples, scores, weights)
            result = metrics(prediction, target)
            candidates.append((result["hit20"], -result["ne"], weights))
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def metrics(prediction, target, samples=None, map_meters=410.0):
    distance = torch.linalg.vector_norm(prediction - target, dim=-1) * map_meters
    result = {
        "ne": distance.mean().item(),
        "median_ne": distance.median().item(),
        "p90_ne": torch.quantile(distance, 0.9).item(),
        "hit10": (distance <= 10).float().mean().mul(100).item(),
        "hit20": (distance <= 20).float().mean().mul(100).item(),
        "hit50": (distance <= 50).float().mean().mul(100).item(),
    }
    if samples is not None:
        candidate_distance = torch.linalg.vector_norm(
            samples - target.unsqueeze(1), dim=-1) * map_meters
        result["oracle20_hit20"] = (candidate_distance.min(-1).values <= 20).float().mean().mul(100).item()
        result["oracle20_ne"] = candidate_distance.min(-1).values.mean().item()
        result["sample_spread_m"] = torch.linalg.vector_norm(
            samples - samples.mean(1, keepdim=True), dim=-1).mean().mul(map_meters).item()
    return result


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_root
    objects = load_json(root / "cityrefer/objects.json")
    processed = load_json(root / "cityrefer/processed_descriptions.json")
    rows = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        trajectories = load_json(root / f"processed_citynav/citynav_{split}.json")
        rows[split] = build_split(split, trajectories, objects, processed, args.grid_size)
        print(split, len(rows[split]))

    text_vectors = encode_text(rows, device)
    data = {split: tensors(split_rows, text_vectors, args.grid_size)
            for split, split_rows in rows.items()}

    point = train_goal_model(GoalModel(args.grid_size, 2, sigmoid=True),
                             data["train_seen"], device, args.epochs, args.batch_size)
    heatmap = train_goal_model(GoalModel(args.grid_size, args.grid_size ** 2),
                               data["train_seen"], device, args.epochs,
                               args.batch_size, heatmap=True)
    diffusion, betas = train_diffusion(
        DiffusionGoal(args.grid_size, args.diffusion_steps), data["train_seen"],
        device, args.diffusion_epochs, args.batch_size, args.diffusion_steps)

    result = {"config": dict(vars(args)), "counts": {key: len(value) for key, value in rows.items()},
              "splits": {}}
    result["config"]["data_root"] = str(args.data_root)
    result["config"]["output"] = str(args.output)
    cached = {}
    for split in ("val_seen", "val_unseen"):
        point_prediction, target = predict_model(point, data[split], device, args.batch_size)
        heatmap_logits = predict_logits(heatmap, data[split], device, args.batch_size)
        heat_index = heatmap_logits.argmax(-1)
        heat_prediction = torch.stack((
            (heat_index % args.grid_size + 0.5) / args.grid_size,
            (heat_index // args.grid_size + 0.5) / args.grid_size,
        ), dim=-1)
        samples, _ = sample_diffusion(
            diffusion, data[split], device, max(1, args.batch_size // args.samples),
            betas, args.samples)
        scores = candidate_scores(samples, point_prediction, heatmap_logits, args.grid_size)
        cached[split] = {
            "point_prediction": point_prediction,
            "heat_prediction": heat_prediction,
            "heatmap_logits": heatmap_logits,
            "samples": samples,
            "scores": scores,
            "target": target,
        }

    rerank_weights = choose_seen_weights(
        cached["val_seen"]["samples"], cached["val_seen"]["target"],
        cached["val_seen"]["scores"])
    result["rerank_weights_selected_on_val_seen"] = rerank_weights
    for split in ("val_seen", "val_unseen"):
        entry = cached[split]
        samples, target = entry["samples"], entry["target"]
        medoid = diffusion_medoid(samples)
        reranked = select_candidates(samples, entry["scores"], rerank_weights)
        result["splits"][split] = {
            "point": metrics(entry["point_prediction"], target),
            "heatmap": metrics(entry["heat_prediction"], target),
            "diffusion": metrics(medoid, target, samples),
            "diffusion_reranked": metrics(reranked, target, samples),
            "selectors": {
                name: metrics(select_candidates(
                    samples, entry["scores"],
                    {key: float(key == name) for key in entry["scores"]}), target)
                for name in entry["scores"]
            },
        }
        print(split, json.dumps(result["splits"][split], indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()
