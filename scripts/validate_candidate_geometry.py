#!/usr/bin/env python3
"""Gate-1 validation for explicit candidate-to-landmark geometry.

The anonymous and named models see identical candidate grids, frozen BERT
instruction features, landmark centers, start positions and supervision.  The
only ablation is whether each landmark geometry token is bound to its name.
Only CityNav development splits are evaluated.
"""

import argparse
import json
import math
import random
import re
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
                        default=Path("runs/candidate_geometry/results.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--models", default="anonymous_geometry,named_geometry")
    parser.add_argument("--flip-loss-weight", type=float, default=0.2)
    parser.add_argument("--candidate-output-dir", type=Path)
    parser.add_argument("--candidate-top-k", type=int, default=5)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalized_xy(x, y, map_name, map_meters=410.0):
    bounds = MAP_BOUNDS[map_name]
    return np.array([(x - bounds.x_min) / map_meters,
                     (bounds.y_max - y) / map_meters], dtype=np.float32)


def normalize_name(value):
    return "".join(value.lower().split())


def polygon_centroid(contour):
    points = np.asarray(contour, dtype=np.float32)
    moments = cv2.moments(points)
    if abs(moments["m00"]) < 1e-8:
        return points.mean(0)
    return np.array([moments["m10"] / moments["m00"],
                     moments["m01"] / moments["m00"]], dtype=np.float32)


def resample_contour(contour, count=16):
    points = np.asarray(contour, dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    closed = np.concatenate((points, points[:1]), axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=-1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] < 1e-6:
        return np.repeat(points[:1], count, axis=0)
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    result = []
    for value in samples:
        segment = min(np.searchsorted(cumulative, value, side="right") - 1,
                      len(points) - 1)
        fraction = (value - cumulative[segment]) / max(lengths[segment], 1e-6)
        result.append(closed[segment] + fraction * (closed[segment + 1] - closed[segment]))
    return np.asarray(result, dtype=np.float32)


def resolve_landmark(query, candidates, exact):
    key = normalize_name(query)
    if key in exact:
        return exact[key]
    return max(candidates, key=lambda obj: SequenceMatcher(
        None, key, normalize_name(obj["name"])).ratio())


def resolve_landmark_group(query, candidates, min_score=0.58, min_margin=0.05):
    """Resolve one semantic landmark to all same-name component contours."""
    grouped = {}
    for obj in candidates:
        grouped.setdefault(normalize_name(obj["name"]), []).append(obj)
    key = normalize_name(query)
    if key in grouped:
        return grouped[key]
    ranked = sorted(
        ((SequenceMatcher(None, key, name).ratio(), name) for name in grouped),
        reverse=True)
    if not ranked:
        return []
    best_score, best_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score < min_score or best_score - second_score < min_margin:
        return []
    return grouped[best_name]


def polygon_group_centroid(contours):
    """Area-weighted centroid for a possibly multi-component landmark."""
    weighted, total = np.zeros(2, dtype=np.float64), 0.0
    fallback = []
    for contour in contours:
        points = np.asarray(contour, dtype=np.float32)
        fallback.append(points)
        moments = cv2.moments(points)
        area = abs(moments["m00"])
        if area > 1e-8:
            center = np.array([moments["m10"] / moments["m00"],
                               moments["m01"] / moments["m00"]])
            weighted += area * center
            total += area
    if total > 0:
        return (weighted / total).astype(np.float32)
    return np.concatenate(fallback).mean(0).astype(np.float32)


def resample_contour_group(contours, count=16):
    """Sample a fixed number of boundary points over all components."""
    starts, ends, lengths = [], [], []
    for contour in contours:
        points = np.asarray(contour, dtype=np.float32)
        if len(points) == 1:
            starts.append(points[0]); ends.append(points[0]); lengths.append(1e-6)
            continue
        closed = np.concatenate((points, points[:1]), axis=0)
        starts.extend(closed[:-1]); ends.extend(closed[1:])
        lengths.extend(np.linalg.norm(np.diff(closed, axis=0), axis=-1))
    starts, ends = np.asarray(starts), np.asarray(ends)
    lengths = np.asarray(lengths, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] < 1e-6:
        return np.repeat(starts[:1], count, axis=0)
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    result = []
    for value in samples:
        index = min(np.searchsorted(cumulative, value, side="right") - 1,
                    len(lengths) - 1)
        fraction = (value - cumulative[index]) / max(lengths[index], 1e-6)
        result.append(starts[index] + fraction * (ends[index] - starts[index]))
    return np.asarray(result, dtype=np.float32)


def _exact_mask(description, value):
    words = re.findall(r"[A-Za-z0-9]+", value)
    if not words:
        return description, False
    pattern = r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b"
    masked, count = re.subn(pattern, "[MASK]", description, count=1,
                            flags=re.IGNORECASE)
    return masked, bool(count)


def mask_landmark_span(description, query, resolved_name=None, fuzzy_threshold=0.58):
    """Mask exact, alias or fuzzy landmark spans without retaining its name."""
    for value in (query, resolved_name):
        if value:
            masked, found = _exact_mask(description, value)
            if found:
                return masked, True

    query_words = re.findall(r"[a-z0-9]+", query.lower())
    matches = list(re.finditer(r"[A-Za-z0-9]+", description))
    if not query_words or not matches:
        return description, False
    query_key = "".join(query_words)
    best = (0.0, None, None)
    for length in range(max(1, len(query_words) - 2), len(query_words) + 3):
        for start in range(0, len(matches) - length + 1):
            end = start + length
            candidate = "".join(match.group(0).lower() for match in matches[start:end])
            ratio = SequenceMatcher(None, query_key, candidate).ratio()
            if ratio > best[0]:
                best = ratio, matches[start].start(), matches[end - 1].end()
    if best[0] < fuzzy_threshold:
        return description, False
    return description[:best[1]] + "[MASK]" + description[best[2]:], True


def has_direction_relation(description):
    words = set(re.findall(r"[a-z]+", description.lower()))
    directional = {
        "north", "northern", "south", "southern", "east", "eastern",
        "west", "western", "left", "right", "behind", "front", "opposite",
        "across", "between", "above", "below", "under", "over",
    }
    return bool(words & directional)


def build_split(trajectories, objects, processed, max_landmarks):
    rows = []
    map_cache, match_cache = {}, {}
    for item in trajectories:
        if item.get("dist_marker_to_target", 0.0) > 30.0:
            continue
        map_name = f'{item["area"]}_block_{item["block"]}'
        object_id = str(item["object_ids"][0])
        ann_id = int(item["ann_ids"][0])
        annotations = processed.get(map_name, {}).get(object_id, [])
        queries = annotations[ann_id].get("landmarks", []) if ann_id < len(annotations) else []
        if map_name not in map_cache:
            candidates = [obj for obj in objects[map_name].values() if obj.get("name")]
            exact = {normalize_name(obj["name"]): obj for obj in candidates}
            map_cache[map_name] = candidates, exact
        candidates, exact = map_cache[map_name]
        names, relation_queries, centers, boundaries = [], [], [], []
        for query in queries[:max_landmarks]:
            key = map_name, normalize_name(query)
            if key not in match_cache:
                match_cache[key] = resolve_landmark_group(query, candidates)
            landmarks = match_cache[key]
            if not landmarks:
                continue
            names.append(landmarks[0]["name"])
            relation_queries.append(query)
            contours = [landmark["contour"] for landmark in landmarks]
            centers.append(normalized_xy(*polygon_group_centroid(contours), map_name))
            boundaries.append(np.stack([
                normalized_xy(*point, map_name)
                for point in resample_contour_group(contours)
            ]))
        rows.append({
            "map_name": map_name,
            "description": item["descriptions"][0],
            "names": names,
            "relation_queries": relation_queries,
            "centers": centers,
            "boundaries": boundaries,
            "start": normalized_xy(*item["trajectory"][0][:2], map_name),
            "target": normalized_xy(*item["target_positions"][-1][:2], map_name),
            "has_direction": has_direction_relation(item["descriptions"][0]),
        })
    return rows


@torch.inference_mode()
def encode_language(rows_by_split, device, batch_size=256):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    bert = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True).to(device).eval()
    descriptions = sorted({row["description"] for rows in rows_by_split.values() for row in rows})
    names = sorted({name for rows in rows_by_split.values() for row in rows for name in row["names"]})
    relation_prompts = set()
    masked_count = 0
    relation_count = 0
    for rows in rows_by_split.values():
        for row in rows:
            for query, name in zip(row["relation_queries"], row["names"]):
                prompt, found = mask_landmark_span(row["description"], query, name)
                relation_prompts.add(prompt)
                masked_count += int(found)
                relation_count += 1
    relation_prompts = sorted(relation_prompts)

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

    def encode_relations(values):
        result = {}
        for offset in range(0, len(values), batch_size):
            texts = values[offset:offset + batch_size]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=128,
                               return_tensors="pt").to(device)
            hidden = bert(**tokens).last_hidden_state
            is_mask = tokens.input_ids == tokenizer.mask_token_id
            weights = tokens.attention_mask.unsqueeze(-1)
            mean_vectors = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            mask_index = is_mask.float().argmax(-1)
            row_index = torch.arange(len(texts), device=device)
            mask_vectors = hidden[row_index, mask_index]
            has_mask = is_mask.any(-1).unsqueeze(-1)
            vectors = torch.where(has_mask, mask_vectors, mean_vectors)
            result.update(zip(texts, vectors.float().cpu()))
        return result

    text_vectors = encode(descriptions)
    encoded_names = encode(names)
    name_vectors = torch.stack([encoded_names[name] for name in names])
    name_vectors = torch.cat((torch.zeros(1, 768), name_vectors), dim=0)
    name_to_id = {name: index + 1 for index, name in enumerate(names)}
    encoded_relations = encode_relations(relation_prompts)
    relation_vectors = torch.stack([encoded_relations[prompt] for prompt in relation_prompts])
    relation_vectors = torch.cat((torch.zeros(1, 768), relation_vectors), dim=0)
    relation_to_id = {prompt: index + 1 for index, prompt in enumerate(relation_prompts)}
    print(f"landmark span mask coverage {masked_count}/{relation_count} "
          f"({100.0 * masked_count / max(relation_count, 1):.2f}%)", flush=True)
    del bert
    torch.cuda.empty_cache()
    return text_vectors, name_vectors, name_to_id, relation_vectors, relation_to_id


def make_tensors(rows, text_vectors, name_to_id, relation_to_id, grid_size, max_landmarks):
    count = len(rows)
    ids = torch.zeros(count, max_landmarks, dtype=torch.long)
    relation_ids = torch.zeros(count, max_landmarks, dtype=torch.long)
    centers = torch.zeros(count, max_landmarks, 2)
    boundaries = torch.zeros(count, max_landmarks, 16, 2)
    valid = torch.zeros(count, max_landmarks)
    for row_index, row in enumerate(rows):
        for landmark_index, (name, query, center, boundary) in enumerate(zip(
                row["names"], row["relation_queries"], row["centers"], row["boundaries"])):
            ids[row_index, landmark_index] = name_to_id[name]
            prompt, _ = mask_landmark_span(row["description"], query, name)
            relation_ids[row_index, landmark_index] = relation_to_id[prompt]
            centers[row_index, landmark_index] = torch.from_numpy(center)
            boundaries[row_index, landmark_index] = torch.from_numpy(boundary)
            valid[row_index, landmark_index] = 1.0
    text = torch.stack([text_vectors[row["description"]] for row in rows])
    start = torch.from_numpy(np.stack([row["start"] for row in rows]))
    target = torch.from_numpy(np.stack([row["target"] for row in rows]))
    cell_xy = torch.clamp((target * grid_size).long(), 0, grid_size - 1)
    labels = cell_xy[:, 1] * grid_size + cell_xy[:, 0]
    has_direction = torch.tensor([row["has_direction"] for row in rows], dtype=torch.float32)
    return (text, ids, relation_ids, centers, boundaries, valid, start, target,
            labels, has_direction)


def candidate_geometry(candidates, landmark_centers, landmark_boundaries=None):
    """Return normalized dx/dy, log-distance and continuous angle features."""
    delta = candidates[None, None] - landmark_centers[:, :, None]
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    centroid_features = torch.cat((
        delta,
        (torch.log1p(distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (delta[..., 1] / distance).unsqueeze(-1),
        (delta[..., 0] / distance).unsqueeze(-1),
    ), dim=-1)
    if landmark_boundaries is None:
        return centroid_features
    boundary_delta = candidates[None, None, :, None] - landmark_boundaries[:, :, None]
    boundary_distance = torch.linalg.vector_norm(boundary_delta, dim=-1)
    nearest = boundary_distance.argmin(-1)
    gather_index = nearest[..., None, None].expand(-1, -1, -1, 1, 2)
    nearest_delta = boundary_delta.gather(3, gather_index).squeeze(3)
    nearest_distance = torch.linalg.vector_norm(nearest_delta, dim=-1).clamp_min(1e-6)
    boundary_features = torch.cat((
        nearest_delta,
        (torch.log1p(nearest_distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (nearest_delta[..., 1] / nearest_distance).unsqueeze(-1),
        (nearest_delta[..., 0] / nearest_distance).unsqueeze(-1),
    ), dim=-1)
    return torch.cat((centroid_features, boundary_features), dim=-1)


class GeometryRanker(nn.Module):
    def __init__(self, name_vectors, relation_vectors, grid_size, mode, hidden=64):
        super().__init__()
        self.mode = mode
        self.register_buffer("name_vectors", name_vectors)
        self.register_buffer("relation_vectors", relation_vectors)
        axis = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("candidates", torch.stack((xx.flatten(), yy.flatten()), dim=-1))
        self.text_relation = nn.Sequential(nn.Linear(768, 192), nn.SiLU(), nn.Linear(192, hidden))
        self.context_relation = nn.Sequential(nn.Linear(768, 192), nn.SiLU(), nn.Linear(192, hidden))
        self.name = nn.Sequential(nn.Linear(768, 128), nn.SiLU(), nn.Linear(128, hidden))
        geometry_dim = 10 if mode.startswith("span_geometry_boundary") else 5
        self.geometry = nn.Sequential(nn.Linear(geometry_dim, 128), nn.SiLU(),
                                      nn.Linear(128, hidden))
        self.absolute = nn.Sequential(nn.Linear(4, 128), nn.SiLU(), nn.Linear(128, hidden))
        self.text_absolute = nn.Sequential(nn.Linear(768, 128), nn.SiLU(), nn.Linear(128, hidden))
        self.temperature = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, text, name_ids, relation_ids, centers, boundaries, valid, start,
                flip_geometry=False):
        geometry = candidate_geometry(
            self.candidates, centers,
            boundaries if self.mode.startswith("span_geometry_boundary") else None)
        if flip_geometry:
            geometry = geometry.clone()
            geometry[..., :2] = -geometry[..., :2]
            geometry[..., 3:5] = -geometry[..., 3:5]
            if geometry.shape[-1] == 10:
                geometry[..., 5:7] = -geometry[..., 5:7]
                geometry[..., 8:10] = -geometry[..., 8:10]
        geometry_key = F.normalize(self.geometry(geometry), dim=-1)
        if self.mode.startswith("span_geometry"):
            query = self.context_relation(self.relation_vectors[relation_ids])
        else:
            query = self.text_relation(text)[:, None]
        if self.mode == "named_geometry":
            query = query + self.name(self.name_vectors[name_ids])
        query = F.normalize(query, dim=-1)
        per_landmark = torch.einsum("bkch,bkh->bkc", geometry_key, query)
        relation_score = (per_landmark * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)

        batch = len(text)
        candidates = self.candidates[None].expand(batch, -1, -1)
        start_delta = candidates - start[:, None]
        absolute_key = F.normalize(self.absolute(torch.cat((candidates, start_delta), dim=-1)), dim=-1)
        absolute_query = F.normalize(self.text_absolute(text), dim=-1)
        absolute_score = torch.einsum("bch,bh->bc", absolute_key, absolute_query)
        return (relation_score + 0.35 * absolute_score) * self.temperature.exp().clamp(max=100)


def loader(data, batch_size, shuffle):
    return DataLoader(TensorDataset(*data), batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0)


def train_model(model, data, device, epochs, batch_size, label, flip_loss_weight):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for text, ids, relation_ids, centers, boundaries, valid, start, target, target_cell, has_direction in loader(
                data, batch_size, True):
            model_args = (text.to(device), ids.to(device), relation_ids.to(device),
                          centers.to(device), boundaries.to(device), valid.to(device),
                          start.to(device))
            logits = model(*model_args)
            if model.mode == "span_geometry_boundary_soft":
                target = target.to(device)
                distance = torch.linalg.vector_norm(
                    model.candidates[None] - target[:, None], dim=-1) * 410.0
                soft_target = F.softmax(-distance / 20.0, dim=-1)
                loss = -(soft_target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            else:
                loss = F.cross_entropy(logits, target_cell.to(device))
            if model.mode == "span_geometry_flip":
                flipped = model(*model_args, flip_geometry=True)
                cells = target_cell.to(device)
                row = torch.arange(len(text), device=device)
                margin = logits[row, cells] - flipped[row, cells]
                direction = has_direction.to(device)
                flip_loss = (F.relu(0.5 - margin) * direction).sum() / direction.sum().clamp_min(1)
                loss = loss + flip_loss_weight * flip_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(text)
        print(f"{label} epoch {epoch + 1}/{epochs} loss={total / len(data[0]):.5f}", flush=True)
    return model.eval()


def distance_metrics(distances):
    result = {
        "count": len(distances),
        "mean_ne": distances[:, 0].mean().item(),
        "median_ne": distances[:, 0].median().item(),
        "hit20": (distances[:, 0] <= 20).float().mean().mul(100).item(),
        "hit50": (distances[:, 0] <= 50).float().mean().mul(100).item(),
    }
    for k in (5, 10, 20):
        nearest = distances[:, :k].amin(-1)
        result[f"top{k}_hit20"] = (nearest <= 20).float().mean().mul(100).item()
        result[f"top{k}_oracle_ne"] = nearest.mean().item()
    return result


@torch.inference_mode()
def evaluate(model, data, device, batch_size):
    logits_all, flipped_all, targets_all, valid_all, direction_all = [], [], [], [], []
    for text, ids, relation_ids, centers, boundaries, valid, start, target, _, has_direction in loader(
            data, batch_size, False):
        args = (text.to(device), ids.to(device), relation_ids.to(device),
                centers.to(device), boundaries.to(device), valid.to(device),
                start.to(device))
        logits_all.append(model(*args).cpu())
        flipped_all.append(model(*args, flip_geometry=True).cpu())
        targets_all.append(target)
        valid_all.append(valid)
        direction_all.append(has_direction)
    logits = torch.cat(logits_all)
    flipped = torch.cat(flipped_all)
    targets = torch.cat(targets_all)
    valid = torch.cat(valid_all)
    has_direction = torch.cat(direction_all).bool()
    indices = logits.topk(20, dim=-1).indices
    candidates = model.candidates.cpu()[indices]
    distances = torch.linalg.vector_norm(candidates - targets[:, None], dim=-1) * 410.0
    target_cells = torch.linalg.vector_norm(
        model.candidates.cpu()[None] - targets[:, None], dim=-1).argmin(-1)
    row = torch.arange(len(targets))
    perturbation_margin = (logits[row, target_cells] - flipped[row, target_cells]).mean().item()
    result = distance_metrics(distances)
    result["geometry_flip_target_margin"] = perturbation_margin
    result["geometry_flip_changes_top1"] = (
        logits.argmax(-1) != flipped.argmax(-1)).float().mean().mul(100).item()
    if has_direction.any():
        direction_row = row[has_direction]
        direction_cells = target_cells[has_direction]
        direction_margin = (
            logits[direction_row, direction_cells] -
            flipped[direction_row, direction_cells]
        ).mean().item()
        result["direction_subset"] = distance_metrics(distances[has_direction])
        result["direction_subset"]["geometry_flip_target_margin"] = direction_margin
    result["by_landmark_count"] = {}
    counts = valid.sum(-1)
    for label, selected in {
        "0": counts == 0, "1": counts == 1, "2": counts == 2, "3_plus": counts >= 3
    }.items():
        if selected.any():
            result["by_landmark_count"][label] = distance_metrics(distances[selected])
    return result


@torch.inference_mode()
def predict_candidates(model, data, device, batch_size, top_k):
    candidates_all = []
    for text, ids, relation_ids, centers, boundaries, valid, start, _, _, _ in loader(
            data, batch_size, False):
        logits = model(text.to(device), ids.to(device), relation_ids.to(device),
                       centers.to(device), boundaries.to(device), valid.to(device),
                       start.to(device))
        indices = logits.topk(top_k, dim=-1).indices.cpu()
        candidates_all.append(model.candidates.cpu()[indices])
    return torch.cat(candidates_all)


def save_candidates(path, rows, candidates):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        for row, values in zip(rows, candidates):
            stream.write(json.dumps({
                "map_name": row["map_name"],
                "description": row["description"],
                "target": row["target"].tolist(),
                "landmark_count": len(row["names"]),
                "candidates": values.tolist(),
            }) + "\n")


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    processed = json.load(open(args.data_root / "cityrefer/processed_descriptions.json"))
    rows = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        trajectories = json.load(open(args.data_root / f"processed_citynav/citynav_{split}.json"))
        rows[split] = build_split(trajectories, objects, processed, args.max_landmarks)
        print(split, len(rows[split]), flush=True)
    text_vectors, name_vectors, name_to_id, relation_vectors, relation_to_id = encode_language(
        rows, device)
    data = {split: make_tensors(values, text_vectors, name_to_id, relation_to_id,
                                args.grid_size, args.max_landmarks)
            for split, values in rows.items()}

    config = {**vars(args), "data_root": str(args.data_root), "output": str(args.output)}
    config["candidate_output_dir"] = (str(args.candidate_output_dir)
                                      if args.candidate_output_dir else None)
    results = {
        "config": config,
        "counts": {split: len(values) for split, values in rows.items()},
        "runs": [],
    }
    for seed in seeds:
        run = {"seed": seed, "splits": {}}
        for label in [value for value in args.models.split(",") if value]:
            if label not in {"anonymous_geometry", "named_geometry",
                             "span_geometry", "span_geometry_flip",
                             "span_geometry_boundary", "span_geometry_boundary_soft"}:
                raise ValueError(f"Unknown model: {label}")
            seed_everything(seed)
            model = GeometryRanker(name_vectors, relation_vectors, args.grid_size, mode=label)
            model = train_model(model, data["train_seen"], device, args.epochs,
                                args.batch_size, f"seed={seed} {label}",
                                args.flip_loss_weight)
            for split in ("val_seen", "val_unseen"):
                run["splits"].setdefault(split, {})[label] = evaluate(
                    model, data[split], device, args.batch_size)
                print(seed, split, label,
                      json.dumps(run["splits"][split][label]), flush=True)
            if (args.candidate_output_dir and seed == seeds[0] and
                    label == "span_geometry_boundary_soft"):
                for split in ("train_seen", "val_seen", "val_unseen"):
                    candidates = predict_candidates(
                        model, data[split], device, args.batch_size,
                        args.candidate_top_k)
                    path = args.candidate_output_dir / f"{split}.jsonl"
                    save_candidates(path, rows[split], candidates)
                    print("saved candidates", path, len(candidates), flush=True)
            del model
            torch.cuda.empty_cache()
        results["runs"].append(run)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()
