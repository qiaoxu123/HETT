#!/usr/bin/env python3
"""Gate candidate-level RGB target/landmark/road observation signatures."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.belief_transformer import BeliefTransformer, TemporalBeliefTransformer, grid_centers
from scripts.build_road_structure_layer import build_binary_layer
from scripts.train_belief_transformer import BeliefDataset, load_rows
from scripts.train_rgb_candidate_ranker import metrics
from scripts.train_temporal_belief_transformer import same_map_shuffle


DEFAULT_SOURCE = Path(
    "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/runs")
TYPE_TO_SEMANTIC = {
    "Building": 1, "Car": 2, "Ground": 3, "TrafficRoad": 3,
    "Footpath": 3, "Parking": 4,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("runs/temporal_belief_candidate_rgb_gate5_large"))
    parser.add_argument("--feature-cache", type=Path, default=Path(
        "runs/temporal_belief_candidate_rgb_gate5_large/semantic_bev_support20.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--type-checkpoint", type=Path, default=Path(
        "runs/temporal_belief_transformer_gate5_large/model.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/rgb_relation_signature_gate6/results.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def road_required(description):
    return bool(re.search(
        r"\b(?:across|opposite|road|street|corner|junction|intersection)\b",
        description.lower()))


def type_catalog(objects):
    result = {}
    for map_name, values in objects.items():
        groups = {}
        for value in values.values():
            if value.get("name"):
                key = "".join(value["name"].lower().split())
                groups.setdefault(key, []).append(value.get("object_type", "Building"))
        result[map_name] = groups
    return result


def expected_landmark_classes(row, catalog, maximum=9):
    output = []
    for name in row.get("landmark_names", [])[:maximum]:
        key = "".join(name.lower().split())
        values = catalog.get(row["map_name"], {}).get(key, [])
        semantic = [TYPE_TO_SEMANTIC[value] for value in values
                    if value in TYPE_TO_SEMANTIC]
        output.append(semantic[0] if semantic else 1)
    return output


def candidate_masks(candidates, radius_m=20.0, grid_size=20):
    cells = grid_centers(grid_size).reshape(-1, 2)
    points = torch.tensor(candidates[:3], dtype=torch.float32)
    distance = torch.linalg.vector_norm(
        cells[None] - points[:, None], dim=-1) * 410.0
    return distance.le(radius_m).reshape(3, grid_size, grid_size)


def top_fraction(value, mask, fraction=0.2):
    selected = value[mask]
    if not len(selected):
        return 0.0
    count = max(1, int(np.ceil(len(selected) * fraction)))
    return float(selected.topk(count).values.mean())


def corridor_mask(candidate, landmarks, valid, grid_size=20):
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    start = np.rint(np.asarray(candidate) * (grid_size - 1)).astype(int)
    for center, is_valid in zip(landmarks, valid):
        if not is_valid:
            continue
        end = np.rint(np.asarray(center) * (grid_size - 1)).astype(int)
        cv2.line(mask, tuple(start), tuple(end), 1, thickness=1)
    return torch.from_numpy(mask.astype(bool))


@torch.inference_mode()
def predicted_types(type_model, base, device, batch_size=128):
    output = []
    for offset in range(0, len(base), batch_size):
        relation = base.relation[offset:offset + batch_size].to(device).float()
        cue = base.cues[offset:offset + batch_size].to(device).float()
        valid = base.valid[offset:offset + batch_size].to(device)
        weight = valid[..., None].float()
        count = weight.sum(1).clamp_min(1.0)
        language = type_model.language((relation * weight).sum(1) / count)
        language = language + type_model.cue((cue * weight).sum(1) / count)
        output.append(type_model.target_type(language).argmax(-1).cpu())
    return torch.cat(output)


def extract_signatures(rows, base, cache, predicted_type, objects, catalog,
                       wrong_landmark=False, wrong_type=False):
    wrong_index = same_map_shuffle(cache["map_names"])
    features, distances = [], []
    road_layers = {}
    for index, row in enumerate(rows):
        semantic = cache["semantic"][index].float()
        view_visible = cache["visibility"][index]
        grid_size = semantic.shape[-1]
        masks = candidate_masks(row["candidates"], grid_size=grid_size)
        reference = int(wrong_index[index]) if wrong_landmark else index
        landmark_masks = base.instance_masks[reference]
        landmark_valid = base.valid[reference]
        landmark_centers = base.centers[reference]
        reference_row = rows[reference]
        landmark_classes = expected_landmark_classes(reference_row, catalog)
        target_class = (int(predicted_type[index]) + int(wrong_type)) % 4 + 1
        name = row["map_name"]
        if name not in road_layers:
            value, _ = build_binary_layer(
                objects, name, resolution=grid_size,
                object_types=("TrafficRoad", "Footpath"))
            road_layers[name] = torch.from_numpy(value.astype(bool))
        candidate_values = []
        for candidate_index in range(3):
            view = min(candidate_index, semantic.shape[0] - 1)
            local = masks[candidate_index] & view_visible[view]
            target_probability = semantic[view, target_class]
            target_mean = float(target_probability[local].mean()) if local.any() else 0.0
            target_top = top_fraction(target_probability, local)
            landmark_scores = []
            for landmark_index, is_valid in enumerate(landmark_valid.tolist()):
                if not is_valid:
                    continue
                visible = landmark_masks[landmark_index] & view_visible[view]
                if not visible.any():
                    landmark_scores.append(0.0)
                    continue
                expected = (landmark_classes[landmark_index]
                            if landmark_index < len(landmark_classes) else 1)
                coverage = visible.sum().float() / landmark_masks[landmark_index].sum().clamp_min(1)
                agreement = semantic[view, expected][visible].mean()
                landmark_scores.append(float(coverage * agreement))
            landmark_mean = float(np.mean(landmark_scores)) if landmark_scores else 1.0
            landmark_min = float(np.min(landmark_scores)) if landmark_scores else 1.0
            corridor = corridor_mask(
                row["candidates"][candidate_index], landmark_centers,
                landmark_valid, grid_size=grid_size)
            corridor_visible = corridor & view_visible[view]
            if road_required(row["description"]):
                expected_road = corridor_visible & road_layers[name]
                road_score = (float(semantic[view, 3][expected_road].mean())
                              if expected_road.any() else 0.0)
            else:
                road_score = 1.0
            visible_fraction = float(local.float().mean())
            candidate_values.append([
                -candidate_index / 2.0,
                target_mean, target_top,
                landmark_mean, landmark_min,
                target_top * landmark_mean,
                road_score, visible_fraction,
            ])
        features.append(candidate_values)
        target = torch.tensor(row["target"])
        candidates = torch.tensor(row["candidates"][:3])
        distances.append(torch.linalg.vector_norm(
            candidates - target[None], dim=-1) * 410.0)
    return torch.tensor(features), torch.stack(distances)


class SignatureRanker(nn.Module):
    def __init__(self, dimension, hidden=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, value):
        return self.network(value).squeeze(-1)


def listwise_loss(score, distance):
    target = F.softmax(-distance / 20.0, dim=-1)
    return -(target * F.log_softmax(score, dim=-1)).sum(-1).mean()


def train_ranker(train_feature, train_distance, val_feature, val_distance,
                 indices, epochs, seed, device):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = SignatureRanker(len(indices)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    dataset = TensorDataset(train_feature[..., indices], train_distance)
    best, state, best_epoch = None, None, None
    for epoch in range(1, epochs + 1):
        model.train()
        for feature, distance in DataLoader(dataset, batch_size=128, shuffle=True):
            loss = listwise_loss(model(feature.to(device)), distance.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = model(val_feature[..., indices].to(device)).cpu()
        value = metrics(val_distance, score.argmax(-1))
        key = value["hit20"], -value["mean_ne"]
        if best is None or key > best:
            best, best_epoch = key, epoch
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model.eval(), best_epoch


@torch.inference_mode()
def evaluate(model, feature, distance, indices, device):
    score = model(feature[..., indices].to(device)).cpu()
    return metrics(distance, score.argmax(-1))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation = torch.load(args.relation_cache, map_location="cpu", weights_only=True)
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    objects = json.loads((args.data_root / "cityrefer" / "objects.json").read_text())
    catalog = type_catalog(objects)
    rows, bases = {}, {}
    grid_size = int(cache["metadata"]["grid_size"])
    for split in ("train_seen", "val_seen"):
        values = load_rows(args.candidate_dir / f"{split}.jsonl")
        source = cache["splits"][split]["source_rows"].tolist()
        rows[split] = [values[index] for index in source]
        bases[split] = BeliefDataset(
            rows[split], relation[split][source], grid_size=grid_size, max_landmarks=9)

    static = BeliefTransformer(grid_size=20, hidden=96, layers=2)
    type_model = TemporalBeliefTransformer(static, hidden=64).to(device)
    saved = torch.load(args.type_checkpoint, map_location="cpu", weights_only=True)
    type_model.load_state_dict(saved["model"])
    type_model.eval()
    predicted = {split: predicted_types(type_model, bases[split], device)
                 for split in ("train_seen", "val_seen")}

    normal, wrong_type_values, wrong_landmark_values, distances = {}, {}, {}, {}
    for split in ("train_seen", "val_seen"):
        split_cache = cache["splits"][split]
        normal[split], distances[split] = extract_signatures(
            rows[split], bases[split], split_cache, predicted[split],
            objects, catalog)
        wrong_type_values[split], _ = extract_signatures(
            rows[split], bases[split], split_cache, predicted[split],
            objects, catalog, wrong_type=True)
        wrong_landmark_values[split], _ = extract_signatures(
            rows[split], bases[split], split_cache, predicted[split],
            objects, catalog, wrong_landmark=True)

    variants = {
        "target_type": [0, 1, 2, 7],
        "target_plus_landmarks": [0, 1, 2, 3, 4, 5, 7],
        "target_landmarks_road": list(range(8)),
    }
    result = {
        "counts": {split: len(rows[split]) for split in rows},
        "candidate_prior": metrics(
            distances["val_seen"], torch.zeros(len(rows["val_seen"]), dtype=torch.long)),
        "top3_oracle": metrics(distances["val_seen"], distances["val_seen"].argmin(-1)),
        "variants": {},
        "anti_leakage": {
            "ranker_training": "train_seen only", "selection": "val_seen only",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
            "view_generation": "predicted candidates and named landmarks only",
            "target_coordinates": "ranking loss and final metrics only",
        },
    }
    for name, indices in variants.items():
        model, epoch = train_ranker(
            normal["train_seen"], distances["train_seen"],
            normal["val_seen"], distances["val_seen"],
            indices, args.epochs, args.seed, device)
        entry = {
            "selected_epoch": epoch,
            "normal": evaluate(model, normal["val_seen"], distances["val_seen"],
                               indices, device),
        }
        if name == "target_landmarks_road":
            entry["wrong_target_type"] = evaluate(
                model, wrong_type_values["val_seen"], distances["val_seen"],
                indices, device)
            entry["wrong_landmarks"] = evaluate(
                model, wrong_landmark_values["val_seen"], distances["val_seen"],
                indices, device)
        result["variants"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
