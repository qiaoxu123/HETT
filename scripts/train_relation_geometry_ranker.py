#!/usr/bin/env python3
"""Train a leakage-controlled language/landmark geometry Top-5 ranker."""

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_rgb_candidate_ranker import (
    encode_landmark_relations,
    metrics,
)


SPLITS = ("train_seen", "val_seen", "val_unseen")
CUE_NAMES = (
    "left", "right", "above", "below", "near", "between", "across_opposite",
    "front", "behind", "along_on", "off", "corner_edge",
)
CUE_PATTERNS = {
    "left": re.compile(r"\b(?:left|west|western)\b"),
    "right": re.compile(r"\b(?:right|east|eastern)\b"),
    "above": re.compile(r"\b(?:above|over|north|northern|top)\b"),
    "below": re.compile(r"\b(?:below|under|south|southern|bottom)\b"),
    "near": re.compile(r"\b(?:near|nearby|next|beside|adjacent|close|alongside|by)\b"),
    "between": re.compile(r"\bbetween\b"),
    "across_opposite": re.compile(r"\b(?:across|opposite)\b"),
    "front": re.compile(r"\bfront\b"),
    "behind": re.compile(r"\bbehind|back\b"),
    "along_on": re.compile(r"\b(?:along|on)\b"),
    "off": re.compile(r"\boff\b"),
    "corner_edge": re.compile(r"\b(?:corner|edge)\b"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-dir", type=Path,
        default=Path("runs/candidate_rgb_data/geometry_top5_oof_grouped_enriched"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/relation_geometry_ranker"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--hit-loss-weight", type=float, default=0.2)
    parser.add_argument("--mismatch-loss-weight", type=float, default=0.0)
    parser.add_argument("--mismatch-margin", type=float, default=0.2)
    parser.add_argument("--use-structured-cues", action="store_true")
    parser.add_argument("--seeds", default="0,1,2")
    return parser.parse_args()


def load_rows(path, top_k):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if len(row["candidates"]) < top_k:
            raise ValueError(f"fewer than {top_k} candidates in {path}")
        row["candidates"] = row["candidates"][:top_k]
    return rows


def rotate_270(delta):
    """Apply the fixed train-selected map-to-language reference transform."""
    return torch.stack((delta[..., 1], -delta[..., 0]), dim=-1)


def relation_cue_vector(prompt, window_before=10, window_after=7):
    """Extract a compact relation vocabulary around the masked landmark."""
    words = re.findall(r"\[mask\]|[a-z]+", prompt.lower())
    try:
        index = words.index("[mask]")
    except ValueError:
        index = len(words) // 2
    local = " ".join(words[max(0, index - window_before):index + window_after + 1])
    return torch.tensor(
        [float(bool(CUE_PATTERNS[name].search(local))) for name in CUE_NAMES])


def structured_relation_cues(rows, max_landmarks):
    from scripts.validate_candidate_geometry import mask_landmark_span

    cue = torch.zeros(len(rows), max_landmarks, len(CUE_NAMES))
    for row_index, row in enumerate(rows):
        for landmark_index, (query, name) in enumerate(zip(
                row.get("relation_queries", [])[:max_landmarks],
                row.get("landmark_names", [])[:max_landmarks])):
            prompt, _ = mask_landmark_span(row["description"], query, name)
            cue[row_index, landmark_index] = relation_cue_vector(prompt)
    return cue


def opposite_direction_cues(cue):
    result = cue.clone()
    indices = {name: index for index, name in enumerate(CUE_NAMES)}
    for first, second in (("left", "right"), ("above", "below"),
                          ("front", "behind")):
        result[..., indices[first]] = cue[..., indices[second]]
        result[..., indices[second]] = cue[..., indices[first]]
    return result


def matched_same_map_switch_indices(rows):
    """Switch descriptions within a map while preserving landmark count."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["map_name"], len(row.get("landmark_names", [])))].append(index)
    result = []
    for index, row in enumerate(rows):
        group = groups[(row["map_name"], len(row.get("landmark_names", [])))]
        start = group.index(index)
        replacement = index
        for offset in range(1, len(group) + 1):
            candidate = group[(start + offset) % len(group)]
            if rows[candidate]["description"] != row["description"]:
                replacement = candidate
                break
        result.append(replacement)
    return torch.tensor(result, dtype=torch.long)


def corrected_landmark_geometry(rows, max_landmarks=9, reference_indices=None):
    """Build inference-visible candidate/landmark features without target coordinates."""
    if reference_indices is None:
        reference_indices = torch.arange(len(rows))
    candidates = torch.tensor([row["candidates"] for row in rows], dtype=torch.float32)
    centers = torch.zeros(len(rows), max_landmarks, 2)
    boundaries = torch.zeros(len(rows), max_landmarks, 16, 2)
    valid = torch.zeros(len(rows), max_landmarks, dtype=torch.bool)
    for row_index, reference_index in enumerate(reference_indices.tolist()):
        reference = rows[reference_index]
        center_values = reference.get("landmark_centers", [])[:max_landmarks]
        boundary_values = reference.get("landmark_boundaries", [])[:max_landmarks]
        count = min(len(center_values), len(boundary_values), max_landmarks)
        if count:
            centers[row_index, :count] = torch.tensor(center_values[:count])
            boundaries[row_index, :count] = torch.tensor(boundary_values[:count])
            valid[row_index, :count] = True

    centroid_delta = rotate_270(candidates[:, :, None] - centers[:, None])
    centroid_distance = torch.linalg.vector_norm(centroid_delta, dim=-1).clamp_min(1e-6)
    centroid = torch.cat((
        centroid_delta,
        (torch.log1p(centroid_distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (centroid_delta[..., 1] / centroid_distance).unsqueeze(-1),
        (centroid_delta[..., 0] / centroid_distance).unsqueeze(-1),
    ), dim=-1)

    boundary_delta = candidates[:, :, None, None] - boundaries[:, None]
    boundary_distance = torch.linalg.vector_norm(boundary_delta, dim=-1)
    nearest = boundary_distance.argmin(-1)
    gather = nearest[..., None, None].expand(-1, -1, -1, 1, 2)
    nearest_delta = rotate_270(boundary_delta.gather(3, gather).squeeze(3))
    nearest_distance = torch.linalg.vector_norm(nearest_delta, dim=-1).clamp_min(1e-6)
    boundary = torch.cat((
        nearest_delta,
        (torch.log1p(nearest_distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (nearest_delta[..., 1] / nearest_distance).unsqueeze(-1),
        (nearest_delta[..., 0] / nearest_distance).unsqueeze(-1),
    ), dim=-1)
    geometry = torch.cat((centroid, boundary), dim=-1)
    geometry = geometry * valid[:, None, :, None]
    return geometry.half(), valid


def build_split(rows, relation, max_landmarks):
    if relation.shape[:2] != (len(rows), max_landmarks):
        raise ValueError(
            f"relation cache mismatch: {tuple(relation.shape)} vs {(len(rows), max_landmarks)}")
    candidates = torch.tensor([row["candidates"] for row in rows])
    targets = torch.tensor([row["target"] for row in rows])
    distance = torch.linalg.vector_norm(candidates - targets[:, None], dim=-1) * 410.0
    geometry, valid = corrected_landmark_geometry(rows, max_landmarks)
    switched = matched_same_map_switch_indices(rows)
    cue = structured_relation_cues(rows, max_landmarks)
    switched_geometry, switched_valid = corrected_landmark_geometry(
        rows, max_landmarks, switched)
    return {
        "geometry": geometry,
        "valid": valid,
        "relation": relation,
        "cue": cue,
        "opposite_cue": opposite_direction_cues(cue),
        "distance": distance,
        "landmark_count": valid.sum(-1),
        "switched_geometry": switched_geometry,
        "switched_valid": switched_valid,
        "switched_relation": relation[switched],
        "switched_cue": cue[switched],
        "switch_changed": switched.ne(torch.arange(len(rows))),
    }


class RelationGeometryRanker(nn.Module):
    """Score each candidate from its relations to a variable number of landmarks."""

    def __init__(self, hidden=128, cue_dim=0):
        super().__init__()
        self.relation = nn.Sequential(
            nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.geometry = nn.Sequential(
            nn.Linear(10, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        self.cue = (nn.Sequential(nn.Linear(cue_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
                    if cue_dim else None)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.LayerNorm(hidden), nn.SiLU())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, relation, geometry, valid, cue=None):
        batch, candidates, landmarks = geometry.shape[:3]
        language = self.relation(relation.float())[:, None].expand(
            batch, candidates, landmarks, -1)
        if self.cue is not None:
            if cue is None:
                raise ValueError("structured cue input is required")
            language = language + self.cue(cue.float())[:, None]
        spatial = self.geometry(geometry.float())
        token = self.fuse(torch.cat((
            language, spatial, language * spatial, torch.abs(language - spatial)), dim=-1))
        token = token.reshape(batch * candidates, landmarks, -1)
        cls = self.cls.expand(batch * candidates, -1, -1)
        sequence = torch.cat((cls, token), dim=1)
        padding = torch.cat((
            torch.zeros(batch, 1, dtype=torch.bool, device=valid.device), ~valid), dim=1)
        padding = padding[:, None].expand(batch, candidates, landmarks + 1).reshape(
            batch * candidates, landmarks + 1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return self.score(encoded[:, 0]).reshape(batch, candidates)


def ranking_loss(score, distance, hit_weight):
    soft_target = F.softmax(-distance / 20.0, dim=-1)
    listwise = -(soft_target * F.log_softmax(score, dim=-1)).sum(-1).mean()
    hit_target = distance.le(20.0).float()
    hit = F.binary_cross_entropy_with_logits(score, hit_target)
    return listwise + hit_weight * hit, listwise, hit


def listwise_loss_per_row(score, distance):
    target = F.softmax(-distance / 20.0, dim=-1)
    return -(target * F.log_softmax(score, dim=-1)).sum(-1)


@torch.inference_mode()
def scores(model, data, device, mode="normal"):
    relation, cue = data["relation"], data["cue"]
    geometry, valid = data["geometry"], data["valid"]
    if mode == "wrong_relation":
        relation, valid = data["switched_relation"], data["switched_valid"]
        cue = data["switched_cue"]
    elif mode == "wrong_landmark":
        geometry, valid = data["switched_geometry"], data["switched_valid"]
    elif mode == "both_switched":
        relation = data["switched_relation"]
        cue = data["switched_cue"]
        geometry, valid = data["switched_geometry"], data["switched_valid"]
    elif mode == "no_relation":
        relation = torch.zeros_like(relation)
        cue = torch.zeros_like(cue)
    elif mode == "no_bert":
        relation = torch.zeros_like(relation)
    elif mode == "no_structured_cue":
        cue = torch.zeros_like(cue)
    elif mode == "opposite_direction":
        cue = data["opposite_cue"]
    return model(relation.to(device), geometry.to(device), valid.to(device),
                 cue.to(device) if model.cue is not None else None).cpu()


def grouped_metrics(distance, rank, landmark_count):
    groups = {
        "one": landmark_count.eq(1),
        "two": landmark_count.eq(2),
        "three_plus": landmark_count.ge(3),
    }
    result = {}
    for name, selected in groups.items():
        result[name] = {"count": int(selected.sum()),
                        **metrics(distance[selected], rank[selected])}
    return result


@torch.inference_mode()
def evaluate(model, data, device, prior_weight):
    normal = scores(model, data, device)
    prior = -torch.arange(normal.shape[1], dtype=normal.dtype)[None]
    learned_rank = normal.argmax(-1)
    hybrid_rank = (normal + prior_weight * prior).argmax(-1)
    distance = data["distance"]
    result = {
        "learned_relation": metrics(distance, learned_rank),
        "hybrid_relation_plus_candidate_prior": metrics(distance, hybrid_rank),
        "wrong_relation_same_map": metrics(
            distance, scores(model, data, device, "wrong_relation").argmax(-1)),
        "wrong_landmark_same_map": metrics(
            distance, scores(model, data, device, "wrong_landmark").argmax(-1)),
        "both_switched_same_map": metrics(
            distance, scores(model, data, device, "both_switched").argmax(-1)),
        "no_relation": metrics(
            distance, scores(model, data, device, "no_relation").argmax(-1)),
        "no_bert": metrics(
            distance, scores(model, data, device, "no_bert").argmax(-1)),
        "no_structured_cue": metrics(
            distance, scores(model, data, device, "no_structured_cue").argmax(-1)),
        "opposite_direction": metrics(
            distance, scores(model, data, device, "opposite_direction").argmax(-1)),
        "candidate_prior_top1": metrics(
            distance, torch.zeros(len(distance), dtype=torch.long)),
        "oracle_top5": metrics(distance, distance.argmin(-1)),
        "switch_coverage_percent": data["switch_changed"].float().mean().mul(100).item(),
        "by_landmark_count": grouped_metrics(
            distance, hybrid_rank, data["landmark_count"]),
    }
    return result


@torch.inference_mode()
def tune_prior(model, data, device):
    learned = scores(model, data, device)
    prior = -torch.arange(learned.shape[1], dtype=learned.dtype)[None]
    best = None
    for weight in (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
        rank = (learned + weight * prior).argmax(-1)
        value = metrics(data["distance"], rank)
        key = value["hit20"], -value["mean_ne"]
        if best is None or key > best[0]:
            best = key, weight
    return best[1]


def train_one(train, val, seed, epochs, batch_size, hit_weight, device,
              use_structured_cues, mismatch_weight, mismatch_margin):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = RelationGeometryRanker(
        cue_dim=len(CUE_NAMES) if use_structured_cues else 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    dataset = TensorDataset(
        train["relation"], train["geometry"], train["valid"], train["cue"],
        train["distance"], train["switched_relation"], train["switched_cue"],
        train["switched_geometry"], train["switched_valid"], train["switch_changed"])
    best_state, best_key, best_epoch = None, None, None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (relation, geometry, valid, cue, distance, switched_relation,
             switched_cue, switched_geometry, switched_valid, switch_changed) in DataLoader(
                dataset, batch_size=batch_size, shuffle=True, pin_memory=True):
            score = model(relation.to(device, non_blocking=True),
                          geometry.to(device, non_blocking=True),
                          valid.to(device, non_blocking=True),
                          cue.to(device, non_blocking=True) if use_structured_cues else None)
            loss, _, _ = ranking_loss(score, distance.to(device), hit_weight)
            if mismatch_weight > 0 and switch_changed.any():
                wrong_relation = model(
                    switched_relation.to(device, non_blocking=True),
                    geometry.to(device, non_blocking=True),
                    valid.to(device, non_blocking=True),
                    switched_cue.to(device, non_blocking=True)
                    if use_structured_cues else None)
                wrong_landmark = model(
                    relation.to(device, non_blocking=True),
                    switched_geometry.to(device, non_blocking=True),
                    switched_valid.to(device, non_blocking=True),
                    cue.to(device, non_blocking=True) if use_structured_cues else None)
                correct_ce = listwise_loss_per_row(score, distance.to(device))
                relation_ce = listwise_loss_per_row(wrong_relation, distance.to(device))
                landmark_ce = listwise_loss_per_row(wrong_landmark, distance.to(device))
                selected = switch_changed.to(device)
                mismatch = 0.5 * (
                    F.softplus(mismatch_margin + correct_ce[selected] - relation_ce[selected])
                    + F.softplus(mismatch_margin + correct_ce[selected] - landmark_ce[selected]))
                loss = loss + mismatch_weight * mismatch.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(relation)
        model.eval()
        with torch.inference_mode():
            val_score = scores(model, val, device)
            val_loss = ranking_loss(
                val_score.to(device), val["distance"].to(device), hit_weight)[0].item()
            val_value = metrics(val["distance"], val_score.argmax(-1))
        key = val_value["hit20"], -val_value["mean_ne"], -val_loss
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        print(
            f"seed={seed} epoch={epoch + 1}/{epochs} "
            f"train_loss={total / len(dataset):.4f} val_loss={val_loss:.4f} "
            f"val_hit20={val_value['hit20']:.2f}", flush=True)
    model.load_state_dict(best_state)
    return model.eval(), best_epoch


def mean_summary(runs):
    keys = ("learned_relation", "hybrid_relation_plus_candidate_prior",
            "wrong_relation_same_map", "wrong_landmark_same_map",
            "both_switched_same_map", "no_relation", "no_bert",
            "no_structured_cue", "opposite_direction", "candidate_prior_top1",
            "oracle_top5")
    result = {}
    for key in keys:
        result[key] = {
            metric: sum(run["val_unseen"][key][metric] for run in runs) / len(runs)
            for metric in ("mean_ne", "median_ne", "hit20", "hit50")
        }
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {split: load_rows(args.candidate_dir / f"{split}.jsonl", args.top_k)
            for split in SPLITS}
    cache_path = args.output_dir / "relation_features.pt"
    if cache_path.exists():
        relations = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        relations = {}
        for split in SPLITS:
            print(f"encoding relations: {split} ({len(rows[split])})", flush=True)
            relations[split] = encode_landmark_relations(
                rows[split], device, max_landmarks=args.max_landmarks)
        torch.save(relations, cache_path)
    data = {split: build_split(rows[split], relations[split], args.max_landmarks)
            for split in SPLITS}

    seeds = [int(value) for value in args.seeds.split(",") if value]
    result = {
        "config": {
            **vars(args), "candidate_dir": str(args.candidate_dir),
            "output_dir": str(args.output_dir), "seeds": seeds,
            "coordinate_transform": "fixed rotate_270 selected on train_seen",
            "language_encoder": "frozen bert-base-uncased [MASK] landmark token",
        },
        "counts": {split: len(rows[split]) for split in SPLITS},
        "structured_cue_coverage_percent": {
            split: data[split]["cue"].any(-1)[data[split]["valid"]].float().mean().mul(100).item()
            for split in SPLITS},
        "anti_leakage": {
            "train_candidates": "map-level out-of-fold predictions",
            "model_inputs": "masked relation embeddings plus candidate-to-landmark geometry",
            "target_coordinates": "training labels and evaluation distances only",
            "absolute_candidate_coordinates": False,
            "rgb_features": False,
            "target_type_features": False,
            "validation_rule": "train_seen training, val_seen selection, val_unseen final only",
        },
        "runs": [],
    }
    for seed in seeds:
        model, best_epoch = train_one(
            data["train_seen"], data["val_seen"], seed, args.epochs,
            args.batch_size, args.hit_loss_weight, device, args.use_structured_cues,
            args.mismatch_loss_weight, args.mismatch_margin)
        prior_weight = tune_prior(model, data["val_seen"], device)
        run = {
            "seed": seed, "best_epoch": best_epoch, "prior_weight": prior_weight,
            "val_seen": evaluate(model, data["val_seen"], device, prior_weight),
            "val_unseen": evaluate(model, data["val_unseen"], device, prior_weight),
        }
        result["runs"].append(run)
        torch.save(model.state_dict(), args.output_dir / f"model_seed{seed}.pt")
        del model
        torch.cuda.empty_cache()
    result["val_unseen_mean"] = mean_summary(result["runs"])
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["val_unseen_mean"], indent=2))


if __name__ == "__main__":
    main()
