#!/usr/bin/env python3
"""Rank candidates from relation language and shape-aware named-landmark masks.

Unlike the centroid baseline, this model retains the oriented extent of every
landmark mask.  Target coordinates remain labels only and are never included
in an inference feature.
"""

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, RelationGeometryRanker, build_split, evaluate, listwise_loss_per_row,
    load_rows, ranking_loss, rotate_270, scores, tune_prior,
)


ROOT = Path("runs")
SPLITS = ("train_seen", "val_seen", "val_unseen")


def shape_aware_geometry(rows, max_landmarks=9, reference_indices=None):
    """Describe the full landmark support around each candidate.

    The old representation retained only a centroid and the nearest boundary
    point.  Here eight fixed directions preserve the mask's oriented extent,
    plus size, area and containment.  The representation is invariant to the
    arbitrary first point of a sampled contour.
    """
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
        torch.log1p(centroid_distance * 410.0).div(torch.log(torch.tensor(411.0))).unsqueeze(-1),
        (centroid_delta[..., 1] / centroid_distance).unsqueeze(-1),
        (centroid_delta[..., 0] / centroid_distance).unsqueeze(-1),
    ), dim=-1)

    boundary_delta = rotate_270(candidates[:, :, None, None] - boundaries[:, None])
    boundary_distance = torch.linalg.vector_norm(boundary_delta, dim=-1)
    nearest_index = boundary_distance.argmin(-1)
    gather = nearest_index[..., None, None].expand(-1, -1, -1, 1, 2)
    nearest_delta = boundary_delta.gather(3, gather).squeeze(3)
    nearest_distance = torch.linalg.vector_norm(nearest_delta, dim=-1).clamp_min(1e-6)
    nearest = torch.cat((
        nearest_delta,
        torch.log1p(nearest_distance * 410.0).div(torch.log(torch.tensor(411.0))).unsqueeze(-1),
        (nearest_delta[..., 1] / nearest_distance).unsqueeze(-1),
        (nearest_delta[..., 0] / nearest_distance).unsqueeze(-1),
    ), dim=-1)

    angles = torch.arange(8, dtype=torch.float32) * (2.0 * torch.pi / 8.0)
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    projection = torch.einsum("nklpd,ad->nklpa", boundary_delta, directions)
    support = torch.cat((projection.amin(3), projection.amax(3)), dim=-1)

    minimum = boundary_delta.amin(3)
    maximum = boundary_delta.amax(3)
    size = maximum - minimum
    diagonal = torch.linalg.vector_norm(size, dim=-1, keepdim=True)
    inside_box = ((minimum <= 0) & (maximum >= 0)).all(-1, keepdim=True).float()
    points = boundaries
    shifted = torch.roll(points, shifts=-1, dims=2)
    area = 0.5 * torch.abs(
        (points[..., 0] * shifted[..., 1] - points[..., 1] * shifted[..., 0]).sum(-1))
    area = area[:, None, :, None].expand(len(rows), candidates.shape[1], -1, -1)
    shape = torch.cat((size, diagonal, area, inside_box), dim=-1)

    geometry = torch.cat((centroid, nearest, support, shape), dim=-1)
    geometry = geometry * valid[:, None, :, None]
    return geometry.half(), valid


def build_mask_split(rows, relation, max_landmarks=9):
    data = build_split(rows, relation, max_landmarks)
    data["geometry"], data["valid"] = shape_aware_geometry(rows, max_landmarks)
    switched = torch.empty(len(rows), dtype=torch.long)
    # build_split already created aligned switched geometry but not its indices;
    # reproduce the deterministic switch through its public helper.
    from scripts.train_relation_geometry_ranker import matched_same_map_switch_indices
    switched[:] = matched_same_map_switch_indices(rows)
    data["switched_geometry"], data["switched_valid"] = shape_aware_geometry(
        rows, max_landmarks, switched)
    return data


class LandmarkMaskRelationRanker(RelationGeometryRanker):
    def __init__(self, geometry_dim=31, hidden=128):
        super().__init__(hidden=hidden, cue_dim=len(CUE_NAMES))
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))


def train_one(train, validation, seed, device, epochs=15):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = LandmarkMaskRelationRanker(train["geometry"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    dataset = TensorDataset(
        train["relation"], train["geometry"], train["valid"], train["cue"],
        train["distance"], train["switched_relation"], train["switched_cue"],
        train["switched_geometry"], train["switched_valid"], train["switch_changed"])
    best_state, best_key, best_epoch = None, None, 0
    for epoch in range(epochs):
        model.train()
        for (relation, geometry, valid, cue, distance, switched_relation,
             switched_cue, switched_geometry, switched_valid, switch_changed) in DataLoader(
                dataset, batch_size=256, shuffle=True, pin_memory=True):
            relation = relation.to(device); geometry = geometry.to(device)
            valid = valid.to(device); cue = cue.to(device); distance = distance.to(device)
            score = model(relation, geometry, valid, cue)
            loss = ranking_loss(score, distance, 0.2)[0]
            if switch_changed.any():
                selected = switch_changed.to(device)
                wrong_relation = model(switched_relation.to(device), geometry, valid,
                                       switched_cue.to(device))
                wrong_landmark = model(relation, switched_geometry.to(device),
                                       switched_valid.to(device), cue)
                correct_ce = listwise_loss_per_row(score, distance)
                relation_ce = listwise_loss_per_row(wrong_relation, distance)
                landmark_ce = listwise_loss_per_row(wrong_landmark, distance)
                mismatch = 0.5 * (
                    F.softplus(0.2 + correct_ce[selected] - relation_ce[selected])
                    + F.softplus(0.2 + correct_ce[selected] - landmark_ce[selected]))
                loss = loss + 0.2 * mismatch.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = scores(model, validation, device)
            from scripts.train_rgb_candidate_ranker import metrics
            value = metrics(validation["distance"], prediction.argmax(-1))
        key = value["hit20"], -value["mean_ne"]
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        print(f"mask seed={seed} epoch={epoch + 1}/{epochs} "
              f"val_hit20={value['hit20']:.2f}", flush=True)
    model.load_state_dict(best_state)
    return model.eval(), best_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    relation_dir = ROOT / "relation_geometry_ranker_contrastive"
    output_dir = ROOT / "landmark_mask_relation_ranker"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {split: load_rows(candidate_dir / f"{split}.jsonl", 5) for split in SPLITS}
    relation = torch.load(
        relation_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    data = {split: build_mask_split(rows[split], relation[split], 9) for split in SPLITS}
    runs = []
    for seed in (0, 1, 2):
        model, epoch = train_one(data["train_seen"], data["val_seen"], seed, device)
        prior_weight = tune_prior(model, data["val_seen"], device)
        run = {"seed": seed, "best_epoch": epoch, "prior_weight": prior_weight,
               "val_seen": evaluate(model, data["val_seen"], device, prior_weight),
               "val_unseen": evaluate(model, data["val_unseen"], device, prior_weight)}
        runs.append(run)
        torch.save(model.state_dict(), output_dir / f"model_seed{seed}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    keys = ("learned_relation", "wrong_relation_same_map", "wrong_landmark_same_map",
            "no_relation", "no_structured_cue", "opposite_direction",
            "candidate_prior_top1", "oracle_top5")
    mean = {key: {metric: sum(run["val_unseen"][key][metric] for run in runs) / len(runs)
                  for metric in ("mean_ne", "median_ne", "hit20", "hit50")}
            for key in keys}
    result = {
        "architecture": "shape-aware named-landmark mask support plus relation transformer",
        "geometry_dim": int(data["train_seen"]["geometry"].shape[-1]),
        "runs": runs,
        "val_unseen_mean": mean,
        "anti_leakage": {
            "training": "train_seen only", "checkpoint_selection": "val_seen only",
            "val_unseen_used_for_training_or_selection": False,
            "target_coordinates": "ranking labels and final metrics only",
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
