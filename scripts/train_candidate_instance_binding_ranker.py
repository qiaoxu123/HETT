#!/usr/bin/env python3
"""Test candidate-object-instance to named-landmark relation binding.

Each grid candidate is bound to the nearest annotated object component of the
language-predicted target class.  The target object id and target coordinate
are never used to choose that component; coordinates are labels only.
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
from scripts.audit_target_type_alignment import CLASSES, train_probe
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, RelationGeometryRanker, evaluate, listwise_loss_per_row, load_rows,
    matched_same_map_switch_indices, opposite_direction_cues, ranking_loss,
    rotate_270, scores, structured_relation_cues, tune_prior,
)
from scripts.train_rgb_candidate_ranker import metrics
from scripts.validate_candidate_geometry import normalized_xy, polygon_centroid, resample_contour


ROOT = Path("runs")
SPLITS = ("train_seen", "val_seen", "val_unseen")


def components_by_map(objects):
    result = {}
    for map_name, values in objects.items():
        grouped = [[] for _ in CLASSES]
        for obj in values.values():
            if obj.get("object_type") not in CLASSES:
                continue
            contour_world = resample_contour(obj["contour"], 16)
            contour = torch.tensor([
                normalized_xy(float(x), float(y), map_name).tolist()
                for x, y in contour_world], dtype=torch.float32)
            center = torch.tensor(normalized_xy(
                *polygon_centroid(obj["contour"]), map_name), dtype=torch.float32)
            grouped[CLASSES.index(obj["object_type"])].append((center, contour))
        result[map_name] = grouped
    return result


def bind_candidate_instances(rows, predicted_class, components):
    centers = torch.zeros(len(rows), 5, 2)
    boundaries = torch.zeros(len(rows), 5, 16, 2)
    valid = torch.zeros(len(rows), 5, dtype=torch.bool)
    unique_counts, offsets = [], []
    for index, row in enumerate(rows):
        candidate = torch.tensor(row["candidates"], dtype=torch.float32)
        available = components[row["map_name"]][int(predicted_class[index])]
        if not available:
            centers[index] = candidate
            boundaries[index] = candidate[:, None].expand(-1, 16, -1)
            unique_counts.append(5)
            offsets.extend([0.0] * 5)
            continue
        component_centers = torch.stack([item[0] for item in available])
        nearest = torch.cdist(candidate, component_centers).argmin(-1)
        centers[index] = component_centers[nearest]
        boundaries[index] = torch.stack([available[int(value)][1] for value in nearest])
        valid[index] = True
        unique_counts.append(len(set(nearest.tolist())))
        offsets.extend((torch.linalg.vector_norm(
            centers[index] - candidate, dim=-1) * 410.0).tolist())
    stats = {
        "mean_unique_instances_per_top5": sum(unique_counts) / len(unique_counts),
        "mean_candidate_to_instance_offset_m": sum(offsets) / len(offsets),
    }
    return centers, boundaries, valid, stats


def _landmarks(rows, max_landmarks, reference_indices=None):
    if reference_indices is None:
        reference_indices = torch.arange(len(rows))
    centers = torch.zeros(len(rows), max_landmarks, 2)
    boundaries = torch.zeros(len(rows), max_landmarks, 16, 2)
    valid = torch.zeros(len(rows), max_landmarks, dtype=torch.bool)
    for row_index, reference_index in enumerate(reference_indices.tolist()):
        row = rows[reference_index]
        count = min(len(row.get("landmark_centers", [])),
                    len(row.get("landmark_boundaries", [])), max_landmarks)
        if count:
            centers[row_index, :count] = torch.tensor(row["landmark_centers"][:count])
            boundaries[row_index, :count] = torch.tensor(row["landmark_boundaries"][:count])
            valid[row_index, :count] = True
    return centers, boundaries, valid


def pair_geometry(rows, instance_centers, instance_boundaries, instance_valid,
                  max_landmarks=9, reference_indices=None):
    landmark_centers, landmark_boundaries, landmark_valid = _landmarks(
        rows, max_landmarks, reference_indices)
    # target-instance center relative to each named landmark
    center_delta = rotate_270(
        instance_centers[:, :, None] - landmark_centers[:, None])
    center_distance = torch.linalg.vector_norm(center_delta, dim=-1).clamp_min(1e-6)
    center = torch.cat((
        center_delta,
        torch.log1p(center_distance * 410.0).div(torch.log(torch.tensor(411.0))).unsqueeze(-1),
        (center_delta[..., 1] / center_distance).unsqueeze(-1),
        (center_delta[..., 0] / center_distance).unsqueeze(-1),
    ), -1)

    # Closest visible edges of the candidate instance and landmark instance.
    target_edge = instance_boundaries[:, :, None, :, None, :]
    landmark_edge = landmark_boundaries[:, None, :, None, :, :]
    edge_delta = rotate_270(target_edge - landmark_edge)
    edge_distance = torch.linalg.vector_norm(edge_delta, dim=-1)
    flat_distance = edge_distance.flatten(3)
    flat_delta = edge_delta.flatten(3, 4)
    nearest_index = flat_distance.argmin(-1)
    nearest_delta = flat_delta.gather(
        3, nearest_index[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(3)
    nearest_distance = torch.linalg.vector_norm(nearest_delta, dim=-1).clamp_min(1e-6)
    edge = torch.cat((
        nearest_delta,
        torch.log1p(nearest_distance * 410.0).div(torch.log(torch.tensor(411.0))).unsqueeze(-1),
        (nearest_delta[..., 1] / nearest_distance).unsqueeze(-1),
        (nearest_delta[..., 0] / nearest_distance).unsqueeze(-1),
    ), -1)

    target_min = instance_boundaries.amin(2)[:, :, None]
    target_max = instance_boundaries.amax(2)[:, :, None]
    landmark_min = landmark_boundaries.amin(2)[:, None]
    landmark_max = landmark_boundaries.amax(2)[:, None]
    gaps = rotate_270(torch.cat((target_min - landmark_max,
                                 target_max - landmark_min), -1).reshape(
                                     len(rows), 5, max_landmarks, 2, 2)).flatten(-2)
    target_size = (target_max - target_min).expand(-1, -1, max_landmarks, -1)
    landmark_size = (landmark_max - landmark_min).expand(-1, 5, -1, -1)
    geometry = torch.cat((center, edge, gaps, target_size, landmark_size), -1)
    valid = landmark_valid & instance_valid.any(-1, keepdim=True)
    geometry = geometry * valid[:, None, :, None]
    return geometry.half(), valid


def build_instance_split(rows, relation, predicted_class, components, max_landmarks=9):
    instance_centers, instance_boundaries, instance_valid, stats = bind_candidate_instances(
        rows, predicted_class, components)
    geometry, valid = pair_geometry(
        rows, instance_centers, instance_boundaries, instance_valid, max_landmarks)
    switched = matched_same_map_switch_indices(rows)
    switched_geometry, switched_valid = pair_geometry(
        rows, instance_centers, instance_boundaries, instance_valid,
        max_landmarks, switched)
    candidates = torch.tensor([row["candidates"] for row in rows])
    targets = torch.tensor([row["target"] for row in rows])
    return {
        "geometry": geometry, "valid": valid, "relation": relation,
        "cue": structured_relation_cues(rows, max_landmarks),
        "opposite_cue": opposite_direction_cues(
            structured_relation_cues(rows, max_landmarks)),
        "distance": torch.linalg.vector_norm(candidates - targets[:, None], dim=-1) * 410.0,
        "landmark_count": valid.sum(-1),
        "switched_geometry": switched_geometry, "switched_valid": switched_valid,
        "switched_relation": relation[switched],
        "switched_cue": structured_relation_cues(rows, max_landmarks)[switched],
        "switch_changed": switched.ne(torch.arange(len(rows))),
    }, stats


class CandidateInstanceBindingRanker(RelationGeometryRanker):
    def __init__(self, geometry_dim=18, hidden=128):
        super().__init__(hidden=hidden, cue_dim=len(CUE_NAMES))
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))


def train_one(train, validation, seed, device, epochs=15):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = CandidateInstanceBindingRanker(train["geometry"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    dataset = TensorDataset(
        train["relation"], train["geometry"], train["valid"], train["cue"],
        train["distance"], train["switched_relation"], train["switched_cue"],
        train["switched_geometry"], train["switched_valid"], train["switch_changed"])
    best_state, best_key, best_epoch = None, None, 0
    for epoch in range(epochs):
        model.train()
        for batch in DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True):
            (relation, geometry, valid, cue, distance, switched_relation,
             switched_cue, switched_geometry, switched_valid, switch_changed) = batch
            relation = relation.to(device); geometry = geometry.to(device)
            valid = valid.to(device); cue = cue.to(device); distance = distance.to(device)
            prediction = model(relation, geometry, valid, cue)
            loss = ranking_loss(prediction, distance, 0.2)[0]
            if switch_changed.any():
                selected = switch_changed.to(device)
                wrong_relation = model(switched_relation.to(device), geometry, valid,
                                       switched_cue.to(device))
                wrong_landmark = model(relation, switched_geometry.to(device),
                                       switched_valid.to(device), cue)
                correct_ce = listwise_loss_per_row(prediction, distance)
                relation_ce = listwise_loss_per_row(wrong_relation, distance)
                landmark_ce = listwise_loss_per_row(wrong_landmark, distance)
                loss += 0.2 * 0.5 * (
                    F.softplus(0.2 + correct_ce[selected] - relation_ce[selected])
                    + F.softplus(0.2 + correct_ce[selected] - landmark_ce[selected])).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        model.eval()
        with torch.inference_mode():
            value = metrics(validation["distance"],
                            scores(model, validation, device).argmax(-1))
        key = value["hit20"], -value["mean_ne"]
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        print(f"instance seed={seed} epoch={epoch + 1}/{epochs} "
              f"val_hit20={value['hit20']:.2f}", flush=True)
    model.load_state_dict(best_state)
    return model.eval(), best_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    relation_dir = ROOT / "relation_geometry_ranker_contrastive"
    output_dir = ROOT / "candidate_instance_binding_ranker"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {split: load_rows(candidate_dir / f"{split}.jsonl", 5) for split in SPLITS}
    relation = torch.load(relation_dir / "relation_features.pt",
                          map_location="cpu", weights_only=True)
    type_feature = torch.load(ROOT / "target_type_alignment/features.pt",
                              map_location="cpu", weights_only=True)
    probabilities = {split: [] for split in SPLITS}
    for seed in (0, 1, 2):
        probe = train_probe(type_feature["text"]["train_seen"],
                            type_feature["text"]["val_seen"], seed, 40, device)
        with torch.inference_mode():
            for split in SPLITS:
                probabilities[split].append(F.softmax(probe(
                    type_feature["text"][split]["features"].float().to(device)).cpu(), -1))
        del probe
    predicted_class = {split: torch.stack(values).mean(0).argmax(-1)
                       for split, values in probabilities.items()}
    components = components_by_map(json.load(open("data/cityrefer/objects.json")))
    data, binding_stats = {}, {}
    for split in SPLITS:
        data[split], binding_stats[split] = build_instance_split(
            rows[split], relation[split], predicted_class[split], components)
    runs = []
    for seed in (0, 1, 2):
        model, epoch = train_one(data["train_seen"], data["val_seen"], seed, device)
        prior = tune_prior(model, data["val_seen"], device)
        runs.append({
            "seed": seed, "best_epoch": epoch, "prior_weight": prior,
            "val_seen": evaluate(model, data["val_seen"], device, prior),
            "val_unseen": evaluate(model, data["val_unseen"], device, prior),
        })
        torch.save(model.state_dict(), output_dir / f"model_seed{seed}.pt")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    keys = ("learned_relation", "wrong_relation_same_map", "wrong_landmark_same_map",
            "no_relation", "no_structured_cue", "opposite_direction",
            "candidate_prior_top1", "oracle_top5")
    mean = {key: {metric: sum(run["val_unseen"][key][metric] for run in runs) / 3
                  for metric in ("mean_ne", "median_ne", "hit20", "hit50")}
            for key in keys}
    result = {
        "status": "ideal candidate semantic-instance binding diagnostic",
        "geometry_dim": int(data["train_seen"]["geometry"].shape[-1]),
        "binding_stats": binding_stats, "runs": runs, "val_unseen_mean": mean,
        "anti_leakage": {
            "instance_selection": "nearest component of language-predicted class",
            "target_object_id_or_coordinate_used_as_input": False,
            "training": "train_seen only", "checkpoint_selection": "val_seen only",
            "val_unseen_used_for_training_or_selection": False,
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
