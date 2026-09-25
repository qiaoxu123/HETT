#!/usr/bin/env python3
"""Rank candidates using a soft set of nearby semantic object instances.

Every candidate retains two nearby components from each semantic class.  Text
class probability and candidate-to-component distance provide an association
prior; relation satisfaction against named landmark masks then reweights the
set.  No target id or target coordinate enters an inference feature.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import train_probe
from scripts.train_candidate_instance_binding_ranker import components_by_map
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, listwise_loss_per_row, load_rows, matched_same_map_switch_indices,
    opposite_direction_cues, ranking_loss, rotate_270, structured_relation_cues,
)
from scripts.train_rgb_candidate_ranker import metrics


ROOT = Path("runs")
SPLITS = ("train_seen", "val_seen", "val_unseen")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "soft_instance_set_ranker")
    return parser.parse_args()


def landmark_tensors(rows, max_landmarks=9, reference_indices=None):
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


def build_component_set(rows, text_probability, components, per_class=2):
    slots = len(text_probability[0]) * per_class
    centers = torch.zeros(len(rows), 5, slots, 2)
    boundaries = torch.zeros(len(rows), 5, slots, 16, 2)
    valid = torch.zeros(len(rows), 5, slots, dtype=torch.bool)
    log_prior = torch.full((len(rows), 5, slots), -1e4)
    candidates = torch.tensor([row["candidates"] for row in rows], dtype=torch.float32)
    represented = []
    for row_index, row in enumerate(rows):
        filled = 0
        for class_index, available in enumerate(components[row["map_name"]]):
            if not available:
                continue
            available_centers = torch.stack([item[0] for item in available])
            distance = torch.cdist(candidates[row_index], available_centers)
            count = min(per_class, len(available))
            nearest_distance, nearest_index = distance.topk(count, largest=False, dim=-1)
            for local_rank in range(count):
                slot = class_index * per_class + local_rank
                chosen = nearest_index[:, local_rank]
                centers[row_index, :, slot] = available_centers[chosen]
                boundaries[row_index, :, slot] = torch.stack(
                    [available[int(value)][1] for value in chosen])
                valid[row_index, :, slot] = True
                class_log_probability = text_probability[row_index, class_index].clamp_min(
                    1e-6).log()
                log_prior[row_index, :, slot] = (
                    class_log_probability - nearest_distance[:, local_rank] * 410.0 / 20.0)
                filled += 1
        represented.append(filled)
    return centers, boundaries, valid, log_prior, candidates, {
        "mean_available_class_rank_slots": sum(represented) / len(represented),
        "set_size": slots,
    }


def relation_geometry(rows, candidates, component_centers, component_boundaries,
                      max_landmarks=9, reference_indices=None):
    landmark_centers, landmark_boundaries, landmark_valid = landmark_tensors(
        rows, max_landmarks, reference_indices)
    # candidate -> possible component, retained to avoid discontinuous snapping
    candidate_delta = rotate_270(candidates[:, :, None] - component_centers)
    candidate_distance = torch.linalg.vector_norm(candidate_delta, dim=-1).clamp_min(1e-6)
    candidate_feature = torch.cat((
        candidate_delta,
        torch.log1p(candidate_distance * 410.0).div(torch.log(torch.tensor(411.0))).unsqueeze(-1),
        (candidate_delta[..., 1] / candidate_distance).unsqueeze(-1),
        (candidate_delta[..., 0] / candidate_distance).unsqueeze(-1),
    ), -1)[:, :, :, None].expand(-1, -1, -1, max_landmarks, -1)

    center_delta = rotate_270(
        component_centers[:, :, :, None] - landmark_centers[:, None, None])
    center_distance = torch.linalg.vector_norm(center_delta, dim=-1).clamp_min(1e-6)
    center_feature = torch.cat((
        center_delta,
        torch.log1p(center_distance * 410.0).div(torch.log(torch.tensor(411.0))).unsqueeze(-1),
        (center_delta[..., 1] / center_distance).unsqueeze(-1),
        (center_delta[..., 0] / center_distance).unsqueeze(-1),
    ), -1)

    component_min = component_boundaries.amin(3)[:, :, :, None]
    component_max = component_boundaries.amax(3)[:, :, :, None]
    landmark_min = landmark_boundaries.amin(2)[:, None, None]
    landmark_max = landmark_boundaries.amax(2)[:, None, None]
    gaps = torch.cat((component_min - landmark_max,
                      component_max - landmark_min), -1)
    component_size = (component_max - component_min).expand(
        -1, -1, -1, max_landmarks, -1)
    landmark_size = (landmark_max - landmark_min).expand(
        -1, candidates.shape[1], component_centers.shape[2], -1, -1)
    geometry = torch.cat((candidate_feature, center_feature, gaps,
                          component_size, landmark_size), -1)
    geometry = geometry * landmark_valid[:, None, None, :, None]
    return geometry.half(), landmark_valid


def build_split(rows, relation, text_probability, components, max_landmarks=9):
    (component_centers, component_boundaries, component_valid, log_prior,
     candidates, stats) = build_component_set(rows, text_probability, components)
    geometry, landmark_valid = relation_geometry(
        rows, candidates, component_centers, component_boundaries, max_landmarks)
    switched = matched_same_map_switch_indices(rows)
    switched_geometry, switched_landmark_valid = relation_geometry(
        rows, candidates, component_centers, component_boundaries,
        max_landmarks, switched)
    cue = structured_relation_cues(rows, max_landmarks)
    targets = torch.tensor([row["target"] for row in rows])
    return {
        "geometry": geometry, "landmark_valid": landmark_valid,
        "component_valid": component_valid, "log_prior": log_prior,
        "relation": relation, "cue": cue, "opposite_cue": opposite_direction_cues(cue),
        "distance": torch.linalg.vector_norm(candidates - targets[:, None], dim=-1) * 410.0,
        "landmark_count": landmark_valid.sum(-1),
        "switched_geometry": switched_geometry,
        "switched_landmark_valid": switched_landmark_valid,
        "switched_relation": relation[switched], "switched_cue": cue[switched],
        "switch_changed": switched.ne(torch.arange(len(rows))),
    }, stats


class SoftInstanceSetRanker(nn.Module):
    def __init__(self, geometry_dim=18, hidden=64):
        super().__init__()
        self.relation = nn.Sequential(
            nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        self.cue = nn.Sequential(
            nn.Linear(len(CUE_NAMES), hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.pair = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        self.aggregate = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.association_scale = nn.Parameter(torch.tensor(1.0))
        self.relation_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, relation, geometry, landmark_valid, component_valid,
                log_prior, cue):
        language = self.relation(relation.float()) + self.cue(cue.float())
        language = language[:, None, None]
        spatial = self.geometry(geometry.float())
        language = language.expand_as(spatial)
        pair = self.pair(torch.cat((language, spatial, language * spatial,
                                    torch.abs(language - spatial)), -1)).squeeze(-1)
        valid = landmark_valid[:, None, None]
        count = valid.sum(-1).clamp_min(1).float()
        mean = (pair * valid).sum(-1) / count
        minimum = pair.masked_fill(~valid, float("inf")).amin(-1)
        maximum = pair.masked_fill(~valid, -float("inf")).amax(-1)
        component_score = self.aggregate(torch.stack((mean, minimum, maximum), -1)).squeeze(-1)
        association = log_prior.float().masked_fill(~component_valid, -1e4)
        association_score = torch.logsumexp(association, -1)
        association_score = ((association_score - association_score.mean(-1, keepdim=True))
                             / association_score.std(-1, keepdim=True).clamp_min(1e-3))
        component_weight = F.softmax(association, -1)
        relation_score = (component_weight * component_score).sum(-1)
        return (F.softplus(self.association_scale) * association_score
                + F.softplus(self.relation_scale) * relation_score)


@torch.inference_mode()
def model_scores(model, data, device, mode="normal"):
    relation, cue = data["relation"], data["cue"]
    geometry, landmark_valid = data["geometry"], data["landmark_valid"]
    if mode == "wrong_relation":
        relation, cue = data["switched_relation"], data["switched_cue"]
    elif mode == "wrong_landmark":
        geometry = data["switched_geometry"]
        landmark_valid = data["switched_landmark_valid"]
    elif mode == "no_relation":
        relation, cue = torch.zeros_like(relation), torch.zeros_like(cue)
    elif mode == "no_structured_cue":
        cue = torch.zeros_like(cue)
    elif mode == "opposite_direction":
        cue = data["opposite_cue"]
    return model(relation.to(device), geometry.to(device), landmark_valid.to(device),
                 data["component_valid"].to(device), data["log_prior"].to(device),
                 cue.to(device)).cpu()


@torch.inference_mode()
def tune_prior(model, data, device):
    learned = model_scores(model, data, device)
    prior = -torch.arange(5, dtype=learned.dtype)[None]
    best = None
    for weight in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        value = metrics(data["distance"], (learned + weight * prior).argmax(-1))
        key = value["hit20"], -value["mean_ne"]
        if best is None or key > best[0]: best = key, weight
    return best[1]


@torch.inference_mode()
def evaluate(model, data, device, prior_weight):
    score = model_scores(model, data, device)
    prior = -torch.arange(5, dtype=score.dtype)[None]
    result = {
        "learned_relation": metrics(data["distance"], score.argmax(-1)),
        "hybrid_relation_plus_candidate_prior": metrics(
            data["distance"], (score + prior_weight * prior).argmax(-1)),
        "wrong_relation_same_map": metrics(
            data["distance"], model_scores(model, data, device, "wrong_relation").argmax(-1)),
        "wrong_landmark_same_map": metrics(
            data["distance"], model_scores(model, data, device, "wrong_landmark").argmax(-1)),
        "no_relation": metrics(
            data["distance"], model_scores(model, data, device, "no_relation").argmax(-1)),
        "no_structured_cue": metrics(
            data["distance"], model_scores(model, data, device, "no_structured_cue").argmax(-1)),
        "opposite_direction": metrics(
            data["distance"], model_scores(model, data, device, "opposite_direction").argmax(-1)),
        "candidate_prior_top1": metrics(
            data["distance"], torch.zeros(len(data["distance"]), dtype=torch.long)),
        "oracle_top5": metrics(data["distance"], data["distance"].argmin(-1)),
    }
    return result


def train_one(train, validation, seed, device, epochs=15):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = SoftInstanceSetRanker(train["geometry"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
    dataset = TensorDataset(
        train["relation"], train["geometry"], train["landmark_valid"],
        train["component_valid"], train["log_prior"], train["cue"], train["distance"],
        train["switched_relation"], train["switched_cue"], train["switched_geometry"],
        train["switched_landmark_valid"], train["switch_changed"])
    best_state, best_key, best_epoch = None, None, 0
    for epoch in range(epochs):
        model.train()
        for batch in DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True):
            (relation, geometry, landmark_valid, component_valid, log_prior, cue, distance,
             switched_relation, switched_cue, switched_geometry,
             switched_landmark_valid, switch_changed) = batch
            args = (geometry.to(device), landmark_valid.to(device), component_valid.to(device),
                    log_prior.to(device))
            prediction = model(relation.to(device), *args, cue.to(device))
            distance = distance.to(device)
            loss = ranking_loss(prediction, distance, 0.2)[0]
            if switch_changed.any():
                selected = switch_changed.to(device)
                wrong_relation = model(switched_relation.to(device), *args,
                                       switched_cue.to(device))
                wrong_landmark = model(
                    relation.to(device), switched_geometry.to(device),
                    switched_landmark_valid.to(device), component_valid.to(device),
                    log_prior.to(device), cue.to(device))
                correct_ce = listwise_loss_per_row(prediction, distance)
                relation_ce = listwise_loss_per_row(wrong_relation, distance)
                landmark_ce = listwise_loss_per_row(wrong_landmark, distance)
                loss += 0.15 * 0.5 * (
                    F.softplus(0.2 + correct_ce[selected] - relation_ce[selected])
                    + F.softplus(0.2 + correct_ce[selected] - landmark_ce[selected])).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        model.eval()
        with torch.inference_mode():
            value = metrics(validation["distance"],
                            model_scores(model, validation, device).argmax(-1))
        key = value["hit20"], -value["mean_ne"]
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        print(f"soft-set seed={seed} epoch={epoch + 1}/{epochs} "
              f"val_hit20={value['hit20']:.2f}", flush=True)
    model.load_state_dict(best_state)
    return model.eval(), best_epoch


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {split: load_rows(candidate_dir / f"{split}.jsonl", 5) for split in SPLITS}
    relation = torch.load(ROOT / "relation_geometry_ranker_contrastive/relation_features.pt",
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
    text_probability = {split: torch.stack(value).mean(0)
                        for split, value in probabilities.items()}
    components = components_by_map(json.load(open("data/cityrefer/objects.json")))
    data, set_stats = {}, {}
    for split in SPLITS:
        data[split], set_stats[split] = build_split(
            rows[split], relation[split], text_probability[split], components)
    runs = []
    seeds = [int(value) for value in args.seeds.split(",") if value]
    for seed in seeds:
        model, epoch = train_one(
            data["train_seen"], data["val_seen"], seed, device, args.epochs)
        prior = tune_prior(model, data["val_seen"], device)
        runs.append({
            "seed": seed, "best_epoch": epoch, "prior_weight": prior,
            "val_seen": evaluate(model, data["val_seen"], device, prior),
            "val_unseen": evaluate(model, data["val_unseen"], device, prior),
        })
        torch.save(model.state_dict(), output_dir / f"model_seed{seed}.pt")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    keys = ("learned_relation", "hybrid_relation_plus_candidate_prior",
            "wrong_relation_same_map", "wrong_landmark_same_map", "no_relation",
            "no_structured_cue", "opposite_direction", "candidate_prior_top1", "oracle_top5")
    mean = {key: {metric: sum(run["val_unseen"][key][metric] for run in runs) / len(runs)
                  for metric in ("mean_ne", "median_ne", "hit20", "hit50")}
            for key in keys}
    result = {
        "status": "ideal soft semantic-instance set diagnostic",
        "set_statistics": set_stats, "runs": runs, "val_unseen_mean": mean,
        "anti_leakage": {
            "instance_source": "two nearest annotated components per semantic class",
            "association_prior": "text class probability and candidate distance only",
            "target_object_id_or_coordinate_used_as_input": False,
            "training": "train_seen only", "checkpoint_selection": "val_seen only",
            "val_unseen_used_for_training_or_selection": False,
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
