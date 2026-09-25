#!/usr/bin/env python3
"""Predict class-component proximity from frozen candidate RGB features.

This is a feasibility probe between coarse type classification and full dense
segmentation.  Supervision is distance to the nearest annotated component of
each class; target identity and target coordinates are never model inputs.
"""

import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_semantic_component_fusion import evaluate, normalized, tune
from scripts.train_candidate_score_calibrator import assemble_frozen_features, split_maps
from scripts.train_rgb_candidate_ranker import metrics
from scripts.validate_candidate_geometry import normalized_xy, polygon_centroid


ROOT = Path("runs")
CLASSES = ("Building", "Car", "Ground", "Parking")
MAX_DISTANCE_M = 120.0


class SemanticDistanceProbe(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.context = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 96), nn.GELU())
        self.detail = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 96), nn.GELU())
        self.output = nn.Sequential(
            nn.Linear(96 * 4, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 4))

    def forward(self, context, detail):
        context = self.context(context.float())
        detail = self.detail(detail.float())
        fused = torch.cat((context, detail, context * detail,
                           torch.abs(context - detail)), dim=-1)
        return self.output(fused)


def component_centers(objects):
    result = {}
    for map_name, values in objects.items():
        grouped = [[] for _ in CLASSES]
        for obj in values.values():
            if obj.get("object_type") not in CLASSES:
                continue
            center = normalized_xy(*polygon_centroid(obj["contour"]), map_name)
            grouped[CLASSES.index(obj["object_type"])].append(center.tolist())
        result[map_name] = [torch.tensor(item) if item else torch.empty(0, 2)
                            for item in grouped]
    return result


def component_distances(rows, centers):
    candidates = torch.tensor([row["candidates"] for row in rows])
    result = torch.full((len(rows), 5, len(CLASSES)), MAX_DISTANCE_M)
    for index, row in enumerate(rows):
        for class_index, values in enumerate(centers[row["map_name"]]):
            if len(values):
                result[index, :, class_index] = torch.cdist(
                    candidates[index], values).amin(-1).mul(410.0).clamp_max(
                        MAX_DISTANCE_M)
    return result


def encode_target(distance):
    return torch.log1p(distance) / math.log1p(MAX_DISTANCE_M)


def decode_prediction(logit):
    value = logit.sigmoid()
    return torch.expm1(value * math.log1p(MAX_DISTANCE_M)).clamp_max(MAX_DISTANCE_M)


def flatten_mask(row_mask):
    return row_mask[:, None].expand(-1, 5).reshape(-1)


def train_probe(feature40, feature20, distance, train_rows, validation_rows,
                seed, max_epochs=120):
    random.seed(seed)
    torch.manual_seed(seed)
    model = SemanticDistanceProbe(feature40.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-2)
    train_indices = torch.where(flatten_mask(train_rows))[0]
    loader = DataLoader(TensorDataset(train_indices), batch_size=512, shuffle=True)
    flat40 = feature40.reshape(-1, feature40.shape[-1]).float()
    flat20 = feature20.reshape(-1, feature20.shape[-1]).float()
    target = encode_target(distance).reshape(-1, len(CLASSES))
    validation = flatten_mask(validation_rows)
    best, best_loss, best_epoch = None, float("inf"), 0
    for epoch in range(max_epochs):
        model.train()
        for (indices,) in loader:
            prediction = model(flat40[indices], flat20[indices])
            loss = F.smooth_l1_loss(prediction.sigmoid(), target[indices], beta=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(flat40[validation], flat20[validation])
            loss = F.smooth_l1_loss(
                prediction.sigmoid(), target[validation], beta=0.1).item()
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch + 1
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    return model.eval(), best_epoch, best_loss


def train_fixed(feature40, feature20, distance, seed, epochs):
    all_rows = torch.ones(len(feature40), dtype=torch.bool)
    # No checkpoint selection here: epoch count is inherited from map-level
    # validation, then all val_seen rows are used.
    random.seed(seed)
    torch.manual_seed(seed)
    model = SemanticDistanceProbe(feature40.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-2)
    indices = torch.where(flatten_mask(all_rows))[0]
    loader = DataLoader(TensorDataset(indices), batch_size=512, shuffle=True)
    flat40 = feature40.reshape(-1, feature40.shape[-1]).float()
    flat20 = feature20.reshape(-1, feature20.shape[-1]).float()
    target = encode_target(distance).reshape(-1, len(CLASSES))
    for _ in range(epochs):
        model.train()
        for (batch,) in loader:
            prediction = model(flat40[batch], flat20[batch])
            loss = F.smooth_l1_loss(prediction.sigmoid(), target[batch], beta=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict(model, feature40, feature20):
    output = []
    flat40 = feature40.reshape(-1, feature40.shape[-1])
    flat20 = feature20.reshape(-1, feature20.shape[-1])
    for offset in range(0, len(flat40), 1024):
        output.append(decode_prediction(model(
            flat40[offset:offset + 1024], flat20[offset:offset + 1024])))
    return torch.cat(output).reshape(len(feature40), 5, len(CLASSES))


def semantic_score(predicted_distance, text_probability):
    return normalized(torch.logsumexp(
        text_probability.clamp_min(1e-6).log()[:, None]
        - predicted_distance / 20.0, dim=-1))


def make_parts(frozen, semantic):
    component = frozen["component"]
    return {
        "relation": component[..., 0],
        "type": component[..., 1],
        "appearance40": component[..., 2],
        "prior": component[..., 4],
        "semantic": semantic,
    }


def proxy_data(rows, frozen):
    return {"distance": frozen["distance"],
            "landmark_count": torch.tensor([row["landmark_count"] for row in rows])}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, frozen = assemble_frozen_features(device)
    candidate40 = {
        "val_seen": torch.load(
            ROOT / "relation_type_fusion/scale40_candidate_features.pt",
            map_location="cpu", weights_only=True)["val_seen"],
        "test_unseen": torch.load(
            ROOT / "test_unseen_fusion/siglip40_features.pt",
            map_location="cpu", weights_only=True)["candidate"],
    }
    candidate20 = torch.load(
        ROOT / "test_unseen_fusion/siglip20_candidate_features.pt",
        map_location="cpu", weights_only=True)
    objects = json.load(open("data/cityrefer/objects.json"))
    centers = component_centers(objects)
    class_distance = {split: component_distances(rows[split], centers) for split in rows}
    train_rows, validation_rows, heldout_rows, validation_maps, heldout_maps = split_maps(
        rows["val_seen"])
    text_probability = {split: frozen[split]["context"][:, -4:] for split in rows}
    val_proxy = proxy_data(rows["val_seen"], frozen["val_seen"])
    test_proxy = proxy_data(rows["test_unseen"], frozen["test_unseen"])
    holdout_runs, test_runs = [], []
    for seed in (0, 1, 2):
        model, epoch, validation_loss = train_probe(
            candidate40["val_seen"], candidate20["val_seen"],
            class_distance["val_seen"], train_rows, validation_rows, seed)
        predicted_val = predict(model, candidate40["val_seen"], candidate20["val_seen"])
        predicted_semantic = semantic_score(predicted_val, text_probability["val_seen"])
        parts = make_parts(frozen["val_seen"], predicted_semantic)
        base_weights = tune(parts, val_proxy["distance"], val_proxy["landmark_count"],
                            validation_rows, False)
        semantic_weights = tune(parts, val_proxy["distance"], val_proxy["landmark_count"],
                                validation_rows, True)
        held_prediction = predicted_val[heldout_rows]
        held_truth = class_distance["val_seen"][heldout_rows]
        holdout_runs.append({
            "seed": seed,
            "selected_epoch": epoch,
            "validation_semantic_loss": validation_loss,
            "heldout_distance_mae_m": (held_prediction - held_truth).abs().mean().item(),
            "base40": evaluate(parts, base_weights, val_proxy, heldout_rows),
            "predicted_semantic_fusion": evaluate(
                parts, semantic_weights, val_proxy, heldout_rows),
            "base_weights": {key: list(value) for key, value in base_weights.items()},
            "semantic_weights": {key: list(value) for key, value in semantic_weights.items()},
        })

        final = train_fixed(candidate40["val_seen"], candidate20["val_seen"],
                            class_distance["val_seen"], seed, epoch)
        predicted_train = predict(
            final, candidate40["val_seen"], candidate20["val_seen"])
        predicted_test = predict(
            final, candidate40["test_unseen"], candidate20["test_unseen"])
        train_parts = make_parts(
            frozen["val_seen"], semantic_score(predicted_train, text_probability["val_seen"]))
        test_parts = make_parts(
            frozen["test_unseen"], semantic_score(predicted_test, text_probability["test_unseen"]))
        final_base_weights = tune(
            train_parts, val_proxy["distance"], val_proxy["landmark_count"],
            torch.ones(len(rows["val_seen"]), dtype=torch.bool), False)
        final_semantic_weights = tune(
            train_parts, val_proxy["distance"], val_proxy["landmark_count"],
            torch.ones(len(rows["val_seen"]), dtype=torch.bool), True)
        all_test = torch.ones(len(rows["test_unseen"]), dtype=torch.bool)
        test_runs.append({
            "seed": seed,
            "selected_epoch": epoch,
            "test_semantic_distance_mae_m": (
                predicted_test - class_distance["test_unseen"]).abs().mean().item(),
            "base40": evaluate(test_parts, final_base_weights, test_proxy, all_test),
            "predicted_semantic_fusion": evaluate(
                test_parts, final_semantic_weights, test_proxy, all_test),
            "semantic_weights": {key: list(value)
                                 for key, value in final_semantic_weights.items()},
        })

    def average(runs, key):
        return {metric: sum(run[key][metric] for run in runs) / len(runs)
                for metric in ("mean_ne", "median_ne", "hit20", "hit50")}

    result = {
        "model": "frozen SigLIP 40m+20m features -> nearest class-component distance",
        "map_partition": {
            "train_rows": int(train_rows.sum()),
            "validation_rows": int(validation_rows.sum()),
            "heldout_rows": int(heldout_rows.sum()),
            "validation_maps": validation_maps,
            "heldout_maps": heldout_maps,
        },
        "map_holdout_mean": {
            "base40": average(holdout_runs, "base40"),
            "predicted_semantic_fusion": average(
                holdout_runs, "predicted_semantic_fusion"),
            "semantic_distance_mae_m": sum(
                run["heldout_distance_mae_m"] for run in holdout_runs) / len(holdout_runs),
        },
        "test_unseen_exploratory_mean": {
            "base40": average(test_runs, "base40"),
            "predicted_semantic_fusion": average(test_runs, "predicted_semantic_fusion"),
            "semantic_distance_mae_m": sum(
                run["test_semantic_distance_mae_m"] for run in test_runs) / len(test_runs),
        },
        "holdout_runs": holdout_runs,
        "test_runs": test_runs,
        "anti_leakage": {
            "model_supervision": "nearest component distances for every class",
            "target_id_or_target_coordinate_used_as_model_input": False,
            "test_targets_used_for_training_or_selection": False,
            "test_targets_used_for": "final exploratory navigation metric only",
        },
    }
    path = ROOT / "test_unseen_fusion/rgb_semantic_component_probe_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
