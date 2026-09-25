#!/usr/bin/env python3
"""Measure the value of ideal class segmentation for Top-5 ranking.

The semantic score uses every annotated object component of the predicted
target class. It never uses the target object id or target coordinate.
Consequently this is an input upper bound for an RGB segmentation model, not a
deployable RGB-only result.
"""

import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_candidate_score_calibrator import assemble_frozen_features, split_maps
from scripts.train_rgb_candidate_ranker import metrics
from scripts.validate_candidate_geometry import normalized_xy, polygon_centroid


ROOT = Path("runs")
PARTS = ("relation", "type", "appearance40", "prior", "semantic")


def semantic_component_score(rows, text_probability, objects, temperature_m=20.0):
    cache = {}
    for map_name, values in objects.items():
        by_class = [[] for _ in range(4)]
        names = ("Building", "Car", "Ground", "Parking")
        for obj in values.values():
            if obj.get("object_type") not in names:
                continue
            center = normalized_xy(*polygon_centroid(obj["contour"]), map_name)
            by_class[names.index(obj["object_type"])].append(center.tolist())
        cache[map_name] = [torch.tensor(value) if value else torch.empty(0, 2)
                           for value in by_class]
    candidates = torch.tensor([row["candidates"] for row in rows])
    class_distance = torch.full((len(rows), 5, 4), 410.0)
    for index, row in enumerate(rows):
        for class_index, centers in enumerate(cache[row["map_name"]]):
            if len(centers):
                class_distance[index, :, class_index] = torch.cdist(
                    candidates[index], centers).amin(-1) * 410.0
    log_probability = text_probability.clamp_min(1e-6).log()[:, None]
    return torch.logsumexp(log_probability - class_distance / temperature_m, dim=-1)


def normalized(value):
    value = value - value.mean(-1, keepdim=True)
    return value / value.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


def score(parts, weights, landmark_count):
    result = torch.empty_like(parts["relation"])
    for group, selected in (("one", landmark_count.eq(1)),
                            ("multi", landmark_count.ne(1))):
        result[selected] = sum(weight * parts[name][selected]
                               for name, weight in zip(PARTS, weights[group]))
    return result


def tune(parts, distance, landmark_count, selected, allow_semantic):
    grids = (
        (0.0, 0.25, 0.5, 1.0, 2.0),
        (0.0, 0.5, 1.0, 2.0),
        (0.0, 0.25, 0.5, 1.0, 2.0),
        (0.0, 0.1, 0.25, 0.5),
        (0.0, 0.25, 0.5, 1.0, 2.0) if allow_semantic else (0.0,),
    )
    result = {}
    for group, group_mask in (("one", landmark_count.eq(1)),
                              ("multi", landmark_count.ne(1))):
        mask = selected & group_mask
        best = None
        for weights in itertools.product(*grids):
            if not any(weights):
                continue
            candidate_score = sum(weight * parts[name][mask]
                                  for name, weight in zip(PARTS, weights))
            value = metrics(distance[mask], candidate_score.argmax(-1))
            key = (value["hit20"], -value["mean_ne"], -sum(weights))
            if best is None or key > best[0]:
                best = key, weights
        result[group] = best[1]
    return result


def evaluate(parts, weights, data, mask):
    rank = score(parts, weights, data["landmark_count"]).argmax(-1)
    return metrics(data["distance"][mask], rank[mask])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, frozen = assemble_frozen_features(device)
    objects = json.load(open("data/cityrefer/objects.json"))
    parts = {}
    for split in rows:
        component = frozen[split]["component"]
        text_probability = frozen[split]["context"][:, -4:]
        semantic = normalized(semantic_component_score(
            rows[split], text_probability, objects))
        parts[split] = {
            "relation": component[..., 0],
            "type": component[..., 1],
            "appearance40": component[..., 2],
            "prior": component[..., 4],
            "semantic": semantic,
        }
    train_mask, validation_mask, heldout_mask, validation_maps, heldout_maps = split_maps(
        rows["val_seen"])
    development = train_mask | validation_mask
    all_validation = torch.ones(len(rows["val_seen"]), dtype=torch.bool)
    all_test = torch.ones(len(rows["test_unseen"]), dtype=torch.bool)
    base_holdout_weights = tune(
        parts["val_seen"], frozen["val_seen"]["distance"],
        torch.tensor([row["landmark_count"] for row in rows["val_seen"]]),
        development, False)
    semantic_holdout_weights = tune(
        parts["val_seen"], frozen["val_seen"]["distance"],
        torch.tensor([row["landmark_count"] for row in rows["val_seen"]]),
        development, True)
    base_all_weights = tune(
        parts["val_seen"], frozen["val_seen"]["distance"],
        torch.tensor([row["landmark_count"] for row in rows["val_seen"]]),
        all_validation, False)
    semantic_all_weights = tune(
        parts["val_seen"], frozen["val_seen"]["distance"],
        torch.tensor([row["landmark_count"] for row in rows["val_seen"]]),
        all_validation, True)
    # build_split already provides the trusted resolved landmark count, while
    # JSON rows store the same count. Use the latter here to keep this script
    # independent of private tensors returned by the feature assembler.
    count = {split: torch.tensor([row["landmark_count"] for row in rows[split]])
             for split in rows}
    proxy = {split: {"distance": frozen[split]["distance"],
                     "landmark_count": count[split]} for split in rows}
    semantic_rank = parts["test_unseen"]["semantic"].argmax(-1)
    result = {
        "status": "ideal semantic-component diagnostic, not RGB-only",
        "semantic_input": "all object class component centroids plus predicted text class; no target id",
        "map_holdout": {
            "validation_maps": validation_maps,
            "heldout_maps": heldout_maps,
            "base40": evaluate(parts["val_seen"], base_holdout_weights,
                               proxy["val_seen"], heldout_mask),
            "with_semantic_components": evaluate(
                parts["val_seen"], semantic_holdout_weights,
                proxy["val_seen"], heldout_mask),
            "base_weights": {key: list(value) for key, value in base_holdout_weights.items()},
            "semantic_weights": {key: list(value)
                                 for key, value in semantic_holdout_weights.items()},
        },
        "test_unseen_exploratory": {
            "base40": evaluate(parts["test_unseen"], base_all_weights,
                               proxy["test_unseen"], all_test),
            "with_semantic_components": evaluate(
                parts["test_unseen"], semantic_all_weights,
                proxy["test_unseen"], all_test),
            "semantic_only": metrics(frozen["test_unseen"]["distance"], semantic_rank),
            "base_weights": {key: list(value) for key, value in base_all_weights.items()},
            "semantic_weights": {key: list(value)
                                 for key, value in semantic_all_weights.items()},
        },
        "anti_leakage": {
            "target_id_or_coordinate_used_to_form_semantic_score": False,
            "test_targets_used_for_weight_selection": False,
            "test_targets_used_for": "final exploratory metrics only",
        },
    }
    path = ROOT / "test_unseen_fusion/semantic_component_fusion_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
