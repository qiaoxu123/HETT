#!/usr/bin/env python3
"""Test relation scoring after binding Top-5 points to semantic instances.

This diagnostic uses all annotated component centers of the predicted target
class, but never the target object id or target coordinate. It isolates whether
perfect instance extraction would improve the existing spatial relation model.
"""

import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import CLASSES, train_probe
from scripts.evaluate_semantic_component_fusion import (
    evaluate, normalized, semantic_component_score, tune,
)
from scripts.fuse_relation_type_scores import type_compatibility
from scripts.train_candidate_score_calibrator import direct_similarity
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, RelationGeometryRanker, build_split, load_rows,
    scores as relation_scores,
)
from scripts.validate_candidate_geometry import normalized_xy, polygon_centroid


ROOT = Path("runs")
SPLITS = ("val_unseen", "test_unseen")


def centers_by_map(objects):
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


def snap_rows(rows, predicted_class, centers):
    snapped = copy.deepcopy(rows)
    changed_distance = []
    unique_counts = []
    for index, row in enumerate(snapped):
        candidates = torch.tensor(row["candidates"])
        components = centers[row["map_name"]][int(predicted_class[index])]
        if not len(components):
            changed_distance.extend([0.0] * len(candidates))
            unique_counts.append(len(candidates))
            continue
        distance = torch.cdist(candidates, components)
        nearest = components[distance.argmin(-1)]
        changed_distance.extend(
            (torch.linalg.vector_norm(nearest - candidates, dim=-1) * 410.0).tolist())
        unique_counts.append(len({tuple(value) for value in nearest.tolist()}))
        row["candidates"] = nearest.tolist()
    return snapped, {
        "mean_snap_distance_m": sum(changed_distance) / len(changed_distance),
        "mean_unique_instances_per_top5": sum(unique_counts) / len(unique_counts),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    relation_dir = ROOT / "relation_geometry_ranker_contrastive"
    test_dir = ROOT / "test_unseen_fusion"
    rows = {
        "val_unseen": load_rows(candidate_dir / "val_unseen.jsonl", 5),
        "test_unseen": load_rows(test_dir / "test_unseen.jsonl", 5),
    }
    relation_feature = torch.load(
        relation_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    relation_embedding = {
        "val_unseen": relation_feature["val_unseen"],
        "test_unseen": torch.load(
            test_dir / "relation_features.pt", map_location="cpu", weights_only=True),
    }
    original_data = {split: build_split(rows[split], relation_embedding[split], 9)
                     for split in SPLITS}
    type_feature = torch.load(
        ROOT / "target_type_alignment/features.pt", map_location="cpu", weights_only=True)
    scale40 = torch.load(
        ROOT / "target_type_alignment/multiscale/scale_40.pt",
        map_location="cpu", weights_only=True)
    test40 = torch.load(
        test_dir / "siglip40_features.pt", map_location="cpu", weights_only=True)
    candidate40 = {
        "val_unseen": torch.load(
            ROOT / "relation_type_fusion/scale40_candidate_features.pt",
            map_location="cpu", weights_only=True)["val_unseen"],
        "test_unseen": test40["candidate"],
    }
    text = {"val_unseen": type_feature["text"]["val_unseen"]["features"],
            "test_unseen": test40["text"]}
    collected = {split: {"original_relation": [], "type": [], "text_probability": []}
                 for split in SPLITS}
    relation_models = []
    for seed in (0, 1, 2):
        text_model = train_probe(type_feature["text"]["train_seen"],
                                 type_feature["text"]["val_seen"], seed, 40, device)
        image_model = train_probe(scale40["target"]["train_seen"],
                                  scale40["target"]["val_seen"], seed, 40, device)
        relation_model = RelationGeometryRanker(cue_dim=len(CUE_NAMES)).to(device)
        relation_model.load_state_dict(torch.load(
            relation_dir / f"model_seed{seed}.pt", map_location="cpu", weights_only=True))
        relation_model.eval()
        for split in SPLITS:
            collected[split]["original_relation"].append(normalized(
                relation_scores(relation_model, original_data[split], device)))
            type_score, text_logits, _ = type_compatibility(
                text_model, image_model, text[split], candidate40[split], device)
            collected[split]["type"].append(normalized(type_score))
            collected[split]["text_probability"].append(F.softmax(text_logits, -1))
        relation_models.append(relation_model.cpu())
        del text_model, image_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    objects = json.load(open("data/cityrefer/objects.json"))
    centers = centers_by_map(objects)
    snapped_rows, snap_stats, snapped_data = {}, {}, {}
    text_probability = {}
    for split in SPLITS:
        text_probability[split] = torch.stack(
            collected[split]["text_probability"]).mean(0)
        predicted_class = text_probability[split].argmax(-1)
        snapped_rows[split], snap_stats[split] = snap_rows(
            rows[split], predicted_class, centers)
        snapped_data[split] = build_split(
            snapped_rows[split], relation_embedding[split], 9)

    snapped_relation = {split: [] for split in SPLITS}
    for model in relation_models:
        model = model.to(device).eval()
        for split in SPLITS:
            snapped_relation[split].append(normalized(
                relation_scores(model, snapped_data[split], device)))
        model.cpu()
    parts_original, parts_snapped, proxy = {}, {}, {}
    for split in SPLITS:
        prior = normalized(-torch.arange(5, dtype=torch.float32)[None].expand(
            len(rows[split]), 5))
        common = {
            "type": torch.stack(collected[split]["type"]).mean(0),
            "appearance40": direct_similarity(text[split], candidate40[split]),
            "prior": prior,
            "semantic": normalized(semantic_component_score(
                rows[split], text_probability[split], objects)),
        }
        parts_original[split] = {
            "relation": torch.stack(collected[split]["original_relation"]).mean(0),
            **common,
        }
        parts_snapped[split] = {
            "relation": torch.stack(snapped_relation[split]).mean(0),
            **common,
        }
        proxy[split] = {"distance": original_data[split]["distance"],
                        "landmark_count": original_data[split]["landmark_count"]}

    all_val = torch.ones(len(rows["val_unseen"]), dtype=torch.bool)
    all_test = torch.ones(len(rows["test_unseen"]), dtype=torch.bool)
    original_weights = tune(
        parts_original["val_unseen"], proxy["val_unseen"]["distance"],
        proxy["val_unseen"]["landmark_count"], all_val, True)
    snapped_weights = tune(
        parts_snapped["val_unseen"], proxy["val_unseen"]["distance"],
        proxy["val_unseen"]["landmark_count"], all_val, True)
    result = {
        "status": "ideal annotated-instance binding diagnostic",
        "snap_statistics": snap_stats,
        "val_unseen": {
            "component_fusion_original_relation": evaluate(
                parts_original["val_unseen"], original_weights,
                proxy["val_unseen"], all_val),
            "component_fusion_instance_snapped_relation": evaluate(
                parts_snapped["val_unseen"], snapped_weights,
                proxy["val_unseen"], all_val),
        },
        "test_unseen": {
            "component_fusion_original_relation": evaluate(
                parts_original["test_unseen"], original_weights,
                proxy["test_unseen"], all_test),
            "component_fusion_instance_snapped_relation": evaluate(
                parts_snapped["test_unseen"], snapped_weights,
                proxy["test_unseen"], all_test),
            "oracle_top5": evaluate(
                {**parts_original["test_unseen"],
                 "relation": -proxy["test_unseen"]["distance"]},
                {"one": (1, 0, 0, 0, 0), "multi": (1, 0, 0, 0, 0)},
                proxy["test_unseen"], all_test),
        },
        "weights": {
            "original": {key: list(value) for key, value in original_weights.items()},
            "snapped": {key: list(value) for key, value in snapped_weights.items()},
        },
        "anti_leakage": {
            "instance_source": "all annotated components of predicted text target class",
            "target_object_id_or_coordinate_used_for_snapping": False,
            "weight_selection": "val_unseen only",
            "test_target_used_for_training_or_selection": False,
        },
    }
    path = test_dir / "instance_snapped_relation_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
