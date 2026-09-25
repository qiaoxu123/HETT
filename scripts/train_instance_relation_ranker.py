#!/usr/bin/env python3
"""Retrain relation geometry on candidates snapped to semantic instances."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_instance_snapped_relation import centers_by_map, snap_rows
from scripts.evaluate_semantic_component_fusion import (
    evaluate as fusion_metrics, normalized, semantic_component_score, tune,
)
from scripts.fuse_relation_type_scores import type_compatibility
from scripts.train_candidate_score_calibrator import direct_similarity
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, build_split, load_rows, scores as relation_scores, train_one,
)
from scripts.train_rgb_candidate_ranker import metrics


ROOT = Path("runs")
TRAIN_SPLITS = ("train_seen", "val_seen", "val_unseen")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    relation_dir = ROOT / "relation_geometry_ranker_contrastive"
    output_dir = ROOT / "instance_relation_ranker"
    test_dir = ROOT / "test_unseen_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {split: load_rows(candidate_dir / f"{split}.jsonl", 5)
            for split in TRAIN_SPLITS}
    rows["test_unseen"] = load_rows(test_dir / "test_unseen.jsonl", 5)
    relation_cache = torch.load(
        relation_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    relation_embedding = {split: relation_cache[split] for split in TRAIN_SPLITS}
    relation_embedding["test_unseen"] = torch.load(
        test_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    original = {split: build_split(rows[split], relation_embedding[split], 9)
                for split in rows}

    type_feature = torch.load(
        ROOT / "target_type_alignment/features.pt", map_location="cpu", weights_only=True)
    scale40 = torch.load(
        ROOT / "target_type_alignment/multiscale/scale_40.pt",
        map_location="cpu", weights_only=True)
    test40 = torch.load(
        test_dir / "siglip40_features.pt", map_location="cpu", weights_only=True)
    text = {split: type_feature["text"][split]["features"] for split in TRAIN_SPLITS}
    text["test_unseen"] = test40["text"]
    text_probabilities = {split: [] for split in rows}
    text_models, image_models = [], []
    for seed in (0, 1, 2):
        text_model = train_probe(type_feature["text"]["train_seen"],
                                 type_feature["text"]["val_seen"], seed, 40, device)
        image_model = train_probe(scale40["target"]["train_seen"],
                                  scale40["target"]["val_seen"], seed, 40, device)
        with torch.inference_mode():
            for split in rows:
                text_probabilities[split].append(F.softmax(
                    text_model(text[split].float().to(device)).cpu(), -1))
        text_models.append(text_model.cpu())
        image_models.append(image_model.cpu())
    text_probability = {split: torch.stack(value).mean(0)
                        for split, value in text_probabilities.items()}

    objects = json.load(open("data/cityrefer/objects.json"))
    centers = centers_by_map(objects)
    snapped_rows, snap_stats, snapped = {}, {}, {}
    for split in rows:
        snapped_rows[split], snap_stats[split] = snap_rows(
            rows[split], text_probability[split].argmax(-1), centers)
        snapped[split] = build_split(snapped_rows[split], relation_embedding[split], 9)
        # Labels remain the distances of the original Top-5 points. Snapped
        # coordinates are inference features only, never replacement targets.
        snapped[split]["distance"] = original[split]["distance"]

    models, relation_result = [], []
    for seed in (0, 1, 2):
        model, best_epoch = train_one(
            snapped["train_seen"], snapped["val_seen"], seed, 15, 256, 0.2,
            device, True, 0.2, 0.2)
        torch.save(model.state_dict(), output_dir / f"model_seed{seed}.pt")
        with torch.inference_mode():
            val_score = relation_scores(model, snapped["val_unseen"], device)
            test_score = relation_scores(model, snapped["test_unseen"], device)
        relation_result.append({
            "seed": seed, "best_epoch": best_epoch,
            "val_unseen": metrics(original["val_unseen"]["distance"],
                                  val_score.argmax(-1)),
            "test_unseen": metrics(original["test_unseen"]["distance"],
                                   test_score.argmax(-1)),
        })
        models.append(model.cpu())

    candidate40 = {
        "val_unseen": torch.load(
            ROOT / "relation_type_fusion/scale40_candidate_features.pt",
            map_location="cpu", weights_only=True)["val_unseen"],
        "test_unseen": test40["candidate"],
    }
    parts, proxy = {}, {}
    for split in ("val_unseen", "test_unseen"):
        relation_values, type_values = [], []
        for seed, model in enumerate(models):
            model = model.to(device).eval()
            relation_values.append(normalized(relation_scores(model, snapped[split], device)))
            model.cpu()
            text_model = text_models[seed].to(device).eval()
            image_model = image_models[seed].to(device).eval()
            type_score, _, _ = type_compatibility(
                text_model, image_model, text[split], candidate40[split], device)
            type_values.append(normalized(type_score))
            text_model.cpu(); image_model.cpu()
        prior = normalized(-torch.arange(5, dtype=torch.float32)[None].expand(
            len(rows[split]), 5))
        parts[split] = {
            "relation": torch.stack(relation_values).mean(0),
            "type": torch.stack(type_values).mean(0),
            "appearance40": direct_similarity(text[split], candidate40[split]),
            "prior": prior,
            "semantic": normalized(semantic_component_score(
                rows[split], text_probability[split], objects)),
        }
        proxy[split] = {"distance": original[split]["distance"],
                        "landmark_count": original[split]["landmark_count"]}
    all_val = torch.ones(len(rows["val_unseen"]), dtype=torch.bool)
    all_test = torch.ones(len(rows["test_unseen"]), dtype=torch.bool)
    weights = tune(parts["val_unseen"], proxy["val_unseen"]["distance"],
                   proxy["val_unseen"]["landmark_count"], all_val, True)
    result = {
        "status": "ideal annotated-instance relation retraining diagnostic",
        "snap_statistics": snap_stats,
        "relation_runs": relation_result,
        "relation_mean": {
            split: {key: sum(run[split][key] for run in relation_result) / 3
                    for key in ("mean_ne", "median_ne", "hit20", "hit50")}
            for split in ("val_unseen", "test_unseen")
        },
        "fusion": {
            "val_unseen": fusion_metrics(parts["val_unseen"], weights,
                                         proxy["val_unseen"], all_val),
            "test_unseen": fusion_metrics(parts["test_unseen"], weights,
                                          proxy["test_unseen"], all_test),
            "weights": {key: list(value) for key, value in weights.items()},
        },
        "anti_leakage": {
            "instance_source": "all annotated components of predicted text class",
            "target_coordinates": "training ranking labels and final metrics only",
            "test_used_for_training_checkpoint_or_weight_selection": False,
            "fusion_weight_selection": "val_unseen only",
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
