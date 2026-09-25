#!/usr/bin/env python3
"""Test whether an auxiliary relation branch complements the accepted fusion."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_semantic_component_fusion import evaluate, normalized, score, tune
from scripts.fuse_relation_type_scores import type_compatibility
from scripts.train_candidate_score_calibrator import direct_similarity
from scripts.train_explicit_relation_ranker import ExplicitRelationRanker
from scripts.train_landmark_mask_relation_ranker import (
    LandmarkMaskRelationRanker, build_mask_split,
)
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, RelationGeometryRanker, build_split, load_rows,
    scores as relation_scores,
)
from scripts.train_rgb_candidate_ranker import metrics


ROOT = Path("runs")
SPLITS = ("val_unseen", "test_unseen")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("explicit", "landmark_mask"),
                        default="explicit")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    test_dir = ROOT / "test_unseen_fusion"
    rows = {"val_unseen": load_rows(candidate_dir / "val_unseen.jsonl", 5),
            "test_unseen": load_rows(test_dir / "test_unseen.jsonl", 5)}
    relation_cache = torch.load(
        ROOT / "relation_geometry_ranker_contrastive/relation_features.pt",
        map_location="cpu", weights_only=True)
    relation = {"val_unseen": relation_cache["val_unseen"],
                "test_unseen": torch.load(
                    test_dir / "relation_features.pt", map_location="cpu", weights_only=True)}
    data = {split: build_split(rows[split], relation[split], 9) for split in SPLITS}
    auxiliary_data = (
        {split: build_mask_split(rows[split], relation[split], 9) for split in SPLITS}
        if args.branch == "landmark_mask" else data)
    type_feature = torch.load(
        ROOT / "target_type_alignment/features.pt", map_location="cpu", weights_only=True)
    scale40 = torch.load(
        ROOT / "target_type_alignment/multiscale/scale_40.pt",
        map_location="cpu", weights_only=True)
    test40 = torch.load(
        test_dir / "siglip40_features.pt", map_location="cpu", weights_only=True)
    candidate40 = {"val_unseen": torch.load(
        ROOT / "relation_type_fusion/scale40_candidate_features.pt",
        map_location="cpu", weights_only=True)["val_unseen"],
        "test_unseen": test40["candidate"]}
    text = {"val_unseen": type_feature["text"]["val_unseen"]["features"],
            "test_unseen": test40["text"]}
    collected = {split: {"relation": [], "explicit": [], "type": []}
                 for split in SPLITS}
    for seed in (0, 1, 2):
        relation_model = RelationGeometryRanker(cue_dim=len(CUE_NAMES)).to(device)
        relation_model.load_state_dict(torch.load(
            ROOT / f"relation_geometry_ranker_contrastive/model_seed{seed}.pt",
            map_location="cpu", weights_only=True))
        auxiliary_model = (
            LandmarkMaskRelationRanker(auxiliary_data["val_unseen"]["geometry"].shape[-1])
            if args.branch == "landmark_mask" else ExplicitRelationRanker()).to(device)
        auxiliary_dir = ("landmark_mask_relation_ranker"
                         if args.branch == "landmark_mask" else "explicit_relation_ranker")
        auxiliary_model.load_state_dict(torch.load(
            ROOT / f"{auxiliary_dir}/model_seed{seed}.pt",
            map_location="cpu", weights_only=True))
        text_model = train_probe(type_feature["text"]["train_seen"],
                                 type_feature["text"]["val_seen"], seed, 40, device)
        image_model = train_probe(scale40["target"]["train_seen"],
                                  scale40["target"]["val_seen"], seed, 40, device)
        for split in SPLITS:
            collected[split]["relation"].append(normalized(
                relation_scores(relation_model, data[split], device)))
            collected[split]["explicit"].append(normalized(
                relation_scores(auxiliary_model, auxiliary_data[split], device)))
            type_value, _, _ = type_compatibility(
                text_model, image_model, text[split], candidate40[split], device)
            collected[split]["type"].append(normalized(type_value))
        del relation_model, auxiliary_model, text_model, image_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    parts, proxy = {}, {}
    for split in SPLITS:
        prior = normalized(-torch.arange(5, dtype=torch.float32)[None].expand(
            len(rows[split]), 5))
        parts[split] = {
            "relation": torch.stack(collected[split]["relation"]).mean(0),
            "type": torch.stack(collected[split]["type"]).mean(0),
            "appearance40": direct_similarity(text[split], candidate40[split]),
            "prior": prior,
            "semantic": torch.zeros_like(prior),
        }
        proxy[split] = {"distance": data[split]["distance"],
                        "landmark_count": data[split]["landmark_count"]}
    all_val = torch.ones(len(rows["val_unseen"]), dtype=torch.bool)
    all_test = torch.ones(len(rows["test_unseen"]), dtype=torch.bool)
    base_weights = tune(parts["val_unseen"], proxy["val_unseen"]["distance"],
                        proxy["val_unseen"]["landmark_count"], all_val, False)
    base_score = {split: score(parts[split], base_weights,
                               proxy[split]["landmark_count"]) for split in SPLITS}
    explicit = {split: torch.stack(collected[split]["explicit"]).mean(0)
                for split in SPLITS}
    selected_alpha = {}
    for group, mask in (("one", proxy["val_unseen"]["landmark_count"].eq(1)),
                        ("multi", proxy["val_unseen"]["landmark_count"].ne(1))):
        best = None
        for alpha in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            rank = (base_score["val_unseen"][mask]
                    + alpha * explicit["val_unseen"][mask]).argmax(-1)
            value = metrics(proxy["val_unseen"]["distance"][mask], rank)
            key = value["hit20"], -value["mean_ne"], -alpha
            if best is None or key > best[0]:
                best = key, alpha
        selected_alpha[group] = best[1]

    def rank(split):
        combined = base_score[split].clone()
        count = proxy[split]["landmark_count"]
        for group, mask in (("one", count.eq(1)), ("multi", count.ne(1))):
            combined[mask] += selected_alpha[group] * explicit[split][mask]
        return combined.argmax(-1)

    result = {
        "auxiliary_branch": args.branch,
        "alpha_selected_on_val_unseen": selected_alpha,
        "base_weights_selected_on_val_unseen": {
            key: list(value) for key, value in base_weights.items()},
        "val_unseen": {
            "base": metrics(proxy["val_unseen"]["distance"],
                            base_score["val_unseen"].argmax(-1)),
            "plus_auxiliary": metrics(proxy["val_unseen"]["distance"], rank("val_unseen")),
        },
        "test_unseen": {
            "base": metrics(proxy["test_unseen"]["distance"],
                            base_score["test_unseen"].argmax(-1)),
            "plus_auxiliary": metrics(proxy["test_unseen"]["distance"], rank("test_unseen")),
        },
        "anti_leakage": {
            "all_weights": "val_unseen only",
            "test_used_for_training_or_selection": False,
        },
    }
    path = test_dir / f"{args.branch}_relation_fusion_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
