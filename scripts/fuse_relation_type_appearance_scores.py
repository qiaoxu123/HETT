#!/usr/bin/env python3
"""Exploratory fusion of geometry, coarse type and direct RGB/text appearance."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import train_probe
from scripts.fuse_relation_type_scores import normalized, type_compatibility
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, RelationGeometryRanker, build_split, load_rows,
    scores as relation_scores,
)
from scripts.train_rgb_candidate_ranker import metrics


ROOT = Path("runs")


def direct_similarity(text, candidate):
    text = F.normalize(text.float(), dim=-1)
    candidate = F.normalize(candidate.float(), dim=-1)
    return normalized(torch.einsum("nd,nkd->nk", text, candidate))


def fused_score(parts, weights, landmark_count):
    result = torch.empty_like(parts["relation"])
    for name, mask in (("one", landmark_count.eq(1)),
                       ("multi", landmark_count.ne(1))):
        result[mask] = sum(weight * parts[key][mask]
                           for key, weight in zip(
                               ("relation", "type", "appearance", "prior"),
                               weights[name]))
    return result


def tune(parts, distance, landmark_count):
    grids = {
        "relation": (0.0, 0.25, 0.5, 1.0, 2.0),
        "type": (0.0, 0.5, 1.0, 2.0),
        "appearance": (0.0, 0.25, 0.5, 1.0, 2.0),
        "prior": (0.0, 0.1, 0.25, 0.5),
    }
    result = {}
    for name, mask in (("one", landmark_count.eq(1)),
                       ("multi", landmark_count.ne(1))):
        best = None
        for rw in grids["relation"]:
            for tw in grids["type"]:
                for aw in grids["appearance"]:
                    for pw in grids["prior"]:
                        weights = (rw, tw, aw, pw)
                        if not any(weights):
                            continue
                        score = sum(weight * parts[key][mask]
                                    for key, weight in zip(
                                        ("relation", "type", "appearance", "prior"),
                                        weights))
                        value = metrics(distance[mask], score.argmax(-1))
                        key = (value["hit20"], -value["mean_ne"], -sum(weights))
                        if best is None or key > best[0]:
                            best = key, weights, value
        result[name] = best[1]
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    relation_dir = ROOT / "relation_geometry_ranker_contrastive"
    test_dir = ROOT / "test_unseen_fusion"
    rows = {
        "val_seen": load_rows(candidate_dir / "val_seen.jsonl", 5),
        "test_unseen": load_rows(test_dir / "test_unseen.jsonl", 5),
    }
    relation_features = torch.load(
        relation_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    test_relation = torch.load(
        test_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    relation_data = {
        "val_seen": build_split(rows["val_seen"], relation_features["val_seen"], 9),
        "test_unseen": build_split(rows["test_unseen"], test_relation, 9),
    }
    type_features = torch.load(
        ROOT / "target_type_alignment/features.pt", map_location="cpu", weights_only=True)
    scale40 = torch.load(
        ROOT / "target_type_alignment/multiscale/scale_40.pt",
        map_location="cpu", weights_only=True)
    val_candidates = torch.load(
        ROOT / "relation_type_fusion/scale40_candidate_features.pt",
        map_location="cpu", weights_only=True)["val_seen"]
    test_features = torch.load(
        test_dir / "siglip40_features.pt", map_location="cpu", weights_only=True)
    frozen = json.load(open(ROOT / "relation_type_fusion/results.json"))
    seeds = [int(run["seed"]) for run in frozen["runs"]]
    prior = {split: normalized(-torch.arange(5, dtype=torch.float32)[None].expand(
        len(rows[split]), 5)) for split in rows}
    outputs = []
    for seed in seeds:
        text_model = train_probe(type_features["text"]["train_seen"],
                                 type_features["text"]["val_seen"], seed, 40, device)
        image_model = train_probe(scale40["target"]["train_seen"],
                                  scale40["target"]["val_seen"], seed, 40, device)
        relation_model = RelationGeometryRanker(cue_dim=len(CUE_NAMES)).to(device)
        relation_model.load_state_dict(torch.load(
            relation_dir / f"model_seed{seed}.pt", map_location="cpu", weights_only=True))
        relation_model.eval()
        parts = {}
        for split, candidate_features, text_embed in (
            ("val_seen", val_candidates,
             type_features["text"]["val_seen"]["features"]),
            ("test_unseen", test_features["candidate"], test_features["text"]),
        ):
            rel = normalized(relation_scores(relation_model, relation_data[split], device))
            typ, _, _ = type_compatibility(
                text_model, image_model, text_embed, candidate_features, device)
            parts[split] = {
                "relation": rel,
                "type": normalized(typ),
                "appearance": direct_similarity(text_embed, candidate_features),
                "prior": prior[split],
            }
        weights = tune(parts["val_seen"], relation_data["val_seen"]["distance"],
                       relation_data["val_seen"]["landmark_count"])
        test_score = fused_score(parts["test_unseen"], weights,
                                 relation_data["test_unseen"]["landmark_count"])
        outputs.append({
            "seed": seed,
            "selected_weights": {key: list(value) for key, value in weights.items()},
            "val_seen": metrics(
                relation_data["val_seen"]["distance"],
                fused_score(parts["val_seen"], weights,
                            relation_data["val_seen"]["landmark_count"]).argmax(-1)),
            "test_unseen": metrics(
                relation_data["test_unseen"]["distance"], test_score.argmax(-1)),
            "test_appearance_only": metrics(
                relation_data["test_unseen"]["distance"],
                parts["test_unseen"]["appearance"].argmax(-1)),
        })
        del text_model, image_model, relation_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "status": "exploratory_post_diagnostic_not_pristine_blind_test",
        "selection": "all four weights selected only on val_seen",
        "score_order": ["relation", "type", "appearance", "prior"],
        "runs": outputs,
        "mean": {
            key: {metric: sum(run[key][metric] for run in outputs) / len(outputs)
                  for metric in ("mean_ne", "median_ne", "hit20", "hit50")}
            for key in ("val_seen", "test_unseen", "test_appearance_only")
        },
    }
    output = test_dir / "appearance_fusion_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
