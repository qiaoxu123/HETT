#!/usr/bin/env python3
"""Evaluate 40 m context plus 20 m candidate-detail appearance fusion.

All fusion weights are selected without test_unseen labels.  A deterministic
map holdout inside val_seen checks whether the extra scale transfers across
maps before weights are refit on all val_seen rows for exploratory test use.
"""

import hashlib
import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_multiscale_type_matching import encode_candidate_crops
from scripts.audit_target_type_alignment import train_probe
from scripts.fuse_relation_type_scores import normalized, type_compatibility
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, RelationGeometryRanker, build_split, load_rows,
    scores as relation_scores,
)
from scripts.train_rgb_candidate_ranker import metrics


ROOT = Path("runs")
DATA = Path("data")
OUTPUT = ROOT / "test_unseen_fusion"
MODEL = "google/siglip2-so400m-patch16-512"
HF_CACHE = Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache")
PARTS = ("relation", "type", "appearance40", "appearance20", "prior")


def direct_similarity(text, candidate):
    text = F.normalize(text.float(), dim=-1)
    candidate = F.normalize(candidate.float(), dim=-1)
    return normalized(torch.einsum("nd,nkd->nk", text, candidate))


def load_or_encode_20m(rows, device):
    path = OUTPUT / "siglip20_candidate_features.pt"
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if all(cached[name].shape[:2] == (len(value), 5)
               for name, value in rows.items()):
            return cached
        raise ValueError("20 m candidate cache shape mismatch")
    processor = AutoProcessor.from_pretrained(
        MODEL, cache_dir=str(HF_CACHE), local_files_only=True)
    encoder = AutoModel.from_pretrained(
        MODEL, cache_dir=str(HF_CACHE), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    cached = {}
    for split in ("val_seen", "test_unseen"):
        print(f"encoding {split} 20 m candidates: {len(rows[split]) * 5}", flush=True)
        cached[split] = encode_candidate_crops(
            rows[split], encoder, processor, DATA, device, 20.0, 24)
    torch.save(cached, path)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cached


def score(parts, weights, landmark_count):
    result = torch.empty_like(parts["relation"])
    for group, selected in (("one", landmark_count.eq(1)),
                            ("multi", landmark_count.ne(1))):
        result[selected] = sum(weight * parts[name][selected]
                               for name, weight in zip(PARTS, weights[group]))
    return result


def tune(parts, distance, landmark_count, selection, allow_20m):
    grids = (
        (0.0, 0.25, 0.5, 1.0, 2.0),
        (0.0, 0.5, 1.0, 2.0),
        (0.0, 0.25, 0.5, 1.0, 2.0),
        (0.0, 0.25, 0.5, 1.0, 2.0) if allow_20m else (0.0,),
        (0.0, 0.1, 0.25, 0.5),
    )
    result = {}
    for group, group_mask in (("one", landmark_count.eq(1)),
                              ("multi", landmark_count.ne(1))):
        mask = selection & group_mask
        best = None
        for weights in itertools.product(*grids):
            if not any(weights):
                continue
            values = sum(weight * parts[name][mask]
                         for name, weight in zip(PARTS, weights))
            value = metrics(distance[mask], values.argmax(-1))
            key = (value["hit20"], -value["mean_ne"], -sum(weights))
            if best is None or key > best[0]:
                best = key, weights
        result[group] = best[1]
    return result


def map_holdout(rows):
    maps = sorted({row["map_name"] for row in rows})
    held_maps = {name for name in maps
                 if int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 4 == 0}
    # Guard against an unexpectedly tiny deterministic partition.
    if len(held_maps) < 4:
        held_maps = set(maps[::4])
    held = torch.tensor([row["map_name"] in held_maps for row in rows])
    return ~held, held, sorted(held_maps)


def selected_metrics(parts, weights, data, mask):
    rank = score(parts, weights, data["landmark_count"]).argmax(-1)
    return metrics(data["distance"][mask], rank[mask])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    relation_dir = ROOT / "relation_geometry_ranker_contrastive"
    rows = {
        "val_seen": load_rows(candidate_dir / "val_seen.jsonl", 5),
        "test_unseen": load_rows(OUTPUT / "test_unseen.jsonl", 5),
    }
    features20 = load_or_encode_20m(rows, device)
    relation_features = torch.load(
        relation_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    relation_test = torch.load(
        OUTPUT / "relation_features.pt", map_location="cpu", weights_only=True)
    data = {
        "val_seen": build_split(rows["val_seen"], relation_features["val_seen"], 9),
        "test_unseen": build_split(rows["test_unseen"], relation_test, 9),
    }
    type_features = torch.load(
        ROOT / "target_type_alignment/features.pt", map_location="cpu", weights_only=True)
    scale40 = torch.load(
        ROOT / "target_type_alignment/multiscale/scale_40.pt",
        map_location="cpu", weights_only=True)
    candidate40 = {
        "val_seen": torch.load(
            ROOT / "relation_type_fusion/scale40_candidate_features.pt",
            map_location="cpu", weights_only=True)["val_seen"],
        "test_unseen": torch.load(
            OUTPUT / "siglip40_features.pt", map_location="cpu", weights_only=True)["candidate"],
    }
    text = {
        "val_seen": type_features["text"]["val_seen"]["features"],
        "test_unseen": torch.load(
            OUTPUT / "siglip40_features.pt", map_location="cpu", weights_only=True)["text"],
    }
    prior = {split: normalized(-torch.arange(5, dtype=torch.float32)[None].expand(
        len(rows[split]), 5)) for split in rows}
    develop, heldout, held_maps = map_holdout(rows["val_seen"])
    seeds = (0, 1, 2)
    runs = []
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
        for split in rows:
            relation = normalized(relation_scores(relation_model, data[split], device))
            type_score, _, _ = type_compatibility(
                text_model, image_model, text[split], candidate40[split], device)
            parts[split] = {
                "relation": relation,
                "type": normalized(type_score),
                "appearance40": direct_similarity(text[split], candidate40[split]),
                "appearance20": direct_similarity(text[split], features20[split]),
                "prior": prior[split],
            }

        base_holdout_weights = tune(
            parts["val_seen"], data["val_seen"]["distance"],
            data["val_seen"]["landmark_count"], develop, False)
        dual_holdout_weights = tune(
            parts["val_seen"], data["val_seen"]["distance"],
            data["val_seen"]["landmark_count"], develop, True)
        base_all_weights = tune(
            parts["val_seen"], data["val_seen"]["distance"],
            data["val_seen"]["landmark_count"], torch.ones(len(develop), dtype=torch.bool), False)
        dual_all_weights = tune(
            parts["val_seen"], data["val_seen"]["distance"],
            data["val_seen"]["landmark_count"], torch.ones(len(develop), dtype=torch.bool), True)
        runs.append({
            "seed": seed,
            "map_holdout": {
                "base40": selected_metrics(parts["val_seen"], base_holdout_weights,
                                           data["val_seen"], heldout),
                "dual40_20": selected_metrics(parts["val_seen"], dual_holdout_weights,
                                              data["val_seen"], heldout),
                "base_weights": {k: list(v) for k, v in base_holdout_weights.items()},
                "dual_weights": {k: list(v) for k, v in dual_holdout_weights.items()},
            },
            "all_val_seen_selected": {
                "base_weights": {k: list(v) for k, v in base_all_weights.items()},
                "dual_weights": {k: list(v) for k, v in dual_all_weights.items()},
                "test_base40": selected_metrics(parts["test_unseen"], base_all_weights,
                                                data["test_unseen"],
                                                torch.ones(len(rows["test_unseen"]), dtype=torch.bool)),
                "test_dual40_20": selected_metrics(parts["test_unseen"], dual_all_weights,
                                                   data["test_unseen"],
                                                   torch.ones(len(rows["test_unseen"]), dtype=torch.bool)),
                "test_appearance20_only": metrics(
                    data["test_unseen"]["distance"],
                    parts["test_unseen"]["appearance20"].argmax(-1)),
            },
        })
        del text_model, image_model, relation_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def average(path):
        values = []
        for run in runs:
            item = run
            for key in path:
                item = item[key]
            values.append(item)
        return {metric: sum(item[metric] for item in values) / len(values)
                for metric in ("mean_ne", "median_ne", "hit20", "hit50")}

    result = {
        "status": "test exploratory; scale decision checked on val_seen map holdout",
        "parts": PARTS,
        "map_holdout": {
            "heldout_maps": held_maps,
            "develop_rows": int(develop.sum()),
            "heldout_rows": int(heldout.sum()),
            "base40_mean": average(("map_holdout", "base40")),
            "dual40_20_mean": average(("map_holdout", "dual40_20")),
        },
        "test_unseen_mean": {
            "base40": average(("all_val_seen_selected", "test_base40")),
            "dual40_20": average(("all_val_seen_selected", "test_dual40_20")),
            "appearance20_only": average(("all_val_seen_selected", "test_appearance20_only")),
            "oracle_top5": metrics(
                data["test_unseen"]["distance"], data["test_unseen"]["distance"].argmin(-1)),
        },
        "runs": runs,
        "anti_leakage": {
            "test_targets_used_for_weight_or_scale_selection": False,
            "test_targets_used_for": "final exploratory metrics only",
        },
    }
    path = OUTPUT / "dualscale_appearance_fusion_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
