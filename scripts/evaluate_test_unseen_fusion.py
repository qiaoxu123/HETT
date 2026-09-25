#!/usr/bin/env python3
"""Fresh test_unseen validation for the frozen relation/type fusion pipeline."""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_multiscale_type_matching import encode_candidate_crops
from scripts.audit_target_type_alignment import first_type_match, train_probe
from scripts.fuse_relation_type_scores import (
    by_landmark_count,
    fused_rank,
    normalized,
    type_compatibility,
)
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES,
    RelationGeometryRanker,
    build_split as build_relation_split,
    encode_landmark_relations,
    load_rows,
    scores as relation_scores,
)
from scripts.train_rgb_candidate_ranker import encode_texts, metrics
from scripts.validate_candidate_geometry import (
    GeometryRanker,
    build_split,
    encode_language,
    make_tensors,
    predict_candidates,
    seed_everything,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--existing-candidate-dir", type=Path,
        default=Path("runs/candidate_rgb_data/geometry_top5_oof_grouped_enriched"))
    parser.add_argument(
        "--relation-dir", type=Path,
        default=Path("runs/relation_geometry_ranker_contrastive"))
    parser.add_argument(
        "--fusion-results", type=Path,
        default=Path("runs/relation_type_fusion/results.json"))
    parser.add_argument(
        "--type-cache", type=Path,
        default=Path("runs/target_type_alignment/features.pt"))
    parser.add_argument(
        "--scale40-cache", type=Path,
        default=Path("runs/target_type_alignment/multiscale/scale_40.pt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/test_unseen_fusion"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument(
        "--hf-cache", type=Path,
        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--feature-batch-size", type=int, default=24)
    return parser.parse_args()


def write_enriched_candidates(path, rows, candidates):
    with open(path, "w") as stream:
        for row, values in zip(rows, candidates):
            stream.write(json.dumps({
                "map_name": row["map_name"],
                "description": row["description"],
                "target": row["target"].tolist(),
                "landmark_count": len(row["names"]),
                "candidates": values.tolist(),
                "landmark_names": row["names"],
                "relation_queries": row["relation_queries"],
                "landmark_centers": [value.tolist() for value in row["centers"]],
                "landmark_boundaries": [value.tolist() for value in row["boundaries"]],
            }) + "\n")


def generate_candidates(args, device):
    output = args.output_dir / "test_unseen.jsonl"
    check_path = args.output_dir / "candidate_reproduction.json"
    if output.exists() and check_path.exists():
        return load_rows(output, 5), json.load(open(check_path))

    objects = json.load(open(args.data_root / "cityrefer" / "objects.json"))
    processed = json.load(open(args.data_root / "cityrefer" / "processed_descriptions.json"))
    rows = {}
    for split in ("train_seen", "val_seen", "val_unseen", "test_unseen"):
        trajectories = json.load(open(
            args.data_root / "processed_citynav" / f"citynav_{split}.json"))
        rows[split] = build_split(trajectories, objects, processed, 9)
        print(f"candidate rows {split}: {len(rows[split])}", flush=True)
    text, names, name_ids, relations, relation_ids = encode_language(rows, device)
    data = {split: make_tensors(values, text, name_ids, relation_ids, 20, 9)
            for split, values in rows.items()}
    seed_everything(0)
    model = GeometryRanker(names, relations, 20, mode="span_geometry_boundary_soft")
    model = train_model(model, data["train_seen"], device, 1, 96,
                        "fresh-test-final-train", 0.0)
    predicted_validation = predict_candidates(model, data["val_unseen"], device, 96, 5)
    predicted_test = predict_candidates(model, data["test_unseen"], device, 96, 5)

    existing = load_rows(args.existing_candidate_dir / "val_unseen.jsonl", 5)
    existing_candidates = torch.tensor([row["candidates"] for row in existing])
    if [row["description"] for row in existing] != [
            row["description"] for row in rows["val_unseen"]]:
        raise ValueError("existing validation candidates have a different row order")
    check = {
        "rows": len(existing),
        "allclose_atol_1e-6": bool(torch.allclose(
            predicted_validation, existing_candidates, atol=1e-6, rtol=0)),
        "top1_exact_percent": 100.0 * predicted_validation[:, 0].eq(
            existing_candidates[:, 0]).all(-1).float().mean().item(),
        "max_absolute_coordinate_difference": (
            predicted_validation - existing_candidates).abs().max().item(),
    }
    if check["top1_exact_percent"] < 99.0:
        raise ValueError(f"candidate generator reproduction failed: {check}")
    write_enriched_candidates(output, rows["test_unseen"], predicted_test)
    check_path.write_text(json.dumps(check, indent=2) + "\n")
    del model
    torch.cuda.empty_cache()
    return load_rows(output, 5), check


def load_or_encode_test_features(args, rows, device):
    cache_path = args.output_dir / "siglip40_features.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached["candidate"].shape[:2] == (len(rows), 5):
            return cached
        raise ValueError("test feature cache shape mismatch")
    processor = AutoProcessor.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    print(f"encoding test instruction text: {len(rows)}", flush=True)
    text = encode_texts(
        encoder, processor, [row["description"] for row in rows], device,
        args.feature_batch_size).half()
    print(f"encoding test 40 m candidates: {len(rows) * 5}", flush=True)
    candidate = encode_candidate_crops(
        rows, encoder, processor, args.data_root, device, 40.0,
        args.feature_batch_size)
    cached = {"text": text, "candidate": candidate}
    torch.save(cached, cache_path)
    del encoder
    torch.cuda.empty_cache()
    return cached


def evaluate_relation_only(args, rows, data, device, seeds, reproduction):
    distance = data["distance"]
    result = {
        "config": {
            "seeds": seeds,
            "candidate_model": "fresh deterministic reproduction of accepted train_seen model",
        },
        "counts": {"test_unseen": len(rows)},
        "candidate_reproduction": reproduction,
        "anti_leakage": {
            "test_target_coordinates": "evaluation distance only",
            "test_used_for_model_or_checkpoint_selection": False,
        },
        "runs": [],
    }
    for seed in seeds:
        model = RelationGeometryRanker(cue_dim=len(CUE_NAMES)).to(device)
        model.load_state_dict(torch.load(
            args.relation_dir / f"model_seed{seed}.pt", map_location="cpu",
            weights_only=True))
        model.eval()
        normal = relation_scores(model, data, device).argmax(-1)
        run = {
            "seed": seed,
            "test_unseen": {
                "relation": metrics(distance, normal),
                "wrong_relation": metrics(
                    distance, relation_scores(model, data, device, "wrong_relation").argmax(-1)),
                "wrong_landmark": metrics(
                    distance, relation_scores(model, data, device, "wrong_landmark").argmax(-1)),
                "opposite_direction": metrics(
                    distance, relation_scores(model, data, device, "opposite_direction").argmax(-1)),
                "no_relation": metrics(
                    distance, relation_scores(model, data, device, "no_relation").argmax(-1)),
                "candidate_prior": metrics(
                    distance, torch.zeros(len(distance), dtype=torch.long)),
                "oracle_top5": metrics(distance, distance.argmin(-1)),
                "by_landmark_count": by_landmark_count(
                    distance, normal, data["landmark_count"]),
            },
        }
        result["runs"].append(run)
        del model
        torch.cuda.empty_cache()
    keys = ("relation", "wrong_relation", "wrong_landmark", "opposite_direction",
            "no_relation", "candidate_prior", "oracle_top5")
    result["test_unseen_mean"] = {
        key: {metric: sum(run["test_unseen"][key][metric] for run in result["runs"])
              / len(result["runs"])
              for metric in ("mean_ne", "median_ne", "hit20", "hit50")}
        for key in keys
    }
    (args.output_dir / "relation_only_results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, reproduction = generate_candidates(args, device)
    relation_cache = args.output_dir / "relation_features.pt"
    if relation_cache.exists():
        relation = torch.load(relation_cache, map_location="cpu", weights_only=True)
    else:
        print(f"encoding test landmark relations: {len(rows)}", flush=True)
        relation = encode_landmark_relations(rows, device)
        torch.save(relation, relation_cache)
    relation_data = build_relation_split(rows, relation, 9)
    selected = json.load(open(args.fusion_results))
    seeds = sorted(int(run["seed"]) for run in selected["runs"])
    relation_result = evaluate_relation_only(
        args, rows, relation_data, device, seeds, reproduction)
    required_maps = sorted({row["map_name"] for row in rows})
    missing_rgb = [name for name in required_maps
                   if not (args.data_root / "rgbd" / f"{name}.png").exists()
                   or not (args.data_root / "rgbd" / f"{name}.tif").exists()]
    if missing_rgb:
        counts = Counter(row["map_name"] for row in rows)
        blocked = {
            "status": "fusion_blocked_missing_rgb",
            "missing_maps": missing_rgb,
            "missing_samples": {name: counts[name] for name in missing_rgb},
            "available_relation_only_result": str(
                args.output_dir / "relation_only_results.json"),
            "relation_only_test_unseen_mean": relation_result["test_unseen_mean"],
        }
        (args.output_dir / "fusion_blocked.json").write_text(
            json.dumps(blocked, indent=2) + "\n")
        print(json.dumps(blocked, indent=2))
        return
    stale_blocker = args.output_dir / "fusion_blocked.json"
    if stale_blocker.exists():
        stale_blocker.unlink()
    test_features = load_or_encode_test_features(args, rows, device)

    type_features = torch.load(args.type_cache, map_location="cpu", weights_only=True)
    scale40 = torch.load(args.scale40_cache, map_location="cpu", weights_only=True)
    weights_by_seed = {
        int(run["seed"]): {name: tuple(values)
                           for name, values in run["selected_weights"].items()}
        for run in selected["runs"]}
    seeds = sorted(weights_by_seed)
    prior = normalized(-torch.arange(5, dtype=torch.float32)[None].expand(len(rows), 5))
    distance = relation_data["distance"]
    result = {
        "config": {
            "seeds": seeds,
            "fusion_weights": "frozen values selected previously on val_seen",
            "candidate_model": "fresh deterministic reproduction of the accepted train_seen model",
        },
        "counts": {"test_unseen": len(rows)},
        "candidate_reproduction": reproduction,
        "anti_leakage": {
            "test_target_coordinates": "evaluation distance only",
            "test_used_for_model_or_weight_selection": False,
            "models_and_fusion_weights_changed_after_test": False,
        },
        "runs": [],
    }
    for seed in seeds:
        text_model = train_probe(
            type_features["text"]["train_seen"], type_features["text"]["val_seen"],
            seed, 40, device)
        image_model = train_probe(
            scale40["target"]["train_seen"], scale40["target"]["val_seen"],
            seed, 40, device)
        relation_model = RelationGeometryRanker(cue_dim=len(CUE_NAMES)).to(device)
        relation_model.load_state_dict(torch.load(
            args.relation_dir / f"model_seed{seed}.pt", map_location="cpu",
            weights_only=True))
        relation_model.eval()

        relation_score = normalized(relation_scores(
            relation_model, relation_data, device))
        type_score, text_logits, image_logits = type_compatibility(
            text_model, image_model, test_features["text"],
            test_features["candidate"], device)
        type_score = normalized(type_score)
        weights = weights_by_seed[seed]
        rank = fused_rank(relation_score, type_score, prior, weights,
                          relation_data["landmark_count"])
        wrong_relation = normalized(relation_scores(
            relation_model, relation_data, device, "wrong_relation"))
        wrong_landmark = normalized(relation_scores(
            relation_model, relation_data, device, "wrong_landmark"))
        permuted_type = normalized(torch.logsumexp(
            F.log_softmax(text_logits.roll(1, -1), -1)[:, None]
            + F.log_softmax(image_logits, -1), -1))
        hard_type, _ = first_type_match(text_logits.argmax(-1), image_logits.argmax(-1))
        result["runs"].append({
            "seed": seed,
            "frozen_weights": {name: list(value) for name, value in weights.items()},
            "test_unseen": {
                "fusion": metrics(distance, rank),
                "wrong_relation": metrics(distance, fused_rank(
                    wrong_relation, type_score, prior, weights,
                    relation_data["landmark_count"])),
                "wrong_landmark": metrics(distance, fused_rank(
                    wrong_landmark, type_score, prior, weights,
                    relation_data["landmark_count"])),
                "permuted_type_classes": metrics(distance, fused_rank(
                    relation_score, permuted_type, prior, weights,
                    relation_data["landmark_count"])),
                "relation_only": metrics(distance, relation_score.argmax(-1)),
                "hard_type_match": metrics(distance, hard_type),
                "candidate_prior": metrics(
                    distance, torch.zeros(len(distance), dtype=torch.long)),
                "oracle_top5": metrics(distance, distance.argmin(-1)),
                "by_landmark_count": by_landmark_count(
                    distance, rank, relation_data["landmark_count"]),
            },
        })
        del text_model, image_model, relation_model
        torch.cuda.empty_cache()

    keys = ("fusion", "wrong_relation", "wrong_landmark", "permuted_type_classes",
            "relation_only", "hard_type_match", "candidate_prior", "oracle_top5")
    result["test_unseen_mean"] = {
        key: {metric: sum(run["test_unseen"][key][metric] for run in result["runs"])
              / len(result["runs"])
              for metric in ("mean_ne", "median_ne", "hit20", "hit50")}
        for key in keys
    }
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["test_unseen_mean"], indent=2))


if __name__ == "__main__":
    main()
