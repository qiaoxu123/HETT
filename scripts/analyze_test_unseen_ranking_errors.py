#!/usr/bin/env python3
"""Explain the gap between frozen fusion ranking and its Top-5 oracle.

The script never uses target coordinates to form a score.  Targets are read only
after inference, to label successful candidates and group ranking failures.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import CLASSES, train_probe
from scripts.evaluate_test_unseen_fusion import load_or_encode_test_features
from scripts.fuse_relation_type_scores import normalized, type_compatibility
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES,
    RelationGeometryRanker,
    build_split,
    load_rows,
    scores as relation_scores,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/test_unseen_fusion"))
    parser.add_argument("--relation-dir", type=Path,
                        default=Path("runs/relation_geometry_ranker_contrastive"))
    parser.add_argument("--fusion-results", type=Path,
                        default=Path("runs/relation_type_fusion/results.json"))
    parser.add_argument("--type-cache", type=Path,
                        default=Path("runs/target_type_alignment/features.pt"))
    parser.add_argument("--scale40-cache", type=Path,
                        default=Path("runs/target_type_alignment/multiscale/scale_40.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/test_unseen_fusion/ranking_error_analysis.json"))
    return parser.parse_args()


def pct(numerator, denominator):
    return 100.0 * numerator / max(denominator, 1)


def mask_summary(mask, success, error, eligible):
    count = int(mask.sum())
    return {
        "count": count,
        "share_of_all_percent": pct(count, len(mask)),
        "hit20_percent": pct(int((mask & success).sum()), count),
        "ranking_error_rate_given_oracle_percent": pct(int((mask & error).sum()),
                                                        int((mask & eligible).sum())),
    }


def source_target_types(data_root, rows):
    objects = json.load(open(data_root / "cityrefer" / "objects.json"))
    trajectories = json.load(open(
        data_root / "processed_citynav" / "citynav_test_unseen.json"))
    labels, descriptions = [], []
    for item in trajectories:
        if item.get("dist_marker_to_target", 0.0) > 30.0:
            continue
        map_name = f'{item["area"]}_block_{item["block"]}'
        labels.append(CLASSES.index(objects[map_name][str(item["object_ids"][0])]["object_type"]))
        descriptions.append(item["descriptions"][0])
    if descriptions != [row["description"] for row in rows]:
        raise ValueError("test source rows and candidate rows are not aligned")
    return torch.tensor(labels)


def final_score(relation, type_score, prior, weights, landmark_count):
    result = torch.empty_like(relation)
    for group, selected in (("one", landmark_count.eq(1)),
                            ("multi", landmark_count.ne(1))):
        rw, tw, pw = weights[group]
        result[selected] = (rw * relation[selected] + tw * type_score[selected]
                            + pw * prior[selected])
    return result


def safe_mean(values):
    return sum(values) / max(len(values), 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.run_dir / "test_unseen.jsonl", 5)
    relation_features = torch.load(
        args.run_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    data = build_split(rows, relation_features, 9)
    test_features = torch.load(
        args.run_dir / "siglip40_features.pt", map_location="cpu", weights_only=True)
    type_features = torch.load(args.type_cache, map_location="cpu", weights_only=True)
    scale40 = torch.load(args.scale40_cache, map_location="cpu", weights_only=True)
    selected = json.load(open(args.fusion_results))
    weights_by_seed = {
        int(run["seed"]): {name: tuple(values)
                           for name, values in run["selected_weights"].items()}
        for run in selected["runs"]
    }

    count = len(rows)
    distance = data["distance"]
    positive = distance.le(20.0)
    eligible = positive.any(-1)
    nearest = distance.argmin(-1)
    prior = normalized(-torch.arange(5, dtype=torch.float32)[None].expand(count, 5))
    target_type = source_target_types(args.data_root, rows)

    seed_records = []
    fusion_scores, relation_scores_all, type_scores_all = [], [], []
    text_predictions, image_predictions = [], []
    for seed, weights in sorted(weights_by_seed.items()):
        text_model = train_probe(type_features["text"]["train_seen"],
                                 type_features["text"]["val_seen"], seed, 40, device)
        image_model = train_probe(scale40["target"]["train_seen"],
                                  scale40["target"]["val_seen"], seed, 40, device)
        relation_model = RelationGeometryRanker(cue_dim=len(CUE_NAMES)).to(device)
        relation_model.load_state_dict(torch.load(
            args.relation_dir / f"model_seed{seed}.pt", map_location="cpu",
            weights_only=True))
        relation_model.eval()
        relation = normalized(relation_scores(relation_model, data, device))
        type_score, text_logits, image_logits = type_compatibility(
            text_model, image_model, test_features["text"],
            test_features["candidate"], device)
        type_score = normalized(type_score)
        fusion = final_score(relation, type_score, prior, weights,
                             data["landmark_count"])
        rank = fusion.argmax(-1)
        error = eligible & ~positive.gather(1, rank[:, None]).squeeze(1)
        seed_records.append({
            "seed": seed,
            "hit20_percent": pct(int(positive.gather(1, rank[:, None]).sum()), count),
            "ranking_errors": int(error.sum()),
            "ranking_error_rate_given_oracle_percent": pct(int(error.sum()), int(eligible.sum())),
        })
        fusion_scores.append(fusion)
        relation_scores_all.append(relation)
        type_scores_all.append(type_score)
        text_predictions.append(text_logits.argmax(-1))
        image_predictions.append(image_logits.argmax(-1))
        del text_model, image_model, relation_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Averaging independently normalized frozen scores gives one deterministic
    # consensus diagnostic. It is not used to report the accepted benchmark.
    fusion = torch.stack(fusion_scores).mean(0)
    relation = torch.stack(relation_scores_all).mean(0)
    type_score = torch.stack(type_scores_all).mean(0)
    text_prediction = torch.mode(torch.stack(text_predictions), dim=0).values
    image_prediction = torch.mode(torch.stack(image_predictions), dim=0).values
    rank = fusion.argmax(-1)
    selected_positive = positive.gather(1, rank[:, None]).squeeze(1)
    error = eligible & ~selected_positive
    impossible = ~eligible
    success = selected_positive

    # Does each branch put any true-positive candidate above every negative?
    minus_inf = torch.full_like(fusion, -torch.inf)
    plus_inf = torch.full_like(fusion, torch.inf)
    best_positive_fusion = torch.where(positive, fusion, minus_inf).max(-1).values
    best_negative_fusion = torch.where(~positive, fusion, minus_inf).max(-1).values
    best_positive_relation = torch.where(positive, relation, minus_inf).max(-1).values
    best_negative_relation = torch.where(~positive, relation, minus_inf).max(-1).values
    best_positive_type = torch.where(positive, type_score, minus_inf).max(-1).values
    best_negative_type = torch.where(~positive, type_score, minus_inf).max(-1).values
    relation_separable = best_positive_relation.gt(best_negative_relation)
    type_separable = best_positive_type.gt(best_negative_type)

    type_match = image_prediction.eq(text_prediction[:, None])
    positive_has_type_match = (type_match & positive).any(-1)
    negative_has_type_match = (type_match & ~positive).any(-1)

    candidate_xy = torch.tensor([row["candidates"] for row in rows])
    selected_xy = candidate_xy[torch.arange(count), rank]
    nearest_xy = candidate_xy[torch.arange(count), nearest]
    selected_to_nearest_m = torch.linalg.vector_norm(selected_xy - nearest_xy, dim=-1) * 410.0
    image = F.normalize(test_features["candidate"].float(), dim=-1)
    text = F.normalize(test_features["text"].float(), dim=-1)
    direct_rgb_text = torch.einsum("nd,nkd->nk", text, image)
    direct_rank = direct_rgb_text.argmax(-1)
    direct_success = positive.gather(1, direct_rank[:, None]).squeeze(1)
    best_positive_direct = torch.where(positive, direct_rgb_text, minus_inf).max(-1).values
    best_negative_direct = torch.where(~positive, direct_rgb_text, minus_inf).max(-1).values
    direct_separable = best_positive_direct.gt(best_negative_direct)
    image_similarity = torch.einsum("nkd,njd->nkj", image, image)
    off_diagonal = ~torch.eye(5, dtype=torch.bool)[None]
    mean_pair_similarity = image_similarity[off_diagonal.expand_as(image_similarity)].reshape(
        count, -1).mean(-1)

    cue_patterns = {
        "direction": r"\b(left|right|above|below|under|over|north|south|east|west|top|bottom)\b",
        "near": r"\b(near|nearby|next|beside|adjacent|close|alongside|by)\b",
        "between": r"\bbetween\b",
        "across_opposite": r"\b(across|opposite)\b",
        "front_behind": r"\b(front|behind|back)\b",
        "ordinal": r"\b(\d+(st|nd|rd|th)|first|second|third|fourth|fifth|last)\b",
        "road": r"\b(road|street|avenue|lane|way|drive|intersection|corner)\b",
        "appearance": r"\b(red|blue|green|gray|grey|black|white|brown|roof|large|small|long)\b",
    }
    group_stats = {}
    for name, pattern in cue_patterns.items():
        mask = torch.tensor([bool(re.search(pattern, row["description"].lower()))
                             for row in rows])
        group_stats[name] = mask_summary(mask, success, error, eligible)
    for index, name in enumerate(CLASSES):
        group_stats[f"target_{name.lower()}"] = mask_summary(
            target_type.eq(index), success, error, eligible)
    for value, name in ((1, "one_landmark"), (2, "two_landmarks")):
        group_stats[name] = mask_summary(
            data["landmark_count"].eq(value), success, error, eligible)
    group_stats["three_plus_landmarks"] = mask_summary(
        data["landmark_count"].ge(3), success, error, eligible)

    # Small anonymized examples: enough to inspect the pattern without cherry-picking.
    hardest = torch.where(error)[0][
        (best_negative_fusion[error] - best_positive_fusion[error]).topk(
            min(12, int(error.sum()))).indices]
    examples = []
    for index in hardest.tolist():
        examples.append({
            "row": index,
            "map": rows[index]["map_name"],
            "description": rows[index]["description"],
            "landmarks": rows[index]["landmark_names"],
            "target_type": CLASSES[target_type[index]],
            "candidate_distance_m": [round(v, 2) for v in distance[index].tolist()],
            "consensus_rank": int(rank[index]),
            "fusion_scores": [round(v, 3) for v in fusion[index].tolist()],
            "predicted_candidate_types": [CLASSES[v] for v in image_prediction[index].tolist()],
        })

    error_count = int(error.sum())
    result = {
        "anti_leakage": {
            "target_coordinates": "used only after frozen inference to label errors",
            "test_used_for_training_or_weight_selection": False,
            "consensus": "diagnostic mean of the three frozen seed scores",
        },
        "counts": {
            "all": count,
            "oracle_hit20": int(eligible.sum()),
            "no_candidate_within_20m": int(impossible.sum()),
            "consensus_success": int(success.sum()),
            "consensus_ranking_error": error_count,
        },
        "rates_percent": {
            "oracle_hit20": pct(int(eligible.sum()), count),
            "candidate_recall_failure": pct(int(impossible.sum()), count),
            "consensus_hit20": pct(int(success.sum()), count),
            "ranking_error_of_all": pct(error_count, count),
            "ranking_error_given_oracle": pct(error_count, int(eligible.sum())),
        },
        "per_seed": seed_records,
        "ranking_error_diagnostics": {
            "relation_can_separate_positive_from_all_negatives_percent": pct(
                int((error & relation_separable).sum()), error_count),
            "type_can_separate_positive_from_all_negatives_percent": pct(
                int((error & type_separable).sum()), error_count),
            "either_branch_can_separate_percent": pct(
                int((error & (relation_separable | type_separable)).sum()), error_count),
            "neither_branch_can_separate_percent": pct(
                int((error & ~(relation_separable | type_separable)).sum()), error_count),
            "positive_candidate_has_predicted_type_match_percent": pct(
                int((error & positive_has_type_match).sum()), error_count),
            "both_positive_and_negative_have_type_match_percent": pct(
                int((error & positive_has_type_match & negative_has_type_match).sum()),
                error_count),
            "only_negative_has_type_match_percent": pct(
                int((error & ~positive_has_type_match & negative_has_type_match).sum()),
                error_count),
            "selected_to_nearest_candidate_m_mean": safe_mean(
                selected_to_nearest_m[error].tolist()),
            "selected_to_nearest_candidate_m_median": selected_to_nearest_m[error].median().item(),
            "candidate_rgb_feature_pair_cosine_mean": safe_mean(
                mean_pair_similarity[error].tolist()),
            "multiple_positive_candidates_percent": pct(
                int((error & positive.sum(-1).gt(1)).sum()), error_count),
            "true_text_type_majority_probe_accuracy_percent": pct(
                int(text_prediction.eq(target_type).sum()), count),
            "raw_siglip_text_rgb_hit20_percent": pct(int(direct_success.sum()), count),
            "raw_siglip_can_separate_ranking_errors_percent": pct(
                int((error & direct_separable).sum()), error_count),
            "any_of_relation_type_raw_siglip_can_separate_percent": pct(
                int((error & (relation_separable | type_separable | direct_separable)).sum()),
                error_count),
        },
        "groups": group_stats,
        "hardest_examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
