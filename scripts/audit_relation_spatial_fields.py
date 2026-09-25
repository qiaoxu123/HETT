#!/usr/bin/env python3
"""Step-1 audit of target-free directional and near landmark fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.relation_spatial_field import (
    FLIPPED_DIRECTION,
    directional_field,
    extract_relation,
    grid_points,
    mask_landmark_span,
    near_field,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "../45-active-altitude/runs/candidate_rgb_data/"
            "geometry_top5_oof_grouped_enriched/val_seen.jsonl"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/relation_spatial_field_step1/results.json")
    )
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--near-scale-m", type=float, default=30.0)
    return parser.parse_args()


def load_rows(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bound_relations(row):
    result = []
    for index, (query, name) in enumerate(
        zip(row["relation_queries"], row["landmark_names"])
    ):
        masked, found = mask_landmark_span(row["description"], query, name)
        if not found:
            continue
        parsed = extract_relation(masked)
        if parsed["directions"] or parsed["near"]:
            result.append({"index": index, "masked": masked, **parsed})
    return result


def percentile(score, grid_scores):
    return float(100.0 * np.mean(np.asarray(grid_scores) <= score))


def hit20(candidate, target):
    return float(np.linalg.norm(candidate - target) * 410.0 <= 20.0)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
    }


def evaluate_directional(row, relation, grid):
    index = relation["index"]
    cue = relation["directions"][0]
    boundary = np.asarray(row["landmark_boundaries"][index], dtype=np.float32)
    target = np.asarray(row["target"], dtype=np.float32)
    candidates = np.asarray(row["candidates"], dtype=np.float32)
    target_score = directional_field(target[None], boundary, cue)[0]
    flipped_cue = FLIPPED_DIRECTION[cue]
    flipped_target_score = directional_field(target[None], boundary, flipped_cue)[0]
    scores = directional_field(candidates, boundary, cue)
    flipped_scores = directional_field(candidates, boundary, flipped_cue)
    grid_scores = directional_field(grid, boundary, cue)
    return {
        "cue": cue,
        "target_percentile": percentile(target_score, grid_scores),
        "target_correct_gt_flipped": float(target_score > flipped_target_score),
        "target_correct_minus_flipped": float(target_score - flipped_target_score),
        "field_hit20": hit20(candidates[int(scores.argmax())], target),
        "flipped_field_hit20": hit20(candidates[int(flipped_scores.argmax())], target),
        "original_top1_hit20": hit20(candidates[0], target),
    }


def evaluate_near(row, relation, grid, scale_m):
    index = relation["index"]
    boundary = np.asarray(row["landmark_boundaries"][index], dtype=np.float32)
    target = np.asarray(row["target"], dtype=np.float32)
    candidates = np.asarray(row["candidates"], dtype=np.float32)
    target_score = near_field(target[None], boundary, scale_m)[0]
    scores = near_field(candidates, boundary, scale_m)
    grid_scores = near_field(grid, boundary, scale_m)
    return {
        "target_percentile": percentile(target_score, grid_scores),
        "field_hit20": hit20(candidates[int(scores.argmax())], target),
        "original_top1_hit20": hit20(candidates[0], target),
    }


def aggregate_directional(rows):
    if not rows:
        return {"count": 0}
    by_cue = {}
    for cue in FLIPPED_DIRECTION:
        cue_rows = [row for row in rows if row["cue"] == cue]
        by_cue[cue] = {
            "count": len(cue_rows),
            "correct_gt_flipped_percent": 100.0 * np.mean(
                [row["target_correct_gt_flipped"] for row in cue_rows]
            ) if cue_rows else None,
        }
    return {
        "count": len(rows),
        "target_field_percentile": summarize(
            [row["target_percentile"] for row in rows]
        ),
        "target_in_top_quartile_percent": 100.0 * np.mean(
            [row["target_percentile"] >= 75.0 for row in rows]
        ),
        "correct_gt_flipped_percent": 100.0 * np.mean(
            [row["target_correct_gt_flipped"] for row in rows]
        ),
        "mean_correct_minus_flipped": float(np.mean(
            [row["target_correct_minus_flipped"] for row in rows]
        )),
        "candidate_hit20": {
            "original_order": 100.0 * np.mean(
                [row["original_top1_hit20"] for row in rows]
            ),
            "correct_field": 100.0 * np.mean(
                [row["field_hit20"] for row in rows]
            ),
            "flipped_field": 100.0 * np.mean(
                [row["flipped_field_hit20"] for row in rows]
            ),
        },
        "by_cue": by_cue,
    }


def aggregate_near(rows):
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "target_field_percentile": summarize(
            [row["target_percentile"] for row in rows]
        ),
        "target_in_top_quartile_percent": 100.0 * np.mean(
            [row["target_percentile"] >= 75.0 for row in rows]
        ),
        "candidate_hit20": {
            "original_order": 100.0 * np.mean(
                [row["original_top1_hit20"] for row in rows]
            ),
            "near_field": 100.0 * np.mean([row["field_hit20"] for row in rows]),
        },
    }


def main():
    args = parse_args()
    grid = grid_points(args.grid_size)
    directional_rows, near_rows = [], []
    exclusion_counts = {
        "no_bound_spatial_relation": 0,
        "multiple_directional_landmarks": 0,
        "multiple_near_landmarks": 0,
        "near_with_direction": 0,
    }

    rows = load_rows(args.input)
    for row in rows:
        relations = bound_relations(row)
        directional = [rel for rel in relations if len(rel["directions"]) == 1]
        near = [rel for rel in relations if rel["near"]]
        if len(directional) == 1:
            directional_rows.append(evaluate_directional(row, directional[0], grid))
            if near:
                exclusion_counts["near_with_direction"] += 1
        elif len(directional) > 1:
            exclusion_counts["multiple_directional_landmarks"] += 1
        elif len(near) == 1:
            near_rows.append(evaluate_near(row, near[0], grid, args.near_scale_m))
        elif len(near) > 1:
            exclusion_counts["multiple_near_landmarks"] += 1
        else:
            exclusion_counts["no_bound_spatial_relation"] += 1

    directional = aggregate_directional(directional_rows)
    near = aggregate_near(near_rows)
    gates = {
        "direction_correct_gt_flipped_at_least_60pct": bool(
            directional.get("correct_gt_flipped_percent", 0.0) >= 60.0
        ),
        "direction_correct_candidate_hit20_exceeds_flipped_by_5pp": bool(
            directional.get("candidate_hit20", {}).get("correct_field", 0.0)
            - directional.get("candidate_hit20", {}).get("flipped_field", 0.0)
            >= 5.0
        ),
        "near_target_top_quartile_at_least_50pct": bool(
            near.get("target_in_top_quartile_percent", 0.0) >= 50.0
        ),
    }
    result = {
        "step": 1,
        "split": "val_seen",
        "input_rows": len(rows),
        "coordinate_transform": "fixed rotate_270 selected previously on train_seen",
        "directional": directional,
        "near_only": near,
        "exclusions": exclusion_counts,
        "gates": gates,
        "passed": all(gates.values()),
        "anti_leakage": {
            "field_function_has_target_argument": False,
            "field_inputs": ["parsed relation", "named landmark boundary", "query points"],
            "target_use": "metrics only after all field and candidate scores are computed",
            "split_used": "val_seen only",
            "val_unseen_touched": False,
            "test_unseen_touched": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
