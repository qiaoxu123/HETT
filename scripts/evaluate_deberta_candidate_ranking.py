#!/usr/bin/env python3
"""Test whether frozen parsed relations causally improve Top-5 candidate ranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_deberta_relation_unseen import build_examples, predict
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE
from scripts.train_deberta_relation_parser import LABELS
from scripts.train_relation_geometry_ranker import (
    audit_consistent_landmark_geometry, load_rows,
)
from scripts.train_rgb_candidate_ranker import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(
        "runs/deberta_relation_parser_broad/model"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--output", type=Path, default=Path(
        "runs/deberta_relation_parser_broad/candidate_ranking.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-landmarks", type=int, default=9)
    return parser.parse_args()


def parse_landmark_relations(model, tokenizer, rows, args, device):
    examples = build_examples(rows, args.max_landmarks)
    predictions = predict(model, tokenizer, examples, device, args.batch_size)
    relation = torch.full(
        (len(rows), args.max_landmarks), LABELS.index("NONE"), dtype=torch.long)
    for prediction, example in zip(predictions, examples):
        _, row_index, landmark_index = example["sample_id"].split(":")
        relation[int(row_index), int(landmark_index)] = prediction
    return relation


def relation_scores(rows, relation, max_landmarks):
    geometry, valid = audit_consistent_landmark_geometry(rows, max_landmarks)
    geometry = geometry.float()
    directional = torch.stack((
        -geometry[..., 0], geometry[..., 0],
        -geometry[..., 1], geometry[..., 1]), dim=-1)
    near = -torch.linalg.vector_norm(geometry[..., 5:7], dim=-1)
    all_scores = torch.cat((directional, near[..., None]), dim=-1)
    active = valid & relation.ne(LABELS.index("NONE"))
    selected = all_scores.gather(
        -1, relation[:, None, :, None].expand(
            -1, geometry.shape[1], -1, 1).clamp_max(4)).squeeze(-1)
    selected *= active[:, None]
    score = selected.sum(-1) / active.sum(-1).clamp_min(1)[:, None]
    score[~active.any(-1)] = 0
    return score, active


def target_distance(rows):
    candidates = torch.tensor([row["candidates"] for row in rows])
    targets = torch.tensor([row["target"] for row in rows])
    return torch.linalg.vector_norm(candidates - targets[:, None], dim=-1) * 410.0


def flip_relations(relation):
    flipped = relation.clone()
    mapping = {0: 1, 1: 0, 2: 3, 3: 2}
    for source, target in mapping.items():
        flipped[relation.eq(source)] = target
    return flipped


def pair_accuracy(distance, score, active):
    selected = distance.le(20).sum(-1).eq(1) & active.any(-1)
    if not selected.any():
        return {"count": 0, "accuracy": None}
    truth = distance.argmin(-1)
    return {"count": int(selected.sum()), "accuracy": float(
        score.argmax(-1)[selected].eq(truth[selected]).float().mean() * 100)}


@torch.inference_mode()
def main():
    args = parse_args(); device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir).to(device).eval()
    prepared = {}
    for split in ("val_seen", "val_unseen"):
        rows = load_rows(args.candidate_dir / f"{split}.jsonl", 5)
        relation = parse_landmark_relations(
            model, tokenizer, rows, args, device)
        score, active = relation_scores(rows, relation, args.max_landmarks)
        flipped, _ = relation_scores(
            rows, flip_relations(relation), args.max_landmarks)
        prepared[split] = {
            "rows": rows, "distance": target_distance(rows),
            "score": score, "flipped": flipped, "active": active,
        }

    seen = prepared["val_seen"]
    prior = -torch.arange(5).float()[None]
    relation_scale = (seen["score"][:, 0] - seen["score"][:, 1]).std().clamp_min(1e-6)
    best = None
    for step in range(81):
        alpha = step / 20.0
        fused = prior + alpha * seen["score"] / relation_scale
        value = metrics(seen["distance"], fused.argmax(-1))
        key = value["hit20"], -value["mean_ne"], -alpha
        if best is None or key > best[0]:
            best = key, alpha
    alpha = best[1]

    results = {}
    for split, data in prepared.items():
        fused = prior + alpha * data["score"] / relation_scale
        flipped_fused = prior + alpha * data["flipped"] / relation_scale
        results[split] = {
            "candidate_prior": metrics(
                data["distance"], torch.zeros(len(data["distance"]), dtype=torch.long)),
            "relation_only": metrics(data["distance"], data["score"].argmax(-1)),
            "prior_plus_relation": metrics(data["distance"], fused.argmax(-1)),
            "prior_plus_flipped_relation": metrics(
                data["distance"], flipped_fused.argmax(-1)),
            "active_episode_percent": float(
                data["active"].any(-1).float().mean() * 100),
            "unique_correct_pair_accuracy": {
                "correct": pair_accuracy(
                    data["distance"], data["score"], data["active"]),
                "flipped": pair_accuracy(
                    data["distance"], data["flipped"], data["active"]),
            },
        }
    result = {
        "experiment": "frozen DeBERTa relations for static Top-5 ranking",
        "selection": {
            "split": "val_seen", "alpha": alpha,
            "relation_scale": float(relation_scale),
        },
        **results,
        "anti_leakage": {
            "model_frozen_before_val_unseen": True,
            "relation_forward_uses_target": False,
            "target_usage": "val_seen alpha selection and final metrics only",
            "test_unseen_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
