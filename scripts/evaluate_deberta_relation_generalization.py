#!/usr/bin/env python3
"""Evaluate relation-parser paraphrase generalization on val_seen only."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_relation_teacher_dataset import geometry_check, target_geometry
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE
from scripts.train_deberta_relation_parser import (
    LABELS, LABEL_TO_ID, RelationDataset, collate,
)
from scripts.train_relation_geometry_ranker import load_rows


PATTERNS = {
    "LEFT_OF": re.compile(r"\b(?:left|west|western)\b", re.I),
    "RIGHT_OF": re.compile(r"\b(?:right|east|eastern)\b", re.I),
    "ABOVE": re.compile(r"\b(?:above|over|north|northern|top)\b", re.I),
    "BELOW": re.compile(r"\b(?:below|under|south|southern|bottom)\b", re.I),
    "NEAR": re.compile(
        r"\b(?:near|nearby|next|beside|adjacent|close|alongside)\b", re.I),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(
        "runs/deberta_relation_parser/model"))
    parser.add_argument("--dataset", type=Path, default=Path(
        "runs/relation_teacher_dataset/val_seen.jsonl"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--output", type=Path, default=Path(
        "runs/deberta_relation_parser/generalization.json"))
    parser.add_argument("--minimum-axis-m", type=float, default=10.0)
    parser.add_argument("--near-distance-m", type=float, default=30.0)
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def local_window(masked, radius=12):
    words = re.findall(r"\[mask\]|[a-z]+", masked.lower())
    try:
        index = words.index("[mask]")
    except ValueError:
        return ""
    return " ".join(words[max(0, index - radius):index + radius + 1])


def broad_relation(masked):
    window = local_window(masked)
    values = [name for name, pattern in PATTERNS.items() if pattern.search(window)]
    return values[0] if len(values) == 1 else None


@torch.inference_mode()
def main():
    args = parse_args(); device = torch.device("cuda")
    records = [json.loads(line) for line in args.dataset.read_text().splitlines()
               if line.strip()]
    rows = load_rows(args.candidate_dir / "val_seen.jsonl", 5)
    geometry, valid = target_geometry(rows, args.max_landmarks)
    examples = []
    for record in records:
        if record["strict_relation"] is not None:
            continue
        relation = broad_relation(record["masked_instruction"])
        if relation is None:
            continue
        row_index, landmark_index = record["row_index"], record["landmark_index"]
        if not valid[row_index, landmark_index]:
            continue
        accepted, _ = geometry_check(
            relation, geometry[row_index, landmark_index],
            args.minimum_axis_m, args.near_distance_m)
        if accepted:
            examples.append({
                "text": record["instruction"],
                "landmark": record["landmark_name"],
                "label": LABEL_TO_ID[relation],
                "sample_id": record["sample_id"],
            })
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir).to(device).eval()
    loader = DataLoader(
        RelationDataset(examples, tokenizer, 192), batch_size=args.batch_size,
        collate_fn=collate(tokenizer))
    logits, truth = [], []
    for batch in loader:
        truth.append(batch.pop("labels"))
        logits.append(model(**{key: value.to(device)
                               for key, value in batch.items()}).logits.cpu())
    logits, truth = torch.cat(logits), torch.cat(truth)
    probability = logits.softmax(-1)
    confidence, prediction = probability.max(-1)
    thresholds = []
    for step in range(50, 100, 5):
        threshold = step / 100.0
        selected = confidence.ge(threshold)
        if not selected.any():
            continue
        thresholds.append({
            "threshold": threshold,
            "count": int(selected.sum()),
            "coverage_percent": float(selected.float().mean() * 100),
            "accuracy": float(prediction[selected].eq(
                truth[selected]).float().mean() * 100),
        })
    eligible = [row for row in thresholds if row["accuracy"] >= 80.0]
    selected_threshold = max(
        eligible, key=lambda row: (row["coverage_percent"], row["accuracy"])) \
        if eligible else max(thresholds, key=lambda row: row["accuracy"])
    present = sorted(truth.unique().tolist())
    result = {
        "experiment": "held-out broad-relation paraphrase gate",
        "count": len(examples),
        "distribution": dict(Counter(LABELS[value] for value in truth.tolist())),
        "unthresholded_accuracy": float(prediction.eq(truth).float().mean() * 100),
        "unthresholded_macro_f1": float(f1_score(
            truth.numpy(), prediction.numpy(), labels=present,
            average="macro", zero_division=0) * 100),
        "thresholds": thresholds,
        "selected": selected_threshold,
        "gate": {
            "minimum_accuracy": 80.0,
            "minimum_count": 100,
            "passed": (selected_threshold["accuracy"] >= 80.0
                       and selected_threshold["count"] >= 100),
        },
        "anti_leakage": {
            "split": "val_seen only",
            "strict_training_templates_excluded": True,
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
