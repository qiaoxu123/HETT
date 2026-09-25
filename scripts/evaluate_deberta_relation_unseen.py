#!/usr/bin/env python3
"""One-shot frozen val_unseen evaluation of landmark-relation parsing."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_relation_teacher_dataset import broad_relation, strict_relation
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE
from scripts.train_relation_geometry_ranker import load_rows
from scripts.train_deberta_relation_parser import (
    LABELS, LABEL_TO_ID, RelationDataset, collate,
)
from scripts.validate_candidate_geometry import mask_landmark_span


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(
        "runs/deberta_relation_parser_broad/model"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--output", type=Path, default=Path(
        "runs/deberta_relation_parser_broad/val_unseen_language.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-landmarks", type=int, default=9)
    return parser.parse_args()


def build_examples(rows, max_landmarks):
    examples = []
    for row_index, row in enumerate(rows):
        for landmark_index, (query, name) in enumerate(zip(
                row.get("relation_queries", [])[:max_landmarks],
                row.get("landmark_names", [])[:max_landmarks])):
            masked, found = mask_landmark_span(row["description"], query, name)
            if not found:
                continue
            label = strict_relation(masked) or broad_relation(masked)
            if label is None:
                continue
            examples.append({
                "text": row["description"],
                "landmark": name,
                "label": LABEL_TO_ID[label],
                "sample_id": f"val_unseen:{row_index}:{landmark_index}",
                "group_id": row_index,
            })
    return examples


@torch.inference_mode()
def predict(model, tokenizer, examples, device, batch_size):
    loader = DataLoader(
        RelationDataset(examples, tokenizer, 192), batch_size=batch_size,
        collate_fn=collate(tokenizer))
    predictions = []
    for batch in loader:
        batch.pop("labels")
        logits = model(**{key: value.to(device)
                          for key, value in batch.items()}).logits
        predictions.extend(logits.argmax(-1).cpu().tolist())
    return predictions


def accuracy(predictions, examples):
    return 100.0 * sum(prediction == row["label"]
                       for prediction, row in zip(predictions, examples)) / len(examples)


def main():
    args = parse_args(); device = torch.device("cuda")
    rows = load_rows(args.candidate_dir / "val_unseen.jsonl", 5)
    examples = build_examples(rows, args.max_landmarks)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir).to(device).eval()
    correct = predict(model, tokenizer, examples, device, args.batch_size)
    blank_examples = [{**row, "landmark": ""} for row in examples]
    blank = predict(model, tokenizer, blank_examples, device, args.batch_size)

    grouped = defaultdict(list)
    for index, row in enumerate(examples):
        grouped[row["group_id"]].append(index)
    eligible = [indices for indices in grouped.values()
                if len(indices) >= 2
                and len({examples[index]["label"] for index in indices}) >= 2]
    binding_indices = [index for indices in eligible for index in indices]
    binding_correct = [correct[index] for index in binding_indices]
    binding_blank = [blank[index] for index in binding_indices]
    binding_examples = [examples[index] for index in binding_indices]
    distinct = sum(len({correct[index] for index in indices}) >= 2
                   for indices in eligible)
    result = {
        "experiment": "frozen val_unseen language-relation and landmark-binding",
        "all_relation_mentions": len(examples),
        "distribution": dict(Counter(
            LABELS[row["label"]] for row in examples)),
        "all_accuracy": accuracy(correct, examples),
        "multi_landmark_instructions": len(eligible),
        "multi_landmark_mentions": len(binding_examples),
        "multi_landmark_correct_focus_accuracy": accuracy(
            binding_correct, binding_examples),
        "multi_landmark_blank_focus_accuracy": accuracy(
            binding_blank, binding_examples),
        "different_focus_produced_different_relation_percent": (
            100.0 * distinct / len(eligible)),
        "frozen_gate": {
            "minimum_all_accuracy": 80.0,
            "minimum_binding_accuracy": 80.0,
            "minimum_binding_drop_points": 20.0,
            "passed": (
                accuracy(correct, examples) >= 80.0
                and accuracy(binding_correct, binding_examples) >= 80.0
                and accuracy(binding_correct, binding_examples)
                - accuracy(binding_blank, binding_examples) >= 20.0),
        },
        "anti_leakage": {
            "model_frozen_before_val_unseen": True,
            "checkpoint_selected_on": "val_seen strict-template subset",
            "val_unseen_used_for_tuning": False,
            "target_coordinates_used_by_metric": False,
            "test_unseen_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
