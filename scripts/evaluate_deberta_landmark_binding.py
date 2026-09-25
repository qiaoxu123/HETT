#!/usr/bin/env python3
"""Check whether relation predictions are causally conditioned on the landmark.

This diagnostic uses only instruction text and per-landmark lexical labels.  It
does not read target coordinates.  Eligible instructions mention at least two
landmarks whose stated relations differ, so a model that ignores the focus
landmark cannot solve the task.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_deberta_relation_parser import (
    LABELS, LABEL_TO_ID, RelationDataset, collate,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(
        "runs/deberta_relation_parser_broad/model"))
    parser.add_argument("--dataset", type=Path, default=Path(
        "runs/relation_teacher_dataset/val_seen.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/deberta_relation_parser_broad/landmark_binding.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def relation_label(row):
    return row["strict_relation"] or row.get("broad_relation")


def make_examples(records):
    groups = defaultdict(list)
    for row in records:
        label = relation_label(row)
        if label is not None:
            groups[row["row_index"]].append((row, label))
    groups = {
        key: values for key, values in groups.items()
        if len(values) >= 2 and len({label for _, label in values}) >= 2
    }
    examples = []
    for group_id, values in groups.items():
        for row, label in values:
            examples.append({
                "text": row["instruction"],
                "landmark": row["landmark_name"],
                "label": LABEL_TO_ID[label],
                "sample_id": row["sample_id"],
                "group_id": group_id,
            })
    return examples, groups


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
    return 100.0 * sum(
        prediction == row["label"]
        for prediction, row in zip(predictions, examples)) / len(examples)


def main():
    args = parse_args(); device = torch.device("cuda")
    records = [json.loads(line) for line in args.dataset.read_text().splitlines()
               if line.strip()]
    examples, groups = make_examples(records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir).to(device).eval()
    correct = predict(model, tokenizer, examples, device, args.batch_size)
    blank_examples = [{**row, "landmark": ""} for row in examples]
    blank = predict(model, tokenizer, blank_examples, device, args.batch_size)
    correct_accuracy = accuracy(correct, examples)
    blank_accuracy = accuracy(blank, examples)

    group_predictions = defaultdict(list)
    for prediction, row in zip(correct, examples):
        group_predictions[row["group_id"]].append((prediction, row["label"]))
    all_correct = sum(all(prediction == truth for prediction, truth in values)
                      for values in group_predictions.values())
    distinct = sum(len({prediction for prediction, _ in values}) >= 2
                   for values in group_predictions.values())
    changed_from_blank = sum(a != b for a, b in zip(correct, blank))
    result = {
        "experiment": "same-instruction multi-landmark causal binding",
        "eligible_instructions": len(groups),
        "eligible_mentions": len(examples),
        "distribution": dict(Counter(
            LABELS[row["label"]] for row in examples)),
        "correct_focus_accuracy": correct_accuracy,
        "blank_focus_accuracy": blank_accuracy,
        "prediction_changed_when_focus_removed_percent": (
            100.0 * changed_from_blank / len(examples)),
        "all_landmarks_correct_instruction_percent": (
            100.0 * all_correct / len(groups)),
        "different_focus_produced_different_relation_percent": (
            100.0 * distinct / len(groups)),
        "gate": {
            "minimum_correct_focus_accuracy": 80.0,
            "minimum_focus_removal_drop_points": 20.0,
            "minimum_different_focus_response": 80.0,
            "passed": (
                correct_accuracy >= 80.0
                and correct_accuracy - blank_accuracy >= 20.0
                and 100.0 * distinct / len(groups) >= 80.0),
        },
        "anti_leakage": {
            "split": "val_seen only",
            "target_coordinates_loaded": False,
            "geometry_used": False,
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
