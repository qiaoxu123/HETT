#!/usr/bin/env python3
"""Fine-tune DeBERTa to normalize per-landmark spatial relations."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ("LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "NEAR", "NONE")
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}
ALLOWED_WORDS = re.compile(
    r"\b(?:left|right|west|western|east|eastern|north|northern|south|southern|"
    r"above|below|under|over|near|nearby|next|beside|adjacent|close|alongside)\b",
    re.I)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path(
        "runs/relation_teacher_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/deberta_relation_parser"))
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--hf-cache", type=Path, default=Path(
        "runs/deberta_relation_parser/hf_cache"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--none-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def local_window(masked_instruction, radius=14):
    words = re.findall(r"\[mask\]|[a-z]+", masked_instruction.lower())
    try:
        index = words.index("[mask]")
    except ValueError:
        return masked_instruction
    return " ".join(words[max(0, index - radius):index + radius + 1])


def build_examples(path, none_ratio, seed, include_broad=False):
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    positive = [row for row in rows
                if row["strict_geometry_accepted"]
                or (include_broad and row.get("broad_geometry_accepted", False))]
    none = [row for row in rows
            if row["strict_relation"] is None
            and not row.get("broad_geometry_accepted", False)
            and not ALLOWED_WORDS.search(local_window(row["masked_instruction"]))]
    random.Random(seed).shuffle(none)
    none = none[:int(len(positive) * none_ratio)]
    examples = []
    for row in positive + none:
        if row["strict_geometry_accepted"]:
            label = row["strict_relation"]
        elif include_broad and row.get("broad_geometry_accepted", False):
            label = row["broad_relation"]
        else:
            label = "NONE"
        examples.append({
            "text": row["instruction"],
            "landmark": row["landmark_name"],
            "label": LABEL_TO_ID[label],
            "sample_id": row["sample_id"],
        })
    random.Random(seed).shuffle(examples)
    return examples


class RelationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        encoded = self.tokenizer(
            row["text"], f'Focus landmark: {row["landmark"]}',
            truncation=True, max_length=self.max_length)
        return {**encoded, "labels": row["label"]}


def collate(tokenizer):
    def apply(batch):
        labels = torch.tensor([row.pop("labels") for row in batch])
        output = tokenizer.pad(batch, padding=True, return_tensors="pt")
        output["labels"] = labels
        return output
    return apply


@torch.inference_mode()
def predict(model, loader, device):
    logits, labels = [], []
    for batch in loader:
        truth = batch.pop("labels")
        output = model(**{key: value.to(device) for key, value in batch.items()})
        logits.append(output.logits.cpu()); labels.append(truth)
    return torch.cat(logits), torch.cat(labels)


def metrics(logits, labels):
    prediction = logits.argmax(-1)
    present = sorted(labels.unique().tolist())
    macro = f1_score(labels.numpy(), prediction.numpy(), labels=present,
                     average="macro", zero_division=0)
    direction = labels.ne(LABEL_TO_ID["NONE"])
    direction_accuracy = prediction[direction].eq(labels[direction]).float().mean()
    return {
        "accuracy": float(prediction.eq(labels).float().mean() * 100),
        "macro_f1": float(macro * 100),
        "direction_accuracy": float(direction_accuracy * 100),
    }


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train = build_examples(
        args.dataset_dir / "train_seen.jsonl", args.none_ratio, args.seed,
        include_broad=True)
    validation = build_examples(
        args.dataset_dir / "val_seen.jsonl", args.none_ratio, args.seed,
        include_broad=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=str(args.hf_cache))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), num_labels=len(LABELS),
        id2label={index: name for index, name in enumerate(LABELS)},
        label2id=LABEL_TO_ID).cuda()
    train_loader = DataLoader(
        RelationDataset(train, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate(tokenizer))
    val_loader = DataLoader(
        RelationDataset(validation, tokenizer, args.max_length),
        batch_size=args.batch_size * 2, collate_fn=collate(tokenizer))
    counts = torch.bincount(
        torch.tensor([row["label"] for row in train]), minlength=len(LABELS)).float()
    weight = counts.sum() / (len(LABELS) * counts.clamp_min(1))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").cuda()
            logits = model(**{key: value.cuda() for key, value in batch.items()}).logits
            loss = nn.functional.cross_entropy(logits, labels, weight=weight.cuda())
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval(); logits, labels = predict(model, val_loader, torch.device("cuda"))
        value = metrics(logits, labels)
        key = value["macro_f1"], value["direction_accuracy"], -epoch
        if best is None or key > best[0]:
            best = key, copy.deepcopy(model.state_dict()), epoch, value
        print(f"epoch={epoch} macro_f1={value['macro_f1']:.2f} "
              f"direction={value['direction_accuracy']:.2f} "
              f"accuracy={value['accuracy']:.2f}", flush=True)
    model.load_state_dict(best[1]); model.eval()
    logits, labels = predict(model, val_loader, torch.device("cuda"))
    prediction = logits.argmax(-1)
    result = {
        "model": args.model,
        "labels": LABELS,
        "counts": {"train": len(train), "val_seen": len(validation)},
        "train_distribution": dict(Counter(LABELS[row["label"]] for row in train)),
        "selected_epoch": best[2],
        "val_seen": best[3],
        "confusion_matrix": confusion_matrix(
            labels.numpy(), prediction.numpy(), labels=range(len(LABELS))).tolist(),
        "classification_report": classification_report(
            labels.numpy(), prediction.numpy(), labels=range(len(LABELS)),
            target_names=LABELS, zero_division=0, output_dict=True),
        "gate": {
            "minimum_macro_f1": 80.0,
            "minimum_direction_accuracy": 80.0,
            "passed": (best[3]["macro_f1"] >= 80.0
                       and best[3]["direction_accuracy"] >= 80.0),
        },
        "anti_leakage": {
            "training": "train_seen strict and broad geometry-verified labels only",
            "selection": "val_seen strict templates only",
            "broad_val_seen": "reserved for the separate generalization gate",
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
    }
    model.save_pretrained(args.output_dir / "model")
    tokenizer.save_pretrained(args.output_dir / "model")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
