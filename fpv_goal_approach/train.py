from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from fpv_goal_approach.dataset import FPVGoalApproachDataset, collate_samples
from fpv_goal_approach.evaluate import evaluate_predictions, save_report
from fpv_goal_approach.model import GoalApproachModel


VARIANTS = {
    "current_fpv_only": dict(frames=1, use_language=False, use_pose=False),
    "current_fpv_language": dict(frames=1, use_language=True, use_pose=False),
    "history_language": dict(frames=4, use_language=True, use_pose=False),
    "history_language_pose": dict(frames=4, use_language=True, use_pose=True),
}
LABEL_KEYS = ("target_match", "arrival", "distance_bin", "bearing_bin", "phase")
ROW_LABEL_KEYS = {"target_match": "target_match", "arrival": "visual_confirmed_arrival",
                  "distance_bin": "distance_bin", "bearing_bin": "bearing_bin", "phase": "phase"}
CLASS_COUNTS = {"target_match": 2, "arrival": 2, "distance_bin": 6, "bearing_bin": 5, "phase": 4}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def weighted_criteria(dataset, device):
    criteria = {}
    for key in LABEL_KEYS:
        values = torch.tensor([row[ROW_LABEL_KEYS[key]] for row in dataset.rows], dtype=torch.long)
        counts = torch.bincount(values, minlength=CLASS_COUNTS[key]).float().clamp_min(1)
        weights = counts.sum() / (len(counts) * counts)
        criteria[key] = nn.CrossEntropyLoss(weight=weights.to(device))
    return criteria


@torch.no_grad()
def run_evaluation(model, loader, device, shuffle_language=False, shuffle_frames=False):
    model.eval()
    labels = {key: [] for key in LABEL_KEYS}
    logits = {key: [] for key in LABEL_KEYS}
    metadata = []
    for batch in loader:
        batch = move_batch(batch, device)
        if shuffle_language and batch["input_ids"].shape[0] > 1:
            batch["input_ids"] = torch.roll(batch["input_ids"], 1, 0)
            batch["attention_mask"] = torch.roll(batch["attention_mask"], 1, 0)
        if shuffle_frames:
            batch["pixel_values"] = batch["pixel_values"].flip(1)
        output = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["pose"])
        for key in LABEL_KEYS:
            labels[key].append(batch[key].cpu().numpy())
            logits[key].append(output[key].float().cpu().numpy())
        metadata.extend(batch["metadata"])
    labels = {key: np.concatenate(value) for key, value in labels.items()}
    logits = {key: np.concatenate(value) for key, value in logits.items()}
    return evaluate_predictions(labels, logits, metadata)


def train(args):
    seed_everything(args.seed)
    config = VARIANTS[args.variant]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name)
    datasets = {
        split: FPVGoalApproachDataset(
            args.data / "metadata" / f"{split}.jsonl", frames=config["frames"],
            use_language=config["use_language"], use_pose=config["use_pose"],
        ) for split in ("train_seen", "val_seen", "val_unseen")
    }
    collate = partial(collate_samples, processor=processor)
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train_seen",
                          num_workers=args.workers, pin_memory=True, collate_fn=collate)
        for split, dataset in datasets.items()
    }
    model = GoalApproachModel(args.model_name, **config).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    criteria = weighted_criteria(datasets["train_seen"], device)
    output = args.output / args.variant
    output.mkdir(parents=True, exist_ok=True)
    history = []
    scaler_enabled = device.type == "cuda"
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        totals = {"total": 0.0, **{key: 0.0 for key in LABEL_KEYS}}
        batches = 0
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        for batch in loaders["train_seen"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler_enabled):
                predictions = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["pose"])
                losses = {key: criteria[key](predictions[key], batch[key]) for key in LABEL_KEYS}
                loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            totals["total"] += float(loss.detach())
            for key, value in losses.items():
                totals[key] += float(value.detach())
            batches += 1
        row = {
            "epoch": epoch,
            "train_loss": totals["total"] / max(1, batches),
            **{f"{key}_loss": totals[key] / max(1, batches) for key in LABEL_KEYS},
            "learning_rate": args.learning_rate,
            "epoch_time_s": time.time() - started,
            "gpu_peak_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0,
        }
        history.append(row)
        print(json.dumps(row))
    with (output / "train_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    trainable = {name: value.cpu() for name, value in model.state_dict().items() if not name.startswith("siglip.")}
    torch.save({"variant": args.variant, "config": config, "trainable_state_dict": trainable}, output / "model.pt")
    reports = {}
    for split in ("val_seen", "val_unseen"):
        report = run_evaluation(model, loaders[split], device)
        reports[split] = report
        save_report(report, output / f"metrics_{split}.json")
    shuffle_language = run_evaluation(model, loaders["val_unseen"], device, shuffle_language=True)
    save_report(shuffle_language, output / "metrics_val_unseen_wrong_language_shuffle.json")
    if config["frames"] > 1:
        shuffle_frames = run_evaluation(model, loaders["val_unseen"], device, shuffle_frames=True)
        save_report(shuffle_frames, output / "metrics_val_unseen_frame_order_shuffle.json")
    summary = {
        "variant": args.variant,
        "config": config,
        "samples": {key: len(value) for key, value in datasets.items()},
        "history": history,
        "metrics": reports,
    }
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model_name", default="google/siglip-base-patch16-224")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
