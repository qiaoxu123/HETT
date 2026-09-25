#!/usr/bin/env python3
"""Test whether target-free point-cloud views improve perspective semantics."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.semantic import CLASSES, ProjectSemanticNet
from scripts.train_independent_semantic_detector import build_patches, split_maps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--orthomosaic-replay-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class PerspectiveDataset(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        values = np.load(self.root / self.rows[index]["file"])
        image = torch.from_numpy(values["rgb"].copy()).permute(2, 0, 1)
        target = torch.from_numpy(values["semantic"].copy()).long()
        return image, target


def confusion_matrix(prediction, target, classes=5):
    valid = target.ne(255)
    values = target[valid] * classes + prediction[valid]
    return torch.bincount(values, minlength=classes * classes).reshape(classes, classes)


def metrics(confusion):
    true_positive = confusion.diag().float()
    union = confusion.sum(0) + confusion.sum(1) - true_positive
    iou = true_positive / union.clamp_min(1)
    recall = true_positive / confusion.sum(1).clamp_min(1)
    support = confusion.sum(1)
    return {
        "mean_iou_foreground": float(iou[1:].mean().mul(100)),
        "mean_iou_all": float(iou.mean().mul(100)),
        "per_class_iou": {name: float(iou[index].mul(100))
                          for index, name in enumerate(CLASSES)},
        "per_class_recall": {name: float(recall[index].mul(100))
                             for index, name in enumerate(CLASSES)},
        "valid_pixels_per_class": {name: int(support[index])
                                   for index, name in enumerate(CLASSES)},
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(5, 5, dtype=torch.long)
    for image, target in loader:
        prediction = model(image.float().to(device).div(255.0)).argmax(1).cpu()
        confusion += confusion_matrix(prediction, target)
    return metrics(confusion)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines()]
    split_rows = {split: [row for row in rows if row["split"] == split]
                  for split in ("train_seen", "val_seen", "val_unseen")}
    loaders = {
        split: DataLoader(PerspectiveDataset(args.dataset, values),
                          batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
        for split, values in split_rows.items()
    }
    train_datasets = [PerspectiveDataset(args.dataset, split_rows["train_seen"])]
    ortho_loaders = None
    if args.orthomosaic_replay_per_class:
        objects = json.load(open(Path("data/cityrefer/objects.json")))
        replay = build_patches(
            Path("data"), split_maps(Path("data"), "train_seen"), objects,
            args.orthomosaic_replay_per_class, 40.0, 128, args.seed + 303, 10.0)
        ortho_seen = build_patches(
            Path("data"), split_maps(Path("data"), "val_seen"), objects,
            200, 40.0, 128, args.seed + 101, 0.0)
        ortho_unseen = build_patches(
            Path("data"), split_maps(Path("data"), "val_unseen"), objects,
            200, 40.0, 128, args.seed + 202, 0.0)
        train_datasets.append(TensorDataset(*replay))
        ortho_loaders = {
            "val_seen": DataLoader(TensorDataset(*ortho_seen), batch_size=32),
            "val_unseen": DataLoader(TensorDataset(*ortho_unseen), batch_size=32),
        }
    train_loader = DataLoader(
        ConcatDataset(train_datasets), batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    model = ProjectSemanticNet().to(device)
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(initial)
    baseline_seen = evaluate(model, loaders["val_seen"], device)
    baseline_unseen = evaluate(model, loaders["val_unseen"], device)
    baseline_ortho = ({split: evaluate(model, loader, device)
                       for split, loader in ortho_loaders.items()}
                      if ortho_loaders else None)
    print(json.dumps({"baseline_val_seen": baseline_seen,
                      "baseline_val_unseen": baseline_unseen}), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    class_weights = torch.tensor([0.5, 1.0, 2.0, 1.0, 1.5], device=device)
    best_score = baseline_seen["mean_iou_foreground"]
    if baseline_ortho:
        best_score = 0.5 * (best_score +
                            baseline_ortho["val_seen"]["mean_iou_foreground"])
    best_epoch, best_state, history = 0, copy.deepcopy(initial), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, target in train_loader:
            logits = model(image.float().to(device).div(255.0))
            loss = F.cross_entropy(
                logits, target.to(device), weight=class_weights, ignore_index=255)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        selection = evaluate(model, loaders["val_seen"], device)
        selection_ortho = (evaluate(model, ortho_loaders["val_seen"], device)
                           if ortho_loaders else None)
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "val_seen": selection, "orthomosaic_val_seen": selection_ortho}
        history.append(row)
        score = selection["mean_iou_foreground"]
        if selection_ortho:
            score = 0.5 * (score + selection_ortho["mean_iou_foreground"])
        print(json.dumps({"epoch": epoch, "loss": row["loss"],
                          "val_seen_miou": score}), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy({name: value.detach().cpu()
                                        for name, value in model.state_dict().items()})

    model.load_state_dict(best_state)
    adapted_seen = evaluate(model, loaders["val_seen"], device)
    adapted_unseen = evaluate(model, loaders["val_unseen"], device)
    adapted_ortho = ({split: evaluate(model, loader, device)
                      for split, loader in ortho_loaders.items()}
                     if ortho_loaders else None)
    result = {
        "config": {**vars(args), "dataset": str(args.dataset),
                   "initial_checkpoint": str(args.initial_checkpoint),
                   "output_dir": str(args.output_dir)},
        "counts": {split: len(values) for split, values in split_rows.items()},
        "selected_epoch_on_val_seen": best_epoch,
        "orthomosaic_model_on_perspective": {
            "val_seen": baseline_seen, "val_unseen": baseline_unseen},
        "orthomosaic_domain_baseline": baseline_ortho,
        "pointcloud_adapted": {"val_seen": adapted_seen, "val_unseen": adapted_unseen},
        "orthomosaic_domain_after_adaptation": adapted_ortho,
        "delta_val_unseen_foreground_miou": (
            adapted_unseen["mean_iou_foreground"] -
            baseline_unseen["mean_iou_foreground"]),
        "anti_leakage": {
            "frame_selection": "trajectory pose only",
            "invalid_pixels": "ignored in both loss and metrics",
            "training": "train_seen only",
            "checkpoint_selection": "val_seen only",
            "val_unseen": "evaluated before training as frozen baseline and once after selection",
            "target_coordinates_or_identity": False,
        },
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
