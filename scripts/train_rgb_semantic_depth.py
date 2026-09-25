#!/usr/bin/env python3
"""Train RGB semantic and depth heads from point-cloud teacher frames."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.semantic import ProjectSemanticDepthNet
from scripts.train_independent_semantic_detector import build_patches, split_maps
from scripts.train_pointcloud_semantic_adaptation import confusion_matrix, metrics


LOG_DEPTH_MAX = float(np.log1p(260.0))


class PointDepthDataset(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        values = np.load(self.root / self.rows[index]["file"])
        return (torch.from_numpy(values["rgb"].copy()).permute(2, 0, 1),
                torch.from_numpy(values["semantic"].copy()).long(),
                torch.from_numpy(values["depth"].astype(np.float32)),
                torch.tensor(True))


class OrthoReplayDataset(Dataset):
    def __init__(self, tensors):
        self.images, self.targets = tensors

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        shape = self.targets[index].shape
        return (self.images[index], self.targets[index],
                torch.full(shape, float("nan")), torch.tensor(False))


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(5, 5, dtype=torch.long)
    absolute, relative, depth_pixels = 0.0, 0.0, 0
    for image, target, depth, has_depth in loader:
        semantic, normalized = model.predict_all(image.float().to(device).div(255.0))
        prediction = semantic.argmax(1).cpu()
        confusion += confusion_matrix(prediction, target)
        depth = depth.to(device)
        valid = torch.isfinite(depth) & has_depth.to(device)[:, None, None]
        if valid.any():
            predicted_depth = torch.expm1(normalized * LOG_DEPTH_MAX)
            error = (predicted_depth - depth).abs()
            absolute += float(error[valid].sum())
            relative += float((error[valid] / depth[valid].clamp_min(1.0)).sum())
            depth_pixels += int(valid.sum())
    result = metrics(confusion)
    result["depth_mae_m"] = absolute / max(depth_pixels, 1)
    result["depth_abs_rel"] = relative / max(depth_pixels, 1)
    result["depth_pixels"] = depth_pixels
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--orthomosaic-replay-per-class", type=int, default=144)
    parser.add_argument("--depth-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [json.loads(line) for line in
            (args.dataset / "manifest.jsonl").read_text().splitlines()]
    split_rows = {split: [row for row in rows if row["split"] == split]
                  for split in ("train_seen", "val_seen", "val_unseen")}
    point = {split: PointDepthDataset(args.dataset, values)
             for split, values in split_rows.items()}

    objects = json.load(open("data/cityrefer/objects.json"))
    replay = build_patches(
        Path("data"), split_maps(Path("data"), "train_seen"), objects,
        args.orthomosaic_replay_per_class, 40.0, 128, args.seed + 303, 10.0)
    ortho_seen = build_patches(
        Path("data"), split_maps(Path("data"), "val_seen"), objects,
        200, 40.0, 128, args.seed + 101, 0.0)
    train_loader = DataLoader(
        ConcatDataset([point["train_seen"], OrthoReplayDataset(replay)]),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    seen_loader = DataLoader(point["val_seen"], batch_size=args.batch_size)
    unseen_loader = DataLoader(point["val_unseen"], batch_size=args.batch_size)
    ortho_loader = DataLoader(OrthoReplayDataset(ortho_seen), batch_size=32)

    model = ProjectSemanticDepthNet().to(device)
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(initial, strict=False)
    assert set(missing) == {"depth_output.weight", "depth_output.bias"} and not unexpected
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    class_weights = torch.tensor([0.5, 1.0, 2.0, 1.0, 1.5], device=device)
    best_score, best_epoch, best_state, history = -1e9, None, None, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, target, depth, has_depth in train_loader:
            semantic, normalized = model.predict_all(image.float().to(device).div(255.0))
            target, depth = target.to(device), depth.to(device)
            semantic_loss = F.cross_entropy(
                semantic, target, weight=class_weights, ignore_index=255)
            valid = torch.isfinite(depth) & has_depth.to(device)[:, None, None]
            if valid.any():
                normalized_target = torch.log1p(depth.clamp_min(0)) / LOG_DEPTH_MAX
                depth_loss = F.smooth_l1_loss(normalized[valid], normalized_target[valid])
            else:
                depth_loss = normalized.sum() * 0.0
            loss = semantic_loss + args.depth_weight * depth_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        perspective = evaluate(model, seen_loader, device)
        ortho = evaluate(model, ortho_loader, device)
        score = (perspective["mean_iou_foreground"] + ortho["mean_iou_foreground"]
                 - perspective["depth_mae_m"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "score": score, "perspective_val_seen": perspective,
                        "orthomosaic_val_seen": ortho})
        print(json.dumps({"epoch": epoch, "loss": history[-1]["loss"],
                          "perspective_miou": perspective["mean_iou_foreground"],
                          "depth_mae_m": perspective["depth_mae_m"],
                          "orthomosaic_miou": ortho["mean_iou_foreground"]}), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy({name: value.detach().cpu()
                                        for name, value in model.state_dict().items()})

    model.load_state_dict(best_state)
    result = {
        "config": {**vars(args), "dataset": str(args.dataset),
                   "initial_checkpoint": str(args.initial_checkpoint),
                   "output_dir": str(args.output_dir)},
        "selected_epoch_on_val_seen": best_epoch,
        "val_seen": evaluate(model, seen_loader, device),
        "val_unseen": evaluate(model, unseen_loader, device),
        "orthomosaic_val_seen": evaluate(model, ortho_loader, device),
        "anti_leakage": {
            "training": "train_seen only", "selection": "val_seen only",
            "target_coordinates_or_identity": False,
            "depth_teacher": "point-cloud z-buffer depth on valid pixels",
        },
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
