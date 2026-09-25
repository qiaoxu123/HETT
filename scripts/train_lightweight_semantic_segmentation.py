#!/usr/bin/env python3
"""Train a small RGB semantic-segmentation feasibility baseline."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_candidate_geometry import polygon_centroid


CLASSES = ("Background", "Building", "Car", "Ground", "Parking")
TYPE_TO_INDEX = {name: index for index, name in enumerate(CLASSES) if index}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/lightweight_semantic_segmentation"))
    parser.add_argument("--crop-side-m", type=float, default=80.0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def split_maps(data_root, split):
    rows = json.load(open(data_root / "processed_citynav" / f"citynav_{split}.json"))
    return sorted({f'{row["area"]}_block_{row["block"]}' for row in rows})


def rasterize_map(rgb_path, tif_path, objects):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    with rasterio.open(tif_path) as raster:
        transform = raster.transform
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    # Large support surfaces first, small foreground objects last.
    for object_type in ("Ground", "Parking", "Building", "Car"):
        for obj in objects.values():
            if obj.get("object_type") != object_type:
                continue
            contour = np.asarray(obj["contour"], dtype=np.float64)
            rows, cols = rasterio.transform.rowcol(
                transform, contour[:, 0], contour[:, 1])
            polygon = np.stack((cols, rows), axis=-1).astype(np.int32)
            cv2.fillPoly(mask, [polygon], TYPE_TO_INDEX[object_type])
    return rgb, mask, transform


def crop_at(array, center_xy, side_pixels, resolution, interpolation):
    patch = cv2.getRectSubPix(array, (side_pixels, side_pixels), center_xy)
    return cv2.resize(patch, (resolution, resolution), interpolation=interpolation)


def selected_entries(map_names, objects, per_class, seed):
    grouped = defaultdict(list)
    for map_name in map_names:
        for obj in objects[map_name].values():
            if obj.get("object_type") in TYPE_TO_INDEX:
                grouped[obj["object_type"]].append((map_name, obj))
    rng = random.Random(seed)
    result = []
    for object_type in TYPE_TO_INDEX:
        values = grouped[object_type]
        if not values:
            continue
        result.extend(rng.choice(values) for _ in range(per_class))
    rng.shuffle(result)
    return result


def build_patches(data_root, map_names, objects, per_class, side_m, resolution, seed,
                  jitter_m):
    entries = selected_entries(map_names, objects, per_class, seed)
    by_map = defaultdict(list)
    for index, (map_name, obj) in enumerate(entries):
        by_map[map_name].append((index, obj))
    images = np.empty((len(entries), resolution, resolution, 3), dtype=np.uint8)
    masks = np.empty((len(entries), resolution, resolution), dtype=np.uint8)
    rng = random.Random(seed + 17)
    for map_name, values in sorted(by_map.items()):
        rgb, mask, transform = rasterize_map(
            data_root / "rgbd" / f"{map_name}.png",
            data_root / "rgbd" / f"{map_name}.tif", objects[map_name])
        side_pixels = max(8, int(round(side_m / abs(transform.a))))
        for index, obj in values:
            x, y = polygon_centroid(obj["contour"])
            x += rng.uniform(-jitter_m, jitter_m)
            y += rng.uniform(-jitter_m, jitter_m)
            row, col = rasterio.transform.rowcol(transform, x, y)
            center = (float(col), float(row))
            images[index] = crop_at(
                rgb, center, side_pixels, resolution, cv2.INTER_AREA)
            masks[index] = crop_at(
                mask, center, side_pixels, resolution, cv2.INTER_NEAREST)
        print(f"patches {map_name}: {len(values)}", flush=True)
    # OpenCV is BGR; channel order is irrelevant from-scratch but RGB is clearer.
    images = images[..., ::-1].copy()
    return (torch.from_numpy(images).permute(0, 3, 1, 2),
            torch.from_numpy(masks).long())


class Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels), nn.SiLU())

    def forward(self, value):
        return self.layers(value)


class TinyUNet(nn.Module):
    def __init__(self, classes=5, width=24):
        super().__init__()
        self.enc1 = Block(3, width)
        self.enc2 = Block(width, width * 2)
        self.enc3 = Block(width * 2, width * 4)
        self.middle = Block(width * 4, width * 8)
        self.dec3 = Block(width * 8 + width * 4, width * 4)
        self.dec2 = Block(width * 4 + width * 2, width * 2)
        self.dec1 = Block(width * 2 + width, width)
        self.output = nn.Conv2d(width, classes, 1)

    def forward(self, value):
        first = self.enc1(value)
        second = self.enc2(F.max_pool2d(first, 2))
        third = self.enc3(F.max_pool2d(second, 2))
        value = self.middle(F.max_pool2d(third, 2))
        value = self.dec3(torch.cat((F.interpolate(value, scale_factor=2), third), 1))
        value = self.dec2(torch.cat((F.interpolate(value, scale_factor=2), second), 1))
        value = self.dec1(torch.cat((F.interpolate(value, scale_factor=2), first), 1))
        return self.output(value)


def confusion_matrix(prediction, target, classes=5):
    valid = target.ge(0) & target.lt(classes)
    values = target[valid] * classes + prediction[valid]
    return torch.bincount(values, minlength=classes * classes).reshape(classes, classes)


def segmentation_metrics(confusion):
    true_positive = confusion.diag().float()
    union = confusion.sum(0) + confusion.sum(1) - true_positive
    iou = true_positive / union.clamp_min(1)
    recall = true_positive / confusion.sum(1).clamp_min(1)
    return {
        "mean_iou_foreground": 100.0 * iou[1:].mean().item(),
        "mean_recall_foreground": 100.0 * recall[1:].mean().item(),
        "per_class_iou": {name: 100.0 * iou[index].item()
                          for index, name in enumerate(CLASSES)},
        "per_class_recall": {name: 100.0 * recall[index].item()
                             for index, name in enumerate(CLASSES)},
        "confusion": confusion.tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(5, 5, dtype=torch.long)
    for image, target in loader:
        prediction = model(image.float().to(device).div(255.0)).argmax(1).cpu()
        confusion += confusion_matrix(prediction, target)
    return segmentation_metrics(confusion)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    train_maps = split_maps(args.data_root, "train_seen")
    validation_maps = split_maps(args.data_root, "val_unseen")
    train_images, train_masks = build_patches(
        args.data_root, train_maps, objects, args.train_per_class,
        args.crop_side_m, args.resolution, args.seed, jitter_m=10.0)
    val_images, val_masks = build_patches(
        args.data_root, validation_maps, objects, args.val_per_class,
        args.crop_side_m, args.resolution, args.seed + 1, jitter_m=0.0)
    train_loader = DataLoader(TensorDataset(train_images, train_masks),
                              batch_size=24, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_masks), batch_size=32)
    model = TinyUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    class_weights = torch.tensor([0.2, 1.0, 2.0, 1.0, 1.5], device=device)
    best, best_metric, history = None, -1.0, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for image, target in train_loader:
            logits = model(image.float().to(device).div(255.0))
            loss = F.cross_entropy(logits, target.to(device), weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        value = evaluate(model, val_loader, device)
        value["epoch"] = epoch + 1
        value["train_loss"] = sum(losses) / len(losses)
        history.append(value)
        print(json.dumps({"epoch": epoch + 1, "loss": value["train_loss"],
                          "val_miou_fg": value["mean_iou_foreground"]}), flush=True)
        if value["mean_iou_foreground"] > best_metric:
            best_metric = value["mean_iou_foreground"]
            best = {key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best, args.output_dir / "model.pt")
    model.load_state_dict(best)
    final = evaluate(model, val_loader, device)
    result = {
        "config": vars(args) | {"data_root": str(args.data_root),
                                "output_dir": str(args.output_dir)},
        "classes": CLASSES,
        "train_maps": train_maps,
        "validation_maps": validation_maps,
        "counts": {"train_patches": len(train_images),
                   "validation_patches": len(val_images)},
        "best_validation": final,
        "history": history,
        "anti_leakage": {
            "training_maps": "train_seen only",
            "validation_maps": "val_unseen only",
            "target_identity_or_target_coordinate": "not used",
        },
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
