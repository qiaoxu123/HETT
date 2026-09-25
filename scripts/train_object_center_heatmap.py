#!/usr/bin/env python3
"""Train class-specific object-center heatmaps on 40 m aerial RGB tiles."""

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
from scripts.train_lightweight_semantic_segmentation import (
    TinyUNet, crop_at, selected_entries, split_maps,
)
from scripts.validate_candidate_geometry import polygon_centroid


CLASSES = ("Building", "Car", "Ground", "Parking")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/object_center_heatmap_40m"))
    parser.add_argument("--crop-side-m", type=float, default=40.0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def gaussian_kernel(sigma=1.75, radius=5):
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))


def draw_gaussian(heatmap, x, y, kernel):
    radius = kernel.shape[0] // 2
    center_x, center_y = int(round(x)), int(round(y))
    x0, x1 = max(0, center_x - radius), min(heatmap.shape[1], center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(heatmap.shape[0], center_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    kernel_x0 = x0 - (center_x - radius)
    kernel_y0 = y0 - (center_y - radius)
    patch = kernel[kernel_y0:kernel_y0 + y1 - y0,
                   kernel_x0:kernel_x0 + x1 - x0]
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)


def object_pixel_centers(values, transform):
    result = []
    for obj in values.values():
        if obj.get("object_type") not in CLASSES:
            continue
        x, y = polygon_centroid(obj["contour"])
        row, col = rasterio.transform.rowcol(transform, x, y)
        result.append((CLASSES.index(obj["object_type"]), float(col), float(row)))
    return result


def build_patches(data_root, map_names, objects, per_class, side_m, resolution,
                  seed, jitter_m):
    entries = selected_entries(map_names, objects, per_class, seed)
    grouped = defaultdict(list)
    for index, (map_name, obj) in enumerate(entries):
        grouped[map_name].append((index, obj))
    images = np.empty((len(entries), resolution, resolution, 3), dtype=np.uint8)
    heatmaps = np.zeros((len(entries), len(CLASSES), resolution, resolution),
                        dtype=np.uint8)
    kernel = gaussian_kernel()
    rng = random.Random(seed + 37)
    for map_name, selected in sorted(grouped.items()):
        rgb = cv2.imread(str(data_root / "rgbd" / f"{map_name}.png"), cv2.IMREAD_COLOR)
        with rasterio.open(data_root / "rgbd" / f"{map_name}.tif") as raster:
            transform = raster.transform
        side_pixels = max(8, int(round(side_m / abs(transform.a))))
        centers = object_pixel_centers(objects[map_name], transform)
        for index, obj in selected:
            x, y = polygon_centroid(obj["contour"])
            x += rng.uniform(-jitter_m, jitter_m)
            y += rng.uniform(-jitter_m, jitter_m)
            center_row, center_col = rasterio.transform.rowcol(transform, x, y)
            images[index] = crop_at(
                rgb, (float(center_col), float(center_row)), side_pixels,
                resolution, cv2.INTER_AREA)
            scale = resolution / side_pixels
            target = np.zeros((len(CLASSES), resolution, resolution), dtype=np.float32)
            for class_index, object_col, object_row in centers:
                local_x = (object_col - center_col) * scale + resolution / 2
                local_y = (object_row - center_row) * scale + resolution / 2
                if -5 <= local_x < resolution + 5 and -5 <= local_y < resolution + 5:
                    draw_gaussian(target[class_index], local_x, local_y, kernel)
            heatmaps[index] = np.round(target * 255).astype(np.uint8)
        print(f"center patches {map_name}: {len(selected)}", flush=True)
    images = images[..., ::-1].copy()
    return (torch.from_numpy(images).permute(0, 3, 1, 2),
            torch.from_numpy(heatmaps))


def focal_heatmap_loss(logits, targets):
    prediction = logits.sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = targets.eq(1.0)
    negative = ~positive
    negative_weight = (1.0 - targets).pow(4)
    positive_loss = -(prediction.log() * (1.0 - prediction).pow(2) * positive).sum(
        dim=(0, 2, 3))
    negative_loss = -((1.0 - prediction).log() * prediction.pow(2)
                      * negative_weight * negative).sum(dim=(0, 2, 3))
    raw_positive_count = positive.sum(dim=(0, 2, 3))
    valid_class = raw_positive_count.gt(0)
    positive_count = raw_positive_count.clamp_min(1)
    # Equalize classes: dense car centers must not suppress sparse ground and
    # parking components or arbitrary building-center supervision.
    return ((positive_loss + negative_loss) / positive_count)[valid_class].mean()


def peak_mask(heatmap, threshold=0.05):
    maximum = F.max_pool2d(heatmap, 5, stride=1, padding=2)
    return heatmap.eq(maximum) & heatmap.ge(threshold)


@torch.inference_mode()
def center_metrics(model, loader, device, radius_px=8.0):
    model.eval()
    true_positive = torch.zeros(len(CLASSES))
    false_positive = torch.zeros(len(CLASSES))
    false_negative = torch.zeros(len(CLASSES))
    localization = []
    for image, target_u8 in loader:
        prediction = model(image.float().to(device).div(255.0)).sigmoid().cpu()
        predicted_peak = peak_mask(prediction)
        true_peak = target_u8.eq(255)
        for batch_index in range(len(image)):
            for class_index in range(len(CLASSES)):
                predicted = torch.nonzero(predicted_peak[batch_index, class_index]).float()
                truth = torch.nonzero(true_peak[batch_index, class_index]).float()
                used = torch.zeros(len(predicted), dtype=torch.bool)
                matched = 0
                for point in truth:
                    if not len(predicted):
                        continue
                    distance = torch.linalg.vector_norm(predicted - point, dim=-1)
                    distance[used] = float("inf")
                    value, index = distance.min(0)
                    if value <= radius_px:
                        used[index] = True
                        matched += 1
                        localization.append(value.item())
                true_positive[class_index] += matched
                false_negative[class_index] += len(truth) - matched
                false_positive[class_index] += len(predicted) - matched
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "macro_f1": 100.0 * f1.mean().item(),
        "macro_precision": 100.0 * precision.mean().item(),
        "macro_recall": 100.0 * recall.mean().item(),
        "mean_localization_error_pixels": (
            sum(localization) / len(localization) if localization else None),
        "per_class_f1": {name: 100.0 * f1[index].item()
                         for index, name in enumerate(CLASSES)},
        "per_class_recall": {name: 100.0 * recall[index].item()
                             for index, name in enumerate(CLASSES)},
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    train_maps = split_maps(args.data_root, "train_seen")
    validation_maps = split_maps(args.data_root, "val_unseen")
    train_images, train_heatmaps = build_patches(
        args.data_root, train_maps, objects, args.train_per_class,
        args.crop_side_m, args.resolution, args.seed, jitter_m=10.0)
    val_images, val_heatmaps = build_patches(
        args.data_root, validation_maps, objects, args.val_per_class,
        args.crop_side_m, args.resolution, args.seed + 1, jitter_m=0.0)
    train_loader = DataLoader(TensorDataset(train_images, train_heatmaps),
                              batch_size=24, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_heatmaps), batch_size=32)
    model = TinyUNet(classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best, best_f1, history = None, -1.0, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for image, target_u8 in train_loader:
            logits = model(image.float().to(device).div(255.0))
            target = target_u8.float().to(device).div(255.0)
            loss = focal_heatmap_loss(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        value = center_metrics(model, val_loader, device)
        value["epoch"] = epoch + 1
        value["train_loss"] = sum(losses) / len(losses)
        history.append(value)
        print(json.dumps({"epoch": epoch + 1, "loss": value["train_loss"],
                          "val_macro_f1": value["macro_f1"]}), flush=True)
        if value["macro_f1"] > best_f1:
            best_f1 = value["macro_f1"]
            best = {key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best, args.output_dir / "model.pt")
    model.load_state_dict(best)
    final = center_metrics(model, val_loader, device)
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
            "target_identity_or_navigation_target_coordinate": "not used",
            "supervision": "centers of all annotated objects in each RGB tile",
        },
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
