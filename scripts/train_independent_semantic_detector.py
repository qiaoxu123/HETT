#!/usr/bin/env python3
"""Train the independent RGB semantic detector without val_unseen selection."""

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
import torch.nn.functional as F
from shapely.geometry import Polygon
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.semantic import CLASSES, TinyUNet


TYPE_TO_INDEX = {name: index for index, name in enumerate(CLASSES) if index}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-side-m", type=float, default=40.0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--selection-per-class", type=int, default=200)
    parser.add_argument("--final-per-class", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def split_maps(data_root, split):
    rows = json.load(open(data_root / "processed_citynav" / f"citynav_{split}.json"))
    return sorted({f'{row["area"]}_block_{row["block"]}' for row in rows})


def centroid(contour):
    value = Polygon(contour)
    if not value.is_valid:
        value = value.buffer(0)
    return value.centroid.x, value.centroid.y


def rasterize_map(data_root, map_name, objects):
    rgb = cv2.imread(str(data_root / "rgbd" / f"{map_name}.png"), cv2.IMREAD_COLOR)
    with rasterio.open(data_root / "rgbd" / f"{map_name}.tif") as raster:
        transform = raster.transform
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for object_type in ("Ground", "Parking", "Building", "Car"):
        for obj in objects[map_name].values():
            if obj.get("object_type") != object_type:
                continue
            contour = np.asarray(obj["contour"], dtype=np.float64)
            rows, cols = rasterio.transform.rowcol(
                transform, contour[:, 0], contour[:, 1])
            cv2.fillPoly(mask, [np.stack((cols, rows), axis=-1).astype(np.int32)],
                         TYPE_TO_INDEX[object_type])
    return rgb, mask, transform


def selected_entries(map_names, objects, per_class, seed):
    grouped = defaultdict(list)
    for map_name in map_names:
        for obj in objects[map_name].values():
            if obj.get("object_type") in TYPE_TO_INDEX:
                grouped[obj["object_type"]].append((map_name, obj))
    rng = random.Random(seed)
    result = []
    for object_type in TYPE_TO_INDEX:
        result.extend(rng.choice(grouped[object_type]) for _ in range(per_class))
    rng.shuffle(result)
    return result


def build_patches(data_root, map_names, objects, per_class, side_m, resolution,
                  seed, jitter_m):
    entries = selected_entries(map_names, objects, per_class, seed)
    by_map = defaultdict(list)
    for index, (map_name, obj) in enumerate(entries):
        by_map[map_name].append((index, obj))
    images = np.empty((len(entries), resolution, resolution, 3), dtype=np.uint8)
    masks = np.empty((len(entries), resolution, resolution), dtype=np.uint8)
    rng = random.Random(seed + 17)
    for map_name, values in sorted(by_map.items()):
        rgb, mask, transform = rasterize_map(data_root, map_name, objects)
        side_pixels = max(8, int(round(side_m / abs(transform.a))))
        for index, obj in values:
            x, y = centroid(obj["contour"])
            x += rng.uniform(-jitter_m, jitter_m)
            y += rng.uniform(-jitter_m, jitter_m)
            row, col = rasterio.transform.rowcol(transform, x, y)
            center = (float(col), float(row))
            image = cv2.getRectSubPix(rgb, (side_pixels, side_pixels), center)
            target = cv2.getRectSubPix(mask, (side_pixels, side_pixels), center)
            images[index] = cv2.resize(
                image, (resolution, resolution), interpolation=cv2.INTER_AREA)
            masks[index] = cv2.resize(
                target, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    return (torch.from_numpy(images[..., ::-1].copy()).permute(0, 3, 1, 2),
            torch.from_numpy(masks).long())


def confusion_matrix(prediction, target, classes=5):
    valid = target.ge(0) & target.lt(classes)
    values = target[valid] * classes + prediction[valid]
    return torch.bincount(values, minlength=classes * classes).reshape(classes, classes)


def metrics(confusion):
    true_positive = confusion.diag().float()
    union = confusion.sum(0) + confusion.sum(1) - true_positive
    iou = true_positive / union.clamp_min(1)
    recall = true_positive / confusion.sum(1).clamp_min(1)
    return {
        "mean_iou_foreground": float(iou[1:].mean().mul(100)),
        "per_class_iou": {name: float(iou[index].mul(100))
                          for index, name in enumerate(CLASSES)},
        "per_class_recall": {name: float(recall[index].mul(100))
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    train_maps = split_maps(args.data_root, "train_seen")
    selection_maps = split_maps(args.data_root, "val_seen")
    final_maps = split_maps(args.data_root, "val_unseen")
    train = build_patches(
        args.data_root, train_maps, objects, args.train_per_class,
        args.crop_side_m, args.resolution, args.seed, 10.0)
    selection = build_patches(
        args.data_root, selection_maps, objects, args.selection_per_class,
        args.crop_side_m, args.resolution, args.seed + 101, 0.0)
    final = build_patches(
        args.data_root, final_maps, objects, args.final_per_class,
        args.crop_side_m, args.resolution, args.seed + 202, 0.0)
    train_loader = DataLoader(TensorDataset(*train), batch_size=24, shuffle=True)
    selection_loader = DataLoader(TensorDataset(*selection), batch_size=32)
    final_loader = DataLoader(TensorDataset(*final), batch_size=32)

    model = TinyUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    weights = torch.tensor([0.2, 1.0, 2.0, 1.0, 1.5], device=device)
    best_state, best_score, best_epoch, history = None, -1.0, None, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for image, target in train_loader:
            logits = model(image.float().to(device).div(255.0))
            loss = F.cross_entropy(logits, target.to(device), weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        selection_value = evaluate(model, selection_loader, device)
        row = {"epoch": epoch + 1, "loss": sum(losses) / len(losses),
               "selection": selection_value}
        history.append(row)
        score = selection_value["mean_iou_foreground"]
        print(json.dumps({"epoch": epoch + 1, "loss": row["loss"],
                          "val_seen_miou": score}), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    final_value = evaluate(model, final_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_dir / "model.pt")
    result = {
        "config": {**vars(args), "data_root": str(args.data_root),
                   "output_dir": str(args.output_dir)},
        "best_epoch_selected_on_val_seen": best_epoch,
        "val_seen_selection": history[best_epoch - 1]["selection"],
        "val_unseen_final": final_value,
        "history": history,
        "anti_leakage": {
            "training": "train_seen maps only",
            "model_selection": "val_seen only",
            "final_evaluation": "val_unseen exactly once after selection",
            "target_coordinates_or_navigation_success": False,
        },
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
