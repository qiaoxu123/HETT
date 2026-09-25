#!/usr/bin/env python3
"""Render representative unseen-map type predictions at 40 m and 160 m."""

import argparse
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import (
    CLASSES,
    evaluate_probe,
    load_examples,
    train_probe,
)
from scripts.train_rgb_candidate_ranker import extract_crop
from scripts.validate_candidate_geometry import polygon_centroid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--base-cache", type=Path,
                        default=Path("runs/target_type_alignment/features.pt"))
    parser.add_argument("--scale-40-cache", type=Path,
                        default=Path("runs/target_type_alignment/multiscale/scale_40.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/target_type_alignment/type_matching_examples.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    objects, examples = load_examples(args.data_root)
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True)
    scale40 = torch.load(args.scale_40_cache, map_location="cpu", weights_only=True)["target"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model40 = train_probe(scale40["train_seen"], scale40["val_seen"], 0, 40, device)
    model160 = train_probe(base["image"]["train_seen"], base["image"]["val_seen"],
                           0, 40, device)
    _, pred40 = evaluate_probe(model40, scale40["val_unseen"], device)
    _, pred160 = evaluate_probe(model160, base["image"]["val_unseen"], device)
    labels = scale40["val_unseen"]["labels"]
    keys = scale40["val_unseen"]["keys"]
    first_text = {}
    for key, text in zip(examples["val_unseen"]["text_object_keys"],
                         examples["val_unseen"]["text"]):
        first_text.setdefault(key, text)

    selected = []
    for class_index in range(len(CLASSES)):
        preferred = [index for index in range(len(keys))
                     if labels[index] == class_index and pred40[index] == class_index
                     and pred160[index] != class_index]
        fallback = [index for index in range(len(keys))
                    if labels[index] == class_index and pred40[index] == class_index]
        selected.append((preferred or fallback)[0])

    figure, axes = plt.subplots(len(CLASSES), 3, figsize=(13, 12),
                                gridspec_kw={"width_ratios": [1, 1, 1.6]})
    for row, index in enumerate(selected):
        map_name, object_id = keys[index].rsplit(":", 1)
        obj = objects[map_name][object_id]
        rgb = cv2.imread(str(args.data_root / "rgbd" / f"{map_name}.png"),
                         cv2.IMREAD_COLOR)
        with rasterio.open(args.data_root / "rgbd" / f"{map_name}.tif") as raster:
            transform = raster.transform
        center = polygon_centroid(obj["contour"])
        image40 = extract_crop(rgb, transform, center, 40.0)
        image160 = extract_crop(rgb, transform, center, 160.0)
        truth = CLASSES[labels[index].item()]
        p40 = CLASSES[pred40[index].item()]
        p160 = CLASSES[pred160[index].item()]
        axes[row, 0].imshow(image40)
        axes[row, 0].set_title(f"40 m: {p40}  ✓" if p40 == truth else f"40 m: {p40}  ✗")
        axes[row, 1].imshow(image160)
        axes[row, 1].set_title(f"160 m: {p160}  ✓" if p160 == truth else f"160 m: {p160}  ✗")
        axes[row, 2].axis("off")
        axes[row, 2].text(
            0.0, 0.82, f"Target type: {truth}\n\n"
            + textwrap.fill(first_text[keys[index]], width=46),
            va="top", fontsize=10)
        for column in (0, 1):
            axes[row, column].axis("off")
    figure.suptitle("Unseen maps: instruction target type and RGB scale", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
