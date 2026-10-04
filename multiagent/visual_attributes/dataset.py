"""Manifest helpers and leakage checks for object-grouped visual attributes."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


ALLOWED_SPLITS = ("train_seen", "val_seen", "val_unseen")


def load_representation(dataset_root: Path, row: dict, representation: str):
    bgr = cv2.imread(str(Path(dataset_root) / row["image"])); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if representation == "whole": return rgb
    polygon = np.asarray(row["contour_uv"], np.int32)
    mask = cv2.fillPoly(np.zeros(rgb.shape[:2], np.uint8), [polygon], 1)
    if representation == "masked":
        result = np.full_like(rgb, 127); result[mask.astype(bool)] = rgb[mask.astype(bool)]; return result
    if representation == "crop":
        x, y, width, height = cv2.boundingRect(polygon); padding = max(4, round(max(width, height) * 0.25))
        x0, y0 = max(0, x - padding), max(0, y - padding); x1, y1 = min(rgb.shape[1], x + width + padding), min(rgb.shape[0], y + height + padding)
        crop = rgb[y0:y1, x0:x1]
        return cv2.resize(crop if crop.size else rgb, rgb.shape[:2][::-1], interpolation=cv2.INTER_AREA)
    raise ValueError(f"unknown representation: {representation}")


def read_manifest(path: Path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if any(row["split"] not in ALLOWED_SPLITS for row in rows):
        raise ValueError("manifest contains a forbidden split")
    return rows


def leakage_report(rows):
    by_split = {split: set() for split in ALLOWED_SPLITS}
    episodes = {split: set() for split in ALLOWED_SPLITS}
    for row in rows:
        by_split[row["split"]].add(row["object_key"])
        episodes[row["split"]].update(row["episode_ids"])
    object_overlap, episode_overlap = {}, {}
    for index, left in enumerate(ALLOWED_SPLITS):
        for right in ALLOWED_SPLITS[index + 1:]:
            object_overlap[f"{left}/{right}"] = len(by_split[left] & by_split[right])
            episode_overlap[f"{left}/{right}"] = len(episodes[left] & episodes[right])
    return {"object_overlap": object_overlap, "episode_overlap": episode_overlap,
            "forbidden_test_unseen": not all(row["split"] != "test_unseen" for row in rows)}


class FeatureDataset(Dataset):
    def __init__(self, payload, split, label_name, representation="crop"):
        self.features = payload["features"][representation]
        self.rows = [index for index, row in enumerate(payload["rows"]) if row["split"] == split and row["labels"].get(label_name) is not None]
        self.label_name = label_name
        self.classes = payload["classes"][label_name]
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}

    def __len__(self): return len(self.rows)

    def __getitem__(self, item):
        index = self.rows[item]; row = self.payload_row(index)
        return self.features[index].float(), self.class_to_index[row["labels"][self.label_name]], index

    def payload_row(self, index):
        # Stored separately to keep tensors compact; attached by loader factory.
        return self._all_rows[index]

    def attach_rows(self, rows):
        self._all_rows = rows; return self
