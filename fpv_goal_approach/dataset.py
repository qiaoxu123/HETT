from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class FPVGoalApproachDataset(Dataset):
    def __init__(self, metadata_path: Path, processor=None, frames: int = 4, use_language: bool = True, use_pose: bool = False):
        self.metadata_path = Path(metadata_path)
        self.root = self.metadata_path.parents[1]
        self.rows = read_jsonl(self.metadata_path)
        self.processor = processor
        self.frames = frames
        self.use_language = use_language
        self.use_pose = use_pose

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        paths = row["frames"][-self.frames:]
        images = [Image.open(self.root / path).convert("RGB") for path in paths]
        while len(images) < self.frames:
            images.insert(0, images[0].copy())
        item = {
            "images": images,
            "instruction": row["instruction"] if self.use_language else "",
            "pose": torch.tensor(row["poses"][-1], dtype=torch.float32),
            "target_match": torch.tensor(row["target_match"], dtype=torch.long),
            "arrival": torch.tensor(row["visual_confirmed_arrival"], dtype=torch.long),
            "distance_bin": torch.tensor(row["distance_bin"], dtype=torch.long),
            "bearing_bin": torch.tensor(row["bearing_bin"], dtype=torch.long),
            "phase": torch.tensor(row["phase"], dtype=torch.long),
            "metadata": row,
        }
        return item


def collate_samples(samples, processor):
    flat_images = [image for sample in samples for image in sample["images"]]
    image_inputs = processor(images=flat_images, return_tensors="pt")
    texts = [sample["instruction"] or "an aerial city view" for sample in samples]
    text_inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt")
    batch = {
        "pixel_values": image_inputs["pixel_values"].reshape(len(samples), -1, *image_inputs["pixel_values"].shape[1:]),
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs.get("attention_mask", torch.ones_like(text_inputs["input_ids"])),
        "pose": torch.stack([sample["pose"] for sample in samples]),
    }
    for key in ("target_match", "arrival", "distance_bin", "bearing_bin", "phase"):
        batch[key] = torch.stack([sample[key] for sample in samples])
    batch["metadata"] = [sample["metadata"] for sample in samples]
    return batch
