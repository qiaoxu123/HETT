#!/usr/bin/env python3
"""Train a controlled RGB + named-landmark-mask spatial relation probe."""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.mapdata import MAP_BOUNDS
from scripts.audit_spatial_reference_frames import build_audit_examples, normalize_xy
from scripts.audit_spatial_relation_probe import encode_prompts
from scripts.train_rgb_candidate_ranker import extract_crop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path,
                        default=Path("runs/visual_spatial_relation_probe/features_v2.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/visual_spatial_relation_probe/results.json"))
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--minimum-axis-m", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", default="0,1,2")
    return parser.parse_args()


def landmark_mask(contours, map_name, resolution):
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    for contour in contours:
        normalized = np.stack([normalize_xy(point, map_name) for point in contour])
        pixels = np.rint(normalized * (resolution - 1)).astype(np.int32)
        if len(pixels) >= 3:
            cv2.fillPoly(mask, [pixels], 255)
    return torch.from_numpy(mask)


def symmetric_candidates(center, delta):
    center = np.asarray(center, dtype=np.float32)
    dx, dy = float(delta[0]), float(delta[1])
    offsets = np.array([[dx, dy], [-dx, dy], [dx, -dy], [-dx, -dy]],
                       dtype=np.float32)
    return center[None] + offsets


CUE_TO_INDEX = {"left": 0, "right": 1, "above": 2, "below": 3}


def candidate_boundary_directions(candidates, contours, map_name):
    segments = []
    for contour in contours:
        points = np.stack([normalize_xy(point, map_name) for point in contour])
        closed = np.concatenate((points, points[:1]), axis=0)
        segments.extend(zip(closed[:-1], closed[1:]))
    result = []
    for candidate in candidates:
        projections = []
        for start, end in segments:
            edge = end - start
            fraction = np.dot(candidate - start, edge) / max(np.dot(edge, edge), 1e-9)
            projections.append(start + np.clip(fraction, 0.0, 1.0) * edge)
        projections = np.stack(projections)
        nearest = projections[np.linalg.norm(projections - candidate[None], axis=-1).argmin()]
        delta = candidate - nearest
        result.append(delta / max(np.linalg.norm(delta), 1e-6))
    return np.stack(result).astype(np.float32)


def normalized_map_rgb(data_root, map_name, resolution):
    bounds = MAP_BOUNDS[map_name]
    center = (bounds.x_min + 205.0, bounds.y_max - 205.0)
    rgb = cv2.imread(str(data_root / "rgbd" / f"{map_name}.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(map_name)
    with rasterio.open(data_root / "rgbd" / f"{map_name}.tif") as raster:
        transform = raster.transform
    image = extract_crop(rgb, transform, center, 410.0, out_size=resolution)
    return torch.from_numpy(image.copy()).permute(2, 0, 1)


def same_map_shift(maps):
    grouped = {}
    for index, map_name in enumerate(maps):
        grouped.setdefault(map_name, []).append(index)
    shifted = []
    for index, map_name in enumerate(maps):
        group = grouped[map_name]
        position = group.index(index)
        shifted.append(group[(position + 1) % len(group)])
    return torch.tensor(shifted, dtype=torch.long)


def build_features(data_root, resolution, minimum_axis_m, device):
    examples = {}
    prompts = []
    map_names = set()
    for split in ("train_seen", "val_seen", "val_unseen"):
        rows = build_audit_examples(data_root, split, minimum_axis_m)
        valid = []
        for row in rows:
            candidates = symmetric_candidates(row["center"], row["centroid_delta"])
            if ((candidates >= 0.0) & (candidates <= 1.0)).all():
                row["candidates"] = candidates
                valid.append(row)
                map_names.add(row["map_name"])
                prompts.append(row["prompt"])
        examples[split] = valid
        print(f"{split}: {len(valid)} in-bounds visual relation groups", flush=True)
    encoded = encode_prompts(prompts, device)
    map_rgb = {name: normalized_map_rgb(data_root, name, resolution)
               for name in sorted(map_names)}
    features = {}
    for split, rows in examples.items():
        maps = [row["map_name"] for row in rows]
        features[split] = {
            "text": torch.stack([encoded[row["prompt"]] for row in rows]).half(),
            "rgb": torch.stack([map_rgb[row["map_name"]] for row in rows]),
            "mask": torch.stack([landmark_mask(
                row["landmark_contours"], row["map_name"], resolution) for row in rows]),
            "candidates": torch.tensor(
                np.stack([row["candidates"] for row in rows]), dtype=torch.float32),
            "directions": torch.tensor(np.stack([
                candidate_boundary_directions(row["candidates"],
                                              row["landmark_contours"],
                                              row["map_name"])
                for row in rows]), dtype=torch.float32),
            "cue": torch.tensor([CUE_TO_INDEX[row["cue"]] for row in rows],
                                dtype=torch.long),
            "wrong_index": same_map_shift(maps),
            "maps": maps,
        }
    return features


class VisualSpatialProbe(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        axis = torch.linspace(0.0, 1.0, resolution)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("pixel_grid", torch.stack((xx, yy), dim=-1))
        # Selected on train_seen only. It maps the instruction frame to the
        # north-up map frame and is then frozen for every validation split.
        self.register_buffer("cue_vectors", torch.tensor([
            [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]
        ]))
        self.visual = nn.Sequential(
            nn.Conv2d(5, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 64))
        self.text = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128), nn.SiLU(),
                                  nn.Linear(128, 64))
        self.score = nn.Sequential(nn.Linear(129, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, text, rgb, mask, candidates, directions, cue,
                use_rgb=True, use_mask=True, use_relation=True):
        batch, count = candidates.shape[:2]
        rgb = rgb.float().div(255.0)
        if not use_rgb:
            rgb = torch.zeros_like(rgb)
        mask = mask.float().div(255.0)
        if not use_mask:
            mask = torch.zeros_like(mask)
        rgb = rgb[:, None].expand(-1, count, -1, -1, -1)
        mask = mask[:, None, None].expand(-1, count, -1, -1, -1)
        delta = self.pixel_grid[None, None] - candidates[:, :, None, None]
        marker = torch.exp(-delta.square().sum(-1) / (2.0 * 0.018 ** 2))[:, :, None]
        image = torch.cat((rgb, mask, marker), dim=2).flatten(0, 1)
        visual = F.normalize(self.visual(image), dim=-1).reshape(batch, count, -1)
        text = F.normalize(self.text(text.float()), dim=-1)[:, None].expand_as(visual)
        expected = self.cue_vectors[cue][:, None]
        alignment = (directions * expected).sum(-1, keepdim=True)
        if not use_relation:
            alignment = torch.zeros_like(alignment)
        return self.score(torch.cat((visual * text, torch.abs(visual - text), alignment),
                                    dim=-1)).squeeze(-1)


def loader(data, batch_size, shuffle):
    return DataLoader(TensorDataset(
        data["text"], data["rgb"], data["mask"], data["candidates"],
        data["directions"], data["cue"], data["wrong_index"]),
        batch_size=batch_size, shuffle=shuffle)


@torch.inference_mode()
def evaluate(model, data, device, train_mode):
    logits, wrong_text, wrong_mask, wrong_cue = [], [], [], []
    no_rgb, no_mask, no_relation = [], [], []
    rgb_enabled = train_mode == "rgb_mask"
    for text, rgb, mask, candidates, directions, cue, wrong_index in loader(data, 64, False):
        text, rgb = text.to(device), rgb.to(device)
        mask, candidates = mask.to(device), candidates.to(device)
        directions, cue = directions.to(device), cue.to(device)
        logits.append(model(text, rgb, mask, candidates, directions, cue,
                            use_rgb=rgb_enabled).cpu())
        wrong_text.append(model(data["text"][wrong_index].to(device), rgb, mask,
                                candidates, directions, cue,
                                use_rgb=rgb_enabled).cpu())
        wrong_mask.append(model(text, rgb, data["mask"][wrong_index].to(device),
                                candidates, directions, cue,
                                use_rgb=rgb_enabled).cpu())
        wrong_cue.append(model(text, rgb, mask, candidates, directions,
                               data["cue"][wrong_index].to(device),
                               use_rgb=rgb_enabled).cpu())
        no_rgb.append(model(text, rgb, mask, candidates, directions, cue,
                            use_rgb=False).cpu())
        no_mask.append(model(text, rgb, mask, candidates, directions, cue,
                             use_rgb=rgb_enabled, use_mask=False).cpu())
        no_relation.append(model(text, rgb, mask, candidates, directions, cue,
                                 use_rgb=rgb_enabled, use_relation=False).cpu())

    def accuracy(values):
        values = torch.cat(values)
        return values.argmax(-1).eq(0).float().mean().mul(100).item()

    return {
        "accuracy": accuracy(logits),
        "same_map_wrong_text_accuracy": accuracy(wrong_text),
        "same_map_wrong_landmark_mask_accuracy": accuracy(wrong_mask),
        "same_map_wrong_relation_cue_accuracy": accuracy(wrong_cue),
        "no_rgb_accuracy": accuracy(no_rgb),
        "no_landmark_mask_accuracy": accuracy(no_mask),
        "no_explicit_relation_accuracy": accuracy(no_relation),
    }


def train_one(train, val, resolution, mode, seed, epochs, device):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = VisualSpatialProbe(resolution).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    best_state, best_accuracy = None, -1.0
    for _ in range(epochs):
        model.train()
        for text, rgb, mask, candidates, directions, cue, _ in loader(train, 32, True):
            logits = model(text.to(device), rgb.to(device), mask.to(device),
                           candidates.to(device), directions.to(device), cue.to(device),
                           use_rgb=mode == "rgb_mask")
            loss = F.cross_entropy(
                logits, torch.zeros(len(logits), dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        accuracy = evaluate(model, val, device, mode)["accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


def summarize(runs, key):
    values = [run[key] for run in runs]
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cache.exists():
        features = torch.load(args.cache, map_location="cpu", weights_only=False)
    else:
        features = build_features(
            args.data_root, args.resolution, args.minimum_axis_m, device)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, args.cache)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    result = {"config": {"resolution": args.resolution, "seeds": seeds,
                         "chance_accuracy": 25.0,
                         "selection": "val_seen accuracy"},
              "counts": {split: len(data["text"]) for split, data in features.items()},
              "runs": {"mask_only": [], "rgb_mask": []}}
    for mode in ("mask_only", "rgb_mask"):
        for seed in seeds:
            model = train_one(features["train_seen"], features["val_seen"],
                              args.resolution, mode, seed, args.epochs, device)
            result["runs"][mode].append({
                "seed": seed,
                "val_seen": evaluate(model, features["val_seen"], device, mode),
                "val_unseen": evaluate(model, features["val_unseen"], device, mode),
            })
            del model
            torch.cuda.empty_cache()
    result["summary"] = {}
    for mode, runs in result["runs"].items():
        unseen = [run["val_unseen"] for run in runs]
        result["summary"][mode] = {key: summarize(unseen, key) for key in (
            "accuracy", "same_map_wrong_text_accuracy",
            "same_map_wrong_landmark_mask_accuracy",
            "same_map_wrong_relation_cue_accuracy", "no_rgb_accuracy",
            "no_landmark_mask_accuracy", "no_explicit_relation_accuracy")}
    result["anti_leakage"] = {
        "train": "train_seen maps", "selection": "val_seen",
        "test": "map-disjoint val_unseen",
        "target_input": False,
        "target_use": "constructing the supervised true/mirrored candidate set only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
