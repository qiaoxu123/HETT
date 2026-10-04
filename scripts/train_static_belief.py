#!/usr/bin/env python3
"""Train/evaluate B0 without RGB, depth, rollout, controller, or ET stages."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiagent.models.static_belief import StaticBeliefModel  # noqa: E402
from multiagent.navigation_state import nms_topk_from_belief  # noqa: E402
from multiagent.static_belief_dataset import (  # noqa: E402
    VARIANTS, StaticBeliefDataset, StaticMapRasterizer, load_static_samples,
)
from multiagent.static_belief_metrics import belief_entropy  # noqa: E402


KS = (1, 4, 8, 16)
RADII_M = (20.0, 40.0)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def segment_tokens(hidden, attention_mask, segments=4):
    """Compress BERT tokens into a few ordered phrase tokens for a small cache."""
    rows = []
    for values, mask in zip(hidden, attention_mask):
        values = values[mask.bool()]
        boundaries = torch.linspace(0, len(values), segments + 1, device=values.device).round().long()
        pooled = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            pooled.append(values[start:max(int(end), int(start) + 1)].mean(0))
        rows.append(torch.stack(pooled))
    return torch.stack(rows)


def encode_languages(texts, output, device, batch_size=128):
    cache = output / "language_features.pt"
    ordered = sorted(set(texts))
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved.get("texts") == ordered:
            features = saved["features"]
            return {text: features[index] for index, text in enumerate(ordered)}
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    encoder = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True).to(device).eval()
    encoded = []
    for start in range(0, len(ordered), batch_size):
        batch = tokenizer(
            ordered[start:start + batch_size], padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = encoder(**batch).last_hidden_state
            pooled = segment_tokens(hidden, batch["attention_mask"], 4)
        encoded.append(pooled.to(dtype=torch.float16, device="cpu"))
    features = torch.cat(encoded)
    torch.save({"texts": ordered, "features": features}, cache)
    del encoder
    if device.type == "cuda": torch.cuda.empty_cache()
    return {text: features[index] for index, text in enumerate(ordered)}


def gaussian_targets(goal_row_col, height, width, map_size, sigma_cells, device):
    goal = goal_row_col.to(device)
    goal[:, 0] = goal[:, 0].mul(height / map_size).clamp(0, height - 1)
    goal[:, 1] = goal[:, 1].mul(width / map_size).clamp(0, width - 1)
    rows = torch.arange(height, device=device).view(1, height, 1)
    cols = torch.arange(width, device=device).view(1, 1, width)
    target = torch.exp(-((rows - goal[:, 0, None, None]) ** 2 + (cols - goal[:, 1, None, None]) ** 2) / (2 * sigma_cells ** 2))
    return target / target.sum((1, 2), keepdim=True).clamp_min(1e-8), goal


def make_loader(samples, rasterizer, language, variant, args, shuffle):
    dataset = StaticBeliefDataset(samples, rasterizer, language, variant, args.map_size)
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(), generator=generator if shuffle else None,
    )


def train_epoch(model, loader, optimizer, device, args):
    model.train(); losses = []
    for batch in loader:
        maps = batch["static_map"].to(device, non_blocking=True)
        language = batch["language"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(maps, language)
            cell_m = args.map_meters / output.logits.shape[-1]
            targets, _ = gaussian_targets(
                batch["goal_row_col"], *output.logits.shape[-2:], args.map_size,
                args.sigma_m / cell_m, device,
            )
            log_probs = torch.log_softmax(output.logits.flatten(1), dim=-1)
            loss = -(targets.flatten(1) * log_probs).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def evaluate(model, loader, device, args):
    model.eval(); losses, entropies, top1 = [], [], []
    oracle = {k: [] for k in KS}
    recall = {(k, radius): [] for k in KS for radius in RADII_M}
    for batch in loader:
        maps = batch["static_map"].to(device, non_blocking=True)
        language = batch["language"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(maps, language)
        probabilities = output.probabilities.float()
        cell_m = args.map_meters / probabilities.shape[-1]
        targets, goals = gaussian_targets(
            batch["goal_row_col"], *probabilities.shape[-2:], args.map_size,
            args.sigma_m / cell_m, device,
        )
        losses.extend((-(targets.flatten(1) * probabilities.flatten(1).clamp_min(1e-8).log()).sum(-1)).cpu().tolist())
        entropies.extend(belief_entropy(probabilities).cpu().tolist())
        for belief, goal in zip(probabilities.cpu(), goals.cpu().numpy()):
            candidates = nms_topk_from_belief(belief, top_k=max(KS), kernel_size=args.nms_kernel)
            distances = np.asarray([math.hypot(item.row - goal[0], item.col - goal[1]) * cell_m for item in candidates])
            top1.append(float(distances[0]))
            for k in KS:
                value = float(distances[:k].min())
                oracle[k].append(value)
                for radius in RADII_M:
                    recall[k, radius].append(float(value <= radius))
    result = {
        "samples": len(top1), "loss": float(np.mean(losses)),
        "top1_peak_distance_m": float(np.mean(top1)),
        "top1_peak_distance_median_m": float(np.median(top1)),
        "belief_entropy": float(np.mean(entropies)),
        "belief_entropy_normalized": float(np.mean(entropies) / math.log(probabilities[0].numel())),
    }
    for k in KS:
        result[f"oracle_distance@{k}_m"] = float(np.mean(oracle[k]))
        for radius in RADII_M:
            result[f"recall@{k}/{int(radius)}m"] = float(np.mean(recall[k, radius]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--map-size", type=int, default=240)
    parser.add_argument("--map-meters", type=float, default=410.0)
    parser.add_argument("--sigma-m", type=float, default=20.0)
    parser.add_argument("--nms-kernel", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {
        split: load_static_samples(args.data_root, split, args.map_size, args.map_meters)
        for split in ("train_seen", "val_seen", "val_unseen")
    }
    if args.max_train_samples: splits["train_seen"] = splits["train_seen"][:args.max_train_samples]
    if args.max_val_samples:
        splits["val_seen"] = splits["val_seen"][:args.max_val_samples]
        splits["val_unseen"] = splits["val_unseen"][:args.max_val_samples]
    language = encode_languages(
        [sample.instruction for values in splits.values() for sample in values],
        args.output, device,
    )
    rasterizer = StaticMapRasterizer(args.data_root, args.map_size, args.map_meters)
    report = {"config": vars(args) | {"data_root": str(args.data_root), "output": str(args.output), "device": str(device)}, "variants": {}}
    for variant in args.variants:
        set_seed(args.seed)
        model = StaticBeliefModel(
            input_channels=5, language_dim=768, hidden_dim=args.hidden_dim,
            attention_heads=4,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        train_loader = make_loader(splits["train_seen"], rasterizer, language, variant, args, True)
        val_loaders = {
            split: make_loader(splits[split], rasterizer, language, variant, args, False)
            for split in ("val_seen", "val_unseen")
        }
        history = []
        for epoch in range(1, args.epochs + 1):
            row = {"epoch": epoch, "train_loss": train_epoch(model, train_loader, optimizer, device, args)}
            for split, loader in val_loaders.items(): row[split] = evaluate(model, loader, device, args)
            history.append(row)
            report["variants"][variant] = history
            write_json(args.output / "metrics.json", report)
            print(json.dumps({"variant": variant, **row}, ensure_ascii=False), flush=True)
        torch.save({"model": model.state_dict(), "variant": variant, "config": report["config"]}, args.output / f"{variant}.pt")
    write_json(args.output / "metrics.json", report)


if __name__ == "__main__":
    main()
