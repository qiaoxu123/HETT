#!/usr/bin/env python3
"""Phase-1 hand-crafted probes; later encoders use the same metric protocol."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.models.attribute_baseline import AttributeProbe  # noqa: E402
from multiagent.visual_attributes.metrics import classification_metrics, grouped_consistency  # noqa: E402


def evaluate(model, x, y, rows, indices, device):
    model.eval()
    with torch.inference_mode(): probabilities = torch.softmax(model(x.to(device)).float(), -1).cpu().numpy()
    metrics = classification_metrics(probabilities, y.numpy())
    metrics["height"] = {}
    for height in sorted({rows[index]["height_m"] for index in indices}):
        selected = np.asarray([rows[index]["height_m"] == height for index in indices])
        metrics["height"][str(height)] = classification_metrics(probabilities[selected], y.numpy()[selected])
    metrics["distance"] = {}
    distance_buckets = (("0-20m", 0, 20), ("20-40m", 20, 40), ("40-80m", 40, 80), ("80m+", 80, float("inf")))
    for name, lower, upper in distance_buckets:
        selected = np.asarray([lower <= rows[index].get("distance_m", 0) < upper for index in indices])
        if selected.any(): metrics["distance"][name] = classification_metrics(probabilities[selected], y.numpy()[selected])
    metrics["cross_height"] = grouped_consistency(probabilities, [rows[index]["object_key"] for index in indices], [rows[index]["height_m"] for index in indices])
    metrics["cross_distance"] = grouped_consistency(
        probabilities, [rows[index]["object_key"] for index in indices],
        [rows[index].get("distance_m", 0) for index in indices],
    )
    metrics["cross_observation"] = grouped_consistency(
        probabilities, [rows[index]["object_key"] for index in indices],
        [(rows[index].get("view", "unknown"), rows[index]["height_m"], rows[index].get("distance_m", 0)) for index in indices],
    )
    grouped = {}
    for position, index in enumerate(indices): grouped.setdefault(rows[index]["object_key"], []).append(position)
    mean_probabilities = np.stack([probabilities[position].mean(0) for position in grouped.values()])
    confidence_probabilities = np.stack([
        probabilities[position][np.argmax(probabilities[position].max(1))] for position in grouped.values()
    ])
    grouped_targets = np.asarray([y.numpy()[position[0]] for position in grouped.values()])
    oracle_probabilities = []
    for positions, target in zip(grouped.values(), grouped_targets):
        correct = [position for position in positions if probabilities[position].argmax() == target]
        pool = correct if correct else positions
        oracle_probabilities.append(probabilities[max(pool, key=lambda position: probabilities[position].max())])
    oracle_probabilities = np.stack(oracle_probabilities)
    metrics["multi_view_mean"] = classification_metrics(mean_probabilities, grouped_targets)
    metrics["multi_view_max_confidence"] = classification_metrics(confidence_probabilities, grouped_targets)
    metrics["oracle_active_view"] = classification_metrics(oracle_probabilities, grouped_targets)
    return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path)
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--representations", nargs="+", default=("whole", "crop", "masked")); parser.add_argument("--labels", nargs="+", default=("color", "size", "shape", "semantic", "road_context", "roof_presence"))
    parser.add_argument("--probe-types", nargs="+", choices=("linear", "mlp"), default=("linear", "mlp"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.feature_file or args.dataset / "handcrafted_features.pt", map_location="cpu", weights_only=False); rows = payload["rows"]
    result = {"config": vars(args) | {"dataset": str(args.dataset), "output": str(args.output), "device": str(device)}, "models": {}}
    for representation in args.representations:
        all_x = payload["features"][representation].float(); train_rows = [i for i, row in enumerate(rows) if row["split"] == "train_seen"]
        mean = all_x[train_rows].mean(0); std = all_x[train_rows].std(0).clamp_min(1e-5); all_x = (all_x - mean) / std
        for label in args.labels:
            classes = payload["classes"][label]; class_to_index = {name: i for i, name in enumerate(classes)}
            split_data = {}
            for split in ("train_seen", "val_seen", "val_unseen"):
                indices = [i for i, row in enumerate(rows) if row["split"] == split and row["labels"].get(label) in class_to_index]
                split_data[split] = (all_x[indices], torch.tensor([class_to_index[rows[i]["labels"][label]] for i in indices]), indices)
            train_x, train_y, _ = split_data["train_seen"]; loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
            counts = torch.bincount(train_y, minlength=len(classes)).float(); weights = counts.sum() / counts.clamp_min(1) / len(classes)
            for probe_type in args.probe_types:
                torch.manual_seed(args.seed)
                model = AttributeProbe(all_x.shape[1], len(classes), 128 if probe_type == "mlp" else 0).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3); started = time.time()
                for _ in range(args.epochs):
                    model.train()
                    for x, y in loader:
                        optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(model(x.to(device)), y.to(device), weight=weights.to(device)); loss.backward(); optimizer.step()
                encoder_name = payload.get("encoder", "handcrafted")
                key = f"{encoder_name}_{probe_type}/{representation}/{label}"; entry = {"parameters": sum(p.numel() for p in model.parameters()), "train_seconds": time.time() - started, "classes": classes}
                for split in ("val_seen", "val_unseen"):
                    x, y, indices = split_data[split]; entry[split] = evaluate(model, x, y, rows, indices, device)
                result["models"][key] = entry
                (args.output / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
                print(key, entry["val_unseen"]["macro_f1"], flush=True)


if __name__ == "__main__": main()
