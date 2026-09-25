#!/usr/bin/env python3
"""Report perspective semantic performance by point-cloud valid-pixel ratio."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.semantic import ProjectSemanticNet
from scripts.train_pointcloud_semantic_adaptation import PerspectiveDataset, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.names):
        parser.error("--checkpoints and --names must have equal length")
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["split"] == args.split]
    bins = {
        "sparse_lt_30pct": [row for row in rows if row["valid_fraction"] < 0.30],
        "medium_30_to_60pct": [row for row in rows
                                if 0.30 <= row["valid_fraction"] < 0.60],
        "dense_ge_60pct": [row for row in rows if row["valid_fraction"] >= 0.60],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"split": args.split, "bins": {}}
    for bin_name, values in bins.items():
        entry = {"frames": len(values), "models": {}}
        if values:
            loader = DataLoader(PerspectiveDataset(args.dataset, values), batch_size=16)
            for name, checkpoint in zip(args.names, args.checkpoints):
                model = ProjectSemanticNet().to(device)
                model.load_state_dict(torch.load(
                    checkpoint, map_location="cpu", weights_only=True))
                entry["models"][name] = evaluate(model, loader, device)
        result["bins"][bin_name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
