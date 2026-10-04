#!/usr/bin/env python3
"""Export the instruction-controlled same-map hard-negative diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--distance-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.distance_metrics.read_text())
    result = {"gate": source["gate"], "representations": {}, "methods": source["methods"], "test_unseen_read": False}
    for representation, components in source["representations"].items():
        result["representations"][representation] = {
            component: {split: {key: metrics[key] for key in ("samples", "positive_negative_margin", "auroc", "same_map_hard_negative_accuracy")}
                        for split, metrics in splits.items()}
            for component, splits in components.items()
        }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__": main()
