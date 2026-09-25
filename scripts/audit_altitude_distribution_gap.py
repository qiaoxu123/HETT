#!/usr/bin/env python3
"""Audit the low-altitude train/evaluation exposure gap.

This is a read-only aggregate diagnostic. It reads one epoch from HETT's
``epoch_metrics.jsonl`` and compares the oracle-controlled training exposure
with a validation split controlled from model predictions. It does not run a
rollout, load a checkpoint, or mutate experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_epoch(path: Path, epoch: int) -> dict[str, Any]:
    """Return exactly one requested epoch from a JSONL metrics file."""
    matches: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if record.get("epoch") == epoch:
                matches.append(record)

    if not matches:
        raise ValueError(f"epoch {epoch} is absent from {path}")
    if len(matches) > 1:
        raise ValueError(f"epoch {epoch} occurs {len(matches)} times in {path}")
    return matches[0]


def summarize_gap(record: dict[str, Any], split: str) -> dict[str, float | bool]:
    """Calculate comparable percentages and a conservative mismatch flag."""
    validation = record.get("validation", {})
    if split not in validation:
        raise ValueError(f"validation split {split!r} is absent")

    train_fraction = float(record["train_low_altitude_fraction"])
    # Training stores a [0, 1] fraction; validation metrics store a percentage.
    train_percent = train_fraction * 100.0
    validation_percent = float(validation[split]["low_altitude_fraction"])
    train_changes = float(record["train_altitude_changes"])
    validation_changes = float(validation[split]["altitude_changes"])
    ratio = validation_percent / train_percent if train_percent > 0.0 else float("inf")

    return {
        "train_low_altitude_percent": train_percent,
        "validation_low_altitude_percent": validation_percent,
        "percentage_point_gap": validation_percent - train_percent,
        "validation_to_train_ratio": ratio,
        "train_altitude_changes_per_trajectory": train_changes,
        "validation_altitude_changes_per_trajectory": validation_changes,
        "distribution_mismatch": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument(
        "--max-ratio", type=float, default=2.0,
        help="largest acceptable validation/train low-altitude ratio",
    )
    parser.add_argument(
        "--max-point-gap", type=float, default=10.0,
        help="largest acceptable absolute percentage-point gap",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_gap(load_epoch(args.metrics, args.epoch), args.split)
    ratio = float(summary["validation_to_train_ratio"])
    point_gap = abs(float(summary["percentage_point_gap"]))
    summary["distribution_mismatch"] = (
        ratio > args.max_ratio or point_gap > args.max_point_gap
    )
    summary["epoch"] = args.epoch
    summary["split"] = args.split
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Non-zero status makes the audit usable as a future experiment gate.
    return 2 if summary["distribution_mismatch"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
