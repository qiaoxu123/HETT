#!/usr/bin/env python3
"""Summarize stop behavior from one saved HETT prediction artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--success-dist", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=False)
    rows = list(predictions.values())
    summary = {
        "episodes": len(rows),
        "final_success": 0,
        "oracle_success": 0,
        "reached_then_left": 0,
        "progress_stops": 0,
        "progress_stop_success": 0,
        "horizon_stops": 0,
    }
    progress_values = []
    for row in rows:
        goal = row["goal"]
        distances = [pose.xy.dist_to(goal) for pose in row["trajectory"]]
        final_success = distances[-1] <= args.success_dist
        oracle_success = min(distances) <= args.success_dist
        reason = row.get("stop_reason", "unknown")
        summary["final_success"] += int(final_success)
        summary["oracle_success"] += int(oracle_success)
        summary["reached_then_left"] += int(oracle_success and not final_success)
        summary["progress_stops"] += int(reason == "progress")
        summary["progress_stop_success"] += int(reason == "progress" and final_success)
        summary["horizon_stops"] += int(reason == "horizon")
        if row.get("stop_progress") is not None:
            progress_values.append(float(row["stop_progress"]))

    count = max(len(rows), 1)
    result = {
        **summary,
        "final_sr": 100.0 * summary["final_success"] / count,
        "oracle_sr": 100.0 * summary["oracle_success"] / count,
        "reached_then_left_percent": 100.0 * summary["reached_then_left"] / count,
        "progress_stop_precision": (
            100.0 * summary["progress_stop_success"] / summary["progress_stops"]
            if summary["progress_stops"] else 0.0
        ),
        "mean_progress_at_stop": (
            sum(progress_values) / len(progress_values) if progress_values else None
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
