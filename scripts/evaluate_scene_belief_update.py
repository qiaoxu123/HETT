#!/usr/bin/env python3
"""Gate guard for attaching real scene evidence to Static B0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--gate-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    gate = json.loads(args.gate_metrics.read_text())["gate"]
    if not gate["passed"]:
        result = {"status": "skipped", "reason": "instruction_specific_scene_match_failed",
                  "static_b0_modified": False, "actual_b0_delta": None,
                  "controller_integrated": False, "test_unseen_read": False}
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2)); return
    raise RuntimeError("Gate passed: candidate expected-scene calibration must be tuned on val_seen before B0 update")


if __name__ == "__main__": main()
