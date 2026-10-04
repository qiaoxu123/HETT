#!/usr/bin/env python3
"""Gate-aware entry point for learned/contrastive scene matchers.

The current experiment intentionally records a skipped phase when explicit
scene evidence fails Phase 4.  This prevents a larger matcher from fitting
camera/distance shortcuts after the instruction-specific test has failed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--gate-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    gate = json.loads(args.gate_metrics.read_text())["gate"]
    if not gate["passed"]:
        result = {"status": "skipped", "reason": "phase4_explicit_scene_evidence_gate_failed",
                  "trained": False, "gate": gate, "test_unseen_read": False}
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2)); return
    raise RuntimeError("Gate passed: configure the matcher study in a new supervised run before training")


if __name__ == "__main__": main()
