#!/usr/bin/env python3
"""Concatenate the resolution-control shards, de-duplicating by sample.

Sharded workers each skip what was already on disk when they started, but a
worker that started earlier cannot see what a later one adds, so the union can
contain a sample twice.  First occurrence wins; the records are deterministic,
so any copy is as good as another.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parent.parent / "artifacts" / "resolution_control"
out = root / "resolution_control.jsonl"
seen, records = set(), []
for path in sorted(root.glob("resolution_control*.jsonl")):
    if path.name.endswith(".merged.jsonl"):
        continue
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (rec["split"], rec["episode_index"], rec["step"])
        if key in seen:
            continue
        seen.add(key)
        records.append(line)
out.write_text("\n".join(records) + "\n")
print(f"merged {len(records)} unique samples from "
      f"{len(list(root.glob('resolution_control*.jsonl')))} files -> {out}")
