#!/usr/bin/env python3
"""Report valid view groups only; this dataset currently supplies ortho RGB."""
import json
from pathlib import Path

if __name__ == "__main__":
    stats = {}
    for split in ("val_seen", "val_unseen"):
        p = Path(f"../artifacts/visual_goal_abstraction/dataset/dataset_{split}_stats.json")
        stats[split] = json.loads(p.read_text()) if p.exists() else None
    out = Path("../artifacts/visual_goal_abstraction/eval/cross_view_availability.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "no_oblique_or_fpv_renderer", "splits": stats}, indent=2))
    print(out)

