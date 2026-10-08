#!/usr/bin/env python3
"""Normalize offline metadata/mask orientation after dataset materialization."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ORIENTATION = "hett_yaw0_world_positive_x_maps_to_image_left_v1"


def _jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def _write_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=Path("../artifacts/visual_goal_abstraction/dataset_v2"))
    args = p.parse_args()
    fixed = 0
    for path in (args.dataset_dir / "masks").rglob("*.npz"):
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        if data.get("_orientation_version") is not None and str(data["_orientation_version"].item()) == ORIENTATION:
            continue
        data = {k: np.flip(v, axis=1).copy() if k != "_orientation_version" else v
                for k, v in data.items()}
        data["_orientation_version"] = np.asarray(ORIENTATION)
        np.savez_compressed(path, **data)
        fixed += 1
    metadata_changes = 0
    for path in args.dataset_dir.glob("templates_*.jsonl"):
        rows = _jsonl(path)
        for row in rows:
            if "anchor_ids" in row:
                row.pop("anchor_ids", None)
                metadata_changes += 1
        _write_jsonl(path, rows)
    for path in args.dataset_dir.glob("queries_*.jsonl"):
        rows = _jsonl(path)
        for row in rows:
            if not row.get("annotation_key"):
                stable = f"{row.get('scene_key','')}|{row.get('instruction','').casefold().strip()}"
                row["annotation_key"] = hashlib.sha256(stable.encode()).hexdigest()[:20]
                metadata_changes += 1
        _write_jsonl(path, rows)
    for path in args.dataset_dir.glob("dataset_*_stats.json"):
        stats = json.loads(path.read_text())
        stats["mask_world_to_image_orientation"] = ORIENTATION
        stats["raw_anchor_object_ids_in_candidate_metadata"] = False
        path.write_text(json.dumps(stats, indent=2))
    print(json.dumps({"masks_horizontally_aligned_to_crop_transform": fixed,
                      "metadata_records_sanitized_or_annotated": metadata_changes,
                      "orientation": ORIENTATION}, indent=2))


if __name__ == "__main__":
    main()

