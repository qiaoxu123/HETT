#!/usr/bin/env python3
"""Add inference-visible landmark geometry to existing candidate JSONL files."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_candidate_geometry import build_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args()


def enrich(candidate_rows, source_rows):
    if len(candidate_rows) != len(source_rows):
        raise ValueError(f"row count mismatch: {len(candidate_rows)} != {len(source_rows)}")
    enriched = []
    for index, (candidate, source) in enumerate(zip(candidate_rows, source_rows)):
        if (candidate["map_name"] != source["map_name"] or
                candidate["description"] != source["description"] or
                not np.allclose(candidate["target"], source["target"], atol=1e-6)):
            raise ValueError(f"candidate/source mismatch at row {index}")
        enriched.append({
            **candidate,
            "landmark_names": source["names"],
            "relation_queries": source["relation_queries"],
            "landmark_centers": [value.tolist() for value in source["centers"]],
            "landmark_boundaries": [value.tolist() for value in source["boundaries"]],
        })
    return enriched


def main():
    args = parse_args()
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    processed = json.load(open(args.data_root / "cityrefer/processed_descriptions.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train_seen", "val_seen", "val_unseen"):
        trajectories = json.load(open(
            args.data_root / f"processed_citynav/citynav_{split}.json"))
        source_rows = build_split(trajectories, objects, processed, max_landmarks=9)
        path = args.candidate_dir / f"{split}.jsonl"
        candidate_rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = enrich(candidate_rows, source_rows)
        output = args.output_dir / f"{split}.jsonl"
        with open(output, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        print(split, len(rows), flush=True)
    manifest = {
        "source": str(args.candidate_dir),
        "rule": "Only inference-visible landmark names, relation text, centers and boundaries were added.",
    }
    (args.output_dir / "enrichment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
