#!/usr/bin/env python3
"""Export target-free per-landmark tensors for HETT relation fusion."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_relation_geometry_ranker import (
    structured_relation_cues,
    load_rows,
)


SPLITS = ("train_seen", "val_seen", "val_unseen")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--relation-features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-landmarks", type=int, default=9)
    return parser.parse_args()


def episode_ids(data_root, split):
    rows = json.load(open(
        data_root / "processed_citynav" / f"citynav_{split}.json"))
    return [[f'{row["area"]}_block_{row["block"]}', str(row["object_ids"][0]),
             int(row["ann_ids"][0])]
            for row in rows if row.get("dist_marker_to_target", 0.0) <= 30.0]


def geometry_tensors(rows, max_landmarks):
    centers = torch.zeros(len(rows), max_landmarks, 2, dtype=torch.float16)
    boundaries = torch.zeros(len(rows), max_landmarks, 16, 2, dtype=torch.float16)
    valid = torch.zeros(len(rows), max_landmarks, dtype=torch.bool)
    for row_index, row in enumerate(rows):
        count = min(len(row.get("landmark_centers", [])),
                    len(row.get("landmark_boundaries", [])), max_landmarks)
        if count:
            centers[row_index, :count] = torch.tensor(
                row["landmark_centers"][:count], dtype=torch.float16)
            boundaries[row_index, :count] = torch.tensor(
                row["landmark_boundaries"][:count], dtype=torch.float16)
            valid[row_index, :count] = True
    return centers, boundaries, valid


def main():
    args = parse_args()
    relation = torch.load(args.relation_features, map_location="cpu", weights_only=True)
    payload = {
        "schema": 1,
        "max_landmarks": args.max_landmarks,
        "anti_leakage": {
            "included": "episode id, landmark centers/boundaries, masked relation BERT, cues",
            "excluded": "target coordinate, target identity, RGB, success labels",
        },
        "splits": {},
    }
    for split in SPLITS:
        rows = load_rows(args.candidate_dir / f"{split}.jsonl", 5)
        ids = episode_ids(args.data_root, split)
        if len(ids) != len(rows) or relation[split].shape[:2] != (
                len(rows), args.max_landmarks):
            raise ValueError(f"alignment mismatch for {split}")
        centers, boundaries, valid = geometry_tensors(rows, args.max_landmarks)
        payload["splits"][split] = {
            "episode_ids": ids,
            "centers": centers,
            "boundaries": boundaries,
            "valid": valid,
            "relation": relation[split].half(),
            "cue": structured_relation_cues(rows, args.max_landmarks).half(),
        }
        print(f"prepared {split}: {len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved target-free artifact: {args.output}", flush=True)


if __name__ == "__main__":
    main()
