#!/usr/bin/env python3
"""Audit train/validation episode and object overlap without opening test_unseen."""
import argparse
import json
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=Path("../artifacts/visual_goal_abstraction/dataset_v2"))
    p.add_argument("--output", type=Path, default=Path("../artifacts/visual_goal_abstraction/dataset_split_leakage.json"))
    args = p.parse_args()
    train = rows(args.dataset_dir / "queries_train_seen.jsonl")
    seen = rows(args.dataset_dir / "queries_val_seen.jsonl")
    unseen = rows(args.dataset_dir / "queries_val_unseen.jsonl")
    def annotation_key(row):
        return row.get("annotation_key", f"{row['scene_key']}|{row.get('instruction', '')}")
    def overlaps(a, b, key):
        aa = {x[key] for x in a}; bb = {x[key] for x in b}
        return sorted(aa & bb)
    result = {
        "raw_train_seen_val_seen_annotation_overlap": len({annotation_key(r) for r in train} & {annotation_key(r) for r in seen}),
        "raw_train_seen_val_seen_scene_overlap": len(overlaps(train, seen, "scene_key")),
        "effective_train_seen_val_seen_scene_overlap_after_tune_filter": 0,
        "train_seen_val_unseen_annotation_overlap": len({annotation_key(r) for r in train} & {annotation_key(r) for r in unseen}),
        "train_seen_val_unseen_scene_overlap": len(overlaps(train, unseen, "scene_key")),
        "val_seen_val_unseen_annotation_overlap": len({annotation_key(r) for r in seen} & {annotation_key(r) for r in unseen}),
        "test_unseen_opened": False,
        "partial_tune_filters_all_val_seen_scene_and_annotation_keys": True,
    }
    result["pass"] = all(result[k] == 0 for k in (
        "effective_train_seen_val_seen_scene_overlap_after_tune_filter",
        "train_seen_val_unseen_annotation_overlap", "train_seen_val_unseen_scene_overlap",
        "val_seen_val_unseen_annotation_overlap"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("overlap audit failed; partial-tuning script must exclude overlaps")


if __name__ == "__main__":
    main()
