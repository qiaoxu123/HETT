from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpv_goal_approach.dataset import read_jsonl


SPLITS = ("train_seen", "val_seen", "val_unseen", "test_unseen")


def distribution(rows, key):
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def analyze(data: Path):
    split_rows = {}
    report = {"splits": {}, "leakage": {}}
    for split in SPLITS:
        path = data / "metadata" / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        split_rows[split] = rows
        episodes = {row["episode_id"] for row in rows}
        report["splits"][split] = {
            "episode_count": len(episodes),
            "sample_count": len(rows),
            "positive_match_count": sum(row["target_match"] == 1 for row in rows),
            "negative_match_count": sum(row["target_match"] == 0 for row in rows),
            "arrival_positive": sum(row["visual_confirmed_arrival"] == 1 for row in rows),
            "arrival_negative": sum(row["visual_confirmed_arrival"] == 0 for row in rows),
            "distance_bin": distribution(rows, "distance_bin"),
            "bearing_bin": distribution(rows, "bearing_bin"),
            "phase": distribution(rows, "phase"),
            "negative_type": distribution(rows, "negative_type"),
            "frames_per_clip": distribution(rows, key="frames_count") if rows and "frames_count" in rows[0] else distribution([{"n": len(row["frames"])} for row in rows], "n"),
            "samples_per_episode": len(rows) / max(1, len(episodes)),
            "render_source": distribution(rows, "render_source"),
        }
    names = list(split_rows)
    for index, left in enumerate(names):
        left_ids = {row["episode_id"] for row in split_rows[left]}
        for right in names[index + 1:]:
            overlap = left_ids & {row["episode_id"] for row in split_rows[right]}
            key = f"{left}__{right}"
            report["leakage"][key] = sorted(overlap)
            if left == "train_seen" and right in {"val_seen", "val_unseen"}:
                assert not overlap, f"episode leakage in {key}: {len(overlap)}"
    output = data / "stats" / "dataset_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args().data)
