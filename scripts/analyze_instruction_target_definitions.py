#!/usr/bin/env python3
"""Summarize landmark counts and target-definition cues in CityNav instructions."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SPLITS = ("train_seen", "val_seen", "val_unseen")

PATTERNS = {
    "appearance": re.compile(
        r"\b(?:black|white|gr[ae]y|red|blue|green|brown|tan|beige|yellow|orange|"
        r"silver|gold|pink|purple|cream|multi[- ]?colou?red|dark|light|bright|"
        r"large|small|big|tiny|tall|short|long|wide|narrow|square|rectangular|"
        r"round|circular|triangular|flat|pitched|slanted|gabled|roof|brick|"
        r"glass|stone|metal|wood(?:en)?|colou?r(?:ed)?)\b"
    ),
    "direction": re.compile(
        r"\b(?:north(?:ern)?|south(?:ern)?|east(?:ern)?|west(?:ern)?|left|right|"
        r"front|behind|above|below|under|over|top|bottom)\b"
    ),
    "near_adjacent": re.compile(
        r"\b(?:near|nearby|beside|adjacent|alongside|close to|next to|by the|"
        r"connected to|attached to|immediately)\b"
    ),
    "between": re.compile(r"\b(?:between|in between)\b"),
    "across_opposite": re.compile(r"\b(?:across(?: from)?|opposite)\b"),
    "road_reference": re.compile(
        r"\b(?:road|street|avenue|lane|highway|motorway|footpath|pathway|sidewalk)\b"
    ),
    "absolute_map_position": re.compile(
        r"\b(?:corner|edge|centre|center|top|bottom|left|right|north|south|east|west)"
        r"\s+(?:of|on)\s+(?:the\s+)?(?:map|block|image)\b"
    ),
    "order_or_group": re.compile(
        r"\b(?:first|second|third|fourth|last|middle|end|row|group|cluster|one of|"
        r"among)\b"
    ),
}


def cue_flags(description, landmark_count, surroundings):
    """Return overlapping language/annotation cues used to identify the target."""
    text = description.lower()
    flags = {name: bool(pattern.search(text)) for name, pattern in PATTERNS.items()}
    flags["named_landmark_relation"] = landmark_count > 0
    flags["surrounding_context"] = bool(surroundings)
    return flags


def bucket_landmarks(count):
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def percent(count, total):
    return round(100.0 * count / total, 2) if total else 0.0


def summarize_counter(counter, total):
    return {
        key: {"count": counter[key], "percent": percent(counter[key], total)}
        for key in counter
    }


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def raw_rows(data_root, split, processed, objects):
    rows = []
    trajectories = json.load(open(
        data_root / "processed_citynav" / f"citynav_{split}.json"))
    for item in trajectories:
        # Candidate generation and current experiments use the same quality filter.
        if item.get("dist_marker_to_target", 0.0) > 30.0:
            continue
        map_name = f'{item["area"]}_block_{item["block"]}'
        object_id = str(item["object_ids"][0])
        ann_id = int(item["ann_ids"][0])
        annotations = processed.get(map_name, {}).get(object_id, [])
        annotation = annotations[ann_id] if ann_id < len(annotations) else {}
        object_record = objects.get(map_name, {}).get(object_id, {})
        rows.append({
            "description": item["descriptions"][0],
            "landmarks": annotation.get("landmarks", []),
            "surroundings": annotation.get("surroundings", []),
            "target_phrase": annotation.get("target", "unknown"),
            "target_type": object_record.get("object_type", "Unknown"),
        })
    return rows


def summarize_split(raw, effective):
    if len(raw) != len(effective):
        raise ValueError(f"row count mismatch: raw={len(raw)}, effective={len(effective)}")
    raw_counts = Counter(bucket_landmarks(len(row["landmarks"])) for row in raw)
    effective_counts = Counter(bucket_landmarks(len(row["landmark_names"])) for row in effective)
    cues = Counter()
    targets = Counter()
    combinations = Counter()
    resolved_mentions = raw_mentions = 0
    for source, model_row in zip(raw, effective):
        if source["description"] != model_row["description"]:
            raise ValueError("raw/effective descriptions are not aligned")
        raw_count = len(source["landmarks"])
        effective_count = len(model_row["landmark_names"])
        raw_mentions += raw_count
        resolved_mentions += effective_count
        flags = cue_flags(source["description"], raw_count, source["surroundings"])
        active = tuple(sorted(name for name, enabled in flags.items() if enabled))
        cues.update(active)
        combinations[" + ".join(active) if active else "none"] += 1
        targets[source["target_type"]] += 1
    total = len(raw)
    return {
        "instructions": total,
        "raw_annotated_landmark_count": summarize_counter(raw_counts, total),
        "effective_resolved_landmark_count": summarize_counter(effective_counts, total),
        "landmark_mentions": {
            "raw": raw_mentions,
            "resolved": resolved_mentions,
            "resolution_rate_percent": percent(resolved_mentions, raw_mentions),
        },
        "target_types": summarize_counter(targets, total),
        "definition_cues_overlapping": summarize_counter(cues, total),
        "top_cue_combinations": [
            {"combination": name, "count": count, "percent": percent(count, total)}
            for name, count in combinations.most_common(15)
        ],
    }


def merge_split_rows(split_rows):
    raw = []
    effective = []
    for raw_part, effective_part in split_rows:
        raw.extend(raw_part)
        effective.extend(effective_part)
    return raw, effective


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--candidate-dir", type=Path,
        default=Path("runs/candidate_rgb_data/geometry_top5_oof_grouped_enriched"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/instruction_target_definitions/results.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    processed = json.load(open(args.data_root / "cityrefer" / "processed_descriptions.json"))
    objects = json.load(open(args.data_root / "cityrefer" / "objects.json"))
    result = {
        "unit": "trajectory instruction after the current <=30 m marker-quality filter",
        "note": "Definition cues overlap; their percentages must not be added together.",
        "splits": {},
    }
    parts = []
    for split in SPLITS:
        raw = raw_rows(args.data_root, split, processed, objects)
        effective = load_jsonl(args.candidate_dir / f"{split}.jsonl")
        result["splits"][split] = summarize_split(raw, effective)
        parts.append((raw, effective))
    raw_all, effective_all = merge_split_rows(parts)
    result["all_three_splits"] = summarize_split(raw_all, effective_all)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
