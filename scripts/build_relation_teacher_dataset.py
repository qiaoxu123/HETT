#!/usr/bin/env python3
"""Build leakage-controlled teacher inputs and geometry-verified seed labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_spatial_relation_probe import cue_names
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE
from scripts.train_relation_geometry_ranker import (
    audit_consistent_landmark_geometry,
    load_rows,
)
from scripts.validate_candidate_geometry import mask_landmark_span


RELATIONS = ("LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "NEAR", "NONE")
DIRECTION_INDEX = {name: index for index, name in enumerate(
    ("left", "right", "above", "below"))}
OUTPUT_NAME = {
    "left": "LEFT_OF", "right": "RIGHT_OF",
    "above": "ABOVE", "below": "BELOW",
}
NEAR_PATTERN = re.compile(
    r"\b(?:near|next to|beside|adjacent to|close to|alongside)\s+(?:the\s+)?"
    r"\[mask\]", re.I)
BROAD_PATTERNS = {
    "LEFT_OF": re.compile(r"\b(?:left|west|western)\b", re.I),
    "RIGHT_OF": re.compile(r"\b(?:right|east|eastern)\b", re.I),
    "ABOVE": re.compile(r"\b(?:above|over|north|northern|top)\b", re.I),
    "BELOW": re.compile(r"\b(?:below|under|south|southern|bottom)\b", re.I),
    "NEAR": re.compile(
        r"\b(?:near|nearby|next|beside|adjacent|close|alongside)\b", re.I),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/relation_teacher_dataset"))
    parser.add_argument("--minimum-axis-m", type=float, default=10.0)
    parser.add_argument("--near-distance-m", type=float, default=30.0)
    parser.add_argument("--max-landmarks", type=int, default=9)
    return parser.parse_args()


def target_geometry(rows, max_landmarks):
    target_rows = [{
        "candidates": [row["target"]],
        "landmark_centers": row.get("landmark_centers", []),
        "landmark_boundaries": row.get("landmark_boundaries", []),
    } for row in rows]
    geometry, valid = audit_consistent_landmark_geometry(
        target_rows, max_landmarks)
    return geometry[:, 0].float(), valid


def strict_relation(masked_prompt):
    directions = cue_names(masked_prompt)
    near = bool(NEAR_PATTERN.search(masked_prompt))
    if len(directions) + int(near) != 1:
        return None
    return OUTPUT_NAME[directions[0]] if directions else "NEAR"


def local_window(masked_instruction, radius=12):
    words = re.findall(r"\[mask\]|[a-z]+", masked_instruction.lower())
    try:
        index = words.index("[mask]")
    except ValueError:
        return ""
    return " ".join(words[max(0, index - radius):index + radius + 1])


def broad_relation(masked_instruction):
    """Return an unambiguous relation cue close to this landmark mention."""
    window = local_window(masked_instruction)
    values = [name for name, pattern in BROAD_PATTERNS.items()
              if pattern.search(window)]
    return values[0] if len(values) == 1 else None


def geometry_check(relation, geometry, minimum_axis_m, near_distance_m):
    if relation in OUTPUT_NAME.values():
        inverse = {value: key for key, value in OUTPUT_NAME.items()}
        direction = inverse[relation]
        x, y = float(geometry[0]), float(geometry[1])
        projection = {
            "left": -x, "right": x, "above": -y, "below": y,
        }[direction] * 410.0
        return projection >= minimum_axis_m, projection
    if relation == "NEAR":
        boundary_distance = float(torch.linalg.vector_norm(geometry[5:7])) * 410.0
        return boundary_distance <= near_distance_m, boundary_distance
    return False, None


def teacher_prompt(instruction, landmark_name):
    schema = (
        '{"landmark":"<exact name>","relation":"LEFT_OF|RIGHT_OF|ABOVE|'
        'BELOW|NEAR|NONE","confidence":0.0,"evidence":"<text span>"}')
    return (
        "Extract only the spatial relation between the final target and the "
        f"named landmark {landmark_name!r}. Use the target's relation to the "
        "landmark, never the inverse. Map west/left to LEFT_OF, east/right to "
        "RIGHT_OF, north/above to ABOVE, south/below to BELOW, and next-to/"
        "beside/adjacent to NEAR. Examples: 'target left of Library' => "
        "LEFT_OF; 'target north of Library' => ABOVE; 'Library left of target' "
        "=> RIGHT_OF. If no allowed relation is explicitly stated for this "
        "landmark, return NONE. Output exactly one single-line JSON object, "
        "with no analysis, markdown, or extra text, matching this schema: "
        f"{schema}\nInstruction: {instruction}")


def build_split(rows, split, args):
    geometry, valid = target_geometry(rows, args.max_landmarks)
    records = []
    for row_index, row in enumerate(rows):
        for landmark_index, (query, name) in enumerate(zip(
                row.get("relation_queries", [])[:args.max_landmarks],
                row.get("landmark_names", [])[:args.max_landmarks])):
            if not valid[row_index, landmark_index]:
                continue
            masked, found = mask_landmark_span(row["description"], query, name)
            relation = strict_relation(masked) if found else None
            accepted, diagnostic = geometry_check(
                relation, geometry[row_index, landmark_index],
                args.minimum_axis_m, args.near_distance_m) if relation else (False, None)
            broad = broad_relation(masked) if found and relation is None else None
            broad_accepted, broad_diagnostic = geometry_check(
                broad, geometry[row_index, landmark_index],
                args.minimum_axis_m, args.near_distance_m) if broad else (False, None)
            records.append({
                "sample_id": f"{split}:{row_index}:{landmark_index}",
                "split": split,
                "row_index": row_index,
                "map_name": row["map_name"],
                "instruction": row["description"],
                "landmark_index": landmark_index,
                "landmark_name": name,
                "masked_instruction": masked,
                "strict_relation": relation,
                "strict_geometry_accepted": accepted,
                "geometry_diagnostic_m": diagnostic,
                "broad_relation": broad,
                "broad_geometry_accepted": broad_accepted,
                "broad_geometry_diagnostic_m": broad_diagnostic,
                "teacher_input": teacher_prompt(row["description"], name),
                # Target coordinates and candidate distances are intentionally
                # excluded. Geometry is used only to accept/reject labels.
            })
    return records


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def summarize(records):
    trusted = [row for row in records if row["strict_geometry_accepted"]]
    rejected = [row for row in records
                if row["strict_relation"] and not row["strict_geometry_accepted"]]
    broad_trusted = [row for row in records
                     if row["broad_geometry_accepted"]]
    return {
        "total_landmark_mentions": len(records),
        "trusted_seed_count": len(trusted),
        "strict_but_geometry_rejected": len(rejected),
        "broad_trusted_count": len(broad_trusted),
        "teacher_candidate_count": len(records) - len(trusted) - len(broad_trusted),
        "trusted_by_relation": dict(Counter(
            row["strict_relation"] for row in trusted)),
        "broad_trusted_by_relation": dict(Counter(
            row["broad_relation"] for row in broad_trusted)),
    }


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = {}
    for split in ("train_seen", "val_seen"):
        rows = load_rows(args.candidate_dir / f"{split}.jsonl", 5)
        records = build_split(rows, split, args)
        all_records[split] = records
        write_jsonl(args.output_dir / f"{split}.jsonl", records)
        write_jsonl(args.output_dir / f"{split}_trusted.jsonl", [
            row for row in records if (row["strict_geometry_accepted"]
                                       or row["broad_geometry_accepted"])])
        write_jsonl(args.output_dir / f"{split}_teacher_inputs.jsonl", [
            {key: row[key] for key in (
                "sample_id", "instruction", "landmark_name", "teacher_input")}
            for row in records if not (row["strict_geometry_accepted"]
                                       or row["broad_geometry_accepted"])])
    result = {
        "relations": RELATIONS,
        "geometry_frame": "audit-consistent M^T",
        "minimum_axis_m": args.minimum_axis_m,
        "near_distance_m": args.near_distance_m,
        "splits": {split: summarize(records)
                   for split, records in all_records.items()},
        "anti_leakage": {
            "teacher_input_contains_target_coordinate": False,
            "teacher_input_contains_candidate_distance": False,
            "train_seen_target_use": "seed-label filtering only",
            "val_seen_target_use": "development diagnostics only",
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
