"""Evaluate lexical and Qwen relation graphs with fixed contour geometry operators."""

import argparse
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.train_relational_heatmap import (ANGLE_NAMES, DISTANCE_METERS, MAX_LANDMARKS,
    landmark_basis, load_objects, load_split, nms_points, pair_field, point_error)

PATTERNS = [
    ("between", r"\b(between|in the middle of|middle of)\b"),
    ("across", r"\b(across|opposite|other side of)\b"),
    ("northeast", r"\b(north[ -]?east|upper right)\b"),
    ("northwest", r"\b(north[ -]?west|upper left)\b"),
    ("southeast", r"\b(south[ -]?east|lower right)\b"),
    ("southwest", r"\b(south[ -]?west|lower left)\b"),
    ("left", r"\b(left|left-hand)\b"), ("right", r"\b(right|right-hand)\b"),
    ("north", r"\b(north|above|top of)\b"), ("south", r"\b(south|below|bottom of)\b"),
    ("front", r"\b(front of|in front)\b"), ("behind", r"\b(behind|back of|rear of)\b"),
    ("near", r"\b(near|nearby|next to|adjacent|beside|close to|by the)\b"),
    ("inside", r"\b(inside|within|in the grounds)\b"), ("along", r"\b(along|on the .* road)\b"),
]
ANGLE = {"left": 1, "right": 2, "north": 3, "south": 4,
         "northwest": 5, "northeast": 6, "southwest": 7, "southeast": 8}


def lexical(text):
    lower = text.lower()
    for relation, pattern in PATTERNS:
        if re.search(pattern, lower): return relation
    return "unknown"


def reference_indices(sample, relation):
    if relation == "between": return list(range(min(2, len(sample["landmarks"]))))
    lower = sample["description"].lower()
    hits = []
    for index, name in enumerate(sample["names"]):
        position = lower.find(name.lower())
        if position >= 0: hits.append((position, index))
    if not hits: return list(range(len(sample["landmarks"])))
    relation_positions = [match.start() for _, pattern in PATTERNS for match in re.finditer(pattern, lower)]
    if not relation_positions: return [hits[0][1]]
    position = min(relation_positions, key=lambda value: min(abs(value - hit[0]) for hit in hits))
    return [min(hits, key=lambda hit: abs(hit[0] - position))[1]]


def constraint_field(sample, bases, relation):
    indices = reference_indices(sample, relation)
    if relation == "between" and len(sample["landmarks"]) >= 2:
        return pair_field(sample["map"], sample["landmarks"])[0].astype(np.float32)
    if not indices: return np.zeros((64, 64), dtype=np.float32)
    if relation in ANGLE:
        angles, distances = [ANGLE[relation]], [1, 2, 3]
    elif relation in {"near", "inside", "along"}:
        angles, distances = [0], [0, 1, 2]
    elif relation == "across":
        angles, distances = [0], [2, 3, 4]
    elif relation in {"front", "behind"}:
        angles, distances = [0], [1, 2, 3]
    else:
        angles, distances = [0], [0, 1, 2, 3]
    fields = []
    for index in indices:
        landmark_id = sample["landmarks"][index][0]
        angle_fields, distance_fields = bases[(sample["map"], landmark_id)]
        angle = np.mean(angle_fields[angles].astype(np.float32), axis=0)
        distance = np.mean(distance_fields[distances].astype(np.float32), axis=0)
        fields.append(angle * distance)
    return np.maximum.reduce(fields)


def metric(rows):
    top1 = np.asarray([row["top1_m"] for row in rows]); top5 = np.asarray([row["top5_m"] for row in rows])
    return {"n": len(rows), "top1_hit20": round(100 * float(np.mean(top1 <= 20)), 2),
            "top5_recall20": round(100 * float(np.mean(top5 <= 20)), 2),
            "median_top1_m": round(float(np.median(top1)), 2)}


def evaluate(samples, relations, bases):
    rows = []
    for sample, relation in zip(samples, relations):
        field = constraint_field(sample, bases, relation)
        bounds = __import__("multiagent.mapdata", fromlist=["MAP_BOUNDS"]).MAP_BOUNDS[sample["map"]]
        target_rc = np.asarray([(bounds.y_max - sample["target"][1]) / (bounds.y_max - bounds.y_min) * 63,
                                (sample["target"][0] - bounds.x_min) / (bounds.x_max - bounds.x_min) * 63])
        cell = np.asarray([(bounds.x_max - bounds.x_min) / 63, (bounds.y_max - bounds.y_min) / 63])
        points = nms_points(field, cell)
        errors = [point_error(point, target_rc, cell) for point in points] or [float("inf")]
        rows.append({"relation": relation, "top1_m": errors[0], "top5_m": min(errors[:5])})
    result = {"overall": metric(rows), "by_relation": {}}
    for relation in sorted(set(row["relation"] for row in rows)):
        result["by_relation"][relation] = metric([row for row in rows if row["relation"] == relation])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--qwen-python", default="/home/tenant2/miniconda3/envs/vlmtest/bin/python")
    args = parser.parse_args(); args.run_dir.mkdir(parents=True, exist_ok=True)
    cache = args.run_dir / "qwen_relations.json"
    subprocess.run([args.qwen_python, str(ROOT / "scripts/parse_constraints_qwen.py"),
                    "--output", str(cache)], check=True)
    qwen = json.loads(cache.read_text())
    objects, processed, lookups = load_objects(); splits = {}
    for split in ["val_seen", "val_unseen"]:
        splits[split], _ = load_split(split, objects, processed, lookups)
    bases = {}
    for samples in splits.values():
        for sample in samples:
            for landmark_id, landmark, _ in sample["landmarks"]:
                bases.setdefault((sample["map"], landmark_id), landmark_basis(sample["map"], landmark))
    result = {"protocol": "PROTOCOL.md", "qwen_model": qwen["model"], "splits": {}}
    for split, samples in splits.items():
        qwen_rows = qwen["splits"][split]
        assert [row["description"] for row in qwen_rows] == [sample["description"] for sample in samples]
        result["splits"][split] = {
            "lexical": evaluate(samples, [lexical(sample["description"]) for sample in samples], bases),
            "qwen": evaluate(samples, [row["relation"] for row in qwen_rows], bases),
            "qwen_relation_counts": Counter(row["relation"] for row in qwen_rows),
        }
        print(split, json.dumps({key: value["overall"] for key, value in result["splits"][split].items()
                                 if key in {"lexical", "qwen"}}), flush=True)
    (args.run_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    lines = ["# 显式约束图验证", "", "| 划分 | 解析器 | Top-1 Hit@20 | Top-5 Recall@20 | 中位误差 |",
             "|---|---|---:|---:|---:|"]
    for split in ["val_seen", "val_unseen"]:
        for parser_name in ["lexical", "qwen"]:
            row = result["splits"][split][parser_name]["overall"]
            lines.append(f"| {split} | {parser_name} | {row['top1_hit20']:.2f}% | "
                         f"{row['top5_recall20']:.2f}% | {row['median_top1_m']:.2f} m |")
    lines += ["", "静态目标定位指标；未运行test_unseen，也不是导航成功率。"]
    (args.run_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
