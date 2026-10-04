#!/usr/bin/env python3
"""Evaluate explicit scene evidence and enforce the Phase-4 Gate."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects  # noqa: E402
from multiagent.scene_grounding.metrics import distance_bucket, spearman, summarize_pairs  # noqa: E402
from multiagent.scene_grounding.scene_evidence import explicit_match  # noqa: E402
from multiagent.scene_grounding.scene_template import SceneTemplate  # noqa: E402
from multiagent.scene_grounding.temporal_fusion import fuse_scores  # noqa: E402


def load_template(value):
    return SceneTemplate(instruction=value["instruction"], anchors=tuple(value["anchors"]), target=value["target"],
                         context=tuple(value["context"]), geometry=tuple(value["geometry"]), ordinal=tuple(value["ordinal"]))


def keys(template):
    return {f"{category}:{value}" for category, values in template.visual_requirements.items() for value in values}


def hard_negative(template_id, templates, map_episodes):
    source = keys(templates[template_id]); candidates = sorted(item for item in map_episodes if item != template_id)
    if not candidates: return None
    def rank(item):
        candidate = keys(templates[item]); overlap = len(source & candidate) / max(len(source | candidate), 1)
        return overlap - 0.05 * abs(len(source) - len(candidate))
    return max(candidates, key=rank)


def geometry_score(template, pose_xy, map_objects):
    names = [anchor.get("name") for anchor in template.anchors if anchor.get("name")]
    landmarks = [obj for obj in map_objects if obj.name in names]
    if not landmarks:
        return None
    point = np.asarray(pose_xy, dtype=float)
    distance = np.mean([np.linalg.norm(point - np.asarray(obj.position[:2], dtype=float)) for obj in landmarks])
    return float(-min(distance, 200.0) / 100.0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    templates = {key: load_template(value) for key, value in json.loads((args.dataset / "templates.json").read_text()).items()}
    rows = [json.loads(line) for line in (args.evidence / "evidence.jsonl").read_text().splitlines()]
    manifest = {row["sample_id"]: row for row in (json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines())}
    city_objects = get_city_refer_objects(args.data_root / "cityrefer/objects.json", args.data_root / "cityrefer/processed_descriptions.json")
    if any(row["split"] == "test_unseen" for row in rows): raise ValueError("test_unseen is forbidden")
    map_episodes = defaultdict(set)
    for row in rows: map_episodes[(row["split"], row["map_name"])].add(row["episode_id"])
    report = {"representations": {}, "gate": {}}
    all_scored = {}
    for representation in ("whole", "regions", "combined"):
        report["representations"][representation] = {}
        for component in ("target", "anchor", "all", "color", "size", "shape", "roof", "semantic", "context"):
            report["representations"][representation][component] = {}
            for split in ("val_seen", "val_unseen"):
                scored = []
                for row in rows:
                    if row["split"] != split: continue
                    negative_id = hard_negative(row["template_id"], templates, map_episodes[(split, row["map_name"])])
                    if negative_id is None: continue
                    positive = explicit_match(templates[row["template_id"]], row[representation], component)
                    negative = explicit_match(templates[negative_id], row[representation], component)
                    if positive["coverage"] == 0 or negative["coverage"] == 0: continue
                    scored.append({"episode_id": row["episode_id"], "distance_m": row["distance_m"],
                                   "step": row["step"], "altitude_m": row["altitude_m"],
                                   "positive_score": positive["score"], "negative_score": negative["score"],
                                   "view_confidence": row["view_quality"]["attribute_confidence"]})
                metrics = summarize_pairs(scored)
                episodes = defaultdict(list)
                for item in scored: episodes[item["episode_id"]].append(item)
                correlations = [spearman([x["distance_m"] for x in values], [x["positive_score"] for x in values])
                                for values in episodes.values() if len(values) >= 3]
                correlations = [value for value in correlations if value is not None]
                metrics["episode_mean_spearman"] = float(np.mean(correlations)) if correlations else None
                metrics["view_quality_score_spearman"] = spearman([x["view_confidence"] for x in scored], [x["positive_score"] for x in scored])
                metrics["altitude_score_spearman"] = spearman([x["altitude_m"] for x in scored], [x["positive_score"] for x in scored])
                metrics["altitude_distance_spearman"] = spearman([x["altitude_m"] for x in scored], [x["distance_m"] for x in scored])
                report["representations"][representation][component][split] = metrics
                all_scored[(representation, component, split)] = scored
    report["methods"] = {}
    for split in ("val_seen", "val_unseen"):
        base_rows = all_scored[("combined", "all", split)]
        geometry_rows, fused_rows = [], []
        for item in base_rows:
            source = next(row for row in rows if row["episode_id"] == item["episode_id"] and row["step"] == item["step"])
            negative_id = hard_negative(source["template_id"], templates, map_episodes[(split, source["map_name"])])
            current = manifest[source["sample_id"]]; pose_xy = (current["pose"]["x"], current["pose"]["y"])
            positive_geometry = geometry_score(templates[source["template_id"]], pose_xy, city_objects[source["map_name"]].values())
            negative_geometry = geometry_score(templates[negative_id], pose_xy, city_objects[source["map_name"]].values())
            if positive_geometry is None or negative_geometry is None: continue
            common = {key: item[key] for key in ("episode_id", "distance_m", "step", "altitude_m", "view_confidence")}
            geometry_rows.append(common | {"positive_score": positive_geometry, "negative_score": negative_geometry})
            fused_rows.append(common | {"positive_score": item["positive_score"] + positive_geometry,
                                        "negative_score": item["negative_score"] + negative_geometry})
        report["methods"][split] = {
            "target_only": report["representations"]["combined"]["target"][split],
            "anchor_only": report["representations"]["combined"]["anchor"][split],
            "target_anchor": report["representations"]["combined"]["all"][split],
            "geometry_only": summarize_pairs(geometry_rows),
            "target_anchor_geometry": summarize_pairs(fused_rows),
        }
        histories = {}
        episode_rows = defaultdict(list)
        for item in base_rows: episode_rows[item["episode_id"]].append(item)
        for name, window, mode in (("single_frame", 1, "mean"), ("3_frame", 3, "mean"),
                                   ("5_frame", 5, "mean"), ("confidence_weighted", 5, "confidence")):
            history_rows = []
            for values in episode_rows.values():
                values.sort(key=lambda item: item["step"])
                confidence = [item["view_confidence"] for item in values]
                positive = fuse_scores([item["positive_score"] for item in values], window, mode, confidence)
                negative = fuse_scores([item["negative_score"] for item in values], window, mode, confidence)
                for item, pos, neg in zip(values, positive, negative):
                    history_rows.append(item | {"positive_score": pos, "negative_score": neg})
            histories[name] = summarize_pairs(history_rows)
        report["methods"][split]["history"] = histories
    gate_metrics = report["representations"]["combined"]["all"]["val_seen"]
    buckets = gate_metrics["distance_buckets"]
    near, far = buckets["0-20m"]["positive_mean"], buckets[">80m"]["positive_mean"]
    passed = bool(near is not None and far is not None and near > far + 0.02 and
                  gate_metrics["same_map_hard_negative_accuracy"] > 0.55 and
                  gate_metrics["episode_mean_spearman"] is not None and gate_metrics["episode_mean_spearman"] < -0.1)
    report["gate"] = {"passed": passed, "criteria": {"near_minus_far_gt": 0.02,
                       "same_map_accuracy_gt": 0.55, "episode_spearman_lt": -0.1},
                       "near_minus_far": None if near is None or far is None else near - far,
                       "decision": "continue_learned_matcher" if passed else "stop_large_matcher_training"}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__": main()
