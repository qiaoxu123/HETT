#!/usr/bin/env python3
"""Report 20 m candidate metrics by resolved instruction-landmark count."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import first_type_match, train_probe
from scripts.train_rgb_candidate_ranker import metrics, prepare_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("runs/candidate_rgb_data/geometry_top5_oof_grouped_enriched"))
    parser.add_argument("--geometry-cache", type=Path,
                        default=Path("runs/rgb_candidate_ranker_oof_grouped/features.pt"))
    parser.add_argument("--type-cache", type=Path,
                        default=Path("runs/target_type_alignment/features.pt"))
    parser.add_argument("--scale40-cache", type=Path,
                        default=Path("runs/target_type_alignment/multiscale/scale_40.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/landmark_count_hit20/results.json"))
    return parser.parse_args()


def mean_metrics(runs):
    return {key: sum(run[key] for run in runs) / len(runs) for key in runs[0]}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = torch.load(
        args.geometry_cache, map_location="cpu", weights_only=True)["val_unseen"]
    base = torch.load(args.type_cache, map_location="cpu", weights_only=True)
    scale40 = torch.load(args.scale40_cache, map_location="cpu", weights_only=True)
    rows, _ = prepare_rows(args.candidate_dir / "val_unseen.jsonl",
                           len(geometry["distance"]), 0, 5, False)
    counts = torch.tensor([len(row.get("landmark_names", [])) for row in rows])
    groups = {"one": counts == 1, "two": counts == 2, "three_plus": counts >= 3}
    result = {"groups": {}, "type_match_runs": {name: [] for name in groups},
              "anti_leakage": {
                  "split": "map-disjoint val_unseen",
                  "target_coordinates_used_for": "distance metrics only",
                  "candidate_selection": "geometry order plus predicted text/RGB type"}}
    for name, selected in groups.items():
        distance = geometry["distance"][selected]
        result["groups"][name] = {
            "count": int(selected.sum()),
            "geometry_top1": metrics(
                distance, torch.zeros(len(distance), dtype=torch.long)),
            "oracle_top5": metrics(distance, distance.argmin(-1)),
        }
    for seed in (0, 1, 2):
        text_model = train_probe(base["text"]["train_seen"],
                                 base["text"]["val_seen"], seed, 40, device)
        image_model = train_probe(scale40["target"]["train_seen"],
                                  scale40["target"]["val_seen"], seed, 40, device)
        text_prediction = text_model(
            geometry["text"].float().to(device)).argmax(-1).cpu()
        shape = scale40["candidate"].shape
        image_prediction = image_model(
            scale40["candidate"].float().reshape(-1, shape[-1]).to(device)
        ).argmax(-1).cpu().reshape(shape[:2])
        rank, _ = first_type_match(text_prediction, image_prediction)
        for name, selected in groups.items():
            result["type_match_runs"][name].append(
                metrics(geometry["distance"][selected], rank[selected]))
        del text_model, image_model
    for name in groups:
        result["groups"][name]["type_match_40m"] = mean_metrics(
            result["type_match_runs"][name])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
