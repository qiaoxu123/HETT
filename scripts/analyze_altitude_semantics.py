#!/usr/bin/env python3
"""Summarize exact-paired RSRefSeg2 altitude and digital-zoom results."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


GROUP = re.compile(r"^(real|zoom)_(val_seen|val_unseen)_d([0-9.]+)_h([0-9.]+)$")


def parse_group(value):
    match = GROUP.match(value)
    if not match:
        raise ValueError(f"Unexpected metric group: {value}")
    modality, split, distance, altitude = match.groups()
    return modality, split, float(distance), float(altitude)


def sample_key(path, altitude, modality):
    stem = Path(path).stem
    suffix = f"_{altitude:g}m"
    if modality == "zoom":
        marker = f"{suffix}_zoom_from_"
        return stem.split(marker, 1)[0]
    if not stem.endswith(suffix):
        raise ValueError(f"Cannot parse real-view filename: {path}")
    return stem[:-len(suffix)]


def summarize(rows):
    return {
        "count": len(rows),
        "mean_box_iou": 100 * float(np.mean([row["box_iou"] for row in rows])),
        "box_acc_0.25": 100 * float(np.mean([row["box_iou"] >= 0.25 for row in rows])),
        "box_acc_0.5": 100 * float(np.mean([row["box_iou"] >= 0.5 for row in rows])),
        "center_hit_rate": 100 * float(np.mean([row["center_hit"] for row in rows])),
        "mask_gIoU": 100 * float(np.mean([row["mask_iou"] for row in rows])),
    }


def bootstrap_delta(first, second, seed=0, repetitions=10000):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.size == 0:
        raise ValueError("Paired arrays must be nonempty and have equal shape")
    differences = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    means = differences[indices].mean(axis=1)
    return {
        "count": len(differences),
        "mean_delta_pp": 100 * float(differences.mean()),
        "ci95_low_pp": 100 * float(np.percentile(means, 2.5)),
        "ci95_high_pp": 100 * float(np.percentile(means, 97.5)),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
    }


def paired_report(lookup, modality_a, altitude_a, modality_b, altitude_b, split, distance, metric):
    first = {
        key[4]: row for key, row in lookup.items()
        if key[:4] == (modality_a, split, distance, altitude_a)
    }
    second = {
        key[4]: row for key, row in lookup.items()
        if key[:4] == (modality_b, split, distance, altitude_b)
    }
    shared = sorted(first.keys() & second.keys())
    return bootstrap_delta(
        [first[key][metric] for key in shared],
        [second[key][metric] for key in shared],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--reference-altitude", type=float, default=120)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.metrics.open()]
    grouped = defaultdict(list)
    lookup = {}
    for row in rows:
        modality, split, distance, altitude = parse_group(row["group"])
        key = sample_key(row["img_path"], altitude, modality)
        grouped[(modality, split, distance, altitude)].append(row)
        lookup[(modality, split, distance, altitude, key)] = row

    summaries = {
        "|".join(map(str, key)): summarize(value)
        for key, value in sorted(grouped.items())
    }
    comparisons = []
    real_conditions = sorted(key for key in grouped if key[0] == "real")
    for _, split, distance, altitude in real_conditions:
        if altitude == args.reference_altitude:
            continue
        for metric in ("box_iou", "center_hit", "mask_iou"):
            comparisons.append({
                "comparison": "real_low_minus_real_high",
                "split": split, "distance_band_m": distance, "altitude_m": altitude,
                "reference_altitude_m": args.reference_altitude, "metric": metric,
                **paired_report(
                    lookup, "real", altitude, "real", args.reference_altitude,
                    split, distance, metric,
                ),
            })
            comparisons.append({
                "comparison": "real_low_minus_digital_zoom",
                "split": split, "distance_band_m": distance, "altitude_m": altitude,
                "reference_altitude_m": args.reference_altitude, "metric": metric,
                **paired_report(
                    lookup, "real", altitude, "zoom", altitude,
                    split, distance, metric,
                ),
            })
    result = {"samples": len(rows), "summaries": summaries, "paired_comparisons": comparisons}
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")

    markdown = [
        "# Paired altitude semantic result", "",
        "All deltas are first condition minus second condition on exact paired samples.", "",
        "| comparison | split | distance | altitude | metric | n | delta (pp) | 95% CI | W/T/L |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        markdown.append(
            f"| {item['comparison']} | {item['split']} | {item['distance_band_m']:g} | "
            f"{item['altitude_m']:g} | {item['metric']} | {item['count']} | "
            f"{item['mean_delta_pp']:.2f} | [{item['ci95_low_pp']:.2f}, {item['ci95_high_pp']:.2f}] | "
            f"{item['wins']}/{item['ties']}/{item['losses']} |"
        )
    args.output_md.write_text("\n".join(markdown) + "\n")
    print(json.dumps({"samples": len(rows), "comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
