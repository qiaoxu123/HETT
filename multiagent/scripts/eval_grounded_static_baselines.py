"""Audit oracle reference resolution and geometry-only target localization."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from multiagent.cityreferobject import get_city_refer_objects
from multiagent.grounded_instruction_memory import (
    OracleReferenceResolver, load_static_examples, map_center, reference_mean_xy,
)
from multiagent.landmark_multimodal_map import LandmarkMultimodalMap


def metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "samples": len(errors),
        "hit@10m": float(np.mean(errors <= 10)),
        "hit@20m": float(np.mean(errors <= 20)),
        "hit@30m": float(np.mean(errors <= 30)),
        "median_error_m": float(np.median(errors)),
        "mean_error_m": float(np.mean(errors)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cityrefer = args.data_root / "cityrefer"
    objects = get_city_refer_objects(
        cityrefer / "objects.json", cityrefer / "processed_descriptions.json"
    )
    landmark_map = LandmarkMultimodalMap.from_cityrefer_objects(objects)
    resolver = OracleReferenceResolver(landmark_map)
    examples = load_static_examples(args.data_root, resolver)
    if len(examples) != 32326:
        raise ValueError(f"Expected 32,326 released rows, got {len(examples)}")

    errors = defaultdict(lambda: defaultdict(list))
    resolution = Counter()
    for example in examples:
        memory = example.memory
        target = np.asarray(example.target_xy)
        references = memory.references
        resolution["rows"] += 1
        resolution["reference_mentions"] += len(references) + len(memory.unresolved_names)
        resolution["resolved_references"] += len(references)
        resolution["unresolved_references"] += len(memory.unresolved_names)
        resolution["exact_references"] += sum(reference.match_type == "exact" for reference in references)
        resolution["fuzzy_references"] += sum(reference.match_type == "fuzzy" for reference in references)
        resolution[f"reference_count_{len(references)}"] += 1

        predictions = {
            "map_center": np.asarray(map_center(memory.map_name)),
            "first_reference_center": np.asarray(
                references[0].record.center_xy if references else map_center(memory.map_name)
            ),
            "reference_mean": np.asarray(reference_mean_xy(memory)),
        }
        if references:
            # Label-dependent diagnostic, never a deployable predictor.
            predictions["oracle_nearest_reference_center"] = min(
                (np.asarray(reference.record.center_xy) for reference in references),
                key=lambda center: np.linalg.norm(center - target),
            )
        group = "0" if not references else "1" if len(references) == 1 else "2" if len(references) == 2 else "3+"
        for name, prediction in predictions.items():
            error = float(np.linalg.norm(prediction - target))
            errors[name][example.split].append(error)
            errors[name][f"{example.split}/references_{group}"].append(error)

    report = {
        "scope": "All 32,326 refined CityNav source rows; no Stage-2 or rollout",
        "oracle_input": "processed.landmarks names only; target object excluded from reference lookup",
        "resolution": dict(resolution),
        "baselines": {
            name: {group: metrics(values) for group, values in grouped.items()}
            for name, grouped in errors.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "resolution": report["resolution"],
        "reference_mean": {
            split: report["baselines"]["reference_mean"][split]
            for split in ("train_seen", "val_seen", "val_unseen", "test_unseen")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
