#!/usr/bin/env python3
"""Generate leakage-controlled map-level out-of-fold geometry candidates."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_candidate_geometry import (
    GeometryRanker,
    build_split,
    encode_language,
    make_tensors,
    predict_candidates,
    save_candidates,
    seed_everything,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/candidate_rgb_data/geometry_top5_oof"))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def balanced_map_folds(rows, fold_count):
    counts = {}
    for row in rows:
        counts[row["map_name"]] = counts.get(row["map_name"], 0) + 1
    folds = [[] for _ in range(fold_count)]
    sizes = [0] * fold_count
    for map_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(fold_count), key=lambda index: (sizes[index], index))
        folds[fold].append(map_name)
        sizes[fold] += count
    return folds, sizes


def subset(data, indices):
    index = torch.tensor(indices, dtype=torch.long)
    return tuple(value[index] for value in data)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    processed = json.load(open(args.data_root / "cityrefer/processed_descriptions.json"))
    rows = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        trajectories = json.load(open(
            args.data_root / f"processed_citynav/citynav_{split}.json"))
        rows[split] = build_split(trajectories, objects, processed, args.max_landmarks)
        print(split, len(rows[split]), flush=True)

    text_vectors, name_vectors, name_to_id, relation_vectors, relation_to_id = encode_language(
        rows, device)
    data = {split: make_tensors(values, text_vectors, name_to_id, relation_to_id,
                                args.grid_size, args.max_landmarks)
            for split, values in rows.items()}
    folds, fold_sizes = balanced_map_folds(rows["train_seen"], args.folds)
    manifest = {
        "folds": [{"index": index, "maps": maps, "samples": fold_sizes[index]}
                  for index, maps in enumerate(folds)],
        "rule": "Each train candidate is predicted by a model trained without its map.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fold_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    all_candidates = torch.empty(len(rows["train_seen"]), args.top_k, 2)
    all_indices = set(range(len(rows["train_seen"])))
    for fold_index, heldout_maps in enumerate(folds):
        heldout_set = set(heldout_maps)
        heldout = [i for i, row in enumerate(rows["train_seen"])
                   if row["map_name"] in heldout_set]
        training = sorted(all_indices - set(heldout))
        print(f"fold={fold_index} train={len(training)} heldout={len(heldout)} "
              f"maps={heldout_maps}", flush=True)
        seed_everything(args.seed + fold_index)
        model = GeometryRanker(name_vectors, relation_vectors, args.grid_size,
                               mode="span_geometry_boundary_soft")
        model = train_model(model, subset(data["train_seen"], training), device,
                            args.epochs, args.batch_size, f"oof-fold={fold_index}", 0.0)
        predicted = predict_candidates(model, subset(data["train_seen"], heldout),
                                       device, args.batch_size, args.top_k)
        all_candidates[torch.tensor(heldout)] = predicted
        del model
        torch.cuda.empty_cache()
    save_candidates(args.output_dir / "train_seen.jsonl", rows["train_seen"], all_candidates)

    seed_everything(args.seed)
    final_model = GeometryRanker(name_vectors, relation_vectors, args.grid_size,
                                 mode="span_geometry_boundary_soft")
    final_model = train_model(final_model, data["train_seen"], device, args.epochs,
                              args.batch_size, "final-all-train", 0.0)
    for split in ("val_seen", "val_unseen"):
        predicted = predict_candidates(final_model, data[split], device,
                                       args.batch_size, args.top_k)
        save_candidates(args.output_dir / f"{split}.jsonl", rows[split], predicted)
        print("saved", split, len(predicted), flush=True)

    summary = {
        "config": {**vars(args), "data_root": str(args.data_root),
                   "output_dir": str(args.output_dir)},
        "counts": {split: len(value) for split, value in rows.items()},
        "manifest": str(args.output_dir / "fold_manifest.json"),
    }
    with open(args.output_dir / "summary.json", "w") as stream:
        json.dump(summary, stream, indent=2)


if __name__ == "__main__":
    main()
