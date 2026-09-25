#!/usr/bin/env python3
"""Build target-free perspective RGB/semantic/depth keyframes from PLY blocks."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.pointcloud import PerspectivePointRenderer, read_ply


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--ply-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train_seen", "val_seen", "val_unseen"])
    parser.add_argument("--frames-per-map", type=int, default=8)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--max-range", type=float, default=260.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def choose_keyframes(rows, count, seed):
    """Sample episode/step pairs without reading target coordinates."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    selected = []
    fractions = (0.1, 0.5, 0.9)
    for rank, index in enumerate(order[:count]):
        trajectory = rows[index]["trajectory"]
        fraction = fractions[rank % len(fractions)]
        step = min(len(trajectory) - 1, int(round(fraction * (len(trajectory) - 1))))
        selected.append((index, step, trajectory[step]))
    return selected


def ply_for_map(root, map_name):
    matches = list(root.glob(f"*/{map_name}.ply")) + list(root.glob(f"{map_name}.ply"))
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise FileNotFoundError(f"expected one PLY for {map_name}, found {unique}")
    return unique[0]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split_index, split in enumerate(args.splits):
        rows = json.loads((args.data_root / "processed_citynav" /
                           f"citynav_{split}.json").read_text())
        by_map = defaultdict(list)
        for row in rows:
            by_map[f'{row["area"]}_block_{row["block"]}'].append(row)
        for map_index, (map_name, map_rows) in enumerate(sorted(by_map.items())):
            path = ply_for_map(args.ply_root, map_name)
            xyz, rgb, semantic = read_ply(path, args.point_stride)
            renderer = PerspectivePointRenderer(xyz, rgb, semantic)
            keyframes = choose_keyframes(
                map_rows, min(args.frames_per_map, len(map_rows)),
                args.seed + split_index * 1009 + map_index)
            for local_index, (episode_index, step, pose) in enumerate(keyframes):
                rendered = renderer.render(
                    pose, args.resolution, args.resolution, args.hfov, args.max_range)
                frame_id = f"{split}_{map_name}_{local_index:04d}"
                relative = Path(split) / f"{frame_id}.npz"
                destination = args.output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination, rgb=rendered["rgb"], semantic=rendered["semantic"],
                    depth=rendered["depth"].astype(np.float16),
                    valid=rendered["valid"].astype(np.uint8))
                valid = rendered["valid"]
                counts = np.bincount(
                    rendered["semantic"][valid], minlength=5).tolist() if valid.any() else [0] * 5
                manifest.append({
                    "frame_id": frame_id, "file": str(relative), "split": split,
                    "map_name": map_name, "episode_index_within_map": episode_index,
                    "step": step, "pose": pose, "valid_fraction": float(valid.mean()),
                    "class_pixels": counts,
                    "selection_inputs": "trajectory pose only; no target coordinate or identity",
                })
            del renderer
            print(split, map_name, len(keyframes), flush=True)

    with (args.output_dir / "manifest.jsonl").open("w") as stream:
        for row in manifest:
            stream.write(json.dumps(row) + "\n")
    summary = {"config": {**vars(args), "data_root": str(args.data_root),
                           "ply_root": str(args.ply_root),
                           "output_dir": str(args.output_dir)}, "splits": {}}
    for split in args.splits:
        values = [row for row in manifest if row["split"] == split]
        validity = np.asarray([row["valid_fraction"] for row in values])
        class_pixels = np.asarray([row["class_pixels"] for row in values]).sum(0)
        summary["splits"][split] = {
            "frames": len(values), "maps": len({row["map_name"] for row in values}),
            "valid_fraction_mean": float(validity.mean()),
            "valid_fraction_median": float(np.median(validity)),
            "valid_fraction_p10": float(np.percentile(validity, 10)),
            "class_pixels": class_pixels.tolist(),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
