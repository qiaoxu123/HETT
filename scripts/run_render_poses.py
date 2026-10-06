#!/usr/bin/env python3
"""Steps 6-8, 10-12: render the sampled poses and write their artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.fit_coordinate_transform import (  # noqa: E402
    CoordinateTransform, load_transform,
)
from sensaturban_fpv.pointcloud_renderer import (  # noqa: E402
    Camera, render_cloud_region,
)
from sensaturban_fpv.render_trajectory_fpv import (  # noqa: E402
    render_pose_bundle, write_pose_artifacts,
)


def load_or_identity(path: Path) -> CoordinateTransform:
    if path.exists():
        return load_transform(path)
    return CoordinateTransform.identity()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None,
                    help="path to best_transform.json; identity when absent")
    ap.add_argument("--out", default=None)
    ap.add_argument("--splat-radius", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--far", type=float, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["splat_radius"] = (
        args.splat_radius if args.splat_radius is not None else cfg["render"]["splat_radius"])
    cfg["render"]["width"] = args.width or cfg["render"]["width"]
    cfg["render"]["height"] = args.height or cfg["render"]["height"]
    cfg["render"]["far"] = args.far or cfg["render"]["far"]
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])

    transform_path = Path(args.transform) if args.transform else \
        artifact_dir(cfg) / "transform" / "best_transform.json"
    transform = load_or_identity(transform_path)
    print(f"coordinate transform: {json.dumps(transform.to_json(), sort_keys=True)}")

    out_root = Path(args.out) if args.out else artifact_dir(cfg) / "poses"
    out_root.mkdir(parents=True, exist_ok=True)

    objects_by_map = load_landmarks(cfg)
    sampling = cfg["sampling"]

    episodes_by_split = {
        split: citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split)
        for split in sampling["per_split"]
    }

    # Group the chosen episodes by map so each block is opened once.
    from sensaturban_fpv.render_trajectory_fpv import sample_poses
    samples = sample_poses(episodes_by_split, sampling["per_split"], transform,
                           prefer_landmarks=sampling["require_referenced_landmark"])
    if sampling.get("max_per_map"):
        kept, per_map = [], defaultdict(int)
        for s in samples:
            if per_map[s.map_name] < sampling["max_per_map"]:
                kept.append(s)
                per_map[s.map_name] += 1
        samples = kept

    by_map = defaultdict(list)
    for s in samples:
        by_map[s.map_name].append(s)

    episode_lookup = {}
    for split, episodes in episodes_by_split.items():
        for ep in episodes:
            episode_lookup[(split, ep.index)] = ep

    index = []
    for map_name, map_samples in sorted(by_map.items()):
        t0 = time.time()
        ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)
        landmarks = ctx.landmarks
        for sample in map_samples:
            ep = episode_lookup[(sample.split, sample.episode_index)]
            bundle = render_pose_bundle(
                ctx.cloud, ctx.grid, sample, cfg["render"], landmarks, transform,
                referenced_ids=ep.object_ids,
            )
            written = write_pose_artifacts(
                out_root, sample, bundle["views"], bundle["context"],
                bundle["overlays"], bundle["metadata"],
            )
            index.append({
                "map": map_name, "split": sample.split,
                "episode_index": int(sample.episode_index), "step": int(sample.step),
                "citynav_pose": [float(v) for v in sample.citynav_pose],
                "position_ply": [float(v) for v in sample.position_ply],
                "valid_pixel_ratio": bundle["metadata"]["valid_pixel_ratio"],
                "depth_median": bundle["metadata"]["depth_median"],
                "referenced_ids": list(ep.object_ids),
                "artifacts": written,
            })
            print(
                f"  {sample.split}:{sample.episode_index} step{sample.step} "
                f"{map_name} valid={bundle['metadata']['valid_pixel_ratio']:.3f} "
                f"depth_med={bundle['metadata']['depth_median']}",
                flush=True,
            )
        print(f"{map_name}: {len(map_samples)} poses in {time.time() - t0:.1f}s", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    index_path = out_root / f"pose_index{tag}.json"
    index_path.write_text(json.dumps(
        {"transform": transform.to_json(),
         "render_config": {k: v for k, v in cfg["render"].items() if k != "lod"},
         "lod": [list(x) for x in cfg["render"]["lod"]],
         "poses": index},
        indent=2, sort_keys=True) + "\n")
    ratios = np.array([p["valid_pixel_ratio"] for p in index], dtype=np.float64)
    print(f"\n{len(index)} poses -> {index_path}")
    if ratios.size:
        print(f"valid_pixel_ratio: median={np.median(ratios):.3f} "
              f"min={ratios.min():.3f} max={ratios.max():.3f} "
              f">=0.60: {(ratios >= 0.6).mean():.2%}")


if __name__ == "__main__":
    main()
