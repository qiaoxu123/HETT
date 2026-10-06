#!/usr/bin/env python3
"""Step 12: raw point render vs limited point-size enhancement.

Splatting enlarges each measured point over a few pixels; the nearest point
still wins every pixel, so this fills pinholes without inventing geometry.  It
does *not* invent colour either -- an enhanced pixel always shows a real
measured point.  Both variants are produced and the raw one is never replaced.
"""

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
    Camera, depth_histogram, render_cloud_region,
)
from sensaturban_fpv.render_trajectory_fpv import sample_poses  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--radii", nargs="*", type=int, default=[0, 1, 2, 4])
    ap.add_argument("--poses-per-split", type=int, default=2)
    ap.add_argument("--views", nargs="*", default=["fpv", "oblique", "steep_oblique"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "gates"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform_path = Path(args.transform) if args.transform else \
        artifact_dir(cfg) / "transform" / "best_transform.json"
    transform = load_transform(transform_path) if transform_path.exists() \
        else CoordinateTransform.identity()

    objects_by_map = load_landmarks(cfg)
    per_split = {s: args.poses_per_split for s in cfg["sampling"]["per_split"]}
    episodes_by_split = {
        split: citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split)
        for split in cfg["sampling"]["per_split"]
    }
    samples = sample_poses(episodes_by_split, per_split, transform)
    by_map = defaultdict(list)
    for s in samples:
        by_map[s.map_name].append(s)

    rows = []
    for map_name, map_samples in sorted(by_map.items()):
        ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)
        for sample in map_samples:
            for view in args.views:
                pitch = cfg["render"]["views"][view]
                cam = Camera(
                    position=sample.position_ply, yaw=float(sample.citynav_pose[3]),
                    pitch=np.deg2rad(pitch), width=cfg["render"]["width"],
                    height=cfg["render"]["height"], hfov_deg=cfg["render"]["hfov_deg"],
                    near=cfg["render"]["near"], far=cfg["render"]["far"],
                )
                row = {"map": map_name, "split": sample.split,
                       "episode_index": int(sample.episode_index),
                       "step": int(sample.step), "view": view, "pitch_deg": pitch,
                       "radii": {}}
                for radius in args.radii:
                    t0 = time.time()
                    res = render_cloud_region(
                        ctx.cloud, ctx.grid, cam, splat_radius=radius,
                        lod=cfg["render"]["lod"])
                    hist = depth_histogram(res.depth)
                    row["radii"][str(radius)] = {
                        "valid_pixel_ratio": float(res.valid_ratio),
                        "below_horizon_valid_ratio":
                            res.stats.get("below_horizon_valid_ratio"),
                        "in_tile_valid_ratio": res.stats.get("in_tile_valid_ratio"),
                        "depth_median": hist.get("median"),
                        "depth_min": hist.get("min"),
                        "seconds": round(time.time() - t0, 2),
                    }
                rows.append(row)
                base = row["radii"]["0"]["valid_pixel_ratio"]
                gains = {k: round(v["valid_pixel_ratio"] - base, 4)
                         for k, v in row["radii"].items() if k != "0"}
                print(f"{map_name:22s} {view:14s} raw={base:.3f} gains={gains}",
                      flush=True)

    summary = {}
    for view in args.views:
        sub = [r for r in rows if r["view"] == view]
        if not sub:
            continue
        entry = {"poses": len(sub)}
        for radius in args.radii:
            vals = np.array([r["radii"][str(radius)]["valid_pixel_ratio"] for r in sub])
            entry[f"radius_{radius}"] = {
                "median_valid": float(np.median(vals)),
                "min": float(vals.min()), "max": float(vals.max()),
            }
        raw = np.array([r["radii"]["0"]["valid_pixel_ratio"] for r in sub])
        for radius in args.radii:
            if radius == 0:
                continue
            got = np.array([r["radii"][str(radius)]["valid_pixel_ratio"] for r in sub])
            entry[f"median_gain_radius_{radius}"] = float(np.median(got - raw))
        summary[view] = entry

    report = {"radii": args.radii, "summary": summary, "rows": rows}
    path = out_dir / "splat_comparison.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\n" + json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
