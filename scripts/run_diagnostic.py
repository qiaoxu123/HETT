#!/usr/bin/env python3
"""Step 1-2: per-map coordinate diagnostic (bounds, z, landmark alignment)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.config import (  # noqa: E402
    all_citynav_maps, artifact_dir, build_map_context, load_config,
)
from sensaturban_fpv.coordinate_diagnostic import summarise_map  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--maps", nargs="*", default=None)
    ap.add_argument("--limit-maps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = args.maps or all_citynav_maps(cfg)
    if args.limit_maps:
        maps = maps[: args.limit_maps]

    splits = ("train_seen", "val_seen", "val_unseen", "test_unseen")
    # Each map is appended as it finishes: the sweep covers 34 blocks and takes
    # long enough that losing the whole run to one bad block is not acceptable.
    jsonl_path = out_dir / "coordinate_diagnostic.jsonl"
    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["map_name"])

    reports = []
    for map_name in maps:
        if map_name in done:
            continue
        t0 = time.time()
        ctx = build_map_context(cfg, map_name, splits=splits)
        lo, hi = ctx.bounds
        report = summarise_map(
            map_name, ctx.cloud, ctx.grid, ctx.episodes, ctx.landmarks, lo, hi,
            z_radius=cfg["diagnostic"]["z_search_radius_m"],
            max_poses=cfg["diagnostic"]["max_poses_per_map"],
            max_landmarks=cfg["diagnostic"]["landmark_sample_per_map"],
        )
        report["point_count"] = int(len(ctx.cloud))
        report["ply_path"] = str(ctx.cloud.path)
        report["elapsed_s"] = round(time.time() - t0, 2)
        report["forward_consistency"] = [
            citynav.check_forward_consistency(ep) for ep in ctx.episodes[:3]
        ]
        reports.append(report)
        with jsonl_path.open("a") as handle:
            handle.write(json.dumps(report, sort_keys=True) + "\n")

        a = report["A_bounds_overlap"]["trajectory_vs_cloud"]
        b = report.get("B_z_consistency", {})
        c = report.get("C_landmark_alignment", {})
        print(
            f"{map_name:24s} N={report['point_count']:>9,} "
            f"traj_in_cloud={a['overlap_frac_of_a']:.3f} "
            f"above_ground={b.get('above_local_ground_ratio', float('nan')):.3f} "
            f"lm_nonempty_r5={c.get('nonempty_ratio_r5', float('nan')):.3f} "
            f"({report['elapsed_s']}s)",
            flush=True,
        )

    summary_path = out_dir / "coordinate_diagnostic.json"
    summary_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")

    agg = _aggregate(reports)
    (out_dir / "coordinate_diagnostic_summary.json").write_text(
        json.dumps(agg, indent=2, sort_keys=True) + "\n")
    print("\n" + json.dumps(agg, indent=2, sort_keys=True))
    print(f"\nwrote {summary_path}")


def _aggregate(reports: list) -> dict:
    """Pool the per-map results: the verdict is about the *set* of blocks."""
    traj_overlap = [r["A_bounds_overlap"]["trajectory_vs_cloud"]["overlap_frac_of_a"]
                    for r in reports]
    z_above = [r["B_z_consistency"]["above_local_ground_ratio"]
               for r in reports if "B_z_consistency" in r]
    z_anom = [r["B_z_consistency"]["anomaly_ratio"]
              for r in reports if "B_z_consistency" in r]
    z_alt = [r["B_z_consistency"]["altitude_median"]
             for r in reports if "B_z_consistency" in r
             and r["B_z_consistency"]["altitude_median"] is not None]
    lm_r5 = [r["C_landmark_alignment"]["nonempty_ratio_r5"]
             for r in reports if "C_landmark_alignment" in r]
    lm_r20 = [r["C_landmark_alignment"]["nonempty_ratio_r20"]
              for r in reports if "C_landmark_alignment" in r]
    lm_pts = [r["C_landmark_alignment"]["median_points_r5"]
              for r in reports if "C_landmark_alignment" in r]

    def stat(xs):
        if not xs:
            return None
        a = np.asarray(xs, dtype=np.float64)
        return {"n": int(a.size), "mean": float(a.mean()),
                "median": float(np.median(a)), "min": float(a.min()),
                "max": float(a.max())}

    return {
        "maps": len(reports),
        "A_trajectory_xy_inside_cloud_ratio": stat(traj_overlap),
        "B_uav_above_local_ground_ratio": stat(z_above),
        "B_altitude_anomaly_ratio": stat(z_anom),
        "B_median_altitude_above_ground_m": stat(z_alt),
        "C_landmark_nonempty_ratio_r5": stat(lm_r5),
        "C_landmark_nonempty_ratio_r20": stat(lm_r20),
        "C_landmark_median_points_r5": stat(lm_pts),
    }


if __name__ == "__main__":
    main()
