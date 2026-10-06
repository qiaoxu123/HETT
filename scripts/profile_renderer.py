#!/usr/bin/env python3
"""Locate the per-frame rendering hotspot, on one fixed pose.

Writes ``artifacts/perf/profile_render.prof`` (loadable with
``python -m pstats``) and ``profile_render.txt`` with the top-50 entries by
cumulative and by total time, plus a stage breakdown and a scaling sweep.

The pose is fixed so successive runs are comparable: birmingham_block_1,
episode 0, mid-trajectory, FPV (config pitch), 512x512, default FOV and LOD.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.pointcloud_renderer import (  # noqa: E402
    Camera, StageTimer, render, render_cloud_region,
)


def fixed_pose(cfg, map_name: str = "birmingham_block_1"):
    objects_by_map = load_landmarks(cfg)
    ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)
    ep = ctx.episodes[0]
    step = len(ep.trajectory) // 2
    position = ep.trajectory[step, :3]
    yaw = float(ep.yaw()[step])
    render_cfg = cfg["render"]
    camera = Camera(
        position=position, yaw=yaw,
        pitch=np.deg2rad(render_cfg["views"][render_cfg.get("primary_view", "fpv")]),
        width=render_cfg["width"], height=render_cfg["height"],
        hfov_deg=render_cfg["hfov_deg"], near=render_cfg["near"], far=render_cfg["far"],
    )
    return ctx, camera, render_cfg, (map_name, ep.index, step)


def render_once(ctx, camera, render_cfg, zbuffer, splat_radius=0, lod=None, timer=None):
    return render_cloud_region(
        ctx.cloud, ctx.grid, camera, splat_radius=splat_radius,
        lod=render_cfg["lod"] if lod is None else lod,
        zbuffer=zbuffer, timer=timer,
    )


def profile_frame(ctx, camera, render_cfg, zbuffer, prof_path: Path) -> dict:
    """Warm up, then profile exactly one frame."""
    render_once(ctx, camera, render_cfg, zbuffer)  # warm page cache and imports

    profiler = cProfile.Profile()
    profiler.enable()
    result = render_once(ctx, camera, render_cfg, zbuffer)
    profiler.disable()
    profiler.dump_stats(str(prof_path))
    return result


def write_pstats(prof_path: Path, out_path: Path, top: int = 50) -> None:
    buf = io.StringIO()
    stats = pstats.Stats(str(prof_path), stream=buf)
    buf.write("=" * 78 + "\nSORTED BY CUMULATIVE TIME\n" + "=" * 78 + "\n")
    stats.sort_stats("cumulative").print_stats(top)
    buf.write("\n" + "=" * 78 + "\nSORTED BY TOTAL TIME\n" + "=" * 78 + "\n")
    stats.sort_stats("tottime").print_stats(top)
    out_path.write_text(buf.getvalue())


def stage_report(result) -> dict:
    stats = result.stats
    timings = stats["timings_s"]
    counts = stats["counts"]
    order = ["T_query", "T_load", "T_range_filter", "T_project", "T_zbuffer",
             "T_density", "T_splat", "T_finalize"]
    lines = []
    for name in order:
        if name in timings:
            lines.append(f"  {name:18s} {timings[name]:7.3f} s")
    lines.append(f"  {'T_total':18s} {stats['T_total']:7.3f} s")
    return {"timings": timings, "counts": counts, "lines": lines}


def scaling_sweep(ctx, camera, render_cfg, zbuffer, fractions) -> list:
    """Render with a growing point budget to reveal the complexity in N."""
    rows = []
    for frac in fractions:
        lod = ((np.inf, max(int(round(1.0 / frac)), 1)),)
        timer = StageTimer()
        res = render_once(ctx, camera, render_cfg, zbuffer, lod=lod, timer=timer)
        rows.append({
            "target_fraction": frac,
            "N_loaded": res.stats["counts"].get("N_loaded"),
            "N_projected": res.stats["counts"].get("N_projected"),
            **res.stats["timings_s"],
            "T_total": res.stats["T_total"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--map", default="birmingham_block_1")
    ap.add_argument("--zbuffer", default="argsort",
                    choices=["lexsort", "argsort", "minimum_at"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-scaling", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    render_cfg = dict(cfg["render"])
    render_cfg["lod"] = tuple(tuple(x) for x in render_cfg["lod"])

    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "perf"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ctx, camera, render_cfg, pose_id = fixed_pose(cfg, args.map)
    context_build_s = time.time() - t0

    prof_path = out_dir / "profile_render.prof"
    result = profile_frame(ctx, camera, render_cfg, args.zbuffer, prof_path)
    write_pstats(prof_path, out_dir / "profile_render.txt")

    stages = stage_report(result)

    print(f"pose: map={pose_id[0]} episode={pose_id[1]} step={pose_id[2]}")
    print(f"camera: pos={np.round(camera.position, 2).tolist()} "
          f"yaw={np.degrees(camera.yaw):.1f} pitch={np.degrees(camera.pitch):.1f} "
          f"hfov={camera.hfov_deg} {camera.width}x{camera.height}")
    print(f"context build (one-off): {context_build_s:.1f} s")
    print(f"zbuffer backend: {args.zbuffer}")
    print("\n--- stages ---")
    print("\n".join(stages["lines"]))
    print("\n--- counts ---")
    for k, v in stages["counts"].items():
        print(f"  {k:18s} {v:>12,}")
    print(f"\nvalid_pixel_ratio = {result.valid_ratio:.4f}   "
          f"in_tile = {result.stats.get('in_tile_valid_ratio')}")

    report = {
        "pose": {"map": pose_id[0], "episode": pose_id[1], "step": pose_id[2]},
        "camera": camera.to_json(),
        "zbuffer": args.zbuffer,
        "context_build_s": round(context_build_s, 2),
        "stages": stages,
        **result.stats["timings_s"],
        "T_total": result.stats["T_total"],
        "valid_pixel_ratio": float(result.valid_ratio),
    }

    if not args.skip_scaling:
        print("\n--- scaling sweep ---")
        sweep = scaling_sweep(ctx, camera, render_cfg, args.zbuffer,
                              [0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
        for row in sweep:
            print(f"  N_loaded={row['N_loaded']:>10,} proj={row['N_projected']:>9,} "
                  f"project={row.get('T_project', 0):6.3f} "
                  f"zbuf={row.get('T_zbuffer', 0):6.3f} "
                  f"range={row.get('T_range_filter', 0):6.3f} "
                  f"total={row['T_total']:6.3f}")
        report["scaling"] = sweep

    (out_dir / "profile_render.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"\nwrote {prof_path}\n      {out_dir / 'profile_render.txt'}\n"
          f"      {out_dir / 'profile_render.json'}")


if __name__ == "__main__":
    main()
