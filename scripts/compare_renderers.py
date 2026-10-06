#!/usr/bin/env python3
"""Regression and benchmark across renderer configurations.

The reference is the original path: no frustum culling, lexsort z-buffer.  Every
faster configuration must reproduce it where both produce a surface, and the
report says so with numbers rather than with an assertion.

Also reports per-configuration wall time, so the speed claims and the
correctness claims come from the same run.
"""

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
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.pointcloud_renderer import Camera, render_cloud_region  # noqa: E402


def sample_poses(cfg, per_map=1, max_maps=10):
    """Spread over maps and over the configured views, deterministically."""
    objects_by_map = load_landmarks(cfg)
    splits = ("val_seen", "val_unseen")
    episodes = {s: citynav.load_split(Path(cfg["paths"]["citynav_dir"]), s) for s in splits}

    chosen = []
    seen_maps = set()
    for split, eps in episodes.items():
        for ep in eps:
            if not ep.object_ids or ep.map_name in seen_maps:
                continue
            seen_maps.add(ep.map_name)
            chosen.append((split, ep))
            if len(seen_maps) >= max_maps:
                break
        if len(seen_maps) >= max_maps:
            break

    out = []
    for split, ep in chosen:
        ctx = build_map_context(cfg, ep.map_name, objects_by_map=objects_by_map)
        for step_frac in (0.33, 0.66)[:per_map]:
            step = int(len(ep.trajectory) * step_frac)
            step = min(step, len(ep.trajectory) - 1)
            pose4 = np.array([*ep.trajectory[step, :3], float(ep.yaw()[step])])
            out.append({
                "map": ep.map_name, "split": split, "episode": int(ep.index),
                "step": int(step), "ctx": ctx, "pose4": pose4,
            })
    return out


def configure(cfg, sample, view, width, height, far):
    r = cfg["render"]
    return Camera(position=sample["pose4"][:3], yaw=float(sample["pose4"][3]),
                  pitch=np.deg2rad(r["views"][view]), width=width, height=height,
                  hfov_deg=r["hfov_deg"], near=r["near"], far=far)


def field(result, name):
    """Read a field from either a RenderResult or the device backend's dict."""
    if isinstance(result, dict):
        return result[name]
    return getattr(result, name)


def compare_to_reference(ref, other, tol=1e-3) -> dict:
    ref_valid = field(ref, "valid")
    other_valid = field(other, "valid")
    ref_depth = field(ref, "depth")
    other_depth = field(other, "depth")
    ref_rgb = field(ref, "rgb")
    other_rgb = field(other, "rgb")

    both = ref_valid & other_valid
    out = {
        "valid_ref": int(ref_valid.sum()),
        "valid_other": int(other_valid.sum()),
        "both": int(both.sum()),
        "ref_only": int((ref_valid & ~other_valid).sum()),
        "other_only": int((other_valid & ~ref_valid).sum()),
        "valid_overlap_frac": float(both.sum() / max(ref_valid.sum(), 1)),
    }
    if both.any():
        d = other_depth[both].astype(np.float64) - ref_depth[both].astype(np.float64)
        out.update({
            "depth_mae": float(np.abs(d).mean()),
            "depth_median_err": float(np.median(d)),
            "depth_p95_abs": float(np.percentile(np.abs(d), 95)),
            "depth_max_abs": float(np.abs(d).max()),
            "depth_within_1e-3": float((np.abs(d) <= tol).mean()),
        })
        a = ref_rgb[both].astype(np.int16)
        b = other_rgb[both].astype(np.int16)
        out.update({
            "rgb_exact": float((a == b).all(axis=1).mean()),
            "rgb_mean_abs_err": float(np.abs(a - b).mean()),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--poses-per-map", type=int, default=2)
    ap.add_argument("--max-maps", type=int, default=6)
    ap.add_argument("--views", nargs="*", default=["fpv", "oblique"])
    ap.add_argument("--far", type=float, default=None)
    ap.add_argument("--torch", action="store_true", help="also benchmark the GPU backend")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    r = cfg["render"]
    far = args.far if args.far is not None else r["far"]
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "perf"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = sample_poses(cfg, args.poses_per_map, args.max_maps)
    print(f"{len(samples)} poses over {len({s['map'] for s in samples})} maps, far={far}")

    configs = [
        ("reference_lexsort_nocull", dict(cull_frustum=False, zbuffer="lexsort")),
        ("cull_lexsort", dict(cull_frustum=True, zbuffer="lexsort")),
        ("cull_argsort", dict(cull_frustum=True, zbuffer="argsort")),
        ("cull_minimum_at", dict(cull_frustum=True, zbuffer="minimum_at")),
    ]

    cache = None
    if args.torch:
        from sensaturban_fpv.torch_backend import DeviceCache, torch_available
        if torch_available():
            cache = DeviceCache(device="cuda", mode="map")

    rows = []
    for sample in samples:
        for view in args.views:
            cam = configure(cfg, sample, view, r["width"], r["height"], far)
            results = {}
            for name, kwargs in configs:
                t0 = time.time()
                res = render_cloud_region(sample["ctx"].cloud, sample["ctx"].grid, cam,
                                          splat_radius=0, lod=r["lod"], **kwargs)
                dt = time.time() - t0
                results[name] = (res, dt)

            gpu = None
            if cache is not None:
                from sensaturban_fpv.torch_backend import render_cloud_region_torch
                t0 = time.time()
                gpu = render_cloud_region_torch(sample["ctx"].cloud, sample["ctx"].grid,
                                                cam, cache, lod=r["lod"])
                gpu["t"] = time.time() - t0

            ref, ref_t = results["reference_lexsort_nocull"]
            row = {"map": sample["map"], "episode": sample["episode"],
                   "step": sample["step"], "view": view, "far": far,
                   "seconds": {k: round(v[1], 4) for k, v in results.items()},
                   "ref_seconds": round(ref_t, 4)}
            if gpu is not None:
                row["seconds"]["cull_torch_gpu"] = round(gpu["t"], 4)
                row["gpu_peak_bytes"] = gpu["stats"]["gpu_peak_bytes"]
            for name, _ in configs[1:]:
                row[name] = compare_to_reference(ref, results[name][0])
            if gpu is not None:
                row["cull_torch_gpu"] = compare_to_reference(ref, gpu)
            rows.append(row)
            print(f"  {sample['map']:22s} {view:8s} "
                  + " ".join(f"{k.split('_')[-1]}={v:.3f}s" for k, v in row["seconds"].items()))

    summary = {}
    for name, _ in configs[1:]:
        keys = ["valid_overlap_frac", "depth_mae", "depth_p95_abs", "depth_max_abs",
                "rgb_exact", "rgb_mean_abs_err"]
        summary[name] = {k: float(np.mean([row[name][k] for row in rows if k in row[name]]))
                         for k in keys}
    summary["timing"] = {
        name: {"median_s": float(np.median([row["seconds"][name] for row in rows])),
               "max_s": float(np.max([row["seconds"][name] for row in rows]))}
        for name in rows[0]["seconds"]
    }

    report = {"far": far, "poses": len(rows), "summary": summary, "rows": rows}
    path = out_dir / "renderer_comparison.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")

    print("\n--- timing (median / max seconds) ---")
    for name, t in summary["timing"].items():
        print(f"  {name:28s} {t['median_s']:8.3f} / {t['max_s']:8.3f}")
    print("\n--- agreement with the reference renderer ---")
    for name, s in summary.items():
        if name == "timing":
            continue
        print(f"  {name:20s} overlap={s['valid_overlap_frac']:.4f} "
              f"depth_mae={s['depth_mae']:.5f} p95={s['depth_p95_abs']:.5f} "
              f"max={s['depth_max_abs']:.4f} rgb_exact={s['rgb_exact']:.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
