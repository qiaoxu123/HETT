#!/usr/bin/env python3
"""Gate A: is the perspective render usable as a visual input at all?

Picks the poses where a referenced landmark is close and in frame, renders the
three perspective views plus a top-down context, then answers six questions with
measurements rather than with an impression:

A1 structure   -- what the frame is made of, by the cloud's own semantic labels
A2 depth       -- finite, varied, and ordered
A3 landmark    -- the referenced landmark projects where the geometry says
A4 yaw         -- turning the camera rotates the view by the predicted amount
A5 continuity  -- consecutive poses give a smooth, correctly-directed sequence
A6 coverage    -- how much of the frame is backed by measured points
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

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

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
from sensaturban_fpv.pose_selection import camera_for, select_poses  # noqa: E402
from sensaturban_fpv.project_landmarks import (  # noqa: E402
    draw_landmarks, landmark_visibility, summarise_visibility,
)
from sensaturban_fpv.render_trajectory_fpv import topdown_context  # noqa: E402

# SensatUrban label ids.  The mapping is not asserted from memory: it is
# re-derived from the cloud itself in `label_profile`, which shows class 0
# hugging the ground, class 1 elevated like canopy, and class 2 spanning the
# full building height range.
LABEL_NAMES = {
    0: "ground", 1: "high_vegetation", 2: "buildings", 3: "walls", 4: "bridge",
    5: "parking", 6: "rail", 7: "roads", 8: "street_furniture", 9: "vehicles",
    10: "pedestrian", 11: "fence", 12: "traffic_light",
}

VIEWS = {"fpv": -10.0, "oblique30": -30.0, "oblique45": -45.0}


def label_profile(ctx, sample_limit: int = 200000) -> dict:
    """Per-label height above the local ground, measured from the cloud.

    This is what justifies reading label 2 as 'buildings' rather than taking the
    dataset's documentation on trust.
    """
    n = len(ctx.cloud)
    idx = np.arange(0, n, max(n // sample_limit, 1))
    xyz = ctx.cloud.xyz(idx)
    labels = ctx.cloud.labels(idx)
    ground = float(np.percentile(xyz[:, 2], 1.0))
    out = {}
    for label in np.unique(labels):
        z = xyz[labels == label, 2] - ground
        out[int(label)] = {
            "count": int((labels == label).sum()),
            "z_p05": float(np.percentile(z, 5)), "z_median": float(np.median(z)),
            "z_p95": float(np.percentile(z, 95)), "z_max": float(z.max()),
        }
    return out


def rendered_label_mix(result, ctx) -> dict:
    """Semantic composition of the points that actually won pixels."""
    winner = result.point_id
    valid = winner >= 0
    if not valid.any():
        return {}
    slots = winner[valid]
    vertex = ctx.grid.order[slots]
    labels = np.asarray(ctx.cloud.labels(vertex))
    counts = np.bincount(labels, minlength=13)
    total = int(counts.sum())
    return {LABEL_NAMES.get(i, str(i)): round(float(c) / total, 4)
            for i, c in enumerate(counts) if c}


def render_views(ctx, sample, cfg, transform) -> dict:
    position = transform.apply_xyz(np.asarray(sample["position"])[None, :])[0]
    out = {}
    for name, pitch in VIEWS.items():
        cam = camera_for(position, np.deg2rad(sample["yaw_deg"]), pitch, cfg)
        result = render_cloud_region(
            ctx.cloud, ctx.grid, cam, splat_radius=cfg["render"]["splat_radius"],
            lod=cfg["render"]["lod"])
        out[name] = {"result": result, "camera": cam}
    return out, position


def write_pose(pose_dir: Path, sample, views, context, overlays, metadata) -> dict:
    pose_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    if context.get("image") is not None:
        p = pose_dir / "topdown_context.png"
        cv2.imwrite(str(p), cv2.cvtColor(context["image"], cv2.COLOR_RGB2BGR))
        written["topdown_context"] = str(p)
    for name, entry in views.items():
        res = entry["result"]
        rgb_path = pose_dir / f"{name}_rgb.png"
        depth_path = pose_dir / f"{name}_depth.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(res.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(depth_path), depth_to_png(res.depth))
        written[f"{name}_rgb"] = str(rgb_path)
        written[f"{name}_depth"] = str(depth_path)
        if name == "fpv":
            overlay_path = pose_dir / "landmark_overlay.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlays, cv2.COLOR_RGB2BGR))
            written["landmark_overlay"] = str(overlay_path)
    meta_path = pose_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    written["metadata"] = str(meta_path)
    return written


def depth_to_png(depth: np.ndarray, far: float = 400.0) -> np.ndarray:
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    finite = np.isfinite(depth)
    if finite.any():
        norm = np.clip(depth / far, 0.0, 1.0)
        scaled = (255 * (1.0 - norm)).astype(np.uint8)
        out[finite] = cv2.applyColorMap(
            scaled[finite].reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return out


def contact_sheet(rows: list, out_path: Path, tile: int = 256) -> None:
    """One row per pose: top-down | FPV | oblique30 | oblique45, with a caption."""
    if not rows:
        return
    caption_h = 34
    panels = []
    for row in rows:
        tiles = []
        for key in ("topdown_context", "fpv_rgb", "oblique30_rgb", "oblique45_rgb"):
            path = row["artifacts"].get(key)
            img = cv2.imread(path) if path else None
            if img is None:
                img = np.zeros((tile, tile, 3), np.uint8)
            tiles.append(cv2.resize(img, (tile, tile)))
        strip = np.concatenate(tiles, axis=1)
        caption = (f"{row['split']}:{row['episode_index']} step{row['step']} "
                   f"{row['map']} | {row['landmark_name']} d={row['distance']:.0f}m "
                   f"| valid fpv={row['view_stats']['fpv']['valid']:.2f} "
                   f"o30={row['view_stats']['oblique30']['valid']:.2f} "
                   f"o45={row['view_stats']['oblique45']['valid']:.2f} "
                   f"| {row['view_stats']['oblique45']['composition']}")
        bar = np.zeros((caption_h, strip.shape[1], 3), np.uint8)
        cv2.putText(bar, caption, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.concatenate([bar, strip], axis=0))
    sheet = np.concatenate(panels, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quota", nargs="*", default=["val_unseen=10", "val_seen=6",
                                                   "train_seen=6"])
    ap.add_argument("--max-per-map", type=int, default=2)
    ap.add_argument("--continuity-steps", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    out_root = Path(args.out) if args.out else artifact_dir(cfg) / "gate_a"
    out_root.mkdir(parents=True, exist_ok=True)

    transform_path = Path(args.transform) if args.transform else \
        artifact_dir(cfg) / "transform" / "best_transform.json"
    transform = load_transform(transform_path) if transform_path.exists() \
        else CoordinateTransform.identity()

    quota = {}
    for item in args.quota:
        split, value = item.split("=")
        quota[split] = int(value)

    objects_by_map = load_landmarks(cfg)
    episodes_by_split = {
        split: citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split)
        for split in quota
    }

    # Scoring is pure geometry over CityRefer positions, so it needs no point
    # cloud at all.  Opening a block costs seconds to minutes; doing it for all
    # 34 candidate maps to then render 22 of them is most of the runtime.
    landmarks_by_map = {
        map_name: [(o.id, o.name, o.object_type,
                    tuple(float(v) for v in o.position),
                    tuple(float(v) for v in o.dimension))
                   for o in objects.values()]
        for map_name, objects in objects_by_map.items()
    }
    selected = select_poses(episodes_by_split, landmarks_by_map, cfg,
                            quota, VIEWS, max_per_map=args.max_per_map)
    print(f"selected {len(selected)} poses over "
          f"{len({s['map'] for s in selected})} maps: "
          + json.dumps({s: sum(1 for e in selected if e["split"] == s) for s in quota}))
    (out_root / "selected_poses.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True, default=float) + "\n")
    if args.dry_run:
        return

    t0 = time.time()
    contexts = {}
    for map_name in sorted({s["map"] for s in selected}):
        contexts[map_name] = build_map_context(cfg, map_name,
                                               objects_by_map=objects_by_map)
        print(f"  context {map_name} ({time.time() - t0:.0f}s)", flush=True)

    episode_lookup = {(sp, ep.index): ep
                      for sp, eps in episodes_by_split.items() for ep in eps}

    rows = []
    composition = defaultdict(list)
    for i, sample in enumerate(selected):
        ctx = contexts[sample["map"]]
        episode = episode_lookup[(sample["split"], sample["episode_index"])]
        views, position_ply = render_views(ctx, sample, cfg, transform)

        ref_lm = next(lm for lm in ctx.landmarks if lm[0] == sample["landmark_id"])
        overlay, annotated = draw_landmarks(
            views["fpv"]["result"].rgb, ctx.landmarks, views["fpv"]["camera"],
            views["fpv"]["result"], referenced_ids=episode.object_ids,
            cloud=ctx.cloud, grid=ctx.grid)
        vis_summary = summarise_visibility(annotated, episode.object_ids)
        ref_vis = next((a for a in annotated if a["id"] == sample["landmark_id"]), None)

        # Standalone pose record for the top-down context panel.
        from sensaturban_fpv.render_trajectory_fpv import PoseSample
        pose_sample = PoseSample(sample["split"], sample["episode_index"],
                                 sample["map"], sample["step"],
                                 np.array([*sample["position"], np.deg2rad(sample["yaw_deg"])]),
                                 position_ply)
        context = topdown_context(ctx.cloud, ctx.grid, pose_sample,
                                  extent=cfg["render"]["context_extent_m"])

        view_stats = {}
        for name, entry in views.items():
            res = entry["result"]
            hist = depth_histogram(res.depth)
            mix = rendered_label_mix(res, ctx)
            composition[name].append(mix)
            view_stats[name] = {
                "pitch_deg": VIEWS[name],
                "valid": float(res.valid_ratio),
                "below_horizon": res.stats.get("below_horizon_valid_ratio"),
                "in_tile": res.stats.get("in_tile_valid_ratio"),
                "sky_fraction": res.stats.get("sky_pixel_ratio"),
                "depth_min": hist.get("min"), "depth_median": hist.get("median"),
                "depth_max": hist.get("max"), "depth_std": hist.get("std"),
                "depth_unique": hist.get("unique_rounded"),
                "points_projected": res.stats["points_projected"],
                "seconds": round(res.stats["T_total"], 3),
                "composition": mix,
            }

        metadata = {
            "episode_id": f"{sample['split']}:{sample['episode_index']}",
            "split": sample["split"],
            "step": sample["step"],
            "map": sample["map"],
            "pose_xyz": [float(v) for v in sample["position"]],
            "pose_ply_xyz": [float(v) for v in position_ply],
            "yaw": float(np.deg2rad(sample["yaw_deg"])),
            "yaw_deg": float(sample["yaw_deg"]),
            "views_deg": VIEWS,
            "fov_deg": float(cfg["render"]["hfov_deg"]),
            "resolution": [cfg["render"]["width"], cfg["render"]["height"]],
            "referenced_landmark": {
                "id": int(ref_lm[0]), "name": ref_lm[1] or ref_lm[2],
                "type": ref_lm[2], "position": list(ref_lm[3]),
                "distance_m": float(sample["distance"]),
                "in_fov": {k: bool(v) for k, v in sample["in_view"].items()},
                "projection": (ref_vis or {}).get("projection"),
                "occluded_at_centre": (ref_vis or {}).get("occluded_at_centre"),
                "observed_points": (ref_vis or {}).get("observed_points"),
                "approx_visible_ratio": (ref_vis or {}).get("approx_visible_ratio"),
                "visible": (ref_vis or {}).get("visible"),
            },
            "same_class_count": int(sample["same_class_count"]),
            "instruction": episode.description,
            "views": view_stats,
            "landmark_summary": vis_summary,
            "coordinate_transform": transform.to_json(),
        }
        pose_dir = out_root / f"{sample['split']}_{sample['episode_index']:05d}_step{sample['step']:03d}"
        artifacts = write_pose(pose_dir, sample, views, context, overlay, metadata)
        rows.append({
            "split": sample["split"], "episode_index": sample["episode_index"],
            "step": sample["step"], "map": sample["map"],
            "landmark_name": ref_lm[1] or ref_lm[2], "distance": sample["distance"],
            "landmark_visible": bool((ref_vis or {}).get("visible")),
            "same_class_count": sample["same_class_count"],
            "artifacts": artifacts, "view_stats": view_stats,
            "metadata": metadata,
        })
        print(f"  [{i + 1}/{len(selected)}] {sample['split']}:{sample['episode_index']} "
              f"{sample['map']} d={sample['distance']:.0f}m "
              f"valid fpv={view_stats['fpv']['valid']:.2f} "
              f"o30={view_stats['oblique30']['valid']:.2f} "
              f"o45={view_stats['oblique45']['valid']:.2f}", flush=True)

    gates = evaluate_gates(rows, contexts, cfg, transform, episode_lookup, out_root,
                           args.continuity_steps)
    report = {"poses": len(rows), "quota": quota, "gates": gates, "rows": rows}
    (out_root / "gate_a.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    contact_sheet(rows, out_root / "contact_sheet.png")

    print("\n=== Gate A ===")
    for name, gate in gates.items():
        mark = {True: "PASS", False: "FAIL", None: "n/a"}[gate.get("passed")]
        print(f"  {name:24s} {mark}   {gate.get('summary', '')}")
    print(f"\nwrote {out_root / 'gate_a.json'}\n      {out_root / 'contact_sheet.png'}")


def evaluate_gates(rows, contexts, cfg, transform, episode_lookup, out_root, steps):
    gates = {}

    # A1 -- what the frame is made of.
    mixes = defaultdict(list)
    for row in rows:
        for view, stats in row["view_stats"].items():
            if stats["composition"]:
                mixes[view].append(stats["composition"])
    a1 = {}
    for view, entries in mixes.items():
        keys = set().union(*entries)
        a1[view] = {k: float(np.mean([e.get(k, 0.0) for e in entries]))
                    for k in sorted(keys)}
    gates["A1_structure"] = {
        "per_view_mean_composition": a1,
        "passed": all(
            a1.get(v, {}).get("buildings", 0.0) > 0.05
            and (a1.get(v, {}).get("ground", 0.0) + a1.get(v, {}).get("roads", 0.0)) > 0.05
            for v in ("fpv", "oblique30", "oblique45")),
        "summary": "mean rendered share of buildings / ground+roads per view",
    }

    # A2 -- depth.
    depth_ok = {}
    for view in ("fpv", "oblique30", "oblique45"):
        stds = [r["view_stats"][view]["depth_std"] for r in rows
                if r["view_stats"][view]["depth_std"] is not None]
        uniq = [r["view_stats"][view]["depth_unique"] for r in rows
                if r["view_stats"][view]["depth_unique"] is not None]
        depth_ok[view] = {"median_std": float(np.median(stds)) if stds else 0.0,
                          "min_unique": int(min(uniq)) if uniq else 0}
    gates["A2_depth"] = {
        "per_view": depth_ok,
        "passed": all(v["median_std"] > 0.5 and v["min_unique"] > 20
                      for v in depth_ok.values()),
        "summary": "depth std and distinct-value count per view",
    }

    # A3 -- landmark projection.
    in_fov = [r for r in rows if r["metadata"]["referenced_landmark"]["in_fov"].get("fpv")]
    visible = [r for r in in_fov if r["landmark_visible"]]
    gates["A3_landmark_projection"] = {
        "poses": len(rows),
        "referenced_in_fpv_fov": len(in_fov),
        "referenced_visible_in_fpv": len(visible),
        "visible_frac_of_in_fov": len(visible) / max(len(in_fov), 1),
        "passed": bool(len(in_fov) > 0 and len(visible) / max(len(in_fov), 1) >= 0.25),
        "summary": "referenced landmarks inside the FPV frame that are backed by "
                   "unoccluded measured points",
    }

    # A4 and A5 are scored by scripts/run_gate_a_yaw_continuity.py, which is run
    # straight after this one.  Both need an estimator whose valid range is
    # established from the data, and an earlier version of this file scored them
    # with a measure that turned out to be self-consistent rather than
    # falsifiable; keeping one implementation of each avoids two answers to one
    # question.
    gates["A4_yaw_consistency"] = {
        "gate": "A4_yaw_consistency", "passed": None,
        "note": "scored by scripts/run_gate_a_yaw_continuity.py"}
    gates["A5_trajectory_continuity"] = {
        "gate": "A5_trajectory_continuity", "passed": None,
        "note": "scored by scripts/run_gate_a_yaw_continuity.py"}

    # A6 -- coverage.
    per_view = {}
    for view in ("fpv", "oblique30", "oblique45"):
        vals = np.array([r["view_stats"][view]["valid"] for r in rows])
        below = np.array([r["view_stats"][view]["below_horizon"] or np.nan for r in rows])
        per_view[view] = {
            "median_valid": float(np.median(vals)),
            "frac_above_0.4": float((vals > 0.4).mean()),
            "frac_below_0.2": float((vals < 0.2).mean()),
            "median_below_horizon": float(np.nanmedian(below)),
        }
    gates["A6_coverage"] = {
        "per_view": per_view,
        "passed": all(v["frac_above_0.4"] >= 0.5 or v["median_below_horizon"] > 0.4
                      for v in per_view.values()),
        "summary": "whole-frame and below-horizon valid pixel ratios per view",
    }
    return gates


