#!/usr/bin/env python3
"""Is the world-coordinate correspondence between the two views actually right?

Before any model is trained, the mapping has to be checked the only way it can
be: by looking at it, plus one statistic that a wrong mapping cannot pass.

The figure shows, per candidate, the top-down crop and the oblique crop with the
landmark's own projected patch mask outlined on each.  What has to hold is that
both outlines sit on *the same building*, that they do not spill onto a
neighbour, and that the oblique outline excludes what is behind an occluder.

The statistic is :func:`correspondence_coherence`: the mean patch-grid distance
between where a top-down patch's correspondence points and where its neighbours'
correspondences point.  Projecting one set of 3D points through two cameras
gives a locally smooth map, so the real pairing must be far more coherent than
the shuffled pairing that preserves weights, masks and sparsity.  If it is not,
the alignment is not carrying anything and the round stops here.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.landmark_geometry import (  # noqa: E402
    candidate_crops, candidate_geometry, correspondence_coherence,
    draw_mask_overlay, landmark_point_set, oblique_window,
    shuffled_correspondence, topdown_window,
)
from sensaturban_fpv.pointcloud_renderer import Camera, render_cloud_region  # noqa: E402
from run_gate_b import build_samples, topdown_raster  # noqa: E402

OBLIQUE_PITCH = -45.0
OBLIQUE_SIZE = 2048
PANEL = 384


def _fit(image, side: int = PANEL):
    if cv2 is None:
        return image
    return cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)


def _label(canvas, text, origin, colour=(255, 255, 255)):
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0),
                3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour,
                1, cv2.LINE_AA)


def render_candidate_figure(rows, path: Path, title: str) -> None:
    """One row per candidate: top-down + mask, oblique + mask, and the numbers."""
    if cv2 is None:
        return
    header = 46
    row_h = PANEL + 34
    width = 2 * (PANEL + 12) + 420
    canvas = np.full((header + row_h * len(rows), width, 3), 22, np.uint8)
    _label(canvas, title, (10, 30), (240, 240, 240))
    for i, row in enumerate(rows):
        y0 = header + i * row_h
        td = _fit(row["td_crop"])
        ob = _fit(row["o_crop"])
        canvas[y0:y0 + PANEL, 6:6 + PANEL] = td
        canvas[y0:y0 + PANEL, 6 + PANEL + 12:6 + 2 * PANEL + 12] = ob
        tag = "TARGET" if row["is_target"] else "distractor"
        colour = (90, 220, 90) if row["is_target"] else (200, 200, 200)
        _label(canvas, f"cand {row['candidate_id']}  {tag}  d={row['distance_m']:.0f}m",
               (12, y0 + PANEL + 24), colour)
        x = 6 + 2 * PANEL + 24
        lines = [
            f"top-down mask : {row['td_patches']:3d} patches, {row['td_pixels']} pts",
            f"oblique  mask : {row['o_patches']:3d} patches, {row['o_visible']} visible pts",
            f"correspondence: {row['pairs']:4d} patch pairs, "
            f"{row['shared']:.0f} shared pts",
            f"coherence     : {row['coherence']:.2f}   shuffled {row['coherence_shuffled']:.2f}",
            f"oblique crop still contains the landmark: {row['o_crop_ok']}",
        ]
        for j, text in enumerate(lines):
            _label(canvas, text, (x, y0 + 24 + j * 26))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--maps", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=["val_seen"])
    ap.add_argument("--patch-grid", type=int, default=32)
    ap.add_argument("--max-per-map", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "geo_fusion" / "check"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_by_map = load_landmarks(cfg)
    samples = build_samples(cfg, objects_by_map, args.splits, args.samples,
                            max_per_map=args.max_per_map)
    if args.maps:
        want = set(args.maps)
        samples = [s for s in samples if s["map"] in want]

    rng = np.random.default_rng(0)
    fig_rows, records, figure_index = [], [], 0
    cache, rasters = {}, {}
    t0 = time.time()
    for sample in samples:
        map_name = sample["map"]
        if map_name not in cache:
            cache = {map_name: build_map_context(cfg, map_name,
                                                 objects_by_map=objects_by_map)}
            rasters = {map_name: topdown_raster(Path(cfg["paths"]["ortho_dir"]), map_name)}
        ctx = cache[map_name]
        raster_info = rasters[map_name]
        position = np.asarray(sample["position"], dtype=np.float64)
        yaw = float(np.deg2rad(sample["yaw_deg"]))
        cam = Camera(position=position, yaw=yaw, pitch=np.deg2rad(OBLIQUE_PITCH),
                     width=OBLIQUE_SIZE, height=OBLIQUE_SIZE,
                     hfov_deg=cfg["render"]["hfov_deg"], near=cfg["render"]["near"],
                     far=cfg["render"]["far"])
        res = render_cloud_region(ctx.cloud, ctx.grid, cam, splat_radius=0,
                                  lod=cfg["render"]["lod"])
        sorted_xyz, _ = ctx.grid.sorted_arrays()
        objs = objects_by_map[map_name]

        for cid in sample["candidate_ids"]:
            obj = objs[cid]
            pts = landmark_point_set(ctx.grid, sorted_xyz, obj.position,
                                     obj.dimension, footprint=obj.contour)
            geom = candidate_geometry(cid, obj.position, obj.dimension, pts,
                                      raster_info, cam, res, sorted_xyz,
                                      patch_grid=args.patch_grid,
                                      footprint=obj.contour)
            if geom is None:
                continue
            views = candidate_crops(raster_info, cam, res.rgb, obj.position,
                                    geom.extent_m)
            if views["td"] is None or views["oblique"] is None:
                continue
            o_win = views["oblique_window"]
            o_crop, td_crop = views["oblique"], views["td"]
            shuf = shuffled_correspondence(geom, args.patch_grid, seed=int(cid) + 7)
            row = {
                "map": map_name, "episode_index": int(sample["episode_index"]),
                "step": int(sample["step"]), "candidate_id": int(cid),
                "is_target": bool(cid == sample["target_id"]),
                "distance_m": float(np.linalg.norm(
                    np.asarray(obj.position, float) - position)),
                "type": obj.object_type,
                "extent_m": geom.extent_m,
                "td_patches": geom.td_patch_count,
                "o_patches": geom.o_patch_count,
                "td_pixels": geom.td_point_count,
                "o_visible": geom.o_visible_point_count,
                "pairs": int(geom.corr_td.size),
                "shared": float(geom.corr_weight.sum()) if geom.corr_weight.size else 0.0,
                "coherence": correspondence_coherence(geom, args.patch_grid),
                "coherence_shuffled": correspondence_coherence(shuf, args.patch_grid),
                "o_crop_ok": bool(0 <= o_win.centre[0] < res.rgb.shape[1]
                                  and 0 <= o_win.centre[1] < res.rgb.shape[0]),
                "td_crop": draw_mask_overlay(td_crop, geom.td_patches, (255, 60, 60)),
                "o_crop": draw_mask_overlay(o_crop, geom.o_patches, (60, 220, 255)),
            }
            records.append({k: v for k, v in row.items()
                            if k not in ("td_crop", "o_crop")})
            fig_rows.append(row)
            if len(fig_rows) == 5:
                render_candidate_figure(
                    fig_rows, out_dir / f"correspondence_{figure_index:02d}.jpg",
                    f"{map_name}  episode {sample['episode_index']}  step {sample['step']}"
                    f"  patch grid {args.patch_grid}x{args.patch_grid}")
                fig_rows, figure_index = [], figure_index + 1
        print(f"  {map_name} ep{sample['episode_index']}: "
              f"{len(records)} candidates so far ({time.time() - t0:.0f}s)", flush=True)

    if fig_rows:
        render_candidate_figure(fig_rows, out_dir / f"correspondence_{figure_index:02d}.jpg",
                                f"final batch  patch grid {args.patch_grid}")
        figure_index += 1

    coh = np.array([r["coherence"] for r in records], dtype=float)
    coh_s = np.array([r["coherence_shuffled"] for r in records], dtype=float)
    ok = np.isfinite(coh) & np.isfinite(coh_s)
    with_pair = [r for r in records if r["pairs"] > 0]
    summary = {
        "candidates": len(records),
        "figures": figure_index,
        "candidates_with_correspondence": len(with_pair),
        "correspondence_success_rate": (len(with_pair) / len(records)) if records else None,
        "median_pairs": float(np.median([r["pairs"] for r in records])) if records else None,
        "median_td_patches": float(np.median([r["td_patches"] for r in records]))
        if records else None,
        "median_o_patches": float(np.median([r["o_patches"] for r in records]))
        if records else None,
        "coherence_real_median": float(np.nanmedian(coh)) if coh.size else None,
        "coherence_shuffled_median": float(np.nanmedian(coh_s)) if coh_s.size else None,
        "coherence_real_mean": float(np.nanmean(coh)) if coh.size else None,
        "coherence_shuffled_mean": float(np.nanmean(coh_s)) if coh_s.size else None,
        "coherence_paired_n": int(ok.sum()),
        "coherence_real_wins": int(np.sum(coh[ok] < coh_s[ok])) if ok.any() else 0,
        "figures_written": [f"correspondence_{i:02d}.jpg" for i in range(figure_index)],
    }
    (out_dir / "correspondence_check.json").write_text(
        json.dumps({"summary": summary, "candidates": records},
                   indent=2, sort_keys=True, default=float) + "\n")
    print("\n--- correspondence check ---")
    for key in ("candidates", "candidates_with_correspondence",
                "correspondence_success_rate", "median_pairs",
                "coherence_real_median", "coherence_shuffled_median",
                "coherence_real_wins"):
        print(f"  {key:34s} {summary[key]}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
