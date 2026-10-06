#!/usr/bin/env python3
"""Step 4: does our own rasterisation of the PLY reproduce ``data/rgbd``?

This is a provenance and self-consistency check.  The HETT release archive
contains no imagery, and ``rasterize.py`` produced ``data/rgbd`` from the same
PLYs, so agreement here confirms that the map the agent navigates is the point
cloud -- it is *not* an independent check against an Unreal render, and the
report says so explicitly.
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

from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.pointcloud_renderer import rasterize_topdown  # noqa: E402

NODATA = -9999.0


def read_official(ortho_dir: Path, map_name: str):
    import rasterio
    from PIL import Image

    with rasterio.open(ortho_dir / f"{map_name}.tif") as ds:
        height = ds.read(1).astype(np.float64)
        transform = ds.transform
        nodata = ds.nodata
        shape = (ds.height, ds.width)
    rgb = np.asarray(Image.open(ortho_dir / f"{map_name}.png").convert("RGB"))
    origin = (transform.c, transform.f)
    res = transform.a
    return {"height": height, "rgb": rgb, "origin": origin, "resolution": res,
            "shape": shape, "nodata": nodata, "transform": transform}


def compare_rasters(mine: dict, official: dict) -> dict:
    """Cell-by-cell agreement, assuming both grids share origin and shape."""
    z_mine = mine["height"].astype(np.float64)
    z_off = official["height"]
    nodata = official["nodata"] if official["nodata"] is not None else NODATA
    if mine["shape"] != official["shape"]:
        return {"comparable": False,
                "reason": f"shape mismatch {mine['shape']} vs {official['shape']}"}

    valid_mine = np.isfinite(z_mine)
    valid_off = z_off > -9998.0
    both = valid_mine & valid_off
    diff = np.zeros(1)
    stats = {
        "comparable": True,
        "cells": int(z_mine.size),
        "valid_mine_ratio": float(valid_mine.mean()),
        "valid_official_ratio": float(valid_off.mean()),
        "valid_both_ratio": float(both.mean()),
        "mine_valid_official_empty": int((valid_mine & ~valid_off).sum()),
        "official_valid_mine_empty": int((valid_off & ~valid_mine).sum()),
    }
    if both.any():
        diff = z_mine[both] - z_off[both]
        stats.update({
            "height_mae_m": float(np.abs(diff).mean()),
            "height_max_abs_diff_m": float(np.abs(diff).max()),
            "height_exact_fraction": float((diff == 0).mean()),
            "height_corr": float(np.corrcoef(z_mine[both], z_off[both])[0, 1])
            if both.sum() > 1 else None,
        })

    rgb_mine = mine["rgb"].astype(np.float64)
    rgb_off = official["rgb"].astype(np.float64)
    if rgb_mine.shape == rgb_off.shape:
        m = both
        if m.any():
            stats["rgb_mae"] = float(np.abs(rgb_mine[m] - rgb_off[m]).mean())
            stats["rgb_exact_fraction"] = float((rgb_mine[m] == rgb_off[m]).mean())
            for c, name in enumerate("rgb"):
                a, b = rgb_mine[..., c][m], rgb_off[..., c][m]
                if a.std() > 0 and b.std() > 0:
                    stats[f"rgb_corr_{name}"] = float(np.corrcoef(a, b)[0, 1])
    return stats


def best_pixel_offset(off_height: np.ndarray, off_valid: np.ndarray,
                      mine: np.ndarray, mine_valid: np.ndarray,
                      max_shift: int = 40) -> dict:
    """Any residual whole-pixel misregistration between the two grids."""
    rows = min(off_height.shape[0], mine.shape[0])
    cols = min(off_height.shape[1], mine.shape[1])
    a = np.where(off_valid[:rows, :cols], off_height[:rows, :cols], np.nan)
    b = np.where(mine_valid[:rows, :cols], mine[:rows, :cols], np.nan)
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < 1000:
        return {"comparable": False}

    a = a - np.nanmean(a[both])
    b = b - np.nanmean(b[both])
    a = np.where(both, a, 0.0)
    b = np.where(both, b, 0.0)
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    corr = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    corr = np.fft.fftshift(corr)
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = peak[0] - a.shape[0] // 2
    dx = peak[1] - a.shape[1] // 2
    return {"comparable": True, "dy": int(dy), "dx": int(dx),
            "max_abs_shift_allowed": int(max_shift),
            "within_tolerance": bool(abs(dy) <= 1 and abs(dx) <= 1)}


def landmark_overlay(official: dict, landmarks: list, map_name: str, out_path: Path,
                     count: int = 10) -> dict:
    """Draw landmark centres on the official RGB raster and crop around each."""
    import cv2

    rgb = official["rgb"].copy()
    origin = official["origin"]
    res = official["resolution"]
    shape = official["shape"]

    selected = landmarks[:count]
    crops = []
    for lm in selected:
        lm_id, name, obj_type, pos, dim = lm
        col = int(round((pos[0] - origin[0]) / res))
        row = int(round((origin[1] - pos[1]) / res))
        inside = 0 <= row < shape[0] and 0 <= col < shape[1]
        colour = (255, 0, 0) if inside else (255, 255, 0)
        if inside:
            half = max(int(max(dim[0], dim[1]) / res / 2), 30)
            r0, r1 = max(row - half, 0), min(row + half, shape[0])
            c0, c1 = max(col - half, 0), min(col + half, shape[1])
            crop = rgb[r0:r1, c0:c1].copy()
            cv2.rectangle(crop, (col - c0 - 6, row - r0 - 6),
                          (col - c0 + 6, row - r0 + 6), colour, 2, cv2.LINE_AA)
            tag = f"{name or obj_type}#{lm_id}"
            cv2.putText(crop, tag, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            crops.append(crop)
            cv2.circle(rgb, (col, row), 9, colour, 3, cv2.LINE_AA)
        else:
            crops.append(None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Strip of crops, resized to a common height for one readable artifact.
    usable = [c for c in crops if c is not None]
    if usable:
        target = 192
        tiles = []
        for c in usable:
            scale = target / c.shape[0]
            tiles.append(cv2.resize(c, (max(int(c.shape[1] * scale), 32), target)))
        strip = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(out_path.with_name(out_path.stem + "_crops.png")),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    return {"drawn": int(sum(c is not None for c in crops)),
            "outside_raster": int(sum(c is None for c in crops)),
            "overlay": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--maps", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "topdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    ortho_dir = Path(cfg["paths"]["ortho_dir"])
    maps = args.maps or cfg["topdown_regression"]["maps"]
    objects_by_map = load_landmarks(cfg)
    reports = {}

    for map_name in maps:
        t0 = time.time()
        official = read_official(ortho_dir, map_name)
        ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)

        idx = np.arange(len(ctx.cloud), dtype=np.int64)
        # Same grid as the official raster, so cells line up one to one.
        mine = rasterize_topdown(
            ctx.cloud.xyz(idx), ctx.cloud.rgb(idx),
            origin=official["origin"], resolution=official["resolution"],
            shape=official["shape"], z_mode="max",
        )
        stats = compare_rasters(mine, official)
        stats["offset"] = best_pixel_offset(
            official["height"], official["height"] > -9998.0,
            np.nan_to_num(mine["height"], nan=-1e9), np.isfinite(mine["height"]),
        )

        landmarks = [
            (o.id, o.name, o.object_type,
             tuple(float(v) for v in o.position), tuple(float(v) for v in o.dimension))
            for o in objects_by_map.get(map_name, {}).values()
        ][: cfg["topdown_regression"]["landmark_overlay_count"]]
        if landmarks:
            stats["landmark_overlay"] = landmark_overlay(
                official, landmarks, map_name, out_dir / f"{map_name}_landmarks.png",
                count=cfg["topdown_regression"]["landmark_overlay_count"])

        stats["map_name"] = map_name
        stats["elapsed_s"] = round(time.time() - t0, 2)
        stats["official_origin"] = list(official["origin"])
        stats["official_resolution"] = official["resolution"]
        stats["official_shape"] = list(official["shape"])
        reports[map_name] = stats
        print(f"{map_name:22s} " + json.dumps(
            {k: v for k, v in stats.items()
             if k in ("valid_both_ratio", "height_exact_fraction", "height_mae_m",
                      "height_corr", "rgb_exact_fraction", "offset")},
            sort_keys=True), flush=True)

    (out_dir / "topdown_regression.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_dir / 'topdown_regression.json'}")


if __name__ == "__main__":
    main()
