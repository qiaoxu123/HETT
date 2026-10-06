#!/usr/bin/env python3
"""Step 3: search the constrained transform class and record the winner."""

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
    all_citynav_maps, artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.fit_coordinate_transform import (  # noqa: E402
    CoordinateTransform, fit_transform, write_transform,
)


def landmarks_for_map(objects_by_map, map_name, limit):
    objects = objects_by_map.get(map_name, {})
    values = list(objects.values())
    if limit and len(values) > limit:
        # Spread over the whole object list rather than taking a spatial corner.
        picks = np.linspace(0, len(values) - 1, num=limit).round().astype(int)
        values = [values[int(i)] for i in picks]
    return np.array([list(o.position) for o in values], dtype=np.float64), values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--maps", nargs="*", default=None)
    ap.add_argument("--limit-maps", type=int, default=3)
    ap.add_argument("--apply-all", action="store_true",
                    help="write a per-map transform for every map, not just the probe set")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = artifact_dir(cfg) / "transform"
    out_dir.mkdir(parents=True, exist_ok=True)
    search = cfg["transform_search"]

    maps = args.maps or all_citynav_maps(cfg)[: args.limit_maps]
    objects_by_map = load_landmarks(cfg)

    per_map = {}
    for map_name in maps:
        t0 = time.time()
        ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)
        if not ctx.landmarks:
            print(f"{map_name}: no CityRefer landmarks, skipped")
            continue
        lm_xyz, _ = landmarks_for_map(
            objects_by_map, map_name, search["landmark_sample_per_map"])
        traj = (np.concatenate([e.trajectory[:, :3] for e in ctx.episodes], axis=0)
                if ctx.episodes else np.zeros((0, 3)))

        lo, hi = ctx.bounds
        result = fit_transform(
            traj, lm_xyz, lo, hi, ctx.grid,
            radius=search["radius_m"], scales=tuple(search["scales"]),
            coarse_extent=search["coarse_extent_m"], coarse_step=search["coarse_step_m"],
            fine_extent=search["fine_extent_m"], fine_step=search["fine_step_m"],
            target_points=search["target_points_per_landmark"],
        )
        best = result["best"]
        identity = result["identity"]
        entry = {
            "map_name": map_name,
            "elapsed_s": round(time.time() - t0, 2),
            "best": {k: v for k, v in best.items() if k != "transform"} | {
                "transform": best["transform"].to_json()},
            "identity": {k: v for k, v in identity.items() if k != "transform"} | {
                "transform": identity["transform"].to_json()},
            "ranking": [
                {"orientation": r["orientation"], "scale": r["scale"],
                 "score": r["score"], "in_bounds_ratio": r["in_bounds_ratio"],
                 "median_points": r["median_points"]}
                for r in result["all"]
            ],
            "search": result["search"],
        }
        per_map[map_name] = entry
        write_transform(
            out_dir / f"{map_name}.json", best["transform"],
            extra={"source": "run_transform_search", "score": best["score"],
                   "identity_score": identity["score"]},
        )
        print(
            f"{map_name:22s} best={best['orientation']:>12s} s={best['scale']:g} "
            f"score={best['score']:.4f}  identity_score={identity['score']:.4f}  "
            f"t={np.round(best['transform'].t, 3).tolist()}  "
            f"({entry['elapsed_s']}s)",
            flush=True,
        )

    # A transform is only worth publishing if it is the same one everywhere.
    names = {e["best"]["orientation"] for e in per_map.values()}
    scales = {round(e["best"]["scale"], 9) for e in per_map.values()}
    translations = np.array([e["best"]["transform"]["translation"]
                             for e in per_map.values()], dtype=np.float64)
    consensus = {
        "orientations": sorted(names),
        "scales": sorted(scales),
        "translation_spread_m": float(np.max(np.ptp(translations, axis=0))) if len(translations) > 1 else 0.0,
        "translation_median": np.median(translations, axis=0).tolist() if len(translations) else None,
        "uniform": bool(len(names) == 1 and len(scales) == 1),
    }

    report = {"maps": per_map, "consensus": consensus}
    (out_dir / "transform_search.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")

    if consensus["uniform"] and consensus["orientations"] == ["identity"]:
        identity = CoordinateTransform.identity()
        write_transform(out_dir / "best_transform.json", identity,
                        extra={"source": "run_transform_search", "consensus": consensus})
        print("\nidentity wins on every probe map; wrote best_transform.json with identity")
    elif consensus["uniform"]:
        name = consensus["orientations"][0]
        scale = consensus["scales"][0]
        t = np.median(translations, axis=0)
        m = np.asarray(
            next(iter(per_map.values()))["best"]["transform"]["matrix_2x2"],
            dtype=np.float64)
        write_transform(
            out_dir / "best_transform.json",
            CoordinateTransform(m, scale, t, name),
            extra={"source": "run_transform_search", "consensus": consensus},
        )
        print(f"\nuniform transform {name} scale={scale} t={t.tolist()}")
    else:
        print("\nno uniform transform: the orientation differs between maps "
              "(see transform_search.json)")

    print(json.dumps(consensus, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
