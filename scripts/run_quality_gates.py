#!/usr/bin/env python3
"""Step 9: run the six rendering gates and write the verdict."""

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
from sensaturban_fpv.pointcloud_renderer import Camera, render_cloud_region  # noqa: E402
from sensaturban_fpv.project_landmarks import (  # noqa: E402
    landmark_visibility, summarise_visibility,
)
from sensaturban_fpv.render_trajectory_fpv import (  # noqa: E402
    sample_poses, topdown_context,
)
from sensaturban_fpv.validate_rendering import (  # noqa: E402
    check_yaw_measurement, depth_ordering_check, gate1_valid_pixels, gate2_depth,
    gate3_continuity, gate4_yaw, gate5_landmarks, gate6_topdown, overall,
)


def make_camera(sample, pitch_deg, cfg) -> Camera:
    r = cfg["render"]
    return Camera(position=sample.position_ply, yaw=float(sample.citynav_pose[3]),
                  pitch=np.deg2rad(pitch_deg), width=r["width"], height=r["height"],
                  hfov_deg=r["hfov_deg"], near=r["near"], far=r["far"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--poses-per-split", type=int, default=4)
    ap.add_argument("--continuity-steps", type=int, default=8)
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

    episode_lookup = {(sp, ep.index): ep
                      for sp, eps in episodes_by_split.items() for ep in eps}

    results, cameras, vis_summaries, pose_xy, occupancy = [], [], [], [], []
    ordering_checks = []
    gate5_lines = []
    poses_meta = []

    for map_name, map_samples in sorted(by_map.items()):
        t0 = time.time()
        ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)
        for sample in map_samples:
            cam = make_camera(sample, cfg["render"]["views"]["fpv"], cfg)
            res = render_cloud_region(ctx.cloud, ctx.grid, cam,
                                      splat_radius=cfg["render"]["splat_radius"],
                                      lod=cfg["render"]["lod"])
            results.append(res)
            cameras.append(cam)

            ep = episode_lookup[(sample.split, sample.episode_index)]
            annotated = [
                landmark_visibility(lm, cam, res, ctx.cloud, ctx.grid)
                for lm in ctx.landmarks
            ]
            summary = summarise_visibility(annotated, ep.object_ids)
            vis_summaries.append(summary)
            gate5_lines.append({"map": map_name, "split": sample.split,
                                "episode": int(sample.episode_index), **summary})

            # Ordering check on one frame per map, where the cloud is already open.
            if not any(o["map"] == map_name for o in ordering_checks):
                idx = ctx.grid.query_radius(float(cam.position[0]), float(cam.position[1]),
                                            cam.far + 2.0)
                chk = depth_ordering_check(ctx.cloud.xyz(idx), ctx.cloud.rgb(idx), res)
                chk["map"] = map_name
                ordering_checks.append(chk)

            pose_xy.append(sample.position_ply[:2])
            local = ctx.grid.query_radius(float(cam.position[0]), float(cam.position[1]), 5.0)
            occupancy.append(int(local.size))
            poses_meta.append({"map": map_name, "split": sample.split,
                               "episode_index": int(sample.episode_index),
                               "step": int(sample.step),
                               "valid_pixel_ratio": float(res.valid_ratio)})
        print(f"{map_name}: {len(map_samples)} gate poses in {time.time() - t0:.1f}s", flush=True)

    gates = []
    g1 = gate1_valid_pixels(results)
    g2 = gate2_depth(results)
    g2["ordering"] = ordering_checks
    g2["passed"] = bool(g2["passed"] and all(o.get("passed", False) for o in ordering_checks))
    gates += [g1, g2]
    gates.append(gate5_landmarks(vis_summaries))

    # ---- Gate 3: continuity along a real trajectory -------------------------
    continuity = _run_continuity(cfg, transform, objects_by_map, episode_lookup,
                                 by_map, args.continuity_steps, out_dir)
    if continuity:
        gates.append(continuity["gate"])

    # ---- Gate 4: yaw probes -------------------------------------------------
    yaw_report = _run_yaw(cfg, transform, objects_by_map, episode_lookup, by_map, out_dir)
    if yaw_report:
        gates.append(yaw_report)

    # ---- Gate 6: are the cameras inside the map, over structure? ------------
    ctx0 = build_map_context(cfg, poses_meta[0]["map"], objects_by_map=objects_by_map)
    lo, hi = ctx0.bounds
    gates.append(gate6_topdown(pose_xy, lo, hi, occupancy))

    verdict = overall(gates)
    report = {
        "transform": transform.to_json(),
        "render_config": {k: v for k, v in cfg["render"].items() if k != "lod"},
        "lod": [list(x) for x in cfg["render"]["lod"]],
        "poses": poses_meta,
        "gate5_frames": gate5_lines,
        "verdict": verdict,
        "gates": gates,
    }
    if continuity:
        report["continuity_artifacts"] = continuity["artifacts"]
    if yaw_report:
        report["yaw_artifacts"] = yaw_report["artifacts"]

    path = out_dir / "quality_gates.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\n=== gate verdict ===")
    for g in gates:
        mark = {True: "PASS", False: "FAIL", None: "UNDECIDED"}[g.get("passed")]
        print(f"  {g['gate']:26s} {mark}")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"\nwrote {path}")


def _run_continuity(cfg, transform, objects_by_map, episode_lookup, by_map, steps, out_dir):
    """Render consecutive poses from one trajectory and compare against prediction."""
    import cv2

    for map_name, map_samples in sorted(by_map.items()):
        ep = episode_lookup[(map_samples[0].split, map_samples[0].episode_index)]
        if len(ep.trajectory) < steps:
            continue
        ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)
        from sensaturban_fpv.render_trajectory_fpv import PoseSample
        frames, cams, rgbs = [], [], []
        start = max(0, len(ep.trajectory) // 2 - steps // 2)
        for step in range(start, min(start + steps, len(ep.trajectory))):
            yaw = float(ep.yaw()[step])
            pose4 = np.array([*ep.trajectory[step, :3], yaw])
            pos_ply = transform.apply_xyz(ep.trajectory[step:step + 1, :3])[0]
            sample = PoseSample(map_samples[0].split, ep.index, map_name, step, pose4, pos_ply)
            cam = make_camera(sample, cfg["render"]["views"]["fpv"], cfg)
            res = render_cloud_region(ctx.cloud, ctx.grid, cam,
                                      splat_radius=cfg["render"]["splat_radius"],
                                      lod=cfg["render"]["lod"])
            frames.append(res)
            cams.append(cam)
            rgbs.append(res.rgb)

        gate = gate3_continuity(frames, cams)
        gate["map"] = map_name
        strip = np.concatenate(rgbs, axis=1)
        strip_path = out_dir / f"continuity_{map_name}.png"
        cv2.imwrite(str(strip_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        return {"gate": gate, "artifacts": {"strip": str(strip_path),
                                            "map": map_name,
                                            "steps": [int(s) for s in
                                                      range(start, start + len(frames))]}}
    return None


def _run_yaw(cfg, transform, objects_by_map, episode_lookup, by_map, out_dir):
    """Render yaw + 90/180/270 at one position and check the view rotates as predicted."""
    import cv2

    map_name, map_samples = sorted(by_map.items())[0]
    sample0 = map_samples[0]
    ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)
    base_cam = make_camera(sample0, cfg["render"]["views"]["fpv"], cfg)
    base = render_cloud_region(ctx.cloud, ctx.grid, base_cam,
                               splat_radius=cfg["render"]["splat_radius"],
                               lod=cfg["render"]["lod"])

    rotated_frames, rotated_cams, panels = [], [], [base.rgb]
    for delta in cfg["render"]["yaw_probe_deltas_deg"]:
        cam = Camera(position=base_cam.position,
                     yaw=base_cam.yaw + np.deg2rad(delta),
                     pitch=base_cam.pitch, width=base_cam.width,
                     height=base_cam.height, hfov_deg=base_cam.hfov_deg,
                     near=base_cam.near, far=base_cam.far)
        res = render_cloud_region(ctx.cloud, ctx.grid, cam,
                                  splat_radius=cfg["render"]["splat_radius"],
                                  lod=cfg["render"]["lod"])
        rotated_frames.append(res)
        rotated_cams.append(cam)
        panels.append(res.rgb)

    report = check_yaw_measurement(base, base_cam, rotated_frames, rotated_cams)
    report["map"] = map_name
    report["position"] = base_cam.position.tolist()
    strip = np.concatenate(panels, axis=1)
    path = out_dir / f"yaw_probe_{map_name}.png"
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    report["artifacts"] = {"strip": str(path),
                           "order": ["yaw", "+90", "+180", "+270"][:len(panels)]}
    return report


if __name__ == "__main__":
    main()
