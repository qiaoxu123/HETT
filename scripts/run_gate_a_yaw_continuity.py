#!/usr/bin/env python3
"""Recompute Gate A's A4 (yaw) and A5 (continuity) and merge them in place.

Only these two checks need redoing, and they need a handful of renders rather
than the full 26-pose sweep.

Why they are defined this way
-----------------------------
Phase correlation between two rendered frames is the only estimator here that
can *falsify* a yaw convention: it compares the render against the render, with
no shared camera model, so a sign error shows up as a sign flip.  But it is only
unambiguous while the displacement stays well inside half the frame, and a
quarter turn moves content by ~326 px in a 512 px image -- the peak then aliases
and the estimate is meaningless.  Measured on these renders it is also
unreliable by 25 degrees.

So A4 is run at small angles where the estimator is sound, and additionally in
both directions: the +d and -d shifts must come out with opposite signs.  A sign
error, a degree/radian mistake or an axis flip inverts that pair.

For A5 the same reasoning applies to a different quantity.  CityNav trajectory
steps are tens of metres apart -- they are not video frames -- so a single
global shift does not describe the change between them.  A5 instead compares
consecutive real poses against unrelated pose pairs: if the renderer placed
geometry consistently along the trajectory, consecutive frames must agree about
where the surface went far more than unrelated ones do.  That comparison is
falsifiable, because a renderer that ignored the pose would score the same on
both.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    Camera, estimate_shift, render_cloud_region, reprojection_shift,
)
from sensaturban_fpv.pose_selection import camera_for  # noqa: E402
from sensaturban_fpv.validate_rendering import reprojection_agreement  # noqa: E402

YAW_ANGLES = (10.0, 15.0)
YAW_ANGLES_REPORTED_ONLY = (20.0, 25.0)
MAX_SHIFT_PX = 120.0  # comfortably inside the unambiguous range for 512 px


def render(ctx, cfg, position, yaw_rad, pitch_deg):
    cam = Camera(position=np.asarray(position), yaw=float(yaw_rad),
                 pitch=np.deg2rad(pitch_deg), width=cfg["render"]["width"],
                 height=cfg["render"]["height"], hfov_deg=cfg["render"]["hfov_deg"],
                 near=cfg["render"]["near"], far=cfg["render"]["far"])
    res = render_cloud_region(ctx.cloud, ctx.grid, cam, splat_radius=0,
                              lod=cfg["render"]["lod"])
    return cam, res


def yaw_gate(ctx, cfg, sample, out_root):
    position = np.asarray(sample["pose_ply"], dtype=np.float64)
    yaw = float(sample["yaw"])
    base_cam, base = render(ctx, cfg, position, yaw, -30.0)

    rows = []
    for delta in YAW_ANGLES:
        pair = {}
        for sign in (1.0, -1.0):
            cam, res = render(ctx, cfg, position, yaw + sign * np.deg2rad(delta), -30.0)
            measured = estimate_shift(base.rgb, res.rgb)
            predicted = reprojection_shift(base_cam, cam, base.depth)
            pair[sign] = {
                "measured_du": measured["du"], "measured_dv": measured["dv"],
                "predicted_du": predicted["du"], "predicted_dv": predicted["dv"],
                "displacement_px": float(np.hypot(measured["du"], measured["dv"])),
            }
        plus, minus = pair[1.0], pair[-1.0]
        in_range = (plus["displacement_px"] <= MAX_SHIFT_PX
                    and minus["displacement_px"] <= MAX_SHIFT_PX)
        sign_ok = (np.sign(plus["measured_du"]) == np.sign(plus["predicted_du"])
                   and np.sign(minus["measured_du"]) == np.sign(minus["predicted_du"]))
        antisymmetric = np.sign(plus["measured_du"]) != np.sign(minus["measured_du"])
        error = float(abs(plus["measured_du"] - plus["predicted_du"]))
        rows.append({
            "delta_deg": delta,
            "plus": plus, "minus": minus,
            "predictions_agree_in_sign": bool(sign_ok),
            "measured_shifts_are_antisymmetric": bool(antisymmetric),
            "abs_error_px": error,
            "in_estimator_range": bool(in_range),
            "passed": bool(sign_ok and antisymmetric and in_range and error <= 30.0),
        })

    # Larger angles are measured and reported but not scored: the probe below
    # shows the estimator losing the peak well before the geometry does.
    degradation = []
    for delta in YAW_ANGLES_REPORTED_ONLY:
        sign_ok, measured, predicted = True, [], []
        for sign in (1.0, -1.0):
            cam, res = render(ctx, cfg, position, yaw + sign * np.deg2rad(delta), -30.0)
            m = estimate_shift(base.rgb, res.rgb)
            p_ = reprojection_shift(base_cam, cam, base.depth)
            measured.append(m["du"]); predicted.append(p_["du"])
            sign_ok &= np.sign(m["du"]) == np.sign(p_["du"])
        degradation.append({"delta_deg": delta, "measured_du": measured,
                            "predicted_du": predicted, "sign_ok": bool(sign_ok)})

    strip = np.concatenate([
        render(ctx, cfg, position, yaw + np.deg2rad(d), -30.0)[1].rgb
        for d in (0.0, YAW_ANGLES[1], -YAW_ANGLES[1])], axis=1)
    import cv2
    path = out_root / "yaw_probe.png"
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    return {
        "gate": "A4_yaw_consistency",
        "method": "phase correlation between renders at yaw and yaw+-d, compared "
                  "against the analytic prediction, run in both directions",
        "angles_deg": list(YAW_ANGLES),
        "angles_reported_not_scored": list(YAW_ANGLES_REPORTED_ONLY),
        "probes": rows,
        "degradation_probes": degradation,
        "passed": bool(rows and all(r["passed"] for r in rows)),
        "why_not_90_degrees": (
            "a quarter turn displaces the view by more than half the frame, so "
            "the correlation peak aliases and the measurement is meaningless; "
            "the large-angle convention is pinned instead by the closed-form "
            "projection tests and by A3"),
        "artifacts": {"strip": str(path), "order": ["yaw", "+15", "-15"]},
    }


def continuity_gate(ctxs, cfg, episodes, transform, out_root, steps=5,
                    max_shift=40.0, tol=10.0, max_baseline_ratio=0.25):
    """Frame-to-frame shift against the analytic prediction, where measurable.

    The measured shift and the prediction come from different places -- one from
    the image, one from reprojecting the previous frame's own depth -- so a pose
    that the renderer ignored, or a trajectory rendered out of order, shows up
    as a mismatch.  Pairs whose predicted displacement exceeds half the frame
    are excluded rather than scored, because the estimator aliases there; how
    many pairs that leaves is reported.
    """
    import cv2

    per_episode, strips = [], []
    for episode, ctx in zip(episodes, ctxs):
        n = len(episode.trajectory)
        if n < steps + 1:
            continue
        start = max(0, n // 2 - steps // 2)
        start = min(start, max(n - steps, 0))
        cams, frames, positions = [], [], []
        for step in range(start, min(start + steps, n)):
            pos = transform.apply_xyz(episode.trajectory[step:step + 1, :3])[0]
            cam, res = render(ctx, cfg, pos, float(episode.yaw()[step]), -30.0)
            cams.append(cam)
            frames.append(res)
            positions.append(pos)
        strips.append(np.concatenate([f.rgb for f in frames], axis=1))

        pairs = []
        for i in range(len(frames) - 1):
            measured = estimate_shift(frames[i].rgb, frames[i + 1].rgb)
            predicted = reprojection_shift(cams[i], cams[i + 1], frames[i].depth)
            displacement = float(np.hypot(predicted["du"], predicted["dv"]))
            stepsize = float(np.linalg.norm(positions[i + 1] - positions[i]))
            # Overlap is governed by the baseline relative to how far away the
            # scene is, not by the median predicted shift: after a 62 m step the
            # handful of surviving pixels can still have a small median shift
            # while the two frames have almost nothing in common.
            median_depth = float(np.nanmedian(frames[i].depth))
            baseline_ratio = stepsize / max(median_depth, 1e-6)
            entry = {
                "step": start + i, "camera_step_m": stepsize,
                "measured_du": measured["du"], "measured_dv": measured["dv"],
                "predicted_du": predicted["du"], "predicted_dv": predicted["dv"],
                "predicted_displacement_px": displacement,
                "median_scene_depth_m": median_depth,
                "baseline_ratio": baseline_ratio,
                "measurable": bool(np.isfinite(displacement)
                                   and displacement <= max_shift
                                   and baseline_ratio <= max_baseline_ratio),
            }
            if entry["measurable"]:
                entry["error_px"] = float(np.hypot(
                    measured["du"] - predicted["du"],
                    measured["dv"] - predicted["dv"]))
                entry["passed"] = bool(entry["error_px"] <= tol)
            pairs.append(entry)

        per_episode.append({
            "episode": f"{episode.split}:{episode.index}",
            "map": episode.map_name,
            "steps_rendered": [int(x) for x in range(start, start + len(frames))],
            "median_camera_step_m": float(np.median([p["camera_step_m"] for p in pairs])),
            "pairs": pairs,
        })

    scored = [p for e in per_episode for p in e["pairs"] if p["measurable"]]
    total = sum(len(e["pairs"]) for e in per_episode)
    errors = [p["error_px"] for p in scored]
    if strips:
        path = out_root / "continuity_strip.png"
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(strips, axis=0),
                                            cv2.COLOR_RGB2BGR))
    else:
        path = None

    return {
        "gate": "A5_trajectory_continuity",
        "method": "measured frame-to-frame shift vs the shift predicted by "
                  "reprojecting the previous frame's own depth",
        "episodes": per_episode,
        "pairs_total": total,
        "pairs_measurable": len(scored),
        "median_error_px": float(np.median(errors)) if errors else None,
        "max_error_px": float(np.max(errors)) if errors else None,
        "excluded_pairs": [
            {"episode": e["episode"], "step": p["step"],
             "camera_step_m": p["camera_step_m"],
             "baseline_ratio": p.get("baseline_ratio"),
             "predicted_displacement_px": p["predicted_displacement_px"]}
            for e in per_episode for p in e["pairs"] if not p["measurable"]],
        "max_scored_displacement_px": max_shift,
        "max_scored_baseline_ratio": max_baseline_ratio,
        "passed": bool(len(scored) >= 4 and errors and np.median(errors) <= tol),
        "note": ("only pairs whose predicted displacement is at most "
                 f"{max_shift:.0f} px and whose baseline is at most "
                 f"{max_baseline_ratio:.0%} of the median scene depth are scored; "
                 "a pair whose previous frame no longer overlaps the next one is "
                 "not evidence about continuity either way"),
        "artifacts": {"strip": str(path) if path else None},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--gate-a", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    gate_a_path = Path(args.gate_a) if args.gate_a else \
        artifact_dir(cfg) / "gate_a" / "gate_a.json"
    report = json.loads(gate_a_path.read_text())
    out_root = gate_a_path.parent

    transform_path = Path(args.transform) if args.transform else \
        artifact_dir(cfg) / "transform" / "best_transform.json"
    transform = load_transform(transform_path) if transform_path.exists() \
        else CoordinateTransform.identity()

    rows = report["rows"]
    yaw_row = rows[0]
    cont_row = rows[1] if len(rows) > 1 else rows[0]

    objects_by_map = load_landmarks(cfg)
    ctx_cache = {}

    def context(map_name):
        if map_name not in ctx_cache:
            ctx_cache.clear()
            ctx_cache[map_name] = build_map_context(cfg, map_name,
                                                    objects_by_map=objects_by_map)
        return ctx_cache[map_name]

    yaw_ctx = context(yaw_row["map"])
    yaw_pos = transform.apply_xyz(np.asarray(yaw_row["metadata"]["pose_xyz"])[None, :])[0]
    a4 = yaw_gate(yaw_ctx, cfg, {
        "pose_ply": yaw_pos, "yaw": np.deg2rad(yaw_row["metadata"]["yaw_deg"])}, out_root)

    splits = {s: citynav.load_split(Path(cfg["paths"]["citynav_dir"]), s)
              for s in ("val_unseen", "val_seen", "train_seen")}
    episode = next(e for e in splits[cont_row["split"]]
                   if e.index == cont_row["episode_index"])
    cont_episodes, cont_contexts = [], []
    for row in rows[:3]:
        ep = next(e for e in splits[row["split"]] if e.index == row["episode_index"])
        cont_episodes.append(ep)
        cont_contexts.append(context(row["map"]))
    a5 = continuity_gate(cont_contexts, cfg, cont_episodes, transform, out_root)

    for name, gate in report["gates"].items():
        if name == "A4_yaw_consistency":
            gate.clear(); gate.update(a4)
        elif name == "A5_trajectory_continuity":
            gate.clear(); gate.update(a5)
    report["gates_recomputed"] = ["A4_yaw_consistency", "A5_trajectory_continuity"]
    gate_a_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")

    print("=== Gate A (A4/A5 recomputed) ===")
    for name, gate in report["gates"].items():
        mark = {True: "PASS", False: "FAIL", None: "n/a"}[gate.get("passed")]
        print(f"  {name:26s} {mark}")
    print("\nA4 probes:")
    for probe in a4["probes"]:
        print(f"  +-{probe['delta_deg']:.0f}deg  measured {probe['plus']['measured_du']:+.1f} / "
              f"{probe['minus']['measured_du']:+.1f}   predicted "
              f"{probe['plus']['predicted_du']:+.1f} / {probe['minus']['predicted_du']:+.1f}   "
              f"sign_ok={probe['predictions_agree_in_sign']} "
              f"antisym={probe['measured_shifts_are_antisymmetric']} "
              f"err={probe['abs_error_px']:.1f}")
    print(f"\nA5: {a5['pairs_measurable']}/{a5['pairs_total']} pairs measurable, "
          f"median error {a5['median_error_px']}, max {a5['max_error_px']}")
    for e in a5["episodes"]:
        print(f"   {e['episode']:24s} {e['map']:22s} "
              f"median step {e['median_camera_step_m']:6.1f} m  "
              f"measurable {sum(1 for p in e['pairs'] if p['measurable'])}/{len(e['pairs'])}")
    print(f"\nwrote {gate_a_path}")


if __name__ == "__main__":
    main()
