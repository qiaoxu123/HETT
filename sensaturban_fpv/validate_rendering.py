"""The six rendering gates.

A frame that "looks like a city" is not evidence of anything.  Each gate below
tests a property that a genuine first-person view of a known scene must have and
that a plausible-looking but misaligned render would fail:

===========  ==========================================================
Gate 1       enough of the frame is backed by measured points
Gate 2       depth is real, ordered, and finite
Gate 3       consecutive poses give a continuous, correctly-directed view
Gate 4       yaw really rotates the view, by the predicted amount
Gate 5       landmarks project where the geometry says they are
Gate 6       the camera is inside the map, over real structure
===========  ==========================================================
"""

from __future__ import annotations

import numpy as np

from .pointcloud_renderer import (
    Camera,
    depth_histogram,
    estimate_shift,
    render,
    reprojection_shift,
)

MIN_VALID_RATIO = 0.60
MIN_DEPTH_STD = 0.5
MIN_DEPTH_UNIQUE = 20
SHIFT_TOLERANCE_PX = 6.0
YAW_SHIFT_TOLERANCE_FRAC = 0.25  # of the predicted rotation shift


def gate1_valid_pixels(results: list) -> dict:
    """Coverage of the frame by measured points.

    CityNav trajectories are UAV flights, so a level view is largely sky by
    geometry.  Both the literal whole-frame ratio and the below-horizon ratio
    are reported; the literal one decides ``passed`` as specified, and the
    below-horizon one is what says whether a low number means "sparse cloud" or
    just "pointed at the sky".
    """
    ratios = np.array([r.valid_ratio for r in results], dtype=np.float64)
    below = np.array([r.stats.get("below_horizon_valid_ratio", np.nan) for r in results],
                     dtype=np.float64)
    sky = np.array([r.stats.get("sky_pixel_ratio", np.nan) for r in results],
                   dtype=np.float64)
    in_tile = np.array([r.stats.get("in_tile_valid_ratio") or np.nan for r in results],
                       dtype=np.float64)
    finite = np.isfinite(below)
    finite_tile = np.isfinite(in_tile)

    def stat(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return None
        return {"median": float(np.median(a)), "min": float(a.min()),
                "max": float(a.max()), "frac_above_threshold":
                    float((a >= MIN_VALID_RATIO).mean())}

    return {
        "gate": "G1_valid_pixel_ratio",
        "threshold": MIN_VALID_RATIO,
        "per_frame": ratios.tolist(),
        "per_frame_below_horizon": below.tolist(),
        "per_frame_sky_fraction": sky.tolist(),
        "per_frame_in_tile": in_tile.tolist(),
        "whole_frame": stat(ratios),
        "below_horizon": stat(below),
        "in_tile": stat(in_tile),
        "passed": bool(ratios.size and (ratios >= MIN_VALID_RATIO).mean() >= 0.5),
        "below_horizon_passed": bool(finite.any()
                                     and (below[finite] >= MIN_VALID_RATIO).mean() >= 0.5),
        "in_tile_passed": bool(finite_tile.any()
                               and (in_tile[finite_tile] >= MIN_VALID_RATIO).mean() >= 0.5),
        "note": ("passed uses the literal whole-frame ratio as specified. "
                 "below_horizon_passed drops rows that are sky by construction. "
                 "in_tile_passed drops pixels whose ray leaves the block's XY "
                 "footprint before the far plane -- those cannot be filled by "
                 "this block at any density, so a failure there is a statement "
                 "about the size of a SensatUrban tile, not about the renderer."),
    }


def gate2_depth(results: list) -> dict:
    """Non-degenerate, finite depth, and a z-buffer that actually orders."""
    hists = [depth_histogram(r.depth) for r in results]
    stats = [h for h in hists if h.get("count", 0) > 0]
    if not stats:
        return {"gate": "G2_depth", "passed": False, "reason": "no valid depth"}

    stds = np.array([h["std"] for h in stats])
    uniques = np.array([h["unique_rounded"] for h in stats])
    nonfinite = np.array([h["nonfinite_fraction"] for h in stats])

    # The ordering property itself is tested by depth_ordering_check(), which
    # the driver runs on the same frames and attaches under "ordering".
    return {
        "gate": "G2_depth",
        "std_min": float(stds.min()), "std_median": float(np.median(stds)),
        "unique_min": int(uniques.min()),
        "nonfinite_max": float(nonfinite.max()),
        "passed": bool(
            float(np.median(stds)) > MIN_DEPTH_STD
            and int(uniques.min()) >= MIN_DEPTH_UNIQUE
            and float(nonfinite.max()) == 0.0
        ),
        "per_frame": stats,
    }


def depth_ordering_check(xyz, rgb, result, ids=None, far_only_margin: float = 0.0) -> dict:
    """Re-render with far geometry removed; depth may only move farther away."""
    cam_stats = result.stats["camera"]
    camera = Camera(
        position=cam_stats["position"], yaw=cam_stats["yaw_rad"],
        pitch=np.deg2rad(cam_stats["pitch_deg"]), width=cam_stats["width"],
        height=cam_stats["height"], hfov_deg=cam_stats["hfov_deg"],
        near=cam_stats["near"], far=cam_stats["far"],
    )
    xyz = np.asarray(xyz, dtype=np.float64)
    dist = np.linalg.norm(xyz - camera.position[None, :], axis=1)
    keep = dist <= camera.far - far_only_margin
    if not keep.any():
        return {"checked": False, "reason": "nothing left after truncation"}

    truncated = render(
        xyz[keep], np.asarray(rgb)[keep], camera, splat_radius=0,
        ids=None if ids is None else np.asarray(ids)[keep],
        lod=None,
    )
    both = result.valid & truncated.valid
    if not both.any():
        return {"checked": False, "reason": "no pixel valid in both renders"}

    delta = truncated.depth[both].astype(np.float64) - result.depth[both].astype(np.float64)
    return {
        "checked": True,
        "pixels_compared": int(both.sum()),
        "min_delta": float(delta.min()),
        "violations": int((delta < -1e-4).sum()),
        "passed": bool((delta >= -1e-4).all()),
        "note": "removing far points must not pull any pixel nearer than before",
    }


def gate3_continuity(frames: list, cameras: list) -> dict:
    """Consecutive frames must move smoothly and in the predicted direction."""
    if len(frames) < 2:
        return {"gate": "G3_pose_continuity", "passed": False,
                "reason": "need at least two consecutive frames"}
    steps = []
    for i in range(len(frames) - 1):
        measured = estimate_shift(frames[i].rgb, frames[i + 1].rgb)
        predicted = reprojection_shift(cameras[i], cameras[i + 1], frames[i].depth)
        steps.append({
            "step": i,
            "measured_du": measured["du"], "measured_dv": measured["dv"],
            "predicted_du": predicted["du"], "predicted_dv": predicted["dv"],
            "n_predicted": predicted["n"],
            "error_px": float(np.hypot(measured["du"] - predicted["du"],
                                       measured["dv"] - predicted["dv"])),
        })
    errors = np.array([s["error_px"] for s in steps], dtype=np.float64)
    measured_mag = np.array([np.hypot(s["measured_du"], s["measured_dv"]) for s in steps])
    return {
        "gate": "G3_pose_continuity",
        "steps": steps,
        "median_error_px": float(np.median(errors)),
        "max_error_px": float(errors.max()),
        "median_shift_px": float(np.median(measured_mag)),
        "max_shift_px": float(measured_mag.max()),
        "passed": bool(np.median(errors) <= SHIFT_TOLERANCE_PX
                       and measured_mag.max() < 0.5 * frames[0].shape[1]),
        "note": ("each frame-to-frame shift is compared against the shift predicted by "
                 "reprojecting the previous frame's own depth through the next camera"),
    }


def gate4_yaw(frames, cameras, deltas_deg=(90, 180, 270)) -> dict:
    """Rotating the camera by D degrees must move the view by the predicted amount."""
    if not frames:
        return {"gate": "G4_yaw_consistency", "passed": False, "reason": "no frames"}
    probes = []
    base = frames[0]
    base_cam = cameras[0]
    for delta in deltas_deg:
        rotated = Camera(
            position=base_cam.position,
            yaw=base_cam.yaw + np.deg2rad(delta),
            pitch=base_cam.pitch,
            width=base_cam.width, height=base_cam.height,
            hfov_deg=base_cam.hfov_deg, near=base_cam.near, far=base_cam.far,
        )
        predicted = reprojection_shift(base_cam, rotated, base.depth)
        probes.append({
            "delta_deg": delta,
            "predicted_du": predicted["du"], "predicted_dv": predicted["dv"],
            "n": predicted["n"],
            "camera": rotated.to_json(),
        })
    return {
        "gate": "G4_yaw_consistency",
        "probes": probes,
        "passed": None,  # filled in once the rotated renders are supplied
        "note": "call check_yaw_measurement() with the rendered rotations to decide",
    }


def check_yaw_measurement(base_frame, base_camera, rotated_frames, rotated_cameras) -> dict:
    """Compare measured rotation shifts against the reprojection prediction."""
    rows = []
    for frame, cam in zip(rotated_frames, rotated_cameras):
        measured = estimate_shift(base_frame.rgb, frame.rgb)
        predicted = reprojection_shift(base_camera, cam, base_frame.depth)
        delta = np.rad2deg(cam.yaw - base_camera.yaw)
        magnitude = abs(predicted["du"])
        err = float(np.hypot(measured["du"] - predicted["du"],
                             measured["dv"] - predicted["dv"]))
        rows.append({
            "delta_deg": float(delta),
            "measured_du": measured["du"], "measured_dv": measured["dv"],
            "predicted_du": predicted["du"], "predicted_dv": predicted["dv"],
            "error_px": err,
            "within_tolerance": bool(err <= max(SHIFT_TOLERANCE_PX,
                                                YAW_SHIFT_TOLERANCE_FRAC * magnitude)),
        })
    return {
        "gate": "G4_yaw_consistency",
        "probes": rows,
        "passed": bool(rows and all(r["within_tolerance"] for r in rows)),
        "note": ("a yaw change must move the view by the shift predicted from "
                 "reprojecting the base frame's own depth; sign errors in the yaw "
                 "convention, degrees-vs-radians, or axis flips all fail here"),
    }


def reprojection_agreement(cam_a, res_a, cam_b, res_b, colour_tol: float = 40.0,
                           depth_tol: float = 2.0) -> dict:
    """Do two renders of the same scene agree on where the surface went?

    Every valid pixel of A is unprojected to a world point and reprojected into
    B; if the geometry is right, B must show the same colour there, unless B
    sees something nearer (a genuine disocclusion) or nothing at all.

    This is the wrap-free alternative to estimating a shift by phase
    correlation.  A 90-degree turn moves content by more than half the frame, so
    a correlation peak aliases and the estimate is meaningless -- but the
    per-pixel correspondence this measures is defined for any rotation.  A
    wrong yaw sign, a degree/radian mix-up or an axis flip drives agreement to
    chance level instead.
    """
    from .pointcloud_renderer import project_points, unproject

    world = unproject(res_a.depth, cam_a)
    finite = np.isfinite(world).all(axis=-1)
    if not finite.any():
        return {"pixels": 0, "agreement": None}
    rows, cols = np.nonzero(finite)
    pts = world[finite]

    proj = project_points(pts, cam_b)
    if proj["u"].size == 0:
        return {"pixels": 0, "agreement": None,
                "note": "no base pixel reprojects into the second frame"}

    src_rows = rows[proj["pos"]]
    src_cols = cols[proj["pos"]]
    dst_rows = np.clip(np.round(proj["v"]).astype(np.int64), 0, cam_b.height - 1)
    dst_cols = np.clip(np.round(proj["u"]).astype(np.int64), 0, cam_b.width - 1)

    seen = res_b.valid[dst_rows, dst_cols]
    if not seen.any():
        return {"pixels": 0, "agreement": None,
                "note": "nothing reprojected lands on a rendered pixel"}
    src_rows, src_cols = src_rows[seen], src_cols[seen]
    dst_rows, dst_cols = dst_rows[seen], dst_cols[seen]

    # B seeing something nearer than the reprojected point is a disocclusion,
    # which is the expected consequence of moving, not a disagreement.
    depth_b = res_b.depth[dst_rows, dst_cols].astype(np.float64)
    metric_b = res_b.depth[dst_rows, dst_cols].astype(np.float64)
    unoccluded = depth_b >= (res_a.depth[src_rows, src_cols].astype(np.float64)
                             - depth_tol)
    # The reprojected range is the honest comparison for occlusion.
    unoccluded &= metric_b >= (proj["metric"][seen] - depth_tol)
    if not unoccluded.any():
        return {"pixels": 0, "agreement": None,
                "note": "every reprojection is occluded in the second frame"}

    a = res_a.rgb[src_rows[unoccluded], src_cols[unoccluded]].astype(np.int32)
    b = res_b.rgb[dst_rows[unoccluded], dst_cols[unoccluded]].astype(np.int32)
    diff = np.abs(a - b).max(axis=1)
    return {
        "pixels": int(unoccluded.sum()),
        "agreement": float((diff <= colour_tol).mean()),
        "median_colour_error": float(np.median(diff)),
        "base_pixels": int(finite.sum()),
    }


def gate_yaw_reprojection(base_cam, base_res, rotated) -> dict:
    """Yaw check via reprojection, valid at any rotation magnitude."""
    rows = []
    for cam, res in rotated:
        delta = np.rad2deg(cam.yaw - base_cam.yaw)
        m = reprojection_agreement(base_cam, base_res, cam, res)
        rows.append({"delta_deg": float(delta), **m,
                     "passed": bool(m["agreement"] is not None and m["agreement"] >= 0.5)})
    return {
        "gate": "A4_yaw_consistency",
        "method": "unproject the base frame, reproject into the rotated camera, "
                  "compare colours where the rotated frame is unoccluded",
        "probes": rows,
        "passed": bool(rows and all(r["passed"] for r in rows)),
        "note": "phase correlation is not used: a quarter turn exceeds its "
                "unambiguous range",
    }


def gate_continuity_reprojection(cameras, results) -> dict:
    """Pose continuity via reprojection rather than a global shift estimate.

    CityNav trajectory steps are tens of metres apart, not video frames, so a
    single global shift does not describe the change between them.
    """
    rows = []
    for i in range(len(results) - 1):
        m = reprojection_agreement(cameras[i], results[i], cameras[i + 1], results[i + 1])
        step = float(np.linalg.norm(cameras[i + 1].position - cameras[i].position))
        rows.append({"step": i, "camera_step_m": step, **m,
                     "passed": bool(m["agreement"] is not None and m["agreement"] >= 0.5)})
    scored = [r for r in rows if r["agreement"] is not None]
    return {
        "gate": "A5_trajectory_continuity",
        "method": "reproject frame i into frame i+1 and compare colours where "
                  "frame i+1 is unoccluded",
        "steps": rows,
        "median_agreement": (float(np.median([r["agreement"] for r in scored]))
                             if scored else None),
        "median_camera_step_m": float(np.median([r["camera_step_m"] for r in rows])),
        "passed": bool(scored and np.median([r["agreement"] for r in scored]) >= 0.5),
    }


def gate5_landmarks(visibility_summaries: list) -> dict:
    summaries = [s for s in visibility_summaries if s["referenced_total"] > 0]
    if not summaries:
        return {"gate": "G5_landmark_projection", "passed": False,
                "reason": "no frame had a referenced landmark"}
    in_fov = sum(s["referenced_in_fov"] for s in summaries)
    visible = sum(s["referenced_visible"] for s in summaries)
    total = sum(s["referenced_total"] for s in summaries)
    frames_with_fov = sum(1 for s in summaries if s["referenced_in_fov"] > 0)
    frames_with_visible = sum(1 for s in summaries if s["referenced_visible"] > 0)
    return {
        "gate": "G5_landmark_projection",
        "frames": len(summaries),
        "referenced_total": total,
        "referenced_in_fov": in_fov,
        "referenced_visible": visible,
        "visible_frac_of_in_fov": visible / max(in_fov, 1),
        "frames_with_referenced_in_fov": frames_with_fov,
        "frames_with_referenced_visible": frames_with_visible,
        "passed": bool(frames_with_fov > 0 and visible / max(in_fov, 1) >= 0.25),
        "note": ("passes when referenced landmarks that fall inside the frame are "
                 "backed by measured, unoccluded geometry at least a quarter of the time"),
    }


def gate6_topdown(poses_xy, cloud_lo, cloud_hi, occupancy_counts,
                  min_local_points: int = 1) -> dict:
    """The camera must sit inside the map and above real structure."""
    poses_xy = np.asarray(poses_xy, dtype=np.float64).reshape(-1, 2)
    lo, hi = np.asarray(cloud_lo, float)[:2], np.asarray(cloud_hi, float)[:2]
    inside = ((poses_xy[:, 0] >= lo[0]) & (poses_xy[:, 0] <= hi[0])
              & (poses_xy[:, 1] >= lo[1]) & (poses_xy[:, 1] <= hi[1]))
    counts = np.asarray(occupancy_counts, dtype=np.float64)
    occupied = counts >= min_local_points
    return {
        "gate": "G6_topdown_position",
        "poses": int(len(poses_xy)),
        "inside_cloud_bounds_ratio": float(inside.mean()) if len(poses_xy) else 0.0,
        "over_structure_ratio": float(occupied.mean()) if counts.size else 0.0,
        "median_local_points": float(np.median(counts)) if counts.size else 0.0,
        "passed": bool(len(poses_xy) and inside.all()
                       and (occupied.mean() if counts.size else 0.0) >= 0.9),
    }


def overall(gates: list) -> dict:
    decided = [g for g in gates if g.get("passed") is not None]
    return {
        "gates": {g["gate"]: g.get("passed") for g in gates},
        "all_passed": bool(decided and all(g["passed"] for g in decided)),
        "undecided": [g["gate"] for g in gates if g.get("passed") is None],
    }
