"""Is CityNav's coordinate frame the SensatUrban point cloud's frame?

Three independent questions, each answered from data rather than from a visual
impression that "it looks like a city":

A. XY bounds overlap  -- trajectory vs cloud vs landmarks.
B. Z consistency      -- is the UAV above the surface the cloud says is beneath it?
C. Landmark alignment -- is there measured geometry where CityRefer says the
                         landmark is?

The three are deliberately different in kind: A is necessary but weak (two
wrong frames can still overlap), B tests the vertical datum, and C is the one
that can actually falsify the correspondence, because it asks whether an
independently authored landmark ends up inside a real structure.
"""

from __future__ import annotations

import numpy as np

from .plyio import local_ground_height


def bbox_overlap(a_lo, a_hi, b_lo, b_hi, axis_names=("x", "y")) -> dict:
    """Per-axis and area overlap of two axis-aligned boxes."""
    a_lo, a_hi = np.asarray(a_lo, float), np.asarray(a_hi, float)
    b_lo, b_hi = np.asarray(b_lo, float), np.asarray(b_hi, float)

    inter_lo = np.maximum(a_lo, b_lo)
    inter_hi = np.minimum(a_hi, b_hi)
    inter_size = np.clip(inter_hi - inter_lo, 0.0, None)
    a_size = np.clip(a_hi - a_lo, 1e-12, None)
    b_size = np.clip(b_hi - b_lo, 1e-12, None)

    n = min(len(a_lo), len(axis_names))
    per_axis = {}
    for i in range(n):
        per_axis[axis_names[i]] = {
            "a": [float(a_lo[i]), float(a_hi[i])],
            "b": [float(b_lo[i]), float(b_hi[i])],
            "intersection": [float(inter_lo[i]), float(inter_hi[i])],
            "overlap_len": float(inter_size[i]),
            "overlap_frac_of_a": float(inter_size[i] / a_size[i]),
            "overlap_frac_of_b": float(inter_size[i] / b_size[i]),
        }

    k = len(inter_size)
    area_inter = float(np.prod(inter_size[:k]))
    area_a = float(np.prod(a_size[:k]))
    area_b = float(np.prod(b_size[:k]))
    return {
        "per_axis": per_axis,
        "area_intersection": area_inter,
        "area_a": area_a,
        "area_b": area_b,
        "iou": area_inter / max(area_a + area_b - area_inter, 1e-12),
        "overlap_frac_of_a": area_inter / max(area_a, 1e-12),
        "overlap_frac_of_b": area_inter / max(area_b, 1e-12),
    }


def z_consistency(
    cloud,
    grid,
    poses_xyz: np.ndarray,
    radius: float = 6.0,
    max_reasonable_altitude: float = 200.0,
    max_poses: int | None = 600,
) -> dict:
    """Compare each UAV height against the local surface height measured in the cloud.

    For every sampled pose we read the cloud around its XY and compare the UAV's
    ``z`` against the local surface: an altitude below the local minimum would
    mean the camera is under the terrain, and an absurd altitude means the
    vertical datum is off even though XY matches.

    The per-pose grid query dominates the cost, so the full pose set is thinned
    to ``max_poses`` by uniform stride.  The stride is deterministic and the
    sample size is reported, so the statistics stay reproducible.
    """
    poses_xyz = np.asarray(poses_xyz, dtype=np.float64)
    sampled_from = int(len(poses_xyz))
    if max_poses and len(poses_xyz) > max_poses:
        stride = int(np.ceil(len(poses_xyz) / max_poses))
        poses_xyz = poses_xyz[::stride]

    stats = []
    for x, y, z in poses_xyz:
        idx = grid.query_radius(x, y, radius)
        if idx.size == 0:
            stats.append({
                "x": float(x), "y": float(y), "z": float(z),
                "points": 0, "ground_z": float("nan"),
                "surface_z": float("nan"), "altitude_above_ground": float("nan"),
                "altitude_above_surface": float("nan"),
            })
            continue
        pts = cloud.xyz(idx)
        ground = local_ground_height(pts, percentile=2.0)
        surface = float(pts[:, 2].max())
        stats.append({
            "x": float(x), "y": float(y), "z": float(z),
            "points": int(idx.size),
            "ground_z": ground,
            "surface_z": surface,
            "altitude_above_ground": float(z - ground),
            "altitude_above_surface": float(z - surface),
        })

    altitudes = np.array([s["altitude_above_ground"] for s in stats], dtype=np.float64)
    finite = np.isfinite(altitudes)
    covered = np.array([s["points"] > 0 for s in stats])

    above_ground = finite & (altitudes > 0)
    anomalies = finite & ((altitudes <= 0) | (altitudes > max_reasonable_altitude))

    return {
        "poses": len(stats),
        "poses_available": sampled_from,
        "sampled_with_stride": int(np.ceil(sampled_from / max(max_poses or sampled_from, 1))),
        "poses_with_local_points": int(covered.sum()),
        "poses_with_local_points_ratio": float(covered.mean()) if covered.size else 0.0,
        "above_local_ground_ratio": float(above_ground.sum() / max(finite.sum(), 1)),
        "anomaly_ratio": float(anomalies.sum() / max(finite.sum(), 1)),
        "altitude_median": float(np.nanmedian(altitudes)) if finite.any() else None,
        "altitude_min": float(np.nanmin(altitudes)) if finite.any() else None,
        "altitude_max": float(np.nanmax(altitudes)) if finite.any() else None,
        "details": stats,
    }


def landmark_alignment(
    cloud,
    grid,
    landmarks: list,
    radii=(5.0, 10.0, 20.0),
    sample_limit: int | None = None,
) -> dict:
    """Measure the cloud around each CityRefer landmark centre.

    ``landmarks`` are ``(id, name, object_type, position_xyz, dimension_xyz)``
    tuples.  A landmark whose neighbourhood is empty is direct evidence that the
    landmark coordinates and the point cloud are not in the same frame.
    """
    if sample_limit and len(landmarks) > sample_limit:
        # Uniform stride, not a prefix: objects.json is ordered by map, and a
        # prefix would sample one corner of the block.
        stride = int(np.ceil(len(landmarks) / sample_limit))
        items = landmarks[::stride]
    else:
        items = landmarks
    per_landmark = []
    for lm in items:
        lm_id, name, obj_type, pos, dim = lm
        pos = np.asarray(pos, dtype=np.float64)
        entries = {}
        local = None
        for r in radii:
            idx = grid.query_radius(float(pos[0]), float(pos[1]), r)
            entries[f"r{int(r)}"] = int(idx.size)
            if r == radii[-1]:
                local = cloud.xyz(idx) if idx.size else None

        record = {
            "id": int(lm_id), "name": name, "object_type": obj_type,
            "position": pos.tolist(),
            "dimension": np.asarray(dim, dtype=np.float64).tolist(),
            "counts_by_radius": entries,
        }
        if local is not None and local.size:
            z = local[:, 2]
            record.update({
                "z_min": float(z.min()), "z_max": float(z.max()),
                "z_median": float(np.median(z)),
                "z_of_landmark": float(pos[2]),
                "dz_to_local_min": float(pos[2] - z.min()),
                "dz_to_local_max": float(pos[2] - z.max()),
                "local_extent_xy": [float(local[:, 0].min()), float(local[:, 0].max()),
                                    float(local[:, 1].min()), float(local[:, 1].max())],
            })
            if cloud.has_rgb:
                idx_rgb = grid.query_radius(float(pos[0]), float(pos[1]), radii[0])
                if idx_rgb.size:
                    col = cloud.rgb(idx_rgb).astype(np.float64)
                    record["rgb_mean_r5"] = col.mean(axis=0).tolist()
                    record["rgb_std_r5"] = col.std(axis=0).tolist()
        per_landmark.append(record)

    counts = np.array([r["counts_by_radius"][f"r{int(radii[0])}"] for r in per_landmark],
                      dtype=np.float64)
    within = np.array([r["counts_by_radius"][f"r{int(radii[-1])}"] for r in per_landmark],
                      dtype=np.float64)
    return {
        "landmarks_sampled": len(per_landmark),
        "nonempty_ratio_r5": float((counts > 0).mean()) if counts.size else 0.0,
        "nonempty_ratio_r20": float((within > 0).mean()) if within.size else 0.0,
        "median_points_r5": float(np.median(counts)) if counts.size else 0.0,
        "median_points_r20": float(np.median(within)) if within.size else 0.0,
        "per_landmark": per_landmark,
    }


def landmark_table(objects_by_map: dict, map_name: str, ids: list, limit: int | None = None) -> list:
    """Pull ``(id, name, type, position, dimension)`` tuples for one map."""
    objects = objects_by_map.get(map_name, {})
    selected = [objects[i] for i in ids if i in objects] if ids else list(objects.values())
    if limit:
        selected = selected[:limit]
    return [
        (o.id, o.name, o.object_type,
         tuple(float(v) for v in o.position), tuple(float(v) for v in o.dimension))
        for o in selected
    ]


def summarise_map(
    map_name: str,
    cloud,
    grid,
    episodes: list,
    landmarks: list,
    cloud_lo,
    cloud_hi,
    z_radius: float = 6.0,
    max_poses: int | None = 600,
    max_landmarks: int | None = 60,
) -> dict:
    """Run A, B and C for one map and return a JSON-serialisable report."""
    poses = np.concatenate([e.trajectory[:, :3] for e in episodes], axis=0) if episodes \
        else np.zeros((0, 3))
    traj_lo = poses.min(axis=0) if len(poses) else np.zeros(3)
    traj_hi = poses.max(axis=0) if len(poses) else np.zeros(3)

    lm_xyz = np.array([lm[3] for lm in landmarks], dtype=np.float64) if landmarks \
        else np.zeros((0, 3))
    lm_lo = lm_xyz.min(axis=0) if len(lm_xyz) else np.zeros(3)
    lm_hi = lm_xyz.max(axis=0) if len(lm_xyz) else np.zeros(3)

    report = {
        "map_name": map_name,
        "episodes": len(episodes),
        "trajectory_poses": int(len(poses)),
        "cloud_bounds": {"min": cloud_lo.tolist(), "max": cloud_hi.tolist()},
        "trajectory_bounds": {"min": traj_lo.tolist(), "max": traj_hi.tolist()},
        "landmark_bounds": {"min": lm_lo.tolist(), "max": lm_hi.tolist()},
        "A_bounds_overlap": {
            "trajectory_vs_cloud": bbox_overlap(traj_lo[:2], traj_hi[:2], cloud_lo[:2], cloud_hi[:2]),
            "landmark_vs_cloud": bbox_overlap(lm_lo[:2], lm_hi[:2], cloud_lo[:2], cloud_hi[:2]),
            "trajectory_vs_landmark": bbox_overlap(traj_lo[:2], traj_hi[:2], lm_lo[:2], lm_hi[:2]),
            "trajectory_z_vs_cloud_z": bbox_overlap(
                traj_lo[2:3], traj_hi[2:3], cloud_lo[2:3], cloud_hi[2:3], axis_names=("z",)),
        },
    }
    if len(poses):
        report["B_z_consistency"] = z_consistency(
            cloud, grid, poses, radius=z_radius, max_poses=max_poses)
    if landmarks:
        report["C_landmark_alignment"] = landmark_alignment(
            cloud, grid, landmarks, sample_limit=max_landmarks)
    return report
