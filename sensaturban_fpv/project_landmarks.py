"""Project CityRefer landmarks into a rendered frame and estimate their visibility.

A landmark centre landing inside the image is *not* evidence that the landmark
was seen: it may sit behind a wall, or there may be no measured point anywhere
near it.  Visibility here is therefore reported as three separate facts --
inside the frustum, backed by measured points, and unoccluded against the
rendered depth buffer -- rather than as a single optimistic boolean.
"""

from __future__ import annotations

import numpy as np

try:  # cv2 is only needed for the annotated overlays
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .pointcloud_renderer import project_points


def project_centre(position, camera) -> dict | None:
    """Project one world point; ``None`` when it is behind the camera."""
    right, up, forward = camera.basis()
    rel = np.asarray(position, dtype=np.float64) - camera.position
    z_cam = float(rel @ forward)
    if z_cam <= camera.near:
        return None
    metric = float(np.linalg.norm(rel))
    focal = camera.focal
    u = camera.width / 2.0 + focal * float(rel @ right) / z_cam
    v = camera.height / 2.0 - focal * float(rel @ up) / z_cam
    return {
        "u": float(u), "v": float(v), "z_cam": z_cam, "metric": metric,
        "inside": bool(0 <= u < camera.width and 0 <= v < camera.height),
        "offset_from_centre_px": float(np.hypot(u - camera.width / 2.0,
                                                v - camera.height / 2.0)),
    }


def _pixel_index(u: float, v: float, camera) -> tuple:
    """Nearest valid pixel.

    A projection can be inside the frame and still round to ``width`` (e.g.
    ``u = 511.6``), so rounding has to be followed by clamping before the value
    is used to index the depth buffer.
    """
    col = int(np.clip(round(u), 0, camera.width - 1))
    row = int(np.clip(round(v), 0, camera.height - 1))
    return row, col


def _winner_belongs_to(landmark, camera, render_result, row, col, cloud, grid,
                       result: dict, padding: float = 1.0) -> bool:
    """Is the surface drawn at this pixel part of the landmark itself?

    The winning point is recovered through ``grid.order`` when the render came
    from the grid, and tested against the landmark's own bounding box.  Returns
    False whenever that cannot be established, so an unknown winner is still
    treated as an occluder.
    """
    lm_id, name, obj_type, pos, dim = landmark
    if cloud is None or grid is None:
        return False
    if render_result.stats.get("id_space") != "slot":
        return False
    slot = int(render_result.point_id[row, col])
    if slot < 0:
        return False
    try:
        vertex = int(grid.order[slot])
        world = cloud.xyz(np.array([vertex], dtype=np.int64))[0]
    except (IndexError, ValueError):
        return False

    half = np.asarray(dim, dtype=np.float64) / 2.0 + padding
    inside = np.all(np.abs(world - np.asarray(pos, dtype=np.float64)) <= half)
    result["occluder_position"] = [float(v) for v in world]
    result["occluder_in_landmark_box"] = bool(inside)
    return bool(inside)


def landmark_visibility(
    landmark,
    camera,
    render_result,
    cloud=None,
    grid=None,
    local_radius: float = 3.0,
    occlusion_tolerance: float = 1.5,
    frame_margin: float = 0.5,
) -> dict:
    """Full visibility estimate for one landmark.

    ``landmark`` is ``(id, name, object_type, position, dimension)``.

    A landmark projecting well outside the frame is settled without touching the
    point cloud: it cannot be visible either way, and the local-point query is
    the expensive part here.  ``frame_margin`` is a fraction of the frame size.
    """
    lm_id, name, obj_type, pos, dim = landmark
    pos = np.asarray(pos, dtype=np.float64)
    dim = np.asarray(dim, dtype=np.float64)

    centre = project_centre(pos, camera)
    result = {
        "id": int(lm_id), "name": name, "object_type": obj_type,
        "position": pos.tolist(), "dimension": dim.tolist(),
        "camera_distance": float(np.linalg.norm(pos - camera.position)),
        "in_fov": bool(centre and centre["inside"]),
        "geometrically_visible": bool(centre is not None),
        "observed_points": 0,
        "projected_points_in_frame": 0,
        "occluded_at_centre": None,
        "approx_visible_ratio": 0.0,
    }
    if centre is None:
        return result

    result["projection"] = {k: centre[k] for k in
                            ("u", "v", "z_cam", "metric", "offset_from_centre_px")}

    mx = frame_margin * camera.width
    my = frame_margin * camera.height
    if not (-mx <= centre["u"] <= camera.width + mx
            and -my <= centre["v"] <= camera.height + my):
        return result  # off-frame by more than the margin: not visible, and no
        # cloud query is needed to know it

    # Occlusion: is there measured geometry in front of the landmark at its pixel?
    if centre["inside"]:
        row, col = _pixel_index(centre["u"], centre["v"], camera)
        rendered = float(render_result.depth[row, col])
        if np.isfinite(rendered):
            result["rendered_depth_at_centre"] = rendered
            occluding = rendered < centre["metric"] - occlusion_tolerance
            # A CityRefer position sits at the object's base, so looking down at
            # a building puts that building's own roof in front of it.  Being
            # covered by your own surface is not being occluded, so the winning
            # point is resolved and kept if it falls inside the object's box.
            if occluding and _winner_belongs_to(landmark, camera, render_result,
                                                row, col, cloud, grid, result):
                occluding = False
            result["occluded_at_centre"] = bool(occluding)
        else:
            result["rendered_depth_at_centre"] = None
            result["occluded_at_centre"] = None

    # Measured points around the landmark that actually land in the frame.
    if cloud is not None and grid is not None:
        idx = grid.query_radius(float(pos[0]), float(pos[1]), local_radius)
        result["observed_points"] = int(idx.size)
        if idx.size:
            pts = cloud.xyz(idx)
            proj = project_points(pts, camera)
            result["projected_points_in_frame"] = int(proj["u"].size)
            if proj["u"].size:
                depth = render_result.depth
                rows = np.clip(proj["v"], 0, camera.height - 1).astype(np.int64)
                cols = np.clip(proj["u"], 0, camera.width - 1).astype(np.int64)
                rendered = depth[rows, cols]
                unoccluded = ~np.isfinite(rendered) | (
                    rendered >= proj["metric"] - occlusion_tolerance)
                result["visible_point_ratio"] = float(unoccluded.mean())

    # Approximate visible ratio from the object's axis-aligned box corners.
    corners = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                corners.append(pos + dim * np.array([sx, sy, sz]))
    corners = np.asarray(corners)
    n_ok = 0
    for c in corners:
        p = project_centre(c, camera)
        if p is None or not p["inside"]:
            continue
        row, col = _pixel_index(p["u"], p["v"], camera)
        rendered = float(render_result.depth[row, col])
        if np.isfinite(rendered) and rendered < p["metric"] - occlusion_tolerance:
            continue
        n_ok += 1
    result["approx_visible_ratio"] = float(n_ok / len(corners))

    result["visible"] = bool(
        result["in_fov"]
        and result["observed_points"] > 0
        and result["occluded_at_centre"] is not True
        and result["approx_visible_ratio"] > 0.0
    )
    return result


LANDMARK_COLOURS = {
    "referenced": (255, 64, 64),    # red   - named in the instruction
    "other": (64, 128, 255),        # blue  - other CityRefer candidates
    "occluded": (255, 176, 32),     # amber - in frame but blocked
}


def draw_landmarks(rgb: np.ndarray, landmarks: list, camera, render_result,
                   referenced_ids=(), cloud=None, grid=None,
                   local_radius: float = 3.0, show_labels: bool = True,
                   marker_min: int = 8, marker_max: int = 46) -> tuple:
    """Overlay projected landmarks, distinguishing referenced from other candidates.

    Marker size encodes distance so a far landmark is not drawn at the same
    visual weight as a near one.  Landmarks outside the frame are not drawn.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for landmark overlays")
    canvas = rgb.copy()
    annotated = []
    referenced_ids = set(int(i) for i in referenced_ids)

    for lm in landmarks:
        vis = landmark_visibility(lm, camera, render_result, cloud, grid, local_radius)
        annotated.append(vis)
        proj = vis.get("projection")
        if proj is None or not (-80 <= proj["u"] < camera.width + 80
                                and -80 <= proj["v"] < camera.height + 80):
            continue

        u, v = int(round(proj["u"])), int(round(proj["v"]))
        is_ref = int(lm[0]) in referenced_ids
        if vis["occluded_at_centre"] is True:
            colour = LANDMARK_COLOURS["occluded"]
        elif is_ref:
            colour = LANDMARK_COLOURS["referenced"]
        else:
            colour = LANDMARK_COLOURS["other"]

        scale = np.clip(60.0 / max(vis["camera_distance"], 1.0), 0.0, 1.0)
        size = int(marker_min + (marker_max - marker_min) * scale)
        thickness = 3 if is_ref else 2
        cv2.rectangle(canvas, (u - size // 2, v - size // 2),
                      (u + size // 2, v + size // 2), colour, thickness, cv2.LINE_AA)
        if show_labels:
            label = lm[1] or f"{lm[2]}#{lm[0]}"
            tag = f"{label} {vis['camera_distance']:.0f}m"
            cv2.putText(canvas, tag, (u + size // 2 + 4, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    return canvas, annotated


def summarise_visibility(annotated: list, referenced_ids=()) -> dict:
    referenced_ids = set(int(i) for i in referenced_ids)
    ref = [a for a in annotated if a["id"] in referenced_ids]
    other = [a for a in annotated if a["id"] not in referenced_ids]
    return {
        "landmarks_total": len(annotated),
        "referenced_total": len(ref),
        "referenced_in_fov": sum(1 for a in ref if a["in_fov"]),
        "referenced_visible": sum(1 for a in ref if a.get("visible")),
        "other_in_fov": sum(1 for a in other if a["in_fov"]),
        "other_visible": sum(1 for a in other if a.get("visible")),
        "referenced_distance_median": (
            float(np.median([a["camera_distance"] for a in ref])) if ref else None),
        "referenced_projected_points_median": (
            float(np.median([a["projected_points_in_frame"] for a in ref])) if ref else None),
    }
