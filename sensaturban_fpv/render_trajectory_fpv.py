"""Render a CityNav trajectory into perspective RGB-D from the SensatUrban cloud.

One directory per sampled pose holding the three views, a top-down context crop,
the landmark overlay, and a metadata file that records everything needed to
reproduce or refute the frame.  No navigation model, controller or trajectory is
touched: the poses are read, transformed, and rendered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .fit_coordinate_transform import CoordinateTransform
from .pointcloud_renderer import (
    Camera,
    depth_histogram,
    render_cloud_region,
    rasterize_topdown,
)
from .project_landmarks import draw_landmarks, summarise_visibility

VIEWS = {
    "fpv": 0.0,
    "fpv_down10": -10.0,
    "oblique": -30.0,
    "steep_oblique": -45.0,
}


@dataclass
class PoseSample:
    split: str
    episode_index: int
    map_name: str
    step: int
    citynav_pose: np.ndarray      # (4,) x, y, z, yaw
    position_ply: np.ndarray      # (3,) after the coordinate transform

    def to_json(self) -> dict:
        return {
            "split": self.split,
            "episode_index": int(self.episode_index),
            "map": self.map_name,
            "step": int(self.step),
            "citynav_pose": [float(v) for v in self.citynav_pose],
        }


def sample_poses(episodes_by_split: dict, per_split: dict, transform: CoordinateTransform,
                 prefer_landmarks: bool = True) -> list:
    """Pick a small, diverse, reproducible set of poses.

    Sampling is spread over episodes that actually reference a landmark, then
    over steps within the episode, so the set is not dominated by one flight.
    """
    samples = []
    for split, episodes in episodes_by_split.items():
        want = per_split.get(split, 0)
        if want <= 0:
            continue
        pool = [e for e in episodes if e.object_ids] if prefer_landmarks else episodes
        if not pool:
            pool = episodes
        # Spread over episodes deterministically, then take one step from each.
        picks = np.linspace(0, len(pool) - 1, num=min(want, len(pool))).round().astype(int)
        for rank, ep_i in enumerate(picks):
            ep = pool[int(ep_i)]
            n = len(ep.trajectory)
            step = int(round((rank + 1) / (len(picks) + 1) * (n - 1)))
            yaw = float(ep.yaw()[step])
            pose4 = np.array([*ep.trajectory[step, :3], yaw], dtype=np.float64)
            pos_ply = transform.apply_xyz(ep.trajectory[step:step + 1, :3])[0]
            samples.append(PoseSample(split, ep.index, ep.map_name, step, pose4, pos_ply))
    return samples


def render_views(cloud, grid, sample: PoseSample, config: dict) -> dict:
    """Render every configured view for one pose; returns name -> RenderResult."""
    out = {}
    for name, pitch_deg in config["views"].items():
        camera = Camera(
            position=sample.position_ply,
            yaw=float(sample.citynav_pose[3]),
            pitch=np.deg2rad(pitch_deg),
            width=config["width"], height=config["height"],
            hfov_deg=config["hfov_deg"],
            near=config["near"], far=config["far"],
        )
        out[name] = render_cloud_region(
            cloud, grid, camera,
            splat_radius=config.get("splat_radius", 0),
            lod=config.get("lod"),
        )
    return out


def topdown_context(cloud, grid, sample: PoseSample, extent: float = 120.0,
                    resolution: float = 1.0) -> dict:
    """A local orthographic crop centred on the pose, for the context panel."""
    x, y = float(sample.position_ply[0]), float(sample.position_ply[1])
    # Read through the bucket-ordered cache: this crop covers millions of
    # points, and gathering them straight out of the PLY is a per-element
    # random read that dominated the per-pose cost.
    slots = grid.query_radius_positions(x, y, extent, None)
    if slots.size == 0:
        return {"image": None, "counts": 0}
    xyz_sorted, rgb_sorted = grid.sorted_arrays()
    xyz = grid.read_sorted(xyz_sorted, slots)
    rgb = grid.read_sorted(rgb_sorted, slots) if cloud.has_rgb else None
    rows = int(round(2 * extent / resolution))
    shape = (rows, rows)
    origin = (x - extent, y + extent)
    rast = rasterize_topdown(xyz, rgb, origin, resolution, shape, z_mode="max")

    image = rast["rgb"].copy()
    if cv2 is not None:
        col = int(round((x - origin[0]) / resolution))
        row = int(round((origin[1] - y) / resolution))
        cv2.drawMarker(image, (col, row), (255, 0, 0), cv2.MARKER_CROSS, 21, 2, cv2.LINE_AA)
        # Heading arrow: CityNav forward is (cos yaw, sin yaw) in world XY, and
        # the crop's rows grow as y falls, so the y component flips sign.
        tip = (int(col + 40 * np.cos(sample.citynav_pose[3])),
               int(row - 40 * np.sin(sample.citynav_pose[3])))
        cv2.arrowedLine(image, (col, row), tip, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.3)
    return {"image": image, "counts": int(rast["counts"].sum()), "height": rast["height"]}


def metadata_for(sample: PoseSample, views: dict, config: dict,
                 visibility: dict | None, transform: CoordinateTransform) -> dict:
    primary = views[config.get("primary_view", "fpv")]
    hist = depth_histogram(primary.depth)
    return {
        "map": sample.map_name,
        "episode_id": f"{sample.split}:{sample.episode_index}",
        "split": sample.split,
        "step": sample.step,
        "citynav_pose": [float(v) for v in sample.citynav_pose],
        "transformed_pose": [
            float(sample.position_ply[0]), float(sample.position_ply[1]),
            float(sample.position_ply[2]), float(sample.citynav_pose[3]),
        ],
        "coordinate_transform": transform.to_json(),
        "yaw": float(sample.citynav_pose[3]),
        "yaw_deg": float(np.rad2deg(sample.citynav_pose[3])),
        "pitch": float(config["views"][config.get("primary_view", "fpv")]),
        "fov": float(config["hfov_deg"]),
        "resolution": [int(config["width"]), int(config["height"])],
        "valid_pixel_ratio": float(primary.valid_ratio),
        "below_horizon_valid_ratio": primary.stats.get("below_horizon_valid_ratio"),
        "horizon_row": primary.stats.get("horizon_row"),
        "depth_median": hist.get("median"),
        "depth_min": hist.get("min"),
        "depth_max": hist.get("max"),
        "depth_std": hist.get("std"),
        "depth_unique_rounded": hist.get("unique_rounded"),
        "splat_radius_px": int(config.get("splat_radius", 0)),
        "lod": config.get("lod"),
        "views": {
            name: {
                "pitch_deg": float(config["views"][name]),
                "valid_pixel_ratio": float(r.valid_ratio),
                "below_horizon_valid_ratio": r.stats.get("below_horizon_valid_ratio"),
                "sky_pixel_ratio": r.stats.get("sky_pixel_ratio"),
                "in_tile_pixel_ratio": r.stats.get("in_tile_pixel_ratio"),
                "in_tile_valid_ratio": r.stats.get("in_tile_valid_ratio"),
                "points_rendered": int(r.stats["points_rendered"]),
                "points_projected": int(r.stats["points_projected"]),
                "depth_median": (depth_histogram(r.depth).get("median")),
            }
            for name, r in views.items()
        },
        "landmark_visibility": visibility,
    }


def write_pose_artifacts(out_dir: Path, sample: PoseSample, views: dict,
                         context: dict, overlays: dict, metadata: dict) -> dict:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to write pose artifacts")
    pose_dir = out_dir / f"{sample.split}_{sample.episode_index:05d}_step{sample.step:03d}"
    pose_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for name, result in views.items():
        rgb_path = pose_dir / f"{name}_rgb.png"
        depth_path = pose_dir / f"{name}_depth.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(result.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(depth_path), _depth_to_png(result.depth))
        written[f"{name}_rgb"] = str(rgb_path)
        written[f"{name}_depth"] = str(depth_path)

        # Keep the raw render alongside any enhanced one, always.
        if metadata.get("splat_radius_px", 0) > 0:
            raw = views.get(f"{name}_raw")
            if raw is not None:
                raw_path = pose_dir / f"{name}_rgb_raw.png"
                cv2.imwrite(str(raw_path), cv2.cvtColor(raw.rgb, cv2.COLOR_RGB2BGR))
                written[f"{name}_rgb_raw"] = str(raw_path)

    if context.get("image") is not None:
        ctx_path = pose_dir / "topdown_context.png"
        cv2.imwrite(str(ctx_path), cv2.cvtColor(context["image"], cv2.COLOR_RGB2BGR))
        written["topdown_context"] = str(ctx_path)

    for name, overlay in overlays.items():
        path = pose_dir / f"{name}_landmarks.png"
        cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        written[f"{name}_landmarks"] = str(path)

    meta_path = pose_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    written["metadata"] = str(meta_path)
    return written


def _depth_to_png(depth: np.ndarray) -> np.ndarray:
    """Visualise depth with a fixed scale so frames stay comparable."""
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    finite = np.isfinite(depth)
    if finite.any():
        norm = np.clip(depth / 150.0, 0.0, 1.0)
        scaled = (255 * (1.0 - norm)).astype(np.uint8)
        out[finite] = cv2.applyColorMap(scaled[finite].reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return out


def render_pose_bundle(cloud, grid, sample: PoseSample, config: dict,
                       landmarks: list, transform: CoordinateTransform,
                       referenced_ids=()) -> dict:
    """Render one pose and assemble its metadata and overlays."""
    views = render_views(cloud, grid, sample, config)
    context = topdown_context(cloud, grid, sample, extent=config.get("context_extent", 120.0))

    overlays = {}
    vis_summary = None
    for name in config.get("overlay_views", ["fpv"]):
        result = views[name]
        camera = Camera(
            position=sample.position_ply, yaw=float(sample.citynav_pose[3]),
            pitch=np.deg2rad(config["views"][name]),
            width=config["width"], height=config["height"],
            hfov_deg=config["hfov_deg"], near=config["near"], far=config["far"],
        )
        overlay, annotated = draw_landmarks(
            result.rgb, landmarks, camera, result,
            referenced_ids=referenced_ids, cloud=cloud, grid=grid,
        )
        overlays[name] = overlay
        if name == config.get("primary_view", "fpv"):
            vis_summary = {
                "summary": summarise_visibility(annotated, referenced_ids),
                "landmarks": annotated,
            }

    metadata = metadata_for(sample, views, config, vis_summary, transform)
    return {"views": views, "context": context, "overlays": overlays,
            "visibility": vis_summary, "metadata": metadata}
