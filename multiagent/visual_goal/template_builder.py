"""Build georeferenced overhead goal templates and genuine trajectory queries."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rasterio
from rasterio.transform import rowcol


TEMPLATE_EXTENTS_M = (40.0, 80.0, 120.0)
IMAGE_SIZE = 224
MASK_ORIENTATION = "hett_yaw0_world_positive_x_maps_to_image_left_v1"
SEMANTIC_TYPES = {
    "building": {"Building", "Wall"},
    "road": {"TrafficRoad", "Footpath", "Rail", "Bridge"},
    "parking": {"Parking"},
    "vegetation": {"HighVegetation"},
    "open_area": {"Ground", "Water"},
}


def opaque_key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:20]


def world_to_template(points_xy, center_xy, extent_m, size=IMAGE_SIZE):
    """Axis-aligned overhead crop, world XY to image col/row."""
    p = np.asarray(points_xy, dtype=np.float64)
    cx, cy = center_xy
    # HETT yaw=0 defines +x as image-left and +y as image-up, matching the
    # front-left -> front-right -> back-right -> back-left crop corner order.
    col = ((cx + extent_m / 2) - p[..., 0]) / extent_m * (size - 1)
    row = ((cy + extent_m / 2) - p[..., 1]) / extent_m * (size - 1)
    return np.stack([col, row], axis=-1)


def polygon_mask(contour_xy, center_xy, extent_m, size=IMAGE_SIZE):
    mask = np.zeros((size, size), dtype=np.uint8)
    if not contour_xy:
        return mask.astype(bool)
    pixels = np.round(world_to_template(contour_xy, center_xy, extent_m, size)).astype(np.int32)
    cv2.fillPoly(mask, [pixels], 1)
    return mask.astype(bool)


def _scene_key(map_name, object_id):
    return opaque_key("scene", map_name, int(object_id))


@dataclass
class MapRGB:
    name: str
    rgb: np.ndarray
    raster: object

    @classmethod
    def open(cls, image_dir: Path, map_name: str):
        raster = rasterio.open(image_dir / f"{map_name}.tif")
        bgr = cv2.imread(str(image_dir / f"{map_name}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raster.close()
            raise FileNotFoundError(image_dir / f"{map_name}.png")
        return cls(map_name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), raster)

    def close(self):
        self.raster.close()

    def crop_pose(self, xyz_yaw, ground_level, size=IMAGE_SIZE):
        """Render the same orthographic image footprint as the HETT crop client."""
        x, y, z, yaw = map(float, xyz_yaw)
        h = z - float(ground_level)
        if h <= 0:
            raise ValueError("UAV pose must be above ground")
        c, s = np.cos(yaw), np.sin(yaw)
        front, left, center = np.array([c, s]), np.array([-s, c]), np.array([x, y])
        corners = np.asarray([
            center + h * (front + left), center + h * (front - left),
            center + h * (-front - left), center + h * (-front + left),
        ], dtype=np.float64)
        src = np.asarray([self.raster.index(px, py)[::-1] for px, py in corners], dtype=np.float32)
        dst = np.asarray([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(self.rgb, matrix, (size, size), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def crop_goal(self, center_xy, extent_m, size=IMAGE_SIZE):
        # A yaw-zero virtual top-down pose gives a square world footprint of
        # precisely `extent_m`. This is an orthographic raster crop, not FPV.
        x, y = map(float, center_xy)
        return self.crop_pose((x, y, 0.5 * float(extent_m), 0.0), 0.0, size=size)


def _annotation_regions(objects_by_id, target_obj, referenced_names, center_xy, extent, size):
    h, w = (int(size[0]), int(size[1])) if isinstance(size, (tuple, list)) else (int(size), int(size))
    masks = {"target": np.zeros((h, w), bool), "anchor": np.zeros((h, w), bool)}
    masks.update({k: np.zeros((h, w), bool) for k in SEMANTIC_TYPES})
    matched_anchor_ids = set()
    norm_refs = [str(s).casefold().strip() for s in referenced_names if str(s).strip()]
    for oid, obj in objects_by_id.items():
        contour = obj.get("contour") or []
        if len(contour) < 3:
            continue
        poly = polygon_mask(contour, center_xy, extent, size=size[0])
        typ = obj.get("object_type", "")
        for label, types in SEMANTIC_TYPES.items():
            if typ in types:
                masks[label] |= poly
        if int(oid) == int(target_obj["id"]):
            masks["target"] |= poly
        name = str(obj.get("name", "")).casefold().strip()
        if name and any(name in ref or ref in name for ref in norm_refs):
            masks["anchor"] |= poly
            matched_anchor_ids.add(int(oid))
    return masks, sorted(matched_anchor_ids)


def _save_rgb(path, image, quality=92):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"failed writing {path}")


def build_split(split: str, out_dir: Path, data_root: Path, image_dir: Path,
                max_episodes: Optional[int] = None, query_per_episode_bin: int = 1):
    """Write images + metadata JSONL for train_seen/val_seen/val_unseen only."""
    if split not in {"train_seen", "val_seen", "val_unseen"}:
        raise ValueError("test_unseen is intentionally disabled for this experiment")
    cityrefer = data_root / "cityrefer"
    traj_dir = data_root / "processed_citynav"
    objects = json.loads((cityrefer / "objects.json").read_text())
    processed = json.loads((cityrefer / "processed_descriptions.json").read_text())
    trajectories = json.loads((traj_dir / f"citynav_{split}.json").read_text())
    if max_episodes is not None:
        trajectories = trajectories[:int(max_episodes)]
    from multiagent.mapdata import GROUND_LEVEL

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images" / split
    template_rows, query_rows = [], []
    opened = {}
    templates_written = set()
    invalid_query_poses = 0
    trajectories = list(enumerate(trajectories))
    trajectories.sort(key=lambda item: (item[1]["area"], str(item[1]["block"]), item[0]))
    active_map = None
    for ix, tr in trajectories:
        map_name = f"{tr['area']}_block_{tr['block']}"
        target_id = int(tr["object_ids"][0])
        ann_id = int(tr["ann_ids"][0])
        target_obj = objects[map_name].get(str(target_id))
        if target_obj is None or len(tr.get("trajectory", [])) == 0:
            continue
        target_xy = target_obj["position"][:2]
        scene_key = _scene_key(map_name, target_id)
        episode_key = opaque_key(split, map_name, target_id, ann_id, ix)
        annotation_key = opaque_key("annotation", map_name, target_id, ann_id)
        try:
            desc = processed[map_name][str(target_id)][ann_id]
            refs = desc.get("landmarks", [])
        except (KeyError, IndexError, TypeError):
            refs = []
        map_objects = objects[map_name]
        if active_map is not None and active_map != map_name:
            opened.pop(active_map).close()
        if map_name not in opened:
            opened[map_name] = MapRGB.open(image_dir, map_name)
        active_map = map_name
        map_rgb = opened[map_name]
        center = tuple(map(float, target_xy))
        context_key = opaque_key("context", scene_key, *sorted(map(str, refs)))
        for extent in TEMPLATE_EXTENTS_M:
            stable_template_key = opaque_key(context_key, int(extent))
            template_key = (stable_template_key, int(extent))
            if template_key not in templates_written:
                goal = map_rgb.crop_goal(center, extent)
                masks, anchor_ids = _annotation_regions(map_objects, target_obj, refs, center, extent,
                                                        (IMAGE_SIZE, IMAGE_SIZE))
                stem = f"goal_{scene_key}_{int(extent)}m.jpg"
                relpath = Path("images") / split / stem
                _save_rgb(out_dir / relpath, goal)
                mask_path = Path("masks") / split / f"mask_{context_key}_{int(extent)}m.npz"
                (out_dir / mask_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_dir / mask_path, **masks,
                                    _orientation_version=np.asarray(MASK_ORIENTATION))
                template_rows.append({
                    "template_key": stable_template_key,
                    "context_key": context_key,
                    "annotation_key": annotation_key,
                    "scene_key": scene_key,
                    "map_name": map_name,
                    "target_xy": center,
                    "target_type": target_obj.get("object_type", "unknown"),
                    "extent_m": extent,
                    "image_path": str(relpath),
                    "mask_path": str(mask_path),
                    "anchor_count": len(anchor_ids),
                    "referenced_landmarks": list(refs),
                    "view": "orthographic_top_down",
                    "split": split,
                })
                templates_written.add(template_key)
            else:
                continue
        # Trajectory entries are real flown poses, with no candidate-centered
        # crop. Pick <= one observation from each distance stratum.
        poses = tr["trajectory"]
        xy = np.asarray([[p[0], p[1]] for p in poses], dtype=np.float64)
        dists = np.linalg.norm(xy - np.asarray(target_xy, dtype=np.float64)[None, :], axis=1)
        strata = ((80, float("inf"), ">80m"), (40, 80, "40-80m"),
                  (20, 40, "20-40m"), (0, 20, "0-20m"))
        for lo, hi, bin_name in strata:
            inds = np.flatnonzero((dists >= lo) & (dists < hi))
            if not len(inds):
                continue
            midpoint = lo + 10 if not np.isfinite(hi) else (lo + hi) / 2
            chosen = sorted(inds, key=lambda j: abs(dists[j] - midpoint))[:max(1, query_per_episode_bin)]
            for qi in sorted(chosen):
                pose = poses[int(qi)]
                if len(pose) == 5:
                    x, y, z, yaw, pitch = map(float, pose)
                else:
                    x, y, z, dx, dy, dz = map(float, pose)
                    yaw = float(np.arctan2(dy, dx))
                    pitch = float(np.arctan2(dz, np.hypot(dx, dy)))
                ground = float(GROUND_LEVEL[map_name])
                if z - ground <= 0.5:
                    invalid_query_poses += 1
                    continue
                query = map_rgb.crop_pose((x, y, z, yaw), ground)
                relpath = Path("images") / split / f"query_{episode_key}_{int(qi):05d}.jpg"
                _save_rgb(out_dir / relpath, query)
                query_rows.append({
                    "query_key": opaque_key("query", episode_key, qi),
                    "episode_key": episode_key,
                    "annotation_key": annotation_key,
                    "scene_key": scene_key,
                    "map_name": map_name,
                    "target_xy": center,
                    "target_type": target_obj.get("object_type", "unknown"),
                    "instruction": tr.get("descriptions", [""])[0],
                    "trajectory_index": int(qi),
                    "distance_to_goal_m": float(dists[qi]),
                    "distance_bin": bin_name,
                    "altitude_agl_m": z - ground,
                    "yaw_rad": yaw,
                    "pitch_rad": pitch,
                    "brightness": float(np.asarray(query, dtype=np.float32).mean() / 255.0),
                    "image_path": str(relpath),
                    "split": split,
                    "view": "orthographic_trajectory_observation",
                })
        if (ix + 1) % 250 == 0:
            print(f"{split}: processed {ix + 1}/{len(trajectories)} episodes; {len(query_rows)} trajectory queries", flush=True)
    for raster in opened.values():
        raster.close()
    _write_jsonl(out_dir / f"templates_{split}.jsonl", template_rows)
    _write_jsonl(out_dir / f"queries_{split}.jsonl", query_rows)
    stats = {
        "split": split, "annotation_rows": len(trajectories),
        "goal_template_rows": len(template_rows), "trajectory_query_rows": len(query_rows),
        "invalid_or_ground_level_trajectory_poses": invalid_query_poses,
        "unique_scenes": len({r["scene_key"] for r in template_rows}),
        "goal_template_extents_m": list(TEMPLATE_EXTENTS_M),
        "query_source": "actual pose from released CityNav human trajectory",
        "mask_world_to_image_orientation": MASK_ORIENTATION,
        "raw_anchor_object_ids_in_candidate_metadata": False,
        "template_source": "georeferenced RGB orthophoto, offline crop centered on GT target",
        "available_views": ["orthographic_top_down", "orthographic_trajectory_observation"],
        "unavailable_views": ["oblique", "FPV: no calibrated 3D renderer / camera geometry in this data interface"],
    }
    (out_dir / f"dataset_{split}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
