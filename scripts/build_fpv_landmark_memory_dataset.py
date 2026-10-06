#!/usr/bin/env python3
"""Render true AirSim perspective views at CityNav teacher poses.

The source manifest already enforces official split membership. GT geometry is
written only to label NPZ files; ``model_inputs`` intentionally excludes it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multiagent.fpv_landmark_memory.projection import project_world_polygon


SPLITS = ("train_seen", "val_seen", "val_unseen")
VIEWS = {"fpv": 0.0, "oblique30": -30.0, "oblique45": -45.0}
HISTORY_OFFSETS = (0, 1, 3, 5, 7, 9, 11, 13)


def model_inputs(row):
    return {k: row[k] for k in ("instruction", "image", "split", "sample_id") if k in row}


def pose_at(episode, step, pitch_deg):
    raw = episode["trajectory"][max(0, min(int(step), len(episode["trajectory"]) - 1))]
    if len(raw) == 5:
        x, y, z, yaw, pitch = map(float, raw)
    else:
        x, y, z, dx, dy, dz = map(float, raw)
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, math.hypot(dx, dy))
    # Explicit camera configuration is independent of trajectory pitch.
    return [x, y, z, yaw, math.radians(float(pitch_deg))]


def history_steps(current_step):
    current_step = int(current_step)
    return [(current_step - offset, offset) for offset in HISTORY_OFFSETS if current_step - offset >= 0]


def project_candidates(row, object_table, pose, size):
    records, masks = [], {}
    positive = set(map(int, row.get("referenced_landmark_ids", ())))
    for candidate in row.get("candidates", ()):
        oid = int(candidate["landmark_id"])
        obj = object_table.get(str(oid))
        if obj is None:
            continue
        dim = obj.get("dimension", [0, 0, 0])
        pos = obj.get("position", [0, 0, 0])
        height = max(float(dim[2]), 0.2) if str(obj.get("object_type", "")).lower() in {"building", "house", "church", "tower"} else 0.2
        projection = project_world_polygon(obj.get("contour", ()), pose, image_size=(size, size),
                                           object_z=float(pos[2]), object_height=height)
        item = {"landmark_id": oid, "name": obj.get("name", ""), "category": obj.get("object_type", ""),
                "world_xy": list(map(float, pos[:2])), "is_referenced": oid in positive,
                "distance_m": float(math.hypot(float(pose[0]) - float(pos[0]), float(pose[1]) - float(pos[1]))),
                "relative_yaw_rad": float((math.atan2(float(pos[1]) - float(pose[1]), float(pos[0]) - float(pose[0])) - float(pose[3]) + math.pi) % (2 * math.pi) - math.pi),
                "occlusion_ratio": None,
                "visible": projection is not None, "visible_ratio": 0.0, "pixel_area": 0,
                "bbox_xyxy": None, "center_uv": None}
        if projection:
            item.update({k: projection[k] for k in ("visible_ratio", "pixel_area", "bbox_xyxy", "center_uv")})
            masks[f"candidate_{oid}"] = projection["mask"]
        records.append(item)
    return records, masks


class AirSimPerspectiveRenderer:
    def __init__(self, host, port, output, size):
        import airsim
        self.airsim = airsim
        self.client = airsim.VehicleClient(ip=host, port=int(port), timeout_value=60)
        self.client.confirmConnection()
        self.output = Path(output)
        self.size = int(size)
        self.loaded = None

    def render(self, map_name, pose, camera_pitch_deg, key):
        airsim = self.airsim
        scene_code = str(map_name)[0] + str(map_name).split("_")[-1]
        if self.loaded != scene_code:
            self.client.simLoadLevel(scene_code)
            self.loaded = scene_code
            # Unreal level streaming is asynchronous; avoid recording a blank frame.
            time.sleep(0.75)
        x, y, z, yaw, _ = pose
        rotation = airsim.to_quaternion(math.radians(camera_pitch_deg), 0.0, -float(yaw))
        position = airsim.Vector3r(float(x), -float(y), -float(z))
        self.client.simSetVehiclePose(airsim.Pose(position, rotation), True, vehicle_name="Drone_1")
        time.sleep(0.04)
        requests = [
            airsim.ImageRequest("front_0", airsim.ImageType.Scene, pixels_as_float=False, compress=False),
            airsim.ImageRequest("front_0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
        ]
        image, depth = self.client.simGetImages(requests, vehicle_name="Drone_1")
        if image.height <= 0 or not image.image_data_uint8:
            raise RuntimeError(f"AirSim returned empty frame ({map_name}, {key})")
        frame = np.frombuffer(image.image_data_uint8, np.uint8).reshape(image.height, image.width, 3)
        frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        out = self.output / "images" / f"{key}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(out)
        depth_array = np.asarray(depth.image_data_float, dtype=np.float32)
        if depth_array.size:
            valid = depth_array[np.isfinite(depth_array) & (depth_array > 0)]
            depth_info = {"size": [int(depth.width), int(depth.height)],
                          "valid_pixels": int(np.sum(valid < 0.999)),
                          "min": float(valid.min()) if len(valid) else None,
                          "max": float(valid.max()) if len(valid) else None,
                          "usable": bool(len(valid) and np.any(valid < 0.999))}
        else:
            depth_info = {"usable": False, "valid_pixels": 0}
        return out, frame, depth_info


def camera_metadata(pose, pitch_deg, size):
    focal = (float(size) / 2.0) / math.tan(math.radians(90.0) / 2.0)
    yaw = float(pose[3]); offset = 0.5
    return {"intrinsics": {"width": int(size), "height": int(size), "horizontal_fov_deg": 90.0,
                            "fx_px": focal, "fy_px": focal, "cx_px": size / 2.0, "cy_px": size / 2.0},
            "extrinsics": {"camera_name": "front_0", "vehicle_to_camera_xyz_ned_m": [offset, 0.0, 0.0],
                           "camera_pitch_deg": float(pitch_deg), "camera_yaw_rad": yaw,
                           "coordinate_transform": "CityNav ENU (x,y,z,yaw) -> AirSim NED (x,-y,-z,pitch,roll,-yaw)"}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dataset", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--airsim-host", default="127.0.0.1")
    p.add_argument("--airsim-port", type=int, default=30001)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--limit-per-split", type=int, default=0)
    a = p.parse_args()
    if "test_unseen" in " ".join(map(str, vars(a).values())):
        raise ValueError("test_unseen is forbidden")
    output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    source_rows = [json.loads(line) for line in (a.source_dataset / "manifest.jsonl").read_text().splitlines()]
    episodes = {s: json.loads((a.data_root / "processed_citynav" / f"citynav_{s}.json").read_text()) for s in SPLITS}
    cityrefer = json.loads((a.data_root / "cityrefer" / "objects.json").read_text())
    renderer = AirSimPerspectiveRenderer(a.airsim_host, a.airsim_port, output, a.image_size)
    seen_frames, records = {}, []
    depth_stats = Counter(); started = time.time()
    for split in SPLITS:
        selected = [r for r in source_rows if r["split"] == split]
        if a.limit_per_split:
            selected = selected[:a.limit_per_split]
        for row in selected:
            _, epi = row["episode_id"].split(":", 1)
            episode = episodes[split][int(epi)]
            table = cityrefer[row["map_name"]]
            center_step = int(row["step"])
            current = {}
            history = []
            for step, offset in history_steps(center_step):
                fpv_pose = pose_at(episode, step, 0)
                fpv_key = f"{split}_{row['sample_id']}_t{step}_fpv"
                frame_key = (split, row["map_name"], tuple(round(x, 3) for x in fpv_pose), "fpv")
                if frame_key not in seen_frames:
                    path, frame, depth_info = renderer.render(row["map_name"], fpv_pose, 0, fpv_key)
                    seen_frames[frame_key] = {"path": str(path), "depth": depth_info}
                fpv = seen_frames[frame_key]
                depth_stats["usable"] += int(fpv["depth"].get("usable", False))
                depth_stats["frames"] += 1
                candidate_geom, candidate_masks = project_candidates(row, table, fpv_pose, a.image_size)
                label_path = output / "labels" / f"{row['sample_id']}_t{step}_fpv.npz"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(label_path, **candidate_masks)
                obs = {"step": step, "offset": offset, "image": fpv["path"], "pose": fpv_pose,
                       "depth": fpv["depth"], "candidates": candidate_geom,
                       **camera_metadata(fpv_pose, 0.0, a.image_size),
                       "label_file": str(label_path), "camera": "airsim_front_0",
                       "render_source": "CityNav Unreal AirSim scene RGB; Pose5D position/yaw; explicit pitch=0"}
                history.append(obs)
                if offset == 0:
                    current["fpv"] = obs
            for name, pitch in (("oblique30", -30.0), ("oblique45", -45.0)):
                view_pose = pose_at(episode, center_step, pitch)
                key = f"{split}_{row['sample_id']}_{name}"
                path, frame, depth_info = renderer.render(row["map_name"], view_pose, pitch, key)
                geom, masks = project_candidates(row, table, view_pose, a.image_size)
                label_path = output / "labels" / f"{row['sample_id']}_{name}.npz"
                np.savez_compressed(label_path, **masks)
                current[name] = {"step": center_step, "image": str(path), "pose": view_pose,
                                 "depth": depth_info, "candidates": geom,
                                 **camera_metadata(view_pose, pitch, a.image_size),
                                 "label_file": str(label_path), "camera": "airsim_front_0",
                                 "render_source": f"CityNav Unreal AirSim scene RGB; pitch={pitch}deg"}
            primary = next((int(x) for x in row.get("referenced_landmark_ids", ())), None)
            current_fpv = current["fpv"]
            target_geom = next((x for x in current_fpv["candidates"] if x["landmark_id"] == primary), None)
            record = {"sample_id": row["sample_id"], "episode_id": row["episode_id"], "split": split,
                      "scene_id": row["map_name"], "step": center_step, "trajectory_length": row["trajectory_length"],
                      "instruction": row["instruction"], "referenced_landmark_ids": row["referenced_landmark_ids"],
                      "referenced_landmark_names": row["referenced_landmark_names"],
                      "referenced_phrase": row.get("referenced_phrase", ""),
                      "candidate_landmarks": row["candidates"], "candidates": row["candidates"],
                      "view_observations": current,
                      "history_fpv": history, "topdown_image": row["image"],
                      "target_landmark_visible_now": bool(target_geom and target_geom["visible"]),
                      "target_landmark_seen_before": any(any(c["landmark_id"] == primary and c["visible"] for c in h["candidates"]) for h in history[1:]),
                      "target_geometry_label": target_geom}
            records.append(record)
            if len(records) % 100 == 0:
                print(f"rendered {len(records)} observations; unique frames={len(seen_frames)}", flush=True)
    with (output / "manifest.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {"samples": len(records), "frames": len(seen_frames), "by_split": dict(Counter(r["split"] for r in records)),
             "views": {"topdown": len(records), "fpv": len(records), "oblique30": len(records), "oblique45": len(records)},
             "history_frames": sum(len(r["history_fpv"]) for r in records),
             "depth": dict(depth_stats), "depth_usable_fraction": depth_stats["usable"] / max(depth_stats["frames"], 1),
             "projection": "CityRefer footprint + annotated height, pinhole estimate; no semantic mask or occlusion claim",
             "gt_geometry_in_model_input": False, "test_unseen_read": False,
             "seconds": time.time() - started, "source_manifest": str(a.source_dataset / "manifest.jsonl")}
    (output / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
