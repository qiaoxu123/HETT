from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpv_goal_approach.dataset import write_jsonl
from fpv_goal_approach.labels import (
    bearing_bin, clip_indices, distance_bin, geometric_arrival, phase,
    pose_is_duplicate, relative_bearing, wrap_radians,
)
from fpv_goal_approach.renderer import create_renderer, frame_key
from multiagent.dataset.mturk_trajectory import MTurkTrajectory


DISTANCE_ANCHORS = (80.0, 60.0, 40.0, 30.0, 20.0, 15.0, 10.0, 5.0)


def load_trajectories(data_root: Path, split: str) -> list[MTurkTrajectory]:
    path = data_root / "processed_citynav" / f"citynav_{split}.json"
    with path.open() as handle:
        return [MTurkTrajectory(**row) for row in json.load(handle)]


def load_cityrefer(data_root: Path):
    with (data_root / "cityrefer" / "objects.json").open() as handle:
        objects = json.load(handle)
    with (data_root / "cityrefer" / "processed_descriptions.json").open() as handle:
        processed = json.load(handle)
    return objects, processed


def episode_id(trajectory: MTurkTrajectory) -> str:
    start = trajectory.trajectory[0]
    payload = (
        f"{trajectory.map_name}|{trajectory.object_id}|{trajectory.desc_id}|"
        f"{start.x:.3f}|{start.y:.3f}|{start.z:.3f}|{trajectory.descriptions[0]}"
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def pose5d(pose) -> list[float]:
    return [float(pose.x), float(pose.y), float(pose.z), float(pose.yaw), float(pose.pitch)]


def select_anchor_indices(trajectory: MTurkTrajectory) -> list[int]:
    target = trajectory.target_position
    distances = [math.hypot(pose.x - target.x, pose.y - target.y) for pose in trajectory.trajectory]
    low, high = min(distances), max(distances)
    selected = []
    for requested in DISTANCE_ANCHORS:
        if low <= requested <= high:
            selected.append(min(range(len(distances)), key=lambda index: abs(distances[index] - requested)))
    selected.append(len(distances) - 1)
    unique = []
    for index in sorted(set(selected)):
        pose = pose5d(trajectory.trajectory[index])
        if not unique or not pose_is_duplicate(pose5d(trajectory.trajectory[unique[-1]]), pose):
            unique.append(index)
    return unique


def referenced_landmarks(map_name: str, object_id: int, desc_id: int, objects, processed):
    names = []
    descriptions = processed.get(map_name, {}).get(str(object_id), [])
    if desc_id < len(descriptions):
        names = descriptions[desc_id].get("landmarks", [])
    result = []
    for name in names:
        candidates = [obj for obj in objects[map_name].values() if obj.get("name", "").casefold() == name.casefold()]
        if candidates:
            position = candidates[0]["position"]
            result.append({"name": name, "xy": [float(position[0]), float(position[1])]})
    return result


def wrong_language(trajectory: MTurkTrajectory, objects) -> tuple[str, int, int] | None:
    current = objects[trajectory.map_name][str(trajectory.object_id)]
    cx, cy = current["position"][:2]
    candidates = []
    for object_id, obj in objects[trajectory.map_name].items():
        if int(object_id) == trajectory.object_id or not obj.get("descriptions"):
            continue
        same_type = obj.get("object_type") == current.get("object_type")
        ox, oy = obj["position"][:2]
        candidates.append((not same_type, math.hypot(ox - cx, oy - cy), int(object_id), obj))
    if not candidates:
        return None
    _, _, object_id, obj = min(candidates)
    return obj["descriptions"][0], object_id, 0


def render_clip(renderer, output_root: Path, map_name: str, poses: list[list[float]], metadata_only: bool):
    paths = []
    for pose in poses:
        key = frame_key(map_name, pose, renderer.source)
        relative = Path("images") / map_name / f"{key}.png"
        absolute = output_root / relative
        if not metadata_only and not absolute.exists():
            renderer.render(map_name, pose, absolute)
        paths.append(relative.as_posix())
    return paths


def make_sample(
    trajectory, split, ep_id, instruction, instruction_object_id, instruction_description_id,
    clip_poses, frame_paths, objects, processed, success_dist, negative_type,
    target_match, render_source,
):
    current = clip_poses[-1]
    target = trajectory.target_position
    distance = math.hypot(current[0] - target.x, current[1] - target.y)
    arrival = geometric_arrival(distance, success_dist)
    rel = relative_bearing(current[0], current[1], current[3], (target.x, target.y))
    return {
        "episode_id": ep_id,
        "split": split,
        "instruction": instruction,
        "map_name": trajectory.map_name,
        "target_object_id": int(trajectory.object_id),
        "description_id": int(trajectory.desc_id),
        "instruction_object_id": int(instruction_object_id),
        "instruction_description_id": int(instruction_description_id),
        "frames": frame_paths,
        "poses": clip_poses,
        "target_xy": [float(target.x), float(target.y)],
        "target_position": [float(target.x), float(target.y), float(target.z)],
        "referenced_landmarks": referenced_landmarks(
            trajectory.map_name, instruction_object_id, instruction_description_id, objects, processed
        ),
        "distance_to_goal_m": distance,
        "target_bearing_rad": wrap_radians(current[3] + rel),
        "relative_bearing_rad": rel,
        "target_match": int(target_match),
        "geometric_arrival": arrival,
        "visual_confirmed_arrival": int(arrival and target_match),
        "distance_bin": distance_bin(distance),
        "bearing_bin": bearing_bin(rel),
        "phase": phase(distance, success_dist),
        "negative_type": negative_type,
        "render_source": render_source,
    }


def build(args):
    data_root = args.data_root.resolve()
    output_root = args.output.resolve()
    objects, processed = load_cityrefer(data_root)
    renderer = create_renderer(args.render_source, data_root / "rgbd", args.image_size, args.airsim_host, args.airsim_port)
    trajectories = load_trajectories(data_root, args.split)
    rows = []
    accepted = 0
    for trajectory in trajectories:
        if trajectory.dist_marker_to_target > 30.0:
            continue
        if args.max_episodes is not None and accepted >= args.max_episodes:
            break
        accepted += 1
        ep_id = episode_id(trajectory)
        replacement = wrong_language(trajectory, objects)
        for anchor_number, index in enumerate(select_anchor_indices(trajectory)):
            indices = clip_indices(index)
            clip = [pose5d(trajectory.trajectory[i]) for i in indices]
            paths = render_clip(renderer, output_root, trajectory.map_name, clip, args.metadata_only)
            distance = math.hypot(clip[-1][0] - trajectory.target_position.x, clip[-1][1] - trajectory.target_position.y)
            negative_type = "near_goal" if args.success_dist < distance <= 60.0 else "none"
            rows.append(make_sample(
                trajectory, args.split, ep_id, trajectory.descriptions[0], trajectory.object_id,
                trajectory.desc_id, clip, paths, objects, processed, args.success_dist,
                negative_type, 1, renderer.source,
            ))

            offset = math.radians((60.0, -60.0, 120.0)[anchor_number % 3])
            wrong_clip = [list(pose) for pose in clip]
            for pose in wrong_clip:
                pose[3] = wrap_radians(pose[3] + offset)
            wrong_paths = render_clip(renderer, output_root, trajectory.map_name, wrong_clip, args.metadata_only)
            rows.append(make_sample(
                trajectory, args.split, ep_id, trajectory.descriptions[0], trajectory.object_id,
                trajectory.desc_id, wrong_clip, wrong_paths, objects, processed, args.success_dist,
                "wrong_view", 0, renderer.source,
            ))

            if replacement is not None:
                instruction, object_id, desc_id = replacement
                rows.append(make_sample(
                    trajectory, args.split, ep_id, instruction, object_id, desc_id, clip, paths,
                    objects, processed, args.success_dist, "wrong_language", 0, renderer.source,
                ))

    metadata_path = output_root / "metadata" / f"{args.split}.jsonl"
    write_jsonl(metadata_path, rows)
    manifest = {
        "split": args.split,
        "episode_count": accepted,
        "sample_count": len(rows),
        "render_source": renderer.source,
        "metadata_only": args.metadata_only,
        "preserves_raw_pose5d_pitch": True,
        "distance_anchors_m": list(DISTANCE_ANCHORS) + ["final"],
        "clip_offsets": [6, 3, 1, 0],
    }
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    with (output_root / "metadata" / f"{args.split}_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train_seen", "val_seen", "val_unseen", "test_unseen"))
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_episodes", type=int)
    parser.add_argument("--render_source", default="orthophoto_heightfield", choices=("orthophoto_heightfield", "airsim_front", "airsim_slanted"))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--success_dist", type=float, default=20.0)
    parser.add_argument("--airsim_host", default="127.0.0.1")
    parser.add_argument("--airsim_port", type=int, default=41451)
    parser.add_argument("--metadata_only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(0)
    build(parse_args())
