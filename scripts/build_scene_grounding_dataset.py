#!/usr/bin/env python3
"""Build RGB observations at published teacher-trajectory poses."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiagent.cityreferobject import get_city_refer_objects  # noqa: E402
from multiagent.mapdata import GROUND_LEVEL  # noqa: E402
from multiagent.observation import cropclient  # noqa: E402
from multiagent.scene_grounding.dataset import (  # noqa: E402
    ALLOWED_SPLITS, leakage_audit, sample_id, select_distance_steps, trajectory_pose, visible_regions,
)
from multiagent.scene_grounding.scene_template import parse_scene_template  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--regions", type=int, default=5)
    parser.add_argument("--max-episodes-train", type=int, default=600)
    parser.add_argument("--max-episodes-val", type=int, default=300)
    args = parser.parse_args()
    if "test_unseen" in " ".join(map(str, vars(args).values())):
        raise ValueError("test_unseen is forbidden")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "images").mkdir(exist_ok=True)
    (args.output / "regions").mkdir(exist_ok=True)
    objects = get_city_refer_objects(args.data_root / "cityrefer/objects.json",
                                     args.data_root / "cityrefer/processed_descriptions.json")
    cropclient.load_image_cache(args.data_root / "rgbd")
    rows, templates, stats = [], {}, {"splits": {}, "distance_buckets": Counter()}
    for split in ALLOWED_SPLITS:
        episodes = json.loads((args.data_root / "processed_citynav" / f"citynav_{split}.json").read_text())
        maximum = args.max_episodes_train if split == "train_seen" else args.max_episodes_val
        episodes = episodes[:maximum] if maximum else episodes
        split_rows = 0
        for episode_index, episode in enumerate(episodes):
            map_name = f"{episode['area']}_block_{episode['block']}"
            object_id, annotation_id = int(episode["object_ids"][0]), int(episode["ann_ids"][0])
            target = objects[map_name][object_id]
            if annotation_id >= len(target.processed_descriptions):
                continue
            processed = target.processed_descriptions[annotation_id]
            instruction = target.descriptions[annotation_id]
            template = parse_scene_template(instruction, target_phrase=processed.target,
                                            landmark_names=processed.landmarks,
                                            surroundings=processed.surroundings,
                                            target_object_type=target.object_type)
            episode_id = f"{split}:{episode_index}"
            templates[episode_id] = template.to_dict()
            goal = episode["target_positions"][-1]
            for step, distance, bucket in select_distance_steps(episode["trajectory"], goal[:2]):
                pose = trajectory_pose(episode["trajectory"][step])
                if pose.z <= GROUND_LEVEL[map_name] + 1:
                    continue
                rgb = cropclient.crop_image(map_name, pose, (args.image_size, args.image_size), "rgb")
                identifier = sample_id(split, episode_index, step)
                image_path = Path("images") / f"{identifier}.jpg"
                cv2.imwrite(str(args.output / image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                proposals = visible_regions(map_name, pose, objects[map_name].values(), args.image_size, args.regions)
                for proposal_index, proposal in enumerate(proposals):
                    x0, y0, x1, y1 = proposal["bbox_xyxy"]
                    crop = cv2.resize(rgb[y0:y1, x0:x1], (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
                    relative = Path("regions") / f"{identifier}_{proposal_index}.jpg"
                    cv2.imwrite(str(args.output / relative), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    proposal["image"] = str(relative)
                row = {
                    "sample_id": identifier, "episode_id": episode_id, "split": split,
                    "map_name": map_name, "object_key": f"{map_name}:{object_id}",
                    "step": step, "trajectory_length": len(episode["trajectory"]),
                    "instruction": instruction, "template_id": episode_id,
                    "pose": {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw},
                    "altitude_m": float(pose.z - GROUND_LEVEL[map_name]),
                    "goal_distance_m": distance, "distance_bucket": bucket,
                    "image": str(image_path), "regions": proposals,
                    "observation_source": "teacher_trajectory_pose", "camera": "citynav_topdown_crop",
                }
                rows.append(row); split_rows += 1; stats["distance_buckets"][f"{split}:{bucket}"] += 1
        stats["splits"][split] = {"episodes": len(episodes), "frames": split_rows}
    with (args.output / "manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output / "templates.json", templates)
    stats["distance_buckets"] = dict(stats["distance_buckets"])
    stats["leakage"] = leakage_audit(rows)
    stats["frames"] = len(rows)
    write_json(args.output / "dataset_stats.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
