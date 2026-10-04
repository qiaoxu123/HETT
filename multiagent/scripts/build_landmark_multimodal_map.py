"""Build the landmark-centric multimodal cache.

Examples:
  python -m multiagent.scripts.build_landmark_multimodal_map \
      --output ../data/cityrefer/landmark_multimodal_map.json

Add RGB assets:
  python -m multiagent.scripts.build_landmark_multimodal_map \
      --output ../data/cityrefer/landmark_multimodal_map.json \
      --materialize-top-rgb --materialize-ego-rgb --ego-frames 8
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from multiagent.cityreferobject import get_city_refer_objects
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.landmark_multimodal_map import LandmarkMultimodalMap
from multiagent.mapdata import GROUND_LEVEL
from multiagent.observation import cropclient
from multiagent.space import Pose4D


def _relative_asset_path(path: Path, cache_path: Path) -> str:
    try:
        return str(path.relative_to(cache_path.parent))
    except ValueError:
        return str(path)


def materialize_top_views(semantic_map, output_dir: Path, cache_path: Path, altitude: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    cropclient.load_image_cache()
    for record in semantic_map.records:
        pose = Pose4D(
            record.center_xy[0],
            record.center_xy[1],
            GROUND_LEVEL[record.map_name] + altitude,
            0.0,
        )
        rgb = cropclient.crop_image(record.map_name, pose, (224, 224), "rgb")
        path = output_dir / record.map_name / f"{record.landmark_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        record.top_rgb_path = _relative_asset_path(path, cache_path)


def materialize_ego_views(
    semantic_map,
    output_dir: Path,
    cache_path: Path,
    splits,
    ego_frames: int,
    altitude: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cropclient.load_image_cache()
    index = {
        (record.map_name, record.landmark_id): record
        for record in semantic_map.records
    }
    for split in splits:
        trajectories = load_mturk_trajectories(split, "all", altitude)
        for trajectory_index, trajectory in enumerate(trajectories):
            record = index.get((trajectory.map_name, int(trajectory.object_id)))
            if record is None:
                continue
            poses = trajectory.trajectory[-max(int(ego_frames), 1):]
            for local_index, pose5d in enumerate(poses):
                pose = pose5d.xyzyaw
                rgb = cropclient.crop_image(
                    trajectory.map_name, pose, (224, 224), "rgb"
                )
                path = (
                    output_dir
                    / trajectory.map_name
                    / str(record.landmark_id)
                    / f"{split}_{trajectory_index:06d}_{local_index:02d}.jpg"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                record.ego_views.append(
                    {
                        "split": split,
                        "trajectory_index": int(trajectory_index),
                        "rgb_path": _relative_asset_path(path, cache_path),
                        "camera_xy": [float(pose.x), float(pose.y)],
                        "camera_yaw": float(pose.yaw),
                        "distance_m": float(
                            np.linalg.norm(
                                np.asarray(record.center_xy, dtype=np.float32)
                                - np.asarray([pose.x, pose.y], dtype=np.float32)
                            )
                        ),
                    }
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="../data/cityrefer/landmark_multimodal_map.json",
    )
    parser.add_argument("--map-meters", type=float, default=410.0)
    parser.add_argument("--materialize-top-rgb", action="store_true")
    parser.add_argument("--materialize-ego-rgb", action="store_true")
    parser.add_argument("--top-altitude", type=float, default=80.0)
    parser.add_argument("--ego-altitude", type=float, default=50.0)
    parser.add_argument("--ego-frames", type=int, default=8)
    parser.add_argument(
        "--ego-splits",
        nargs="+",
        default=["train_seen"],
        choices=["train_seen", "val_seen", "val_unseen", "test_unseen"],
    )
    args = parser.parse_args()

    output = Path(args.output)
    semantic_map = LandmarkMultimodalMap.from_cityrefer_objects(
        get_city_refer_objects(),
        map_meters=args.map_meters,
    )

    asset_root = output.parent / "landmark_multimodal_assets"
    if args.materialize_top_rgb:
        materialize_top_views(
            semantic_map,
            asset_root / "top_rgb",
            output,
            args.top_altitude,
        )
    if args.materialize_ego_rgb:
        materialize_ego_views(
            semantic_map,
            asset_root / "ego_rgb",
            output,
            args.ego_splits,
            args.ego_frames,
            args.ego_altitude,
        )

    semantic_map.save(output)
    print(
        f"Saved {len(semantic_map.records)} landmarks to {output}; "
        f"top_rgb={args.materialize_top_rgb} ego_rgb={args.materialize_ego_rgb}"
    )


if __name__ == "__main__":
    main()
