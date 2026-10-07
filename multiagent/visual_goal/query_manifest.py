from __future__ import annotations

import json
from pathlib import Path


REQUIRED = {"episode_id", "split", "map_name", "image", "distance_to_goal_m", "renderer_source",
            "renderer_alignment_verified", "image_origin", "trajectory_pose5d"}


def load_verified_queries(path: Path, root: Path) -> list[dict]:
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows: raise ValueError("query manifest is empty")
    for i,row in enumerate(rows):
        missing=REQUIRED-row.keys()
        if missing: raise ValueError(f"query row {i} missing fields: {sorted(missing)}")
        if row["split"] == "test_unseen": raise ValueError("test_unseen is forbidden")
        if row["renderer_source"] not in {"airsim_calibrated", "original_cityflight"} or not row["renderer_alignment_verified"]:
            raise ValueError(f"query row {i} is not a verified real UAV RGB observation")
        if row["image_origin"] != "uav_trajectory_pose" or len(row["trajectory_pose5d"]) != 5:
            raise ValueError(f"query row {i} is not an RGB frame from a recorded UAV trajectory pose")
        if row.get("candidate_centered_crop", False):
            raise ValueError(f"query row {i} uses a forbidden candidate-centered crop")
        if not (Path(root)/row["image"]).is_file(): raise FileNotFoundError(Path(root)/row["image"])
    # An episode/object cannot cross evaluation splits.
    assignments={}
    for row in rows:
        key=(row.get("episode_id"),row.get("target_object_id"))
        prev=assignments.setdefault(key,row["split"])
        if prev!=row["split"]: raise ValueError(f"episode/object split leakage: {key}")
    return rows
