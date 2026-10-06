"""Choosing the poses worth rendering.

The point of Gate A is to judge whether a perspective render carries usable
landmark appearance, so the poses that matter are the ones where a referenced
landmark is close, inside at least one view's frustum, and surrounded by other
landmarks of the same class -- not a random sample of a trajectory.

Selection is geometry only: no rendering happens here, so scoring thousands of
candidates costs seconds.
"""

from __future__ import annotations

import numpy as np

from .pointcloud_renderer import Camera
from .project_landmarks import project_centre


def camera_for(position, yaw, pitch_deg, cfg) -> Camera:
    r = cfg["render"]
    return Camera(position=position, yaw=yaw, pitch=np.deg2rad(pitch_deg),
                  width=r["width"], height=r["height"], hfov_deg=r["hfov_deg"],
                  near=r["near"], far=r["far"])


# A landmark 1 m from the camera is not an identity question, it is a collision;
# one 300 m away is a few pixels of texture.  The useful band is where the object
# fills a meaningful part of the frame without swallowing it.
MIN_USEFUL_DISTANCE = 25.0
MAX_USEFUL_DISTANCE = 220.0
PREFERRED_DISTANCE = 80.0
DISTANCE_BANDWIDTH = 70.0


def distance_preference(distance: float) -> float:
    """Peaks at ``PREFERRED_DISTANCE`` and decays smoothly either side."""
    return float(np.exp(-((distance - PREFERRED_DISTANCE) / DISTANCE_BANDWIDTH) ** 2))


def score_pose(episode, step, landmarks, referenced_ids, cfg, views) -> dict:
    """Geometry-only score for one trajectory step.

    Scores every referenced landmark and keeps the best one, rather than the
    closest: several referenced landmarks may be in view and the useful one is
    the one framed well.
    """
    position = episode.trajectory[step, :3]
    yaw = float(episode.yaw()[step])
    ref = [lm for lm in landmarks if lm[0] in referenced_ids]
    if not ref:
        return None

    cameras = {name: camera_for(position, yaw, pitch, cfg) for name, pitch in views.items()}

    best, best_score = None, -np.inf
    for lm in ref:
        pos = np.asarray(lm[3], dtype=np.float64)
        distance = float(np.linalg.norm(pos - position))
        if not (MIN_USEFUL_DISTANCE <= distance <= MAX_USEFUL_DISTANCE):
            continue
        in_view = {}
        for name, cam in cameras.items():
            proj = project_centre(pos, cam)
            in_view[name] = bool(proj and proj["inside"])
        if not any(in_view.values()):
            continue

        entry = {
            "landmark_id": int(lm[0]),
            "landmark_name": lm[1] or lm[2],
            "object_type": lm[2],
            "distance": distance,
            "in_view": in_view,
            "any_in_view": True,
        }
        # Framed well, framed in more than one view, in a scene with several
        # same-class landmarks (that is where identity actually has to be told
        # apart), and by an identifier whose text is not empty.
        named = 1.0 if (lm[1] or "").strip() else 0.0
        same_class = sum(1 for other in landmarks if other[2] == lm[2])
        entry_score = (
            100.0 * distance_preference(distance)
            + 12.0 * sum(in_view.values())
            + 1.5 * min(same_class, 20)
            + 5.0 * named
        )
        if entry_score > best_score:
            best, best_score = entry, entry_score

    if best is None:
        return None

    best.update({
        "same_class_count": sum(1 for lm in landmarks if lm[2] == best["object_type"]),
        "altitude": float(position[2]),
        "yaw_deg": float(np.rad2deg(yaw)),
        "position": position.tolist(),
        "step": int(step),
        "map": episode.map_name,
        "score": float(best_score),
    })
    return best


def select_poses(episodes_by_split, contexts, cfg,
                 quota, views, max_per_map=2, candidates_per_episode=4):
    """Pick scored poses per split, spread over maps and yaw.

    ``contexts`` maps a map name to its landmarks -- not to a full
    :class:`MapContext` -- so scoring never opens a point cloud for an episode
    that will not be chosen.
    """
    chosen = []
    for split, want in quota.items():
        scored = []
        for episode in episodes_by_split.get(split, ()):
            if not episode.object_ids or episode.map_name not in contexts:
                continue
            landmarks = contexts[episode.map_name]
            n = len(episode.trajectory)
            steps = np.linspace(0, n - 1, num=min(candidates_per_episode, n)).astype(int)
            for step in steps:
                entry = score_pose(episode, int(step), landmarks, episode.object_ids,
                                   cfg, views)
                if entry is None:
                    continue
                entry.update({"split": split, "episode_index": int(episode.index)})
                scored.append(entry)

        scored.sort(key=lambda e: -e["score"])
        taken, per_map, yaw_bins = [], {}, set()
        for entry in scored:
            if len(taken) >= want:
                break
            if per_map.get(entry["map"], 0) >= max_per_map:
                continue
            # Keep the heading spread wide: a set of near-identical yaws would
            # make the contact sheet look consistent for the wrong reason.
            yaw_bin = int(((entry["yaw_deg"] + 360.0) % 360.0) // 45.0)
            if len(taken) >= 4 and yaw_bin in yaw_bins and len(yaw_bins) < 8:
                continue
            yaw_bins.add(yaw_bin)
            per_map[entry["map"]] = per_map.get(entry["map"], 0) + 1
            taken.append(entry)
        chosen.extend(taken)
    return chosen
