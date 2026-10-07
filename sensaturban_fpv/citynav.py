"""Loading CityNav episodes and CityRefer landmarks, without the training stack.

Only reading is done here.  Objects are handed to the repo's own
``gsamllavanav.cityreferobject`` so the landmark geometry (position, dimension,
contour, bbox corners) has exactly one definition in the tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

SPLIT_FILES = {
    "train_seen": "citynav_train_seen.json",
    "val_seen": "citynav_val_seen.json",
    "val_unseen": "citynav_val_unseen.json",
    "test_unseen": "citynav_test_unseen.json",
}


@dataclass
class Episode:
    index: int
    split: str
    area: str
    block: int
    map_name: str
    description: str
    trajectory: np.ndarray        # (T, 6) -> x, y, z, dx, dy, dz
    marker_positions: np.ndarray  # (M, 3)
    target_positions: np.ndarray  # (K, 3)
    object_ids: list              # the *target* ids -- see below
    ann_ids: list = field(default_factory=list)

    # ``object_ids`` is the target, not a reference set.  Every record in the
    # three splits has exactly one, and it always names an object at one of
    # ``target_positions``; upstream agrees (``MTurkTrajectory.object_id``
    # returns ``object_ids[0]`` and ``generate.py`` builds the Episode from
    # ``objects[map][object_id]``).  ``ann_ids`` indexes that object's
    # ``descriptions``/``processed_descriptions``, which is where the
    # instruction's anchor *names* live.

    @property
    def poses(self) -> np.ndarray:
        return self.trajectory[:, :4]

    @property
    def forward(self) -> np.ndarray:
        return self.trajectory[:, 3:6]

    def yaw(self) -> np.ndarray:
        """Heading from the recorded look direction, in CityNav's convention."""
        return np.arctan2(self.trajectory[:, 4], self.trajectory[:, 3])

    def pitch(self) -> np.ndarray:
        d = self.trajectory[:, 3:6]
        horiz = np.linalg.norm(d[:, :2], axis=1)
        return np.arctan2(d[:, 2], horiz)


def _as_array(value, width=None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, width) if width else arr.reshape(0)
    return arr.reshape(-1, width) if width else arr


def normalise_object_ids(raw) -> list:
    """CityNav stores referenced ids either as ints or as ``{id: ...}`` dicts."""
    out = []
    for item in raw:
        if isinstance(item, dict):
            for key in ("object_id", "id", "objectId"):
                if key in item:
                    out.append(int(item[key]))
                    break
            else:
                out.append(int(next(iter(item))))
        else:
            out.append(int(item))
    return out


@lru_cache(maxsize=8)
def _load_split_cached(path_str: str) -> list:
    return json.loads(Path(path_str).read_text())


def load_split(data_dir: Path, split: str) -> list:
    """Load one split as Episodes, memoised per file.

    Both the ~180 MB JSON parse *and* the Episode construction are cached: the
    diagnostics and the renderer walk a map list, and rebuilding ~27k Episodes
    (each converting four nested lists through ``np.asarray``) on every map was
    costing more than the point-cloud work.  The returned Episodes are
    read-only and callers do not mutate them.
    """
    return _load_split_episodes(str(Path(data_dir) / SPLIT_FILES[split]))


@lru_cache(maxsize=8)
def _load_split_episodes(path_str: str) -> tuple:
    path = Path(path_str)
    split = next((k for k, v in SPLIT_FILES.items() if path.name == v), path.stem)
    records = _load_split_cached(path_str)
    episodes = []
    for i, r in enumerate(records):
        episodes.append(
            Episode(
                index=i,
                split=split,
                area=r["area"],
                block=int(r["block"]),
                map_name=f"{r['area']}_block_{int(r['block'])}",
                description=r["descriptions"][0] if r.get("descriptions") else "",
                trajectory=_as_array(r["trajectory"], 6),
                marker_positions=_as_array(r.get("marker_positions"), 3),
                target_positions=_as_array(r.get("target_positions"), 3),
                object_ids=normalise_object_ids(r.get("object_ids", [])),
                ann_ids=[int(a) for a in r.get("ann_ids", [])],
            )
        )
    return tuple(episodes)


def ply_path(ply_root: Path, map_name: str) -> Path:
    """Locate ``<area>_block_<n>.ply`` under a train/test split directory."""
    matches = sorted(Path(ply_root).glob(f"*/{map_name}.ply"))
    if not matches:
        raise FileNotFoundError(f"no PLY for {map_name} under {ply_root}")
    return matches[0]


def check_forward_consistency(episode: Episode) -> dict:
    """Do the recorded look directions agree with the recorded motion?

    The 6-vector's last three components are a unit direction.  If the pose
    frame and the look-direction frame were different, consecutive motion would
    have no consistent relation to them; here we measure how well the direction
    of travel matches the recorded heading.  This is a check on CityNav's own
    internal consistency, independent of the point cloud.
    """
    d = episode.forward
    norms = np.linalg.norm(d, axis=1)
    step = np.diff(episode.trajectory[:, :3], axis=0)
    step_norm = np.linalg.norm(step, axis=1)
    moving = step_norm > 0.5
    cos = np.full(len(step), np.nan)
    if moving.any():
        cos[moving] = np.einsum(
            "ij,ij->i", step[moving] / step_norm[moving, None], d[:-1][moving]
        )
    return {
        "direction_norm_min": float(norms.min()),
        "direction_norm_max": float(norms.max()),
        "moving_steps": int(moving.sum()),
        "total_steps": int(len(step)),
        "cos_travel_vs_look_median": float(np.nanmedian(cos)) if moving.any() else None,
        "cos_travel_vs_look_mean": float(np.nanmean(cos)) if moving.any() else None,
    }
