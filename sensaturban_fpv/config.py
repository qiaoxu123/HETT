"""Configuration loading and per-map context assembly."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import citynav
from .plyio import PlyCloud, XyGridIndex

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sensaturban_fpv.json"


def load_config(path: Path | str | None = None) -> dict:
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    cfg["_path"] = str(Path(path or DEFAULT_CONFIG).resolve())
    return cfg


def ensure_repo_importable() -> None:
    """Put the repo root on sys.path so ``gsamllavanav`` imports as a package."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def artifact_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["artifact_dir"])
    if not d.is_absolute():
        d = REPO_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class MapContext:
    map_name: str
    cloud: PlyCloud
    grid: XyGridIndex
    episodes: list
    landmarks: list                       # (id, name, type, position, dimension)
    objects_by_map: dict = field(default_factory=dict)

    @property
    def bounds(self):
        """Exact cloud bounds, served from the index (no extra cloud rescan)."""
        return self.grid.exact_bounds()

    @property
    def cloud_lo(self) -> np.ndarray:
        return self.bounds[0]

    @property
    def cloud_hi(self) -> np.ndarray:
        return self.bounds[1]

    def referenced_ids_for(self, episode) -> set:
        return set(episode.object_ids)


def load_landmarks(cfg: dict):
    """CityRefer objects, via the repo's own loader so geometry has one definition."""
    ensure_repo_importable()
    from gsamllavanav.cityreferobject import get_city_refer_objects

    cityrefer = Path(cfg["paths"]["cityrefer_dir"])
    return get_city_refer_objects(
        objects_path=cityrefer / "objects.json",
        processed_description_path=cityrefer / "processed_descriptions.json",
    )


def build_map_context(cfg: dict, map_name: str, splits=("train_seen", "val_seen", "val_unseen"),
                      objects_by_map=None, cell: float | None = None) -> MapContext:
    """Assemble everything needed to reason about one block."""
    ply_root = Path(cfg["paths"]["ply_root"])
    ply = citynav.ply_path(ply_root, map_name)

    cloud = PlyCloud(ply)
    cache_dir = artifact_dir(cfg) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    grid = XyGridIndex(
        cloud, cell=cell or cfg["diagnostic"]["grid_cell_m"], cache_dir=cache_dir
    ).build()

    episodes = []
    for split in splits:
        for ep in citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split):
            if ep.map_name == map_name:
                episodes.append(ep)

    if objects_by_map is None:
        objects_by_map = load_landmarks(cfg)
    objects = objects_by_map.get(map_name, {})
    landmarks = [
        (o.id, o.name, o.object_type,
         tuple(float(v) for v in o.position), tuple(float(v) for v in o.dimension))
        for o in objects.values()
    ]
    return MapContext(map_name, cloud, grid, episodes, landmarks, objects_by_map)


def all_citynav_maps(cfg: dict, splits=("train_seen", "val_seen", "val_unseen", "test_unseen")) -> list:
    seen = []
    for split in splits:
        for ep in citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split):
            if ep.map_name not in seen:
                seen.append(ep.map_name)
    return sorted(seen)


def referenced_ids_by_episode(cfg: dict, map_name: str, splits) -> dict:
    out = {}
    for split in splits:
        for ep in citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split):
            if ep.map_name == map_name and ep.object_ids:
                out[(split, ep.index)] = ep.object_ids
    return out
