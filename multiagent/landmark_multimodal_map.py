"""Landmark-centric multimodal semantic map utilities.

The cache binds a CityRefer landmark identity to its name, geometry, normalized
map location, optional top-view RGB crop, and optional ego-view RGB examples.
Runtime navigation only consumes the lightweight metadata/prior; RGB assets are
kept for diagnostics and future Stage-2 visual matching.
"""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_STOPWORDS = {
    "a", "an", "the", "to", "of", "on", "at", "go", "fly", "head",
    "toward", "towards", "near", "next", "beside", "by",
}


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _tokens(text: str) -> set:
    return {token for token in _normalize_text(text).split() if token not in _STOPWORDS}


@dataclass
class LandmarkRecord:
    map_name: str
    landmark_id: int
    name: str
    center_xy: Tuple[float, float]
    normalized_xy: Tuple[float, float]
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    object_type: str = ""
    aliases: List[str] = field(default_factory=list)
    top_rgb_path: Optional[str] = None
    top_feature_path: Optional[str] = None
    ego_views: List[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict):
        data = dict(data)
        data["center_xy"] = tuple(data["center_xy"])
        data["normalized_xy"] = tuple(data["normalized_xy"])
        data["polygon"] = [tuple(point) for point in data.get("polygon", [])]
        return cls(**data)


class LandmarkMultimodalMap:
    VERSION = 1

    def __init__(self, records: Sequence[LandmarkRecord]):
        self.records = list(records)
        self._by_map: Dict[str, List[LandmarkRecord]] = {}
        for record in self.records:
            self._by_map.setdefault(record.map_name, []).append(record)

    @classmethod
    def from_cityrefer_objects(cls, objects, map_meters: float = 410.0):
        from multiagent.mapdata import MAP_BOUNDS

        records = []
        for map_name, map_objects in objects.items():
            bounds = MAP_BOUNDS[map_name]
            for landmark_id, obj in map_objects.items():
                if not obj.name:
                    continue
                nx = (float(obj.position.x) - bounds.x_min) / map_meters
                ny = (bounds.y_max - float(obj.position.y)) / map_meters
                aliases = []
                for processed in getattr(obj, "processed_descriptions", []):
                    target_phrase = str(getattr(processed, "target", "")).strip()
                    if (
                        target_phrase
                        and _normalize_text(target_phrase) != _normalize_text(str(obj.name))
                        and target_phrase not in aliases
                    ):
                        aliases.append(target_phrase)

                records.append(
                    LandmarkRecord(
                        map_name=map_name,
                        landmark_id=int(landmark_id),
                        name=str(obj.name),
                        center_xy=(float(obj.position.x), float(obj.position.y)),
                        normalized_xy=(
                            float(np.clip(nx, 0.0, 1.0 - 1e-6)),
                            float(np.clip(ny, 0.0, 1.0 - 1e-6)),
                        ),
                        polygon=[(float(p.x), float(p.y)) for p in obj.contour],
                        object_type=str(obj.object_type),
                        aliases=aliases,
                    )
                )
        return cls(records)

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = json.loads(path.read_text())
        version = int(payload.get("version", 0))
        if version != cls.VERSION:
            raise ValueError(
                f"Unsupported landmark cache version {version}; expected {cls.VERSION}"
            )
        return cls([LandmarkRecord.from_dict(item) for item in payload["records"]])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "records": [asdict(record) for record in self.records],
        }
        path.write_text(json.dumps(payload, indent=2))

    def records_for_map(self, map_name: str) -> List[LandmarkRecord]:
        return self._by_map.get(map_name, [])

    @staticmethod
    def _text_score(instruction: str, record: LandmarkRecord) -> float:
        normalized_instruction = _normalize_text(instruction)
        instruction_tokens = _tokens(instruction)
        if not instruction_tokens:
            return 0.0

        best = 0.0
        for index, candidate in enumerate([record.name] + list(record.aliases)):
            normalized_candidate = _normalize_text(candidate)
            candidate_tokens = _tokens(candidate)
            if not candidate_tokens:
                continue

            overlap = len(instruction_tokens & candidate_tokens)
            coverage = overlap / max(len(candidate_tokens), 1)
            precision = overlap / max(len(instruction_tokens), 1)
            score = coverage + 0.25 * precision
            if normalized_candidate and normalized_candidate in normalized_instruction:
                score += 2.0 if index == 0 else 1.5
            best = max(best, score)
        return best

    def retrieve_text(
        self,
        instruction: str,
        map_name: str,
        topk: int = 5,
    ) -> List[Tuple[LandmarkRecord, float]]:
        scored = []
        for record in self.records_for_map(map_name):
            score = self._text_score(instruction, record)
            if score > 0:
                scored.append((record, float(score)))
        scored.sort(key=lambda item: (-item[1], item[0].landmark_id))
        return scored[: max(int(topk), 0)]

    @staticmethod
    def scatter_prior(
        retrieved: Sequence[Tuple[LandmarkRecord, float]],
        grid_size: int,
        sigma: float = 0.8,
    ) -> np.ndarray:
        prior = np.zeros((grid_size, grid_size), dtype=np.float32)
        if not retrieved:
            return prior.reshape(-1)

        rows, cols = np.meshgrid(
            np.arange(grid_size, dtype=np.float32),
            np.arange(grid_size, dtype=np.float32),
            indexing="ij",
        )
        sigma = max(float(sigma), 1e-3)
        for record, score in retrieved:
            row = float(record.normalized_xy[0] * grid_size - 0.5)
            col = float(record.normalized_xy[1] * grid_size - 0.5)
            gaussian = np.exp(
                -((rows - row) ** 2 + (cols - col) ** 2) / (2.0 * sigma ** 2)
            )
            prior += float(score) * gaussian.astype(np.float32)

        max_value = float(prior.max())
        if max_value > 0:
            prior /= max_value
        return prior.reshape(-1)

    def build_batch_prior(
        self,
        instructions: Sequence[str],
        map_names: Sequence[str],
        grid_size: int,
        topk: int = 5,
        sigma: float = 0.8,
    ):
        if len(instructions) != len(map_names):
            raise ValueError("instructions and map_names must have the same length")

        priors = []
        retrieved_batch = []
        for instruction, map_name in zip(instructions, map_names):
            retrieved = self.retrieve_text(instruction, map_name, topk=topk)
            priors.append(self.scatter_prior(retrieved, grid_size, sigma=sigma))
            retrieved_batch.append(retrieved)
        return np.stack(priors, axis=0), retrieved_batch

    def find(self, map_name: str, landmark_id: int) -> Optional[LandmarkRecord]:
        for record in self.records_for_map(map_name):
            if record.landmark_id == int(landmark_id):
                return record
        return None
