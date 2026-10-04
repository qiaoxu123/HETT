"""Compile a CityNav instruction once into target and grounded reference tokens.

Only the processed landmark names are oracle inputs. Ground-truth target position
is kept outside GroundedInstructionMemory and never used for reference lookup.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Optional, Sequence, Tuple

from multiagent.landmark_multimodal_map import LandmarkMultimodalMap, LandmarkRecord, _normalize_text
from multiagent.mapdata import MAP_BOUNDS


SPLITS = ("train_seen", "val_seen", "val_unseen", "test_unseen")
RELATIONS = (
    "left", "right", "front", "behind", "between", "near", "across",
    "inside", "around", "ordinal",
)
RELATION_PATTERNS = (
    r"\bleft\b", r"\bright\b", r"\b(?:front|facing)\b",
    r"\b(?:behind|back)\b", r"\bbetween\b",
    r"\b(?:near|nearby|next to|beside|adjacent|close to)\b",
    r"\b(?:across|opposite)\b", r"\b(?:inside|within)\b",
    r"\b(?:around|surround(?:s|ed|ing)?)\b",
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b",
)
_COMPILED_RELATIONS = tuple(re.compile(pattern, re.I) for pattern in RELATION_PATTERNS)


def relation_labels(instruction: str) -> Tuple[int, ...]:
    """Return weak lexical multi-label supervision, not semantic ground truth."""
    return tuple(int(bool(pattern.search(instruction))) for pattern in _COMPILED_RELATIONS)


def polygon_area(points) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )) / 2.0


@dataclass(frozen=True)
class GroundedReference:
    requested_name: str
    record: LandmarkRecord
    match_type: str
    similarity: float


@dataclass(frozen=True)
class GroundedInstructionMemory:
    """Per-episode static inputs; intentionally contains no target coordinates."""

    episode_id: str
    map_name: str
    instruction: str
    target_query: str
    attributes: str
    relation_labels: Tuple[int, ...]
    references: Tuple[GroundedReference, ...]
    unresolved_names: Tuple[str, ...]


@dataclass(frozen=True)
class StaticExample:
    split: str
    source_row: int
    target_type: str
    target_xy: Tuple[float, float]
    memory: GroundedInstructionMemory


class OracleReferenceResolver:
    """Resolve processed reference names by name only, excluding target ID."""

    def __init__(self, landmark_map: LandmarkMultimodalMap, minimum_similarity=0.55):
        self.landmark_map = landmark_map
        self.minimum_similarity = minimum_similarity

    def resolve(self, map_name: str, name: str, target_id: int) -> Optional[GroundedReference]:
        normalized_name = _normalize_text(name)
        candidates = [
            record for record in self.landmark_map.records_for_map(map_name)
            if record.landmark_id != target_id and record.name.strip()
        ]
        if not normalized_name or not candidates:
            return None
        exact = [record for record in candidates if _normalize_text(record.name) == normalized_name]
        if exact:
            record = min(exact, key=lambda item: (-polygon_area(item.polygon), item.landmark_id))
            return GroundedReference(name, record, "exact", 1.0)
        scored = [
            (SequenceMatcher(None, normalized_name, _normalize_text(record.name)).ratio(), record)
            for record in candidates
        ]
        similarity, record = max(
            scored, key=lambda item: (item[0], polygon_area(item[1].polygon), -item[1].landmark_id)
        )
        if similarity < self.minimum_similarity:
            return None
        return GroundedReference(name, record, "fuzzy", similarity)

    def compile(self, episode_id: str, map_name: str, target_id: int,
                instruction: str, processed: dict) -> GroundedInstructionMemory:
        references = []
        unresolved = []
        for name in processed.get("landmarks", []):
            result = self.resolve(map_name, str(name), target_id)
            if result is None:
                unresolved.append(str(name))
            else:
                references.append(result)
        return GroundedInstructionMemory(
            episode_id=episode_id,
            map_name=map_name,
            instruction=instruction,
            target_query=str(processed.get("target", "")),
            attributes="; ".join(str(item) for item in processed.get("surroundings", [])),
            relation_labels=relation_labels(instruction),
            references=tuple(references),
            unresolved_names=tuple(unresolved),
        )


def load_static_examples(data_root: Path, resolver: OracleReferenceResolver) -> list:
    """Load all 32,326 source rows, without rollout/teacher-length filtering."""
    data_root = Path(data_root)
    objects = json.loads((data_root / "cityrefer" / "objects.json").read_text())
    processed = json.loads((data_root / "cityrefer" / "processed_descriptions.json").read_text())
    examples = []
    for split in SPLITS:
        path = data_root / "processed_citynav" / f"citynav_{split}.json"
        trajectories = json.loads(path.read_text())
        for source_row, trajectory in enumerate(trajectories):
            map_name = f"{trajectory['area']}_block_{trajectory['block']}"
            target_id = int(trajectory["object_ids"][0])
            ann_id = int(trajectory["ann_ids"][0])
            obj = objects[map_name][str(target_id)]
            annotation = processed[map_name][str(target_id)][ann_id]
            memory = resolver.compile(
                f"{map_name}:{target_id}:{ann_id}", map_name, target_id,
                obj["descriptions"][ann_id], annotation,
            )
            examples.append(StaticExample(
                split=split,
                source_row=source_row,
                target_type=str(obj["object_type"]),
                target_xy=tuple(float(value) for value in obj["position"][:2]),
                memory=memory,
            ))
    return examples


def map_center(map_name: str) -> Tuple[float, float]:
    bounds = MAP_BOUNDS[map_name]
    return ((bounds.x_min + bounds.x_max) / 2, (bounds.y_min + bounds.y_max) / 2)


def reference_mean_xy(memory: GroundedInstructionMemory) -> Tuple[float, float]:
    if not memory.references:
        return map_center(memory.map_name)
    return tuple(
        sum(reference.record.center_xy[axis] for reference in memory.references)
        / len(memory.references)
        for axis in (0, 1)
    )
