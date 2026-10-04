"""Deterministic instruction-to-scene-template parser.

The parser is deliberately rule based: this benchmark evaluates whether the
representation is useful before introducing an LLM parser as another source of
error.  A template keeps target, anchor/context, geometry, and ordinal evidence
separate so they can be ablated independently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

from multiagent.visual_attributes.parser import parse_attributes


TARGET_SEMANTICS = {
    "building", "house", "church", "library", "college", "school",
    "office", "tower", "hospital", "car", "parking", "ground",
}
SCENE_OBJECTS = {
    "road", "intersection", "crossing", "parking_lot", "field", "grass",
    "sports_field", "water", "river", "bridge", "railway", "open_area",
    "vegetation",
}
RELATIONS = {
    "left", "right", "front", "behind", "near", "between", "before",
    "after", "past", "along", "across", "opposite",
}
ORDINALS = {"first", "second", "third", "nearest", "farthest"}


@dataclass(frozen=True)
class SceneTemplate:
    instruction: str
    anchors: tuple[dict, ...] = field(default_factory=tuple)
    target: dict = field(default_factory=dict)
    context: tuple[dict, ...] = field(default_factory=tuple)
    geometry: tuple[dict, ...] = field(default_factory=tuple)
    ordinal: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ("anchors", "context", "geometry", "ordinal"):
            value[key] = list(value[key])
        return value

    @property
    def visual_requirements(self) -> dict[str, tuple[str, ...]]:
        attributes = self.target.get("visual_attributes", {})
        result = {name: tuple(values) for name, values in attributes.items() if values}
        target_type = self.target.get("type")
        if target_type:
            result["semantic"] = (target_type,)
        contexts = tuple(item["type"] for item in self.context if item.get("type"))
        if contexts:
            result["context"] = contexts
        return result


def _deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def parse_scene_template(
    instruction: str,
    *,
    target_phrase: str = "",
    landmark_names: Iterable[str] = (),
    surroundings: Iterable[str] = (),
    target_object_type: str | None = None,
) -> SceneTemplate:
    parsed = parse_attributes(instruction)
    target_parsed = parse_attributes(target_phrase or instruction)
    visual_attributes = {
        category: _deduplicate(target_parsed.get(category, parsed.get(category, ())))
        for category in ("color", "size", "shape", "roof")
    }
    visual_attributes = {key: list(value) for key, value in visual_attributes.items() if value}

    semantic_mentions = _deduplicate(target_parsed.get("semantic", ()))
    target_type = semantic_mentions[0] if semantic_mentions else None
    if not target_type and target_object_type:
        target_type = target_object_type.casefold().replace(" ", "_")

    names = _deduplicate(landmark_names)
    anchor_types = list(parsed.get("semantic", ()))
    anchors = []
    for index, name in enumerate(names):
        kind = next((value for value in anchor_types if value in name.casefold()), None)
        anchors.append({"name": name, "type": kind or "named_landmark", "role": "reference"})

    context_values = list(parsed.get("context", ()))
    for phrase in surroundings:
        phrase_parsed = parse_attributes(phrase)
        context_values.extend(phrase_parsed.get("context", ()))
        for value in phrase_parsed.get("semantic", ()):
            if value not in TARGET_SEMANTICS or value != target_type:
                context_values.append(value)
    context = tuple({"type": value, "relation": "near", "subject": "target"}
                    for value in _deduplicate(context_values) if value in SCENE_OBJECTS or value in TARGET_SEMANTICS)

    relation_values = _deduplicate(parsed.get("spatial_relation", ()))
    reference = names[0] if names else "reference"
    geometry = tuple({"type": relation, "reference": reference, "subject": "target"}
                     for relation in relation_values if relation in RELATIONS)
    ordinal = tuple(value for value in parsed.get("count_ordinal", ()) if value in ORDINALS)
    return SceneTemplate(
        instruction=instruction,
        anchors=tuple(anchors),
        target={"type": target_type, "visual_attributes": visual_attributes},
        context=context,
        geometry=geometry,
        ordinal=ordinal,
    )
