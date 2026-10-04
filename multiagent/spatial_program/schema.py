"""Strict serialisable schema for executable spatial programs."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    name: str
    role: str
    attributes: tuple[str, ...] = ()
    clause_id: str = ""
    source: str = "lexical"

@dataclass(frozen=True)
class Axis:
    id: str
    definition: str
    anchor: Optional[str] = None
    confidence: float = 0.0
    source: str = "resolver"

@dataclass(frozen=True)
class Clause:
    id: str
    text: str
    relation: str
    subject: str = "target"
    object: Optional[str] = None
    axis: Optional[str] = None
    value: Optional[int] = None
    scope: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    supported: bool = True

@dataclass(frozen=True)
class SpatialProgram:
    instruction: str
    segments: tuple[str, ...]
    entities: tuple[Entity, ...]
    clauses: tuple[Clause, ...]
    axes: tuple[Axis, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SpatialProgram":
        return cls(value["instruction"], tuple(value.get("segments", ())),
                   tuple(Entity(**{**x, "attributes": tuple(x.get("attributes", ()))}) for x in value.get("entities", ())),
                   tuple(Clause(**x) for x in value.get("clauses", ())),
                   tuple(Axis(**x) for x in value.get("axes", ())), value.get("metadata", {}))
