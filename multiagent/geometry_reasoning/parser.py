"""Rule-based geometry parser; it never reads target coordinates."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import re

PATTERNS = (
    ("left_of", r"\b(?:to the )?left(?: of)?\b"), ("right_of", r"\b(?:to the )?right(?: of)?\b"),
    ("front_of", r"\b(?:in )?front of\b|\bahead of\b"), ("behind", r"\bbehind\b|\bback of\b"),
    ("near", r"\bnear\b|\bnext to\b|\bbeside\b|\badjacent(?: to)?\b"),
    ("nearest", r"\bnearest\b|\bclosest\b"), ("farthest", r"\bfarthest\b|\bfurthest\b"),
    ("before", r"\bbefore\b"), ("after", r"\bafter\b"),
    ("past", r"\bpast\b|\bbeyond\b"), ("between", r"\bbetween\b"),
    ("across", r"\bacross\b"), ("opposite", r"\bopposite\b"), ("along", r"\balong\b"),
)
ORDINALS = ((1, r"\bfirst\b|\b1st\b"), (2, r"\bsecond\b|\b2nd\b"), (3, r"\bthird\b|\b3rd\b"))
GENERIC_ANCHORS = ("church", "library", "college", "school", "road", "street", "parking lot", "bridge", "tower", "hospital")
SUPPORTED = {"left_of", "right_of", "front_of", "behind", "near", "nearest", "farthest", "before", "after", "past", "between", "ordinal"}

def _norm(value): return re.sub(r"[^a-z0-9]", "", value.casefold())

@dataclass(frozen=True)
class GeometryProgram:
    instruction: str
    anchors: tuple[dict, ...]
    constraints: tuple[dict, ...]
    def to_dict(self):
        value = asdict(self); value["anchors"] = list(value["anchors"]); value["constraints"] = list(value["constraints"]); return value

def parse_geometry(instruction: str, referenced_names=()) -> GeometryProgram:
    lower = instruction.casefold(); normalized = _norm(instruction)
    anchors = []
    for name in dict.fromkeys(referenced_names):
        if _norm(name) and _norm(name) in normalized:
            anchors.append({"name": name, "role": "reference", "source": "instruction_annotation_link"})
    for term in GENERIC_ANCHORS:
        if re.search(rf"\b{re.escape(term)}\b", lower) and not any(term in item["name"].casefold() for item in anchors):
            anchors.append({"name": term, "role": "reference", "source": "lexical"})
    reference = anchors[0]["name"] if anchors else None
    constraints = []
    for kind, pattern in PATTERNS:
        if re.search(pattern, lower):
            constraints.append({"type": kind, "reference": reference, "supported": kind in SUPPORTED})
    for value, pattern in ORDINALS:
        if re.search(pattern, lower): constraints.append({"type": "ordinal", "value": value, "reference": reference, "supported": True})
    return GeometryProgram(instruction, tuple(anchors), tuple(constraints))
