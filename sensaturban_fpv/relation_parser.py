"""What a CityNav instruction actually says, read as structured relations.

The previous round established that a masked single view reaches Top-4 ≈ 0.76
but Top-1 ≈ 0.32: the model is choosing the right *kind* of thing and not the
right *instance*.  Instructions distinguish instances in ways that are mostly
not appearance -- "the building beside the church", "the car parked behind the
van" -- and a SigLIP embedding of the whole sentence has no mechanism that turns
"beside" into a distance between two entities.

This module is deliberately a lexicon, not a model.  It has one job: report which
relational and attribute words a sentence contains, so that a probe can be given
the same information a person would extract by reading it.  It makes no claim to
understand language; it is a control for "did anyone ever give the model the
relation at all".

Every function here is pure text in, plain data out, so it can be tested without
a scene.
"""

from __future__ import annotations

import re

# Relation families.  Each is a set of surface forms; the parser reports presence
# per family rather than per word, because the probe wants "is a left/right
# relation being asked for", not which synonym was used.
RELATION_LEXICON = {
    "proximity": ("near", "nearby", "close to", "next to", "beside", "alongside",
                  "adjacent", "by the", "at the", "around", "surrounding"),
    "distance_far": ("far from", "away from", "distant", "in the distance",
                     "beyond"),
    "front_back": ("in front of", "behind", "at the back", "back of", "front of",
                   "ahead of"),
    "left_right": ("left of", "right of", "on the left", "on the right",
                   "to the left", "to the right", "left side", "right side"),
    "cardinal": ("north", "south", "east", "west", "northeast", "northwest",
                 "southeast", "southwest"),
    "between": ("between", "in the middle of", "among"),
    "across": ("across from", "opposite", "facing"),
    "containment": ("inside", "within", "on top of", "under", "below", "above",
                    "overlooking"),
    "road_context": ("along the road", "by the road", "beside the road",
                     "side of the road", "roadside", "on the street",
                     "near the road", "along the street", "at the intersection",
                     "junction", "crossroads", "roundabout"),
}

# Attribute families, used for the instruction-type breakdown.
ATTRIBUTE_LEXICON = {
    "color": ("red", "white", "black", "blue", "green", "yellow", "brown",
              "grey", "gray", "orange", "purple", "pink", "silver", "golden",
              "dark", "light", "bright", "coloured", "colored"),
    "size": ("large", "big", "small", "tall", "short", "long", "wide", "narrow",
             "high", "low", "huge", "little", "massive"),
    "shape": ("round", "square", "rectangular", "triangular", "curved",
              "straight", "flat", "sloped", "pyramid", "cylindrical", "L-shaped",
              "shaped", "circular", "oval"),
    "appearance": ("modern", "old", "new", "brick", "glass", "stone", "metal",
                   "concrete", "wooden", "white-painted", "shiny", "rusted",
                   "textured", "painted", "clean", "dirty"),
    "material": ("brick", "glass", "stone", "metal", "concrete", "wood",
                 "tiled", "steel"),
}

NAMED_HINT = re.compile(r"\b(named|called|known as)\b", re.IGNORECASE)


def _hits(text: str, terms) -> bool:
    low = (text or "").lower()
    return any(term in low for term in terms)


def parse_relations(text: str) -> dict:
    """Presence of each relation family in a sentence, plus a summary count."""
    out = {name: _hits(text, terms) for name, terms in RELATION_LEXICON.items()}
    out["any"] = any(out.values())
    out["n_families"] = int(sum(1 for name in RELATION_LEXICON if out[name]))
    return out


def parse_attributes(text: str) -> dict:
    out = {name: _hits(text, terms) for name, terms in ATTRIBUTE_LEXICON.items()}
    out["named"] = bool(NAMED_HINT.search(text or ""))
    return out


def instruction_buckets(text: str) -> set:
    """Which kinds of evidence a sentence is asking for.

    A sentence can land in several buckets -- "the large red building beside the
    church" is size, colour, appearance-shaped and relational -- and the point of
    the breakdown is to see which bucket a feature group actually helps, so
    overlapping membership is the honest representation.
    """
    rel = parse_relations(text or "")
    att = parse_attributes(text or "")
    buckets = set()
    for name in ("color", "size", "shape", "appearance", "material"):
        if att[name]:
            buckets.add(name)
    if rel["any"]:
        buckets.add("relation")
    if rel["cardinal"]:
        buckets.add("direction")
    if rel["proximity"] or rel["distance_far"]:
        buckets.add("distance")
    if att["named"]:
        buckets.add("named")
    if rel["n_families"] >= 2:
        buckets.add("multi_relation")
    if not buckets:
        buckets.add("category_only")
    return buckets


def color_words(text: str) -> list:
    low = (text or "").lower()
    return [w for w in ATTRIBUTE_LEXICON["color"] if w in low]


def relation_feature_vector(text: str) -> list:
    """Fixed-length 0/1 vector over the relation and attribute families."""
    rel = parse_relations(text)
    att = parse_attributes(text)
    return ([1.0 if rel[name] else 0.0 for name in RELATION_LEXICON]
            + [1.0 if att[name] else 0.0 for name in ATTRIBUTE_LEXICON]
            + [1.0 if att["named"] else 0.0])


RELATION_FEATURE_NAMES = (list(RELATION_LEXICON) + list(ATTRIBUTE_LEXICON) + ["named"])
