"""Split a CityRefer instruction into target, anchors, relations and attributes.

The previous round established that the grounding task's binding constraint is
not appearance: the colour and texture of a correctly masked target are at
chance, and an anchor-only oracle nearly doubles Top-1.  That makes the sentence
structure the thing to model, and the first step is to read the sentence the way
the task demands -- as a target, one or more *anchor* entities it is described
relative to, and the relations between them.

Getting this split wrong is the failure mode this round exists to avoid.  In

    "the long white building beside the church"

``church`` is not a property of the target.  Feeding the whole sentence to a
text-image matcher lets the anchor leak into the target's appearance score, and
it also throws away the only structure that could distinguish two similar
buildings.  So the parser's job is to keep the four parts apart, and everything
downstream depends on it having done so.

The vocabulary is built from the dataset's own instructions rather than written
from imagination; :func:`vocabulary_stats` reports what actually occurs and how
much of it the lexicons cover.
"""

from __future__ import annotations

import re
from collections import Counter

# Type words, mapping a surface form to the SensatUrban class it names.  A
# phrase containing one of these is an anchor that can be matched to entities by
# type alone, with no name lookup.
TYPE_WORDS = {
    "Building": ("building", "buildings", "house", "houses", "home", "church",
                 "chapel", "cathedral", "school", "college", "university",
                 "tower", "hall", "hotel", "pub", "restaurant", "cafe", "shop",
                 "store", "supermarket", "warehouse", "factory", "office",
                 "block", "apartment", "flat", "garage", "station", "centre",
                 "center", "court", "lodge", "manor", "mill", "theatre",
                 "cinema", "museum", "library", "hospital", "clinic", "bank",
                 "club", "gym", "pub", "inn", "lodge", "barn", "shed"),
    "Car": ("car", "cars", "vehicle", "vehicles", "van", "vans", "truck",
            "trucks", "lorry", "taxi", "cab", "minibus", "bus"),
    "Bike": ("bike", "bikes", "bicycle", "bicycles", "motorcycle",
             "motorcycles", "scooter"),
    "TrafficRoad": ("road", "roads", "street", "streets", "lane", "lanes",
                    "avenue", "drive", "way", "boulevard", "highway", "route",
                    "junction", "intersection", "roundabout", "crossroads",
                    "carriageway", "motorway"),
    "Footpath": ("path", "paths", "pavement", "pavements", "sidewalk",
                 "sidewalks", "footpath", "walkway", "walk", "alley", "steps",
                 "stairs"),
    "Parking": ("parking", "lot", "lots", "car park", "carpark", "bays",
                "spaces", "space"),
    "Wall": ("wall", "walls", "fence", "fences", "railing", "railings",
             "barrier", "hedge", "hedges"),
    "HighVegetation": ("tree", "trees", "bush", "bushes", "shrub", "shrubs",
                       "vegetation", "greenery", "foliage", "canopy", "woodland",
                       "forest", "grove"),
    "Ground": ("ground", "lawn", "lawns", "grass", "grassland", "field",
               "fields", "garden", "gardens", "yard", "meadow", "green",
               "patch", "area"),
    "Water": ("water", "river", "canal", "lake", "pond", "stream", "dock",
              "harbour", "harbor", "reservoir", "sea"),
    "Bridge": ("bridge", "bridges", "overpass", "underpass", "viaduct"),
    "Rail": ("rail", "rails", "railway", "track", "tracks", "tram"),
    "StreetFurniture": ("bench", "benches", "bin", "bins", "lamp", "lamps",
                        "lamppost", "lampposts", "post", "posts", "sign",
                        "signs", "pole", "poles", "monument", "statue",
                        "bollard", "bollards", "hydrant", "box", "camera",
                        "furniture", "shelter", "advert", "billboard"),
}
WORD_TO_TYPE = {w: t for t, words in TYPE_WORDS.items() for w in words}

# Relations carry a direction, and the direction is expressed in a frame.  The
# frame is written down here rather than left implicit, because "left of" means
# something different depending on whether it is read from the agent's heading
# or from the map's axes, and silently choosing one is how a relation model
# becomes unfalsifiable.
#
#   "agent"  -- the relative bearing from the UAV's heading
#   "global" -- the world axes, +x east and +y north
#   "none"   -- symmetric; direction is not part of the meaning
RELATIONS = {
    "near": ("proximity", "none"),
    "close to": ("proximity", "none"),
    "next to": ("proximity", "none"),
    "beside": ("proximity", "none"),
    "adjacent to": ("proximity", "none"),
    "alongside": ("proximity", "none"),
    "by the": ("proximity", "none"),
    "far from": ("distance", "none"),
    "away from": ("distance", "none"),
    "behind": ("front_back", "agent"),
    "in front of": ("front_back", "agent"),
    "ahead of": ("front_back", "agent"),
    "left of": ("left_right", "agent"),
    "right of": ("left_right", "agent"),
    "to the left": ("left_right", "agent"),
    "to the right": ("left_right", "agent"),
    "north of": ("cardinal", "global"),
    "south of": ("cardinal", "global"),
    "east of": ("cardinal", "global"),
    "west of": ("cardinal", "global"),
    "between": ("between", "none"),
    "across from": ("across", "none"),
    "opposite": ("across", "none"),
    "facing": ("across", "none"),
    "on top of": ("vertical", "global"),
    "under": ("vertical", "global"),
    "below": ("vertical", "global"),
    "above": ("vertical", "global"),
    "bordering": ("proximity", "none"),
    "surrounding": ("proximity", "none"),
    "overlooking": ("vertical", "global"),
    "on": ("on_surface", "none"),
}

# Words that describe the target rather than place it.
COLOR_WORDS = ("red", "white", "black", "blue", "green", "yellow", "brown",
               "grey", "gray", "orange", "purple", "pink", "silver", "golden",
               "beige", "maroon", "turquoise", "cream", "dark", "light",
               "bright", "multicolored", "multicoloured", "coloured", "colored")
SIZE_WORDS = ("large", "big", "small", "tall", "short", "long", "wide",
              "narrow", "high", "low", "huge", "little", "massive", "tiny",
              "medium", "thick", "thin")
SHAPE_WORDS = ("round", "square", "rectangular", "triangular", "curved",
               "straight", "flat", "sloped", "cylindrical", "l-shaped",
               "shaped", "circular", "oval", "conjoined", "three-part")

# A phrase boundary: prepositions and conjunctions that end a noun phrase.
_BOUNDARY = re.compile(
    r"\b(that|which|who|with|and|or|but|is|are|was|were|has|have|had|it|its|"
    r"in|on|at|of|from|by|for|to|when|where|while|there|this|these|those)\b")
_DETERMINER = re.compile(r"^(the|a|an|this|that|these|those|another|other|"
                         r"some|any|two|three|four|five|six|seven|eight|nine|ten)\s+")
_ANCHOR_MAX_WORDS = 7


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", (text or "").lower())).strip()


_NEW_NP = re.compile(r"\b(the|a|an|this|that|these|those|another|other|some|"
                     r"any|its|his|her|their)\b")
_RELATION_ALTERNATION = None


def _phrase_after(text: str, end: int) -> str:
    """The noun phrase a relation points at, starting just after ``end``.

    The phrase ends at the first thing that begins a new one: a clause word, a
    preposition, another relation, or a determiner.  Without the last of those,
    "close to the building the railway track runs beside" reads as one anchor
    made of two entities, and a grounder handed that phrase can only fail.
    """
    global _RELATION_ALTERNATION
    if _RELATION_ALTERNATION is None:
        _RELATION_ALTERNATION = re.compile(
            r"\b(" + "|".join(re.escape(p) for p in sorted(
                RELATIONS, key=len, reverse=True)) + r")\b")

    tail = text[end:end + 120].strip()
    lead = _DETERMINER.match(tail)
    if lead:
        tail = tail[lead.end():]
    cuts = [len(tail)]
    for pattern in (_BOUNDARY, _NEW_NP, _RELATION_ALTERNATION):
        found = pattern.search(tail)
        if found:
            cuts.append(found.start())
    tail = tail[: min(cuts)]
    return " ".join(tail.split()[:_ANCHOR_MAX_WORDS]).strip()


def parse_instruction(text: str) -> dict:
    """Target / anchors / relations / attributes for one instruction.

    An anchor is the noun phrase a relation points at.  ``on`` is deliberately
    the weakest relation in the table: "the building *on* St Peter's Street" and
    "the bin *on* the pavement" both place the target on a surface, and streets
    are the anchor far more often than not in this dataset.  That is recorded as
    its own relation family so a later stage can decide what to do with it.
    """
    low = normalise(text)
    matches = []
    for phrase, (family, frame) in RELATIONS.items():
        for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
            matches.append((m.start(), m.end(), phrase, family, frame))
    matches.sort()

    anchors, relations = [], []
    for start, end, phrase, family, frame in matches:
        anchor_phrase = _phrase_after(low, end)
        relations.append({"phrase": phrase, "family": family, "frame": frame,
                          "start": start, "end": end})
        if not anchor_phrase:
            continue
        types = [WORD_TO_TYPE[w] for w in anchor_phrase.split()
                 if w in WORD_TO_TYPE]
        anchors.append({
            "phrase": anchor_phrase,
            "type": types[0] if types else None,
            "relation": phrase,
            "family": family,
            "frame": frame,
            "is_named": bool(re.search(r"\b(st|saint|street|road|lane|avenue|"
                                       r"house|court|school|church|college)\b",
                                       anchor_phrase)),
        })

    first_relation = matches[0][0] if matches else len(low)
    target_phrase = low[:first_relation].strip()
    colours = [w for w in COLOR_WORDS if re.search(r"\b" + w + r"\b", target_phrase)]
    sizes = [w for w in SIZE_WORDS if re.search(r"\b" + w + r"\b", target_phrase)]
    shapes = [w for w in SHAPE_WORDS if re.search(r"\b" + w + r"\b", target_phrase)]
    target_types = [WORD_TO_TYPE[w] for w in target_phrase.split()
                    if w in WORD_TO_TYPE]

    return {
        "instruction": text,
        "target_phrase": target_phrase,
        "target_type": target_types[0] if target_types else None,
        "attributes": {"color": colours, "size": sizes, "shape": shapes},
        "anchors": anchors,
        "relations": relations,
        "outdoor_anchor_count": sum(1 for a in anchors if a["type"] is not None),
        "n_relations": len(relations),
    }


def vocabulary_stats(instructions) -> dict:
    """What the dataset actually says, so the lexicons can be checked against it.

    Written out because a lexer built from imagination would quietly miss the
    words the corpus uses and the parser's coverage would be a property of the
    author rather than of the data.
    """
    words = Counter()
    relation_hits = Counter()
    type_hits = Counter()
    anchor_types = Counter()
    for text in instructions:
        low = normalise(text)
        words.update(low.split())
        for phrase, (family, _) in RELATIONS.items():
            n = len(re.findall(r"\b" + re.escape(phrase) + r"\b", low))
            if n:
                relation_hits[phrase] += n
        for word in low.split():
            if word in WORD_TO_TYPE:
                type_hits[word] += 1
        parsed = parse_instruction(text)
        for anchor in parsed["anchors"]:
            anchor_types[anchor["type"] or "NONE"] += 1
    total_relations = sum(relation_hits.values())
    return {
        "n_instructions": len(instructions),
        "relation_hits": dict(relation_hits.most_common()),
        "relations_per_instruction": total_relations / max(len(instructions), 1),
        "type_word_hits": dict(type_hits.most_common(40)),
        "anchor_type_distribution": dict(anchor_types.most_common()),
        "top_words": dict(words.most_common(120)),
    }
