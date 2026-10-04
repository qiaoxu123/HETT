"""Auditable lexical taxonomy for CityNav instruction attributes."""
from __future__ import annotations


ATTRIBUTE_TAXONOMY = {
    "color": {
        "white": ("white", "cream", "ivory"), "black": ("black",),
        "gray": ("gray", "grey", "silver"), "red": ("red", "reddish", "maroon"),
        "brown": ("brown", "tan", "beige"), "green": ("green",), "blue": ("blue",),
        "yellow": ("yellow", "gold"), "light": ("light", "pale"),
        "dark": ("dark", "darker"), "bright": ("bright",),
    },
    "size": {
        "large": ("large", "big", "huge"), "small": ("small", "tiny"),
        "tall": ("tall", "high"), "short": ("short", "low"),
        "long": ("long",), "wide": ("wide",), "narrow": ("narrow",),
    },
    "shape": {
        "rectangular": ("rectangular", "rectangle"), "square": ("square",),
        "round": ("round", "circular", "circle"), "elongated": ("elongated",),
        "l_shaped": ("l-shaped", "l shaped"), "u_shaped": ("u-shaped", "u shaped"),
        "irregular": ("irregular",), "courtyard": ("courtyard",),
        "hollow_center": ("hollow-center", "hollow center", "open center"),
    },
    "roof": {
        "flat_roof": ("flat roof", "flat-roof"), "pitched_roof": ("pitched roof", "sloped roof"),
        "roof": ("roof", "roofed",),
    },
    "context": {
        "road": ("road", "street", "lane", "drive", "avenue"),
        "intersection": ("intersection", "junction"), "crossing": ("crossing", "crossroad"),
        "parking_lot": ("parking lot", "car park", "parking area"),
        "field": ("field",), "grass": ("grass", "grassy"),
        "sports_field": ("sports field", "football field", "playing field"),
        "water": ("water", "pond", "lake"), "river": ("river",), "bridge": ("bridge",),
        "railway": ("railway", "railroad", "train tracks"),
        "open_area": ("open area", "open space"),
        "vegetation": ("tree", "trees", "vegetation", "wooded", "bush"),
    },
    "count_ordinal": {
        "one": ("one", "single"), "two": ("two", "couple"), "three": ("three",),
        "first": ("first", "1st"), "second": ("second", "2nd"), "third": ("third", "3rd"),
        "nearest": ("nearest", "closest"), "farthest": ("farthest", "furthest"),
        "multiple": ("multiple", "several"),
    },
    "spatial_relation": {
        "left": ("left",), "right": ("right",), "front": ("front", "ahead"),
        "behind": ("behind", "back of"), "near": ("near", "next to", "beside", "adjacent"),
        "between": ("between",), "before": ("before",), "after": ("after",),
        "past": ("past", "beyond"), "along": ("along",), "across": ("across",),
        "opposite": ("opposite",),
    },
    "semantic": {
        "church": ("church", "chapel", "cathedral"), "library": ("library",),
        "college": ("college", "university"), "school": ("school",), "office": ("office",),
        "tower": ("tower",), "hospital": ("hospital",), "house": ("house", "home"),
        "building": ("building",), "car": ("car", "vehicle", "van"),
        "parking": ("parking lot", "car park"), "ground": ("ground", "field"),
    },
}

VISUAL_CATEGORIES = ("color", "size", "shape", "roof", "context", "semantic")
GEOMETRIC_CATEGORIES = ("count_ordinal", "spatial_relation")

