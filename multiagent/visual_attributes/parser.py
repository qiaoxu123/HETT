"""Deterministic language-to-attribute parser used for census and labels."""
from __future__ import annotations

import re

from .taxonomy import ATTRIBUTE_TAXONOMY


def _pattern(phrase: str) -> re.Pattern:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


COMPILED = {
    category: {name: tuple(_pattern(term) for term in terms) for name, terms in values.items()}
    for category, values in ATTRIBUTE_TAXONOMY.items()
}


def parse_attributes(text: str) -> dict[str, tuple[str, ...]]:
    result = {}
    for category, values in COMPILED.items():
        matched = tuple(name for name, patterns in values.items() if any(pattern.search(text) for pattern in patterns))
        if matched:
            result[category] = matched
    return result


def flattened_attributes(text: str) -> tuple[str, ...]:
    parsed = parse_attributes(text)
    return tuple(f"{category}:{value}" for category, values in parsed.items() for value in values)

