"""Visual-attribute observability utilities (independent of navigation control)."""

from .parser import parse_attributes
from .taxonomy import ATTRIBUTE_TAXONOMY, GEOMETRIC_CATEGORIES, VISUAL_CATEGORIES

__all__ = ["ATTRIBUTE_TAXONOMY", "GEOMETRIC_CATEGORIES", "VISUAL_CATEGORIES", "parse_attributes"]

