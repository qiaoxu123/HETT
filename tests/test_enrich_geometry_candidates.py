import pytest

from scripts.enrich_geometry_candidates import enrich


def test_enrich_rejects_misaligned_candidate_rows():
    candidate = [{"map_name": "a", "description": "x", "target": [0.1, 0.2]}]
    source = [{"map_name": "b", "description": "x", "target": [0.1, 0.2]}]

    with pytest.raises(ValueError, match="mismatch"):
        enrich(candidate, source)
