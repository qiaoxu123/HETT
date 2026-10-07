"""The language stage's contract, tested without a network.

Phase 4 cannot run in this environment -- ``DEEPSEEK_API_KEY`` is unset and the
brief forbids putting a key in the repository -- so what is testable is the part
that matters for trust: what the model is *allowed to see*, that a missing key
fails loudly rather than returning nothing, and that a model's answer is treated
as data and validated rather than believed.

The leakage test is the important one.  Every previous round in this project
lost time to something reading the answer; here the request is assembled in one
function precisely so that a single test can assert none of those strings can
reach it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sensaturban_fpv.spatial_graph.deepseek_parser import (  # noqa: E402
    FORBIDDEN_IN_REQUEST, ONTOLOGY, Cache, api_key, build_request,
    check_no_leakage, normalisation_consistency, require_key, unknown_rate,
    validate,
)

INSTRUCTION = ("the building across the road from the church on Aldridge Road")


def test_the_request_carries_only_the_instruction_and_the_ontology():
    request = build_request(INSTRUCTION)
    blob = json.dumps(request)
    assert INSTRUCTION in blob
    for name in ONTOLOGY:
        assert name in blob


def test_none_of_the_answer_can_reach_the_request():
    """Section 18, as an assertion rather than a promise."""
    request = build_request(INSTRUCTION)
    check_no_leakage(request)          # must not raise
    for token in FORBIDDEN_IN_REQUEST:
        assert token.lower() not in json.dumps(request).lower()


def test_the_leakage_check_actually_fires():
    """A guard that cannot fail is not a guard."""
    with pytest.raises(ValueError):
        check_no_leakage({"messages": [{"content": "target_id: 7"}]})
    with pytest.raises(ValueError):
        check_no_leakage({"candidate_ids": [1, 2, 3]})


def test_an_ontology_permutation_changes_the_prompt():
    a = build_request(INSTRUCTION, shuffle_seed=0)
    b = build_request(INSTRUCTION, shuffle_seed=1)
    assert json.dumps(a) != json.dumps(b)
    # But never the instruction itself.
    assert INSTRUCTION in json.dumps(b)


def test_a_missing_key_fails_loudly_rather_than_returning_nothing():
    """Silence would be indistinguishable from a corpus with no relations."""
    assert api_key({}) is None
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        require_key({})


def test_the_key_is_read_from_the_environment_only():
    assert api_key({"DEEPSEEK_API_KEY": "x"}) == "x"
    # And the module has no other path to one.
    source = (REPO_ROOT / "sensaturban_fpv" / "spatial_graph"
              / "deepseek_parser.py").read_text()
    for forbidden in ("open('.env", "api_key.json", "secrets"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------

def test_the_cache_returns_a_previous_answer_without_a_second_call(tmp_path):
    cache = Cache(tmp_path)
    request = build_request(INSTRUCTION)
    assert cache.get(request) is None
    cache.put(request, {"target_phrase": "building"},
              {"prompt_tokens": 100, "completion_tokens": 20})
    assert cache.get(request) == {"target_phrase": "building"}
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["prompt_tokens"] == 100


def test_the_cache_key_covers_the_whole_request(tmp_path):
    """A key over the instruction alone would collide across prompt changes."""
    cache = Cache(tmp_path)
    a = build_request(INSTRUCTION, shuffle_seed=0)
    b = build_request(INSTRUCTION, shuffle_seed=1)
    assert cache.key(a) != cache.key(b)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def test_a_valid_answer_passes():
    parsed = {"target_phrase": "building",
              "anchors": [{"phrase": "the church", "entity_name": "Church",
                           "entity_type": "Building"}],
              "relations": [{"source": "target", "relation_raw": "across the road from",
                             "relation_normalized": "across_road",
                             "anchor": 0, "confidence": 0.8}]}
    ok, reason = validate(parsed)
    assert ok, reason


def test_a_relation_outside_the_ontology_is_rejected():
    parsed = {"target_phrase": "x", "anchors": [],
              "relations": [{"relation_normalized": "behind", "anchor": 0}]}
    ok, reason = validate(parsed)
    assert not ok and "behind" in reason


def test_unknown_and_associated_with_are_part_of_the_ontology():
    """So that "I cannot map this" is an answer the model can give."""
    assert "UNKNOWN" in ONTOLOGY and "ASSOCIATED_WITH" in ONTOLOGY
    for name in ("UNKNOWN", "ASSOCIATED_WITH"):
        ok, _ = validate({"target_phrase": "x", "anchors": [],
                          "relations": [{"relation_normalized": name}]})
        assert ok


def test_a_malformed_answer_is_rejected_rather_than_used():
    for bad in (None, [], {}, {"target_phrase": "x"},
                {"target_phrase": "x", "anchors": {}, "relations": []},
                {"target_phrase": "x", "anchors": [{"phrase": "a"}],
                 "relations": []},
                {"target_phrase": "x", "anchors": [],
                 "relations": [{"relation_normalized": "near",
                                "confidence": 5.0}]}):
        ok, _ = validate(bad)
        assert not ok, bad


# --------------------------------------------------------------------------
# consistency and coverage
# --------------------------------------------------------------------------

def test_consistency_notices_a_reordered_ontology_changing_the_answer():
    first = {"target_phrase": "b", "anchors": [{"entity_name": "Church"}],
             "relations": [{"relation_normalized": "across_road"}]}
    same = json.loads(json.dumps(first))
    other = {"target_phrase": "b", "anchors": [{"entity_name": "Church"}],
             "relations": [{"relation_normalized": "near"}]}
    assert normalisation_consistency(first, same)["relations_equal"]
    report = normalisation_consistency(first, other)
    assert not report["relations_equal"]
    assert report["relation_jaccard"] == 0.0
    assert report["anchors_equal"]


def test_unknown_rate_counts_what_could_not_be_mapped():
    rows = [
        {"relations": [{"relation_normalized": "near"},
                       {"relation_normalized": "UNKNOWN"}]},
        {"relations": [{"relation_normalized": "ASSOCIATED_WITH"}]},
    ]
    stats = unknown_rate(rows)
    assert stats["relations"] == 3 and stats["mapped"] == 1
    assert stats["unknown"] == 1 and stats["associated_with"] == 1
    assert stats["unknown_rate"] == pytest.approx(1 / 3)
    assert unknown_rate([])["unknown_rate"] == 0.0
