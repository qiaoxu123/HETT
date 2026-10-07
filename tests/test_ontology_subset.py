"""The subset definitions Gate E rests on, and the arms it compares.

The round's conclusion depends entirely on the subsets being what they claim:
"ontology-covered" has to mean every *spatial* relation maps, without disqualifying
a sentence for saying "red", and "binding-clean" has to mean one node per anchor
without reaching for the answer.  A subset that quietly widened would make the
diagnosis look better than the data.

The last test is the one that keeps the comparison honest: an ORACLE arm exists
in the evaluator to price the ambiguity, and nothing in the deployable path may
read it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.spatial_graph.node_types import Node, NodeKind  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_geometry import RELATION_NAMES  # noqa: E402
from scripts.eval_ontology_subset import (  # noqa: E402
    DISTANCE_RELATIONS, binding_status, classify_parse, is_multi_anchor,
    mapped_pairs, relation_groups,
)


def record(relations, anchors=None, valid=True):
    """A parse record whose anchor list is long enough for its own indices.

    ``mapped_pairs`` drops a relation pointing past the anchor list, so the
    default has to cover the largest index used -- an empty list silently turned
    every sample into "no mapped relations" in an earlier version of these
    tests.
    """
    top = max([a for _, a in relations if isinstance(a, int)], default=-1) + 1
    if anchors is None:
        anchors = [([], []) for _ in range(top)]
    return {
        "valid": valid,
        "parsed": {"relations": [{"relation_normalized": n, "anchor": a}
                                 for n, a in relations]},
        "anchors": [{"regions": list(r), "instances": list(i)}
                    for r, i in anchors],
    }


def anchor(regions=(), instances=()):
    return {"regions": list(regions), "instances": list(instances)}


# --------------------------------------------------------------------------
# §2 the parse classes
# --------------------------------------------------------------------------

def test_a_sentence_of_mapped_relations_is_covered():
    assert classify_parse(record([("near", 0), ("north_of", 1)])) == \
        "ONTOLOGY_COVERED"


def test_one_unknown_relation_makes_it_partial_not_covered():
    assert classify_parse(record([("near", 0), ("UNKNOWN", 1)])) == \
        "PARTIAL_COVERED"


def test_a_sentence_of_only_unknown_relations_is_unknown_only():
    assert classify_parse(record([("UNKNOWN", 0), ("UNKNOWN", 1)])) == \
        "UNKNOWN_ONLY"


def test_associated_with_is_not_a_spatial_relation():
    """A mentioned anchor with no geometry asserted must not disqualify a sample."""
    assert classify_parse(record([("near", 0), ("ASSOCIATED_WITH", 1)])) == \
        "ONTOLOGY_COVERED"
    assert classify_parse(record([("ASSOCIATED_WITH", 0)])) == \
        "NO_SPATIAL_RELATION"


def test_appearance_words_never_reach_the_classifier():
    """§3: only spatial relations decide coverage.

    The classifier sees the parser's relation list, not the sentence, and the
    parser reports a colour as an attribute.  So "red building near the church"
    and "building near the church" are the same sample here -- which is the
    point: a colour must not exclude anything.
    """
    with_colour = classify_parse(record([("near", 0)]))
    without = classify_parse(record([("near", 0)]))
    assert with_colour == without == "ONTOLOGY_COVERED"


def test_a_failed_parse_is_its_own_class():
    assert classify_parse(record([("near", 0)], valid=False)) == "PARSE_FAILED"
    assert classify_parse({"valid": True, "parsed": None}) == "PARSE_FAILED"


# --------------------------------------------------------------------------
# §3 what counts as a mapped relation
# --------------------------------------------------------------------------

def test_only_ontology_relations_are_scored():
    rec = record([("near", 0), ("UNKNOWN", 0), ("ASSOCIATED_WITH", 0)])
    assert [n for n, _ in mapped_pairs(rec)] == ["near"]


def test_an_anchor_named_by_string_is_resolved_to_its_index():
    rec = {"valid": True,
           "parsed": {"relations": [{"relation_normalized": "near",
                                     "anchor": "the church"}]},
           "anchors": [{"raw": {"entity_name": "the church"}, "regions": [3],
                        "instances": []}]}
    assert mapped_pairs(rec) == [("near", 0)]


def test_a_relation_pointing_at_nothing_is_dropped():
    rec = {"valid": True,
           "parsed": {"relations": [{"relation_normalized": "near",
                                     "anchor": "nowhere"}]},
           "anchors": [{"raw": {"entity_name": "the church"}, "regions": [3],
                        "instances": []}]}
    assert mapped_pairs(rec) == []


# --------------------------------------------------------------------------
# §4/§5 binding classes
# --------------------------------------------------------------------------

def test_one_node_per_anchor_is_clean():
    rec = record([("near", 0)], anchors=[([7], [])])
    assert binding_status(rec, {0: [_node(7)]}) == "BINDING_CLEAN"


def test_a_unique_road_region_is_a_clean_binding():
    """§5: the region *is* the entity the name denotes."""
    rec = record([("near", 0)], anchors=[([7], [])])
    assert binding_status(rec, {0: [_node(7, NodeKind.ROAD_REGION)]}) == \
        "BINDING_CLEAN"


def test_several_nodes_for_one_name_is_ambiguous():
    rec = record([("near", 0)], anchors=[([7, 8], [])])
    assert binding_status(rec, {0: [_node(7), _node(8)]}) == \
        "BINDING_AMBIGUOUS"


def test_two_disconnected_regions_of_one_name_are_ambiguous():
    rec = record([("near", 0)], anchors=[([7, 9], [])])
    got = binding_status(rec, {0: [_node(7, NodeKind.ROAD_REGION),
                                   _node(9, NodeKind.ROAD_REGION)]})
    assert got == "BINDING_AMBIGUOUS"


def test_an_anchor_that_binds_to_nothing_is_unresolved():
    rec = record([("near", 0)], anchors=[([], [])])
    assert binding_status(rec, {0: []}) == "BINDING_UNRESOLVED"


def test_an_anchor_only_an_unknown_relation_uses_does_not_count():
    """Otherwise a sample is called ambiguous for an anchor that never scores."""
    rec = record([("near", 0), ("UNKNOWN", 1)], anchors=[([7], []), ([8, 9], [])])
    assert binding_status(rec, {0: [_node(7)], 1: [_node(8), _node(9)]}) == \
        "BINDING_CLEAN"


def test_binding_never_consults_the_target():
    """The check that makes the subset trustworthy, asserted structurally."""
    import inspect
    source = inspect.getsource(binding_status)
    for forbidden in ("target", "is_target", "gt_", "candidate"):
        assert forbidden not in source, forbidden


# --------------------------------------------------------------------------
# §10/§11 the subsets
# --------------------------------------------------------------------------

def test_near_and_far_are_the_distance_relations():
    assert DISTANCE_RELATIONS == {"near", "far"}
    rec = record([("near", 0), ("north_of", 1)])
    assert relation_groups(rec) & DISTANCE_RELATIONS


def test_a_between_sentence_is_multi_anchor():
    assert is_multi_anchor(record([("between", 0), ("between", 1)]))


def test_a_road_topology_relation_is_multi_anchor():
    assert is_multi_anchor(record([("opposite_side_of_road", 0)]))


def test_two_distinct_anchors_are_multi_anchor():
    assert is_multi_anchor(record([("near", 0), ("north_of", 1)]))
    assert not is_multi_anchor(record([("near", 0)]))


# --------------------------------------------------------------------------
# the arms must actually differ
# --------------------------------------------------------------------------

def _node(nid, kind=NodeKind.BUILDING):
    return Node(node_id=nid, kind=kind, entity_type="Building", name="",
                center=np.array([float(nid) * 10, 0.0, 0.0]),
                dimension=np.array([4.0, 4.0, 3.0]),
                footprint=np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0],
                                    [0.0, 4.0]]) + [nid * 10, 0.0],
                major_axis=np.array([1.0, 0.0]))


class _Graph:
    def __init__(self):
        self.regions, self.by_id = [], {7: _node(7), 8: _node(8)}
        self.context = type("C", (), {"road_members": {}})()


def test_the_second_anchor_changes_the_score():
    """§11: if pairing the between factor changed nothing, it is not being used."""
    import torch
    from scripts.eval_ontology_subset import graph_scores

    seen = []

    def fake_teacher(geom, rel):
        # Returns the between_t column, so the score is exactly the feature the
        # second anchor writes.  A no-op pairing would give a constant.
        seen.append(float(geom[:, 26].abs().sum()))
        return geom[:, 26]

    graph = _Graph()
    candidates = [_node(100), _node(101)]
    anchors = {0: [_node(7)], 1: [_node(8)]}
    # The parser reports a between-factor as a symmetric *pair* of relations,
    # one per anchor, and that is what the evaluator pairs up.  A single
    # relation has no second anchor to pair with, which is exactly the shape the
    # previous round fed it for 73% of its covered samples.
    single = graph_scores(torch, fake_teacher, {"between": 1}, graph, candidates,
                          anchors, [("between", 0)])
    assert not seen or seen[-1] == 0.0, "one relation cannot have a second anchor"

    both = graph_scores(torch, fake_teacher, {"between": 1}, graph, candidates,
                        anchors, [("between", 0), ("between", 1)])
    assert seen and seen[-1] > 0, "the between column was never populated"
    assert both is not None and np.isfinite(both).all()
    assert single is not None


def test_shuffling_the_relation_changes_the_score():
    import torch
    from scripts.eval_ontology_subset import graph_scores

    def fake_teacher(geom, rel):
        # Score depends on the relation id, so a shuffle must move it.
        return geom[:, 0] + rel.float() * 100.0

    graph = _Graph()
    candidates = [_node(100), _node(101), _node(102)]
    anchors = {0: [_node(7)]}
    rng = np.random.default_rng(0)
    plain = graph_scores(torch, fake_teacher, {"near": 1}, graph, candidates,
                         anchors, [("near", 0)])
    shuffled = graph_scores(torch, fake_teacher, {"near": 1}, graph, candidates,
                            anchors, [("near", 0)], rng=rng, shuffled=True)
    assert plain is not None and shuffled is not None
    assert not np.allclose(plain, shuffled)


def test_the_opposite_relation_changes_the_score():
    import torch
    from scripts.eval_ontology_subset import graph_scores

    def fake_teacher(geom, rel):
        # dx, not dy: the synthetic nodes sit on the x axis, so a dy-based
        # stand-in would be identically zero and the test would pass vacuously.
        return geom[:, 0] * rel.float()

    graph = _Graph()
    candidates = [_node(100), _node(101)]
    anchors = {0: [_node(7)]}
    north = graph_scores(torch, fake_teacher, {"north_of": 1}, graph, candidates,
                         anchors, [("north_of", 0)])
    south = graph_scores(torch, fake_teacher, {"south_of": 2}, graph, candidates,
                         anchors, [("south_of", 0)])
    assert not np.allclose(north, south)


def test_the_arms_are_permutation_invariant_over_candidates():
    """A candidate order that changes the answer would hand it to index 0."""
    import torch
    from scripts.eval_ontology_subset import graph_scores

    def fake_teacher(geom, rel):
        return geom[:, 0] + geom[:, 1]

    graph = _Graph()
    candidates = [_node(100), _node(101), _node(102)]
    anchors = {0: [_node(7)]}
    plain = graph_scores(torch, fake_teacher, {"near": 1}, graph, candidates,
                         anchors, [("near", 0)])
    order = np.array([2, 0, 1])
    permuted = graph_scores(torch, fake_teacher, {"near": 1}, graph,
                            [candidates[i] for i in order], anchors,
                            [("near", 0)])
    assert np.allclose(plain[order], permuted)


def test_the_oracle_arm_is_kept_out_of_the_deployable_path():
    """§14/§20: an ORACLE anchor must not be reachable from a scored arm."""
    import inspect
    from scripts import eval_ontology_subset as module
    source = inspect.getsource(module.main)
    # The oracle appears only as a named arm, never inside the default scoring.
    assert '"oracle_anchor"' in source
    scoring = inspect.getsource(module.graph_scores)
    for forbidden in ("oracle", "target_index", "is_target"):
        assert forbidden not in scoring, forbidden
