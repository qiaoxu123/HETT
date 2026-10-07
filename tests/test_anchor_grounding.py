"""Unit tests for parser, anchor grounding and relation reasoning.

The round turns on language actually reaching the score, so most of these tests
are of the form "change the words, require the ordering to change".  Two are
about not fooling ourselves: that an anchor's frame is declared rather than
assumed, and that the evaluation subset for anchors cannot be silently reused as
a prediction input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.anchor_grounding import (  # noqa: E402
    EntityNode, anchor_log_probabilities, between_score, block_nodes,
    gt_anchor_for, lexical_anchor_scores, marginalise, oriented_relation_score,
    relation_features, relation_geometry, rule_relation_score,
)
from sensaturban_fpv.anchor_parser import (  # noqa: E402
    RELATIONS, normalise, parse_instruction, vocabulary_stats,
)


class _Obj:
    def __init__(self, eid, otype, name, pos, dim=(5.0, 5.0, 5.0)):
        self.id, self.object_type, self.name = eid, otype, name
        self.position, self.dimension = pos, dim


def _nodes():
    return [
        EntityNode(1, "Building", "St Teresa's Court", np.array([0.0, 0.0, 5.0]),
                   np.array([10.0, 10.0, 8.0])),
        EntityNode(2, "Building", "", np.array([30.0, 0.0, 5.0]),
                   np.array([10.0, 10.0, 8.0])),
        EntityNode(3, "TrafficRoad", "Leslie Road", np.array([0.0, 40.0, 0.2]),
                   np.array([6.0, 60.0, 0.2])),
        EntityNode(4, "Car", "", np.array([-20.0, 0.0, 1.0]),
                   np.array([4.0, 2.0, 1.5])),
    ]


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def test_parser_separates_target_from_anchor():
    """The distinction the whole round rests on."""
    parsed = parse_instruction("the long white building beside the church")
    assert parsed["target_type"] == "Building"
    assert "white" in parsed["attributes"]["color"]
    assert "long" in parsed["attributes"]["size"]
    assert [a["phrase"] for a in parsed["anchors"]] == ["church"]
    assert parsed["anchors"][0]["type"] == "Building"
    assert "church" not in parsed["target_phrase"]


def test_parser_keeps_two_anchors_apart():
    parsed = parse_instruction(
        "the car behind the red building near the road")
    phrases = [a["phrase"] for a in parsed["anchors"]]
    # Attributes stay with the noun phrase they modify, which is what makes
    # "the red building" a usable anchor description rather than just "building".
    assert any("building" in p for p in phrases)
    assert any("road" in p for p in phrases)
    assert len(phrases) == 2
    families = [a["family"] for a in parsed["anchors"]]
    assert "front_back" in families and "proximity" in families


def test_anchor_phrase_stops_at_the_next_entity():
    """A phrase that swallows two entities is one no grounder can match."""
    parsed = parse_instruction("close to the building the railway track runs on")
    phrases = [a["phrase"] for a in parsed["anchors"]]
    assert phrases == ["building"], phrases


def test_every_relation_declares_a_frame():
    for phrase, (family, frame) in RELATIONS.items():
        assert frame in ("agent", "global", "none"), (phrase, frame)
        assert family in ("proximity", "distance", "front_back", "left_right",
                          "cardinal", "between", "across", "vertical",
                          "on_surface")
    # The viewpoint-dependent words must not be declared as global.
    assert RELATIONS["left of"][1] == "agent"
    assert RELATIONS["north of"][1] == "global"


def test_vocabulary_stats_reports_what_the_corpus_says():
    stats = vocabulary_stats(["the white car beside the church",
                              "the building near the road"])
    assert stats["n_instructions"] == 2
    assert stats["relation_hits"]["beside"] == 1
    assert stats["relation_hits"]["near"] == 1
    assert stats["anchor_type_distribution"]["Building"] == 1
    assert stats["anchor_type_distribution"]["TrafficRoad"] == 1


def test_normalise_is_idempotent_and_punctuation_free():
    once = normalise("St. Teresa's Court, Birmingham!")
    assert once == normalise(once)
    assert "." not in once and "," not in once


# --------------------------------------------------------------------------
# anchor grounding
# --------------------------------------------------------------------------

def test_lexical_scores_rank_the_named_entity_first():
    nodes = _nodes()
    scores = lexical_anchor_scores("st teresa", nodes, phrase_type="Building")
    assert int(np.argmax(scores)) == 0
    road = lexical_anchor_scores("leslie road", nodes, phrase_type="TrafficRoad")
    assert int(np.argmax(road)) == 2


def test_lexical_scores_use_the_type_when_no_name_matches():
    nodes = _nodes()
    scores = lexical_anchor_scores("some car", nodes, phrase_type="Car")
    assert int(np.argmax(scores)) == 3


def test_lexical_scores_are_zero_for_an_empty_phrase():
    nodes = _nodes()
    assert not lexical_anchor_scores("", nodes).any()


def test_anchor_log_probabilities_are_normalised_over_the_top_k():
    scores = np.array([3.0, 2.0, 1.0, 0.0, -1.0])
    order, logp = anchor_log_probabilities(scores, top_k=3)
    assert order.tolist() == [0, 1, 2]
    assert np.all(np.diff(logp) <= 1e-9)          # descending
    assert np.exp(logp).sum() <= 1.0 + 1e-9


def test_gt_anchor_ignores_the_targets_own_name():
    """Otherwise a sentence that only names its target invents an anchor."""
    nodes = _nodes()
    anchors = [{"phrase": "st teresa", "type": "Building"}]
    assert gt_anchor_for(anchors, nodes, target_id=2) == (0, 0)
    assert gt_anchor_for(anchors, nodes, target_id=1) is None


def test_gt_anchor_refuses_an_ambiguous_name():
    nodes = _nodes() + [EntityNode(5, "Building", "St Teresa's House",
                                   np.array([60.0, 0.0, 5.0]),
                                   np.array([8.0, 8.0, 8.0]))]
    anchors = [{"phrase": "st teresa", "type": "Building"}]
    assert gt_anchor_for(anchors, nodes, target_id=2) is None


def test_block_nodes_covers_the_whole_block():
    objects = {o.id: o for o in
               [_Obj(1, "Building", "A", [0, 0, 0]), _Obj(2, "Car", "", [1, 1, 0])]}
    nodes = block_nodes(objects)
    assert len(nodes) == 2
    assert {n.object_type for n in nodes} == {"Building", "Car"}


# --------------------------------------------------------------------------
# relations
# --------------------------------------------------------------------------

def test_north_of_and_south_of_cannot_both_be_satisfied():
    north = {"phrase": "north of", "family": "cardinal", "frame": "global"}
    south = {"phrase": "south of", "family": "cardinal", "frame": "global"}
    geometry = relation_geometry([0.0, 50.0], [0.0, 0.0], [0.0, -100.0], 0.0)
    assert oriented_relation_score(geometry, north) > 0
    assert oriented_relation_score(geometry, south) < 0


def test_left_and_right_are_opposite_in_the_agent_frame():
    left = {"phrase": "left of", "family": "left_right", "frame": "agent"}
    right = {"phrase": "right of", "family": "left_right", "frame": "agent"}
    # Facing east (yaw 0), right is south, so a candidate to the north is left.
    geometry = relation_geometry([0.0, 30.0], [0.0, 0.0], [0.0, 0.0], 0.0)
    assert oriented_relation_score(geometry, left) > 0
    assert oriented_relation_score(geometry, right) < 0
    # Facing north, the same candidate is ahead rather than to a side.
    turned = relation_geometry([0.0, 30.0], [0.0, 0.0], [0.0, 0.0], np.pi / 2)
    assert abs(oriented_relation_score(turned, left)) < 1e-6


def test_behind_requires_the_candidate_to_be_farther_along_the_heading():
    behind = {"phrase": "behind", "family": "front_back", "frame": "agent"}
    front = {"phrase": "in front of", "family": "front_back", "frame": "agent"}
    geometry = relation_geometry([0.0, 80.0], [0.0, 40.0], [0.0, 0.0], np.pi / 2)
    assert oriented_relation_score(geometry, behind) > 0
    assert oriented_relation_score(geometry, front) < 0


def test_proximity_decreases_with_distance_and_far_from_increases():
    near_geom = relation_geometry([0.0, 10.0], [0.0, 0.0], [0.0, -50.0], 0.0)
    far_geom = relation_geometry([0.0, 120.0], [0.0, 0.0], [0.0, -50.0], 0.0)
    assert rule_relation_score(near_geom, "proximity", "none") > \
        rule_relation_score(far_geom, "proximity", "none")
    assert rule_relation_score(far_geom, "distance", "none") > \
        rule_relation_score(near_geom, "distance", "none")


def test_between_prefers_the_middle_and_rejects_the_outside():
    middle = between_score([5.0, 0.0], [0.0, 0.0], [10.0, 0.0], [0.0, -50.0], 0.0)
    beyond = between_score([30.0, 0.0], [0.0, 0.0], [10.0, 0.0], [0.0, -50.0], 0.0)
    aside = between_score([5.0, 60.0], [0.0, 0.0], [10.0, 0.0], [0.0, -50.0], 0.0)
    assert middle > beyond and middle > aside


def test_relation_features_have_a_fixed_width_and_are_finite():
    relation = {"phrase": "beside", "family": "proximity", "frame": "none"}
    geometry = relation_geometry([10.0, 5.0], [0.0, 0.0], [0.0, -30.0], 0.4)
    feats = relation_features(geometry, relation, np.array([4.0, 2.0, 1.5]),
                              np.array([10.0, 10.0, 8.0]))
    assert feats.shape == (25,) and np.isfinite(feats).all()


def test_marginalise_never_loses_to_the_best_single_hypothesis():
    logp = np.log(np.array([0.6, 0.3, 0.1]))
    compat = np.array([2.0, 1.0, 0.5])
    value = marginalise(logp, compat)
    assert value >= float((logp + compat).max()) - 1e-9


def test_marginalise_lets_a_second_anchor_rescue_a_bad_first_one():
    logp = np.log(np.array([0.9, 0.1]))
    good_second = marginalise(logp, np.array([0.0, 3.0]))
    assert good_second > 3.0 * 0.1 - 0.1


# --------------------------------------------------------------------------
# language must reach the score
# --------------------------------------------------------------------------

def _score_for(text, candidate_xy, anchor_xy, uav_yaw=0.0,
               uav_position=(0.0, -50.0)):
    parsed = parse_instruction(text)
    anchor = parsed["anchors"][0]
    relation = {"phrase": anchor["relation"], "family": anchor["family"],
                "frame": anchor["frame"]}
    geometry = relation_geometry(candidate_xy, anchor_xy, uav_position, uav_yaw)
    return oriented_relation_score(geometry, relation)


def test_changing_beside_to_far_from_changes_the_ordering():
    """The brief's own example, as a test."""
    far_candidate, near_candidate = (0.0, 100.0), (0.0, 10.0)
    anchor = (0.0, 0.0)
    beside_near = _score_for("the building beside the church", near_candidate,
                             anchor)
    beside_far = _score_for("the building beside the church", far_candidate, anchor)
    assert beside_near > beside_far
    farfrom_near = _score_for("the building far from the church", near_candidate,
                              anchor)
    farfrom_far = _score_for("the building far from the church", far_candidate,
                             anchor)
    assert farfrom_far > farfrom_near


def test_changing_the_anchor_entity_changes_the_scores():
    """Otherwise the model is reading the target's words, not the anchor's."""
    nodes = _nodes()
    near_church = lexical_anchor_scores(
        parse_instruction("the building beside the church")["anchors"][0]["phrase"],
        nodes, "Building")
    near_road = lexical_anchor_scores(
        parse_instruction("the building beside the road")["anchors"][0]["phrase"],
        nodes, "TrafficRoad")
    assert int(np.argmax(near_church)) == 0
    assert int(np.argmax(near_road)) == 2
    assert not np.allclose(near_church, near_road)


def test_a_relation_with_no_opposite_is_symmetric_in_sign():
    geometry = relation_geometry([5.0, 5.0], [0.0, 0.0], [0.0, -50.0], 0.0)
    mirrored = relation_geometry([-5.0, -5.0], [0.0, 0.0], [0.0, -50.0], 0.0)
    beside = {"phrase": "beside", "family": "proximity", "frame": "none"}
    assert oriented_relation_score(geometry, beside) == pytest.approx(
        oriented_relation_score(mirrored, beside), rel=1e-6)


def test_parser_never_returns_an_attribute_as_an_anchor():
    parsed = parse_instruction("the long white building beside the church")
    for anchor in parsed["anchors"]:
        assert "white" not in anchor["phrase"]
        assert "long" not in anchor["phrase"]
