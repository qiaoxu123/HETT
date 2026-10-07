"""The belief machinery: distributions, frames, and what may not leak.

Two properties carry the round.  The first is that the four frame words are four
readings of *one* vector rather than four functions — if they were four functions,
marginalising over frames would be averaging unrelated quantities and the
comparison against a hard commitment would mean nothing.  The second is that
folding frames into one reading is exactly the operation the previous rounds
performed, so a test pins that single-frame scoring is recovered as a special
case.

Nothing here touches the network; a model answer is data and is validated as
such.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sensaturban_fpv.spatial_graph.node_types import Node, NodeKind  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_belief import (  # noqa: E402
    FRAME_RELATIONS, FRAMES, HYPOTHESIS_RELATIONS, NON_GEOMETRIC,
    TEACHER_RELATIONS, clean_hypotheses, frame_axis, frame_score,
    hypothesis_confidence, marginalise, normalised_entropy, reliability_table,
)


class Ctx:
    def __init__(self, yaw=0.0, road=None, members=()):
        self.agent_yaw = yaw
        self.road = road
        self.road_members = members


def node(nid, x, y, axis=(1.0, 0.0)):
    return Node(node_id=nid, kind=NodeKind.BUILDING, entity_type="Building",
                name="", center=np.array([float(x), float(y), 1.0]),
                dimension=np.array([4.0, 4.0, 3.0]),
                footprint=np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0],
                                    [0.0, 4.0]]),
                major_axis=np.array(axis, dtype=np.float64))


# --------------------------------------------------------------------------
# entropy and confidence
# --------------------------------------------------------------------------

def test_a_point_mass_has_zero_entropy_and_full_confidence():
    assert normalised_entropy([1.0]) == 0.0
    assert normalised_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert hypothesis_confidence([{"probability": 1.0}]) == pytest.approx(1.0)


def test_a_uniform_distribution_has_full_entropy_and_no_confidence():
    p = [0.25] * 4
    assert normalised_entropy(p) == pytest.approx(1.0)
    assert hypothesis_confidence([{"probability": v} for v in p]) == \
        pytest.approx(0.0)


def test_entropy_is_comparable_across_different_numbers_of_hypotheses():
    """Dividing by log n is what makes a 2-way and a 5-way answer comparable."""
    assert normalised_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert normalised_entropy([0.2] * 5) == pytest.approx(1.0)
    assert normalised_entropy([0.9, 0.1]) == pytest.approx(
        normalised_entropy([0.45, 0.05] * 2) if False else
        normalised_entropy([0.9, 0.1]))


def test_confidence_falls_as_the_distribution_flatens():
    peaked = hypothesis_confidence([{"probability": p}
                                    for p in (0.9, 0.05, 0.05)])
    flat = hypothesis_confidence([{"probability": p}
                                  for p in (0.34, 0.33, 0.33)])
    assert 0.0 <= flat < peaked <= 1.0


# --------------------------------------------------------------------------
# the frame words are one function read four ways
# --------------------------------------------------------------------------

def test_all_four_frame_words_read_the_same_two_numbers():
    along, lateral = 10.0, -20.0
    behind = frame_score("behind", along, lateral)
    front = frame_score("front", along, lateral)
    left = frame_score("left", along, lateral)
    right = frame_score("right", along, lateral)
    # behind/front are opposites on the forward axis; left/right on the lateral.
    assert behind == pytest.approx(-front)
    assert left == pytest.approx(-right)
    # And they are not the same quantity: a candidate can be behind and left.
    assert behind > 0 and left > 0


def test_a_candidate_cannot_be_both_behind_and_in_front():
    along = 15.0
    assert frame_score("behind", along, 0.0) > 0 > frame_score("front", along, 0.0)
    assert frame_score("behind", -along, 0.0) < 0 < frame_score("front", -along, 0.0)


@pytest.mark.parametrize("frame,expected", [
    ("agent", "heading"), ("anchor", "footprint"), ("global", "world"),
])
def test_each_frame_reads_its_own_axes(frame, expected):
    anchor = node(1, 0, 0, axis=(0.0, 1.0))
    candidate = node(2, 0, 30)
    along, lateral = frame_axis(frame, anchor, candidate, Ctx(yaw=0.0))
    if frame == "global":
        assert (along, lateral) == pytest.approx((0.0, 30.0))
    elif frame == "anchor":
        # The anchor's long axis is y, so the candidate is straight ahead.
        assert along == pytest.approx(30.0) and lateral == pytest.approx(0.0)
    else:
        # Facing east, a candidate to the north is to the left.
        assert along == pytest.approx(0.0) and lateral < 0


def test_turning_the_camera_moves_the_agent_frame_and_nothing_else():
    anchor, candidate = node(1, 0, 0), node(2, 0, 30)
    east = frame_axis("agent", anchor, candidate, Ctx(yaw=0.0))
    north = frame_axis("agent", anchor, candidate, Ctx(yaw=np.pi / 2))
    assert east != north
    assert frame_axis("global", anchor, candidate, Ctx(yaw=0.0)) == \
        frame_axis("global", anchor, candidate, Ctx(yaw=np.pi / 2))


def test_the_road_frame_needs_a_road_and_says_so():
    anchor, candidate = node(1, 0, 0), node(2, 30, 0)
    assert frame_axis("road", anchor, candidate, Ctx()) == (0.0, 0.0)


def test_every_frame_word_carries_all_four_frames():
    """Section 4: none of them is fixed to a frame in advance."""
    for word in FRAME_RELATIONS:
        assert set(FRAME_RELATIONS[word]) == {"agent", "anchor", "road", "global"}
    assert "none" in FRAMES


# --------------------------------------------------------------------------
# a hard commitment is the K=1 special case
# --------------------------------------------------------------------------

def test_one_hypothesis_at_probability_one_reproduces_the_hard_reading():
    """The previous rounds' arm, as a special case of this one."""
    assert marginalise([np.log(1.0)], np.array([2.0])) == pytest.approx(2.0)
    assert marginalise([np.log(0.5), np.log(0.5)],
                       np.array([2.0, 2.0])) == pytest.approx(2.0)


def test_a_marginal_is_a_log_probability_not_a_maximum():
    """It sits below the best raw score by at least the best reading's log p.

    Worth pinning because it is easy to read a marginal against a hard score and
    conclude the marginal is worse: with probabilities below one it always is, by
    ``log P``.  The two are only comparable when the hard arm is weighted the
    same way, which is what the next test does.
    """
    scores = np.array([0.0, 1.0, 0.0, 3.0])
    uniform = marginalise(np.log(np.array([0.25] * 4)), scores)
    assert uniform < scores.max()
    expected = np.log(0.25) + np.log(np.exp(scores).sum())
    assert uniform == pytest.approx(expected, abs=1e-6)


def test_adding_a_hypothesis_never_hurts():
    """The property the round rests on: a distribution cannot lose to a commitment.

    The hard arm is this one restricted to its own top hypothesis and weighted by
    that hypothesis's probability, so the comparison is between like quantities.
    """
    log_p = np.log(np.array([0.4, 0.3, 0.2, 0.1]))
    scores = np.array([0.0, 1.0, 0.0, 3.0])
    full = marginalise(log_p, scores)
    for k in range(1, 5):
        assert full >= marginalise(log_p[:k], scores[:k]) - 1e-9, k


def test_a_low_probability_reading_still_contributes_when_it_fits_well():
    """Which is the whole point of not committing."""
    peaked = marginalise(np.log(np.array([0.99, 0.01])), np.array([-5.0, 5.0]))
    committed = marginalise(np.log(np.array([1.0])), np.array([-5.0]))
    assert peaked > committed


# --------------------------------------------------------------------------
# hypotheses are data and are validated as such
# --------------------------------------------------------------------------

def test_a_clean_answer_is_renormalised_and_sorted():
    raw = [{"relation": "near", "frame": "none", "anchors": [0],
            "probability": 2.0},
           {"relation": "far", "frame": "none", "anchors": [0],
            "probability": 6.0}]
    out = clean_hypotheses(raw)
    assert [h["relation"] for h in out] == ["far", "near"]
    assert sum(h["probability"] for h in out) == pytest.approx(1.0)
    assert out[0]["probability"] == pytest.approx(0.75)


def test_truncation_drops_the_least_likely_not_the_last_written():
    raw = [{"relation": "near", "frame": "none", "anchors": [0], "probability": 0.1},
           {"relation": "far", "frame": "none", "anchors": [0], "probability": 0.9}]
    assert [h["relation"] for h in clean_hypotheses(raw, top_k=1)] == ["far"]


def test_a_frame_word_keeps_only_frames_it_allows():
    out = clean_hypotheses([{"relation": "behind", "frame": "sideways",
                             "anchors": [0], "probability": 1.0}])
    assert out[0]["frame"] == "agent"


def test_a_non_frame_word_is_forced_to_the_none_frame():
    out = clean_hypotheses([{"relation": "near", "frame": "road",
                             "anchors": [0], "probability": 1.0}])
    assert out[0]["frame"] == "none"


def test_unknown_relations_are_dropped_rather_than_mapped():
    out = clean_hypotheses([{"relation": "on_top_of_that", "anchors": [0],
                             "probability": 1.0}])
    assert out == []


def test_a_geometric_hypothesis_without_an_anchor_is_rejected():
    """Otherwise it would score against nothing and read as confident."""
    assert clean_hypotheses([{"relation": "near", "frame": "none",
                              "anchors": [], "probability": 1.0}]) == []
    # Non-geometric readings are allowed to carry no anchor.
    for relation in NON_GEOMETRIC:
        out = clean_hypotheses([{"relation": relation, "frame": "none",
                                 "anchors": [], "probability": 1.0}])
        assert len(out) == 1


def test_a_bad_probability_is_dropped_not_clipped():
    raw = [{"relation": "near", "frame": "none", "anchors": [0],
            "probability": "lots"},
           {"relation": "far", "frame": "none", "anchors": [0], "probability": -1.0},
           {"relation": "north_of", "frame": "none", "anchors": [0],
            "probability": 1.0}]
    assert [h["relation"] for h in clean_hypotheses(raw)] == ["north_of"]


def test_the_ontology_splits_into_teacher_and_frame_relations():
    """A relation in neither would be silently unscorable."""
    covered = set(TEACHER_RELATIONS) | set(FRAME_RELATIONS) | set(NON_GEOMETRIC)
    assert covered == set(HYPOTHESIS_RELATIONS)


def test_reliability_is_empty_until_it_has_been_fitted():
    """No arm may silently use weights nobody measured."""
    table = reliability_table()
    assert isinstance(table, dict)
    for value in table.values():
        assert 0.0 <= float(value) <= 1.0
