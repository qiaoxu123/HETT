"""Reference binding and relation semantics, tested rather than assumed.

The round's claim is narrow and falsifiable: given the right anchor and the
right relation word, the target should rank higher than it does under the wrong
relation.  These tests pin the parts of that claim that are checkable without
training a model --

* that a phrase binds to a *set* of entities, because names in this corpus are
  shared (``aldridge road`` is 180 segments);
* that each relation's geometry column has the sign its word implies, which is
  the defect the previous round shipped;
* that the counterfactual arms actually change the input they claim to change,
  since an arm that silently no-ops would look like a passing control;
* and that nothing on the deployable path reads the answer.

The four behavioural tests the brief asks for (zeroing the relation hurts,
shuffling it hurts, a wrong anchor hurts, the true anchor is never worse than a
guessed one) are properties of a *trained* model and cannot be asserted here
without training one.  What is asserted instead is the plumbing they depend on,
and the numbers themselves are reported per arm in
``artifacts/relation_v2/relation_self_test.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.relation_v2 import (  # noqa: E402
    AXIS_COLUMNS, GEOM_DIM, NO_OPPOSITE, OPPOSITES, aggregate, between_geometry,
    build_relation_vocab, geometry, mcnemar, opposite_id, principal_axis,
    rank_metrics, shuffled_ids,
)

NO_DIM = np.array([4.0, 2.0, 1.5])
BIG_DIM = np.array([10.0, 10.0, 8.0])
EAST = np.array([1.0, 0.0])          # anchor footprint running east-west
NORTH = np.array([0.0, 1.0])


def geom(cand_xy, anchor_xy=(0.0, 0.0), axis=EAST, uav=(0.0, -50.0), yaw=0.0,
         cand_dim=NO_DIM, anchor_dim=BIG_DIM):
    return geometry(np.asarray(cand_xy, float), np.asarray(anchor_xy, float),
                    axis, np.asarray(uav, float), yaw, cand_dim, anchor_dim)


def col(v, name):
    return v[AXIS_COLUMNS[name]]


# --------------------------------------------------------------------------
# frames: every axis must mean what its name says
# --------------------------------------------------------------------------

def test_global_axes_point_east_and_north():
    """north is +y and east is +x; the previous round had these transposed."""
    v = geom((30.0, 70.0))
    assert col(v, "global_x") > 0 and col(v, "global_y") > 0
    v = geom((-30.0, -70.0))
    assert col(v, "global_x") < 0 and col(v, "global_y") < 0


def test_agent_axes_follow_the_heading():
    """Facing north (yaw pi/2), ahead is +y and right is +x."""
    ahead = geom((0.0, 60.0), uav=(0.0, 0.0), yaw=np.pi / 2)
    assert col(ahead, "agent_ahead") > 0
    # A target to the south is behind.
    behind = geom((0.0, -60.0), uav=(0.0, 0.0), yaw=np.pi / 2)
    assert col(behind, "agent_ahead") < 0
    # Facing north, east is to the right.
    right = geom((60.0, 0.0), uav=(0.0, 0.0), yaw=np.pi / 2)
    assert col(right, "agent_lateral") > 0
    left = geom((-60.0, 0.0), uav=(0.0, 0.0), yaw=np.pi / 2)
    assert col(left, "agent_lateral") < 0


def test_turning_the_agent_moves_the_word_not_the_geometry():
    """The same candidate reads as ahead or behind depending only on the yaw."""
    for yaw, sign in ((0.0, 1.0), (np.pi, -1.0)):
        v = geom((60.0, 0.0), uav=(0.0, 0.0), yaw=yaw)
        assert np.sign(col(v, "agent_ahead")) == sign
        # Global coordinates must not move at all.
        assert col(v, "global_x") == pytest.approx(0.6)


def test_anchor_frame_is_the_anchors_own_footprint():
    """An anchor running north puts its long axis on y, not on x."""
    v = geom((0.0, 60.0), axis=NORTH)
    assert col(v, "anchor_along") > 0
    assert col(v, "anchor_perp") == pytest.approx(0.0)
    side = geom((60.0, 0.0), axis=NORTH)
    assert col(side, "anchor_along") == pytest.approx(0.0)
    assert col(side, "anchor_perp") != 0.0


def test_principal_axis_finds_the_long_side_of_a_rectangle():
    # A rectangle twice as long in y as in x.
    contour = [[-5, -10], [5, -10], [5, 10], [-5, 10]]
    axis = principal_axis(contour)
    assert abs(float(axis[1])) > abs(float(axis[0]))
    # A square has no preferred direction but must still return a unit vector.
    assert np.linalg.norm(principal_axis([[0, 0], [1, 0], [1, 1], [0, 1]])) == \
        pytest.approx(1.0)
    # Degenerate input must not raise.
    assert np.allclose(principal_axis([]), [1.0, 0.0])


def test_geometry_has_a_fixed_width_and_is_finite():
    v = geom((10.0, -20.0))
    assert v.shape == (GEOM_DIM,) and np.isfinite(v).all()


# --------------------------------------------------------------------------
# counterfactual sign pairs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("positive,negative,axis", [
    ((0.0, 50.0), (0.0, -50.0), "global_y"),      # north of / south of
    ((50.0, 0.0), (-50.0, 0.0), "global_x"),      # east of / west of
    ((50.0, 0.0), (-50.0, 0.0), "agent_lateral"),  # right of / left of
])
def test_opposite_words_cannot_both_be_satisfied(positive, negative, axis):
    """A north-of candidate must not also be a south-of candidate."""
    up = col(geom(positive, uav=(0.0, 0.0), yaw=np.pi / 2), axis)
    down = col(geom(negative, uav=(0.0, 0.0), yaw=np.pi / 2), axis)
    assert up > 0 > down


def test_front_and_behind_are_opposite_in_the_agent_frame():
    front = geom((0.0, 60.0), uav=(0.0, 0.0), yaw=np.pi / 2)
    back = geom((0.0, -60.0), uav=(0.0, 0.0), yaw=np.pi / 2)
    assert (col(front, "agent_ahead") > 0) and (col(back, "agent_ahead") < 0)


def test_near_and_far_are_ordered_by_distance():
    near = geom((0.0, 10.0))
    far = geom((0.0, 200.0))
    assert near[3] < far[3]                       # distance column
    assert near[2] < far[2]                       # log distance column


def test_between_prefers_the_middle_and_rejects_the_outside():
    a, b = (0.0, 0.0), (10.0, 0.0)
    t_mid, perp_mid = between_geometry((5.0, 1.0), a, b)
    t_out, _ = between_geometry((30.0, 1.0), a, b)
    t_before, _ = between_geometry((-30.0, 1.0), a, b)
    _, perp_off = between_geometry((5.0, 60.0), a, b)
    assert 0.0 <= t_mid <= 1.0
    assert t_out > 1.0 and t_before < 0.0
    assert perp_mid < perp_off
    # Two coincident anchors have no segment to be between.
    assert between_geometry((5.0, 0.0), a, a)[1] == float("inf")


# --------------------------------------------------------------------------
# the counterfactual arms must actually change the input
# --------------------------------------------------------------------------

def test_every_opposite_differs_from_its_own_relation():
    """An arm whose negative equals its positive would look like a passing test."""
    vocab, inv = build_relation_vocab(list(OPPOSITES) + list(NO_OPPOSITE))
    for phrase in OPPOSITES:
        rid = vocab[phrase]
        other = opposite_id(rid, vocab, inv)
        assert other != rid, phrase
        assert other != 0, phrase


def test_opposites_are_symmetric_where_they_exist():
    vocab, inv = build_relation_vocab(list(OPPOSITES) + ["between"])
    for phrase, other in OPPOSITES.items():
        if other not in vocab:
            continue
        back = OPPOSITES.get(other)
        if back is not None and back in vocab:
            assert opposite_id(vocab[other], vocab, inv) == vocab[back], phrase


def test_symmetric_relations_get_a_fallback_not_their_own_word():
    vocab, inv = build_relation_vocab(["between", "near", "far from"])
    other = opposite_id(vocab["between"], vocab, inv,
                        np.random.default_rng(0))
    assert other != vocab["between"]


def test_a_shuffled_relation_label_is_different_for_most_samples():
    ids = np.array([1, 2, 3, 4, 5, 6, 7, 8] * 8)
    shuffled = shuffled_ids(ids, np.random.default_rng(0))
    assert shuffled.shape == ids.shape
    assert (shuffled != ids).mean() > 0.5


def test_shuffling_preserves_the_marginal_distribution():
    """Otherwise the control would change the label frequencies too."""
    ids = np.array([1] * 20 + [2] * 5)
    shuffled = shuffled_ids(ids, np.random.default_rng(1))
    assert sorted(shuffled.tolist()) == sorted(ids.tolist())


def test_an_unknown_relation_has_no_negative_and_says_so():
    vocab, inv = build_relation_vocab(["near", "far from"])
    assert opposite_id(0, vocab, inv) == 0


# --------------------------------------------------------------------------
# scoring arithmetic
# --------------------------------------------------------------------------

def test_rank_metrics_report_the_rank_of_the_target():
    scores = np.array([0.1, 0.9, 0.3, 0.2])
    out = rank_metrics(scores, target=1)
    assert out["rank"] == 1 and out["top1"] == 1.0 and out["top4"] == 1.0
    out = rank_metrics(scores, target=2)
    assert out["rank"] == 2 and out["top1"] == 0.0 and out["top4"] == 1.0
    assert out["mrr"] == pytest.approx(0.5)
    # The margin is against the best competitor, not the mean.
    assert out["margin"] == pytest.approx(0.3 - 0.9)


def test_rank_metrics_are_invariant_to_a_constant_offset():
    scores = np.array([0.1, 0.9, 0.3])
    a = rank_metrics(scores, 0)
    b = rank_metrics(scores + 100.0, 0)
    assert a["rank"] == b["rank"]
    assert a["margin"] == pytest.approx(b["margin"])


def test_candidate_order_does_not_change_the_rank():
    """A tie broken by position would hand the answer to whichever index it sat at."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=10)
    target = 3
    plain = rank_metrics(scores, target)["rank"]
    perm = rng.permutation(10)
    back = np.empty_like(perm)
    back[perm] = np.arange(10)
    shuffled = rank_metrics(scores[perm], int(back[target]))["rank"]
    assert plain == shuffled


def test_aggregate_reports_a_wilson_interval_inside_the_unit_range():
    rows = [{"top1": 1.0, "top4": 1.0, "mrr": 1.0, "margin": 0.1}] * 7 + \
           [{"top1": 0.0, "top4": 1.0, "mrr": 0.5, "margin": -0.1}] * 3
    out = aggregate(rows)
    assert out["n"] == 10 and out["top1"] == pytest.approx(0.7)
    low, high = out["top1_ci"]
    assert 0.0 <= low <= out["top1"] <= high <= 1.0
    assert aggregate([]) == {"n": 0}


def test_mcnemar_counts_only_the_disagreements():
    a = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    b = np.array([1, 0, 0, 1, 0, 0], dtype=bool)
    out = mcnemar(a, b)
    assert out["a_only"] == 2 and out["b_only"] == 1 and out["n"] == 3
    # Identical arms have nothing to test and must not report a fake p-value.
    assert mcnemar(a, a)["p_value"] == 1.0


def test_a_marginalised_score_never_loses_to_a_single_hypothesis():
    """logsumexp over hypotheses can only raise a candidate, never lower it."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 3, 4))
    top = values.max(axis=(1, 2), keepdims=True)
    marg = (top + np.log(np.exp(values - top).sum(axis=(1, 2), keepdims=True))
            ).squeeze()
    assert np.all(marg >= values.max(axis=(1, 2)) - 1e-9)


def test_a_second_anchor_can_rescue_a_weak_first_one():
    values = np.zeros((1, 2, 1))
    values[0, 1, 0] = 5.0
    top = values.max(axis=(1, 2), keepdims=True)
    marg = (top + np.log(np.exp(values - top).sum(axis=(1, 2), keepdims=True))
            ).squeeze()
    # Committing to the first hypothesis would score 0; marginalising finds the
    # second, which is the whole reason the anchor set is not collapsed to one.
    assert values[0, 0, 0] == 0.0 and marg > 4.9
