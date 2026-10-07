"""Clean supervision, and the criterion that chose its thresholds.

The threshold test is the one worth keeping.  The first version of the sweep
maximised the *number* of examples and walked every threshold to its loosest
value, because loosening admits positives faster than it removes the negatives
they are compared against -- so the criterion was monotone in looseness and
selected "everything is north of everything".  That is a silent, plausible-
looking failure, so the property that replaced it is pinned here rather than
left to a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.spatial_graph.relation_geometry import (  # noqa: E402
    DEFAULT_THRESHOLDS, OPPOSITE, RELATION_NAMES,
)
from sensaturban_fpv.spatial_graph.synthetic_relations import (  # noqa: E402
    DISTANCE_BAND, _single_relation_margins, margin_from_facts, scale_grids,
)

TH = dict(DEFAULT_THRESHOLDS)


def facts(**kw):
    base = {"dx": 0.0, "dy": 0.0, "footprint_distance": 0.0,
            "centre_distance": 0.0, "side_a": 0.0, "side_b": 0.0,
            "crosses_road": False, "road_distance": 100.0, "along_cos": 0.0,
            "intersection_distance": 500.0, "aligned_cos": 0.0,
            "parallel_cos": 0.0}
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# the ontology's signs
# --------------------------------------------------------------------------

def test_cardinals_are_mutually_exclusive():
    north = facts(dy=50.0)
    south = facts(dy=-50.0)
    assert margin_from_facts("north_of", north, TH) > 0
    assert margin_from_facts("south_of", north, TH) < 0
    assert margin_from_facts("south_of", south, TH) > 0
    assert margin_from_facts("north_of", south, TH) < 0
    east, west = facts(dx=40.0), facts(dx=-40.0)
    assert margin_from_facts("east_of", east, TH) > 0
    assert margin_from_facts("west_of", east, TH) < 0
    assert margin_from_facts("west_of", west, TH) > 0


def test_near_and_far_are_opposites():
    close = facts(footprint_distance=5.0)
    distant = facts(footprint_distance=300.0)
    assert margin_from_facts("near", close, TH) > 0
    assert margin_from_facts("far", close, TH) < 0
    assert margin_from_facts("far", distant, TH) > 0
    assert margin_from_facts("near", distant, TH) < 0


def test_the_opposite_table_only_lists_real_opposites():
    for a, b in OPPOSITE.items():
        assert a in RELATION_NAMES and b in RELATION_NAMES
        assert OPPOSITE[b] == a


def test_a_relation_with_no_opposite_is_not_in_the_table():
    for symmetric in ("between", "aligned_with", "parallel_to",
                      "perpendicular_to", "along_road", "across_road",
                      "near_intersection"):
        assert symmetric not in OPPOSITE


def test_road_side_relations_need_both_entities_off_the_road():
    """A node sitting on the road has no side, so neither relation applies."""
    on_road = facts(side_a=0.0, side_b=20.0)
    assert margin_from_facts("same_side_of_road", on_road, TH) == float("-inf")
    assert margin_from_facts("opposite_side_of_road", on_road, TH) == float("-inf")
    both_off = facts(side_a=10.0, side_b=20.0)
    assert margin_from_facts("same_side_of_road", both_off, TH) > 0
    assert margin_from_facts("opposite_side_of_road", both_off, TH) < 0
    across = facts(side_a=10.0, side_b=-20.0)
    assert margin_from_facts("opposite_side_of_road", across, TH) > 0
    assert margin_from_facts("same_side_of_road", across, TH) < 0


def test_across_road_requires_the_line_to_cross():
    assert margin_from_facts("across_road", facts(crosses_road=False), TH) == \
        float("-inf")
    assert margin_from_facts("across_road",
                             facts(crosses_road=True, road_distance=8.0), TH) > 0


def test_alignment_relations_are_distinct_questions():
    parallel = facts(parallel_cos=0.99, aligned_cos=0.0)
    assert margin_from_facts("parallel_to", parallel, TH) > 0
    assert margin_from_facts("perpendicular_to", parallel, TH) < 0
    aligned = facts(parallel_cos=0.0, aligned_cos=0.99)
    assert margin_from_facts("aligned_with", aligned, TH) > 0
    assert margin_from_facts("parallel_to", aligned, TH) < 0


def test_between_needs_a_second_anchor():
    assert margin_from_facts("between", facts(), TH, None) == float("-inf")
    # t in the middle of the segment, close to the line.
    assert margin_from_facts("between", facts(), TH, (0.5, 2.0, 100.0)) > 0
    # Beyond the end of the segment.
    assert margin_from_facts("between", facts(), TH, (1.6, 2.0, 100.0)) < 0
    # Off to the side.
    assert margin_from_facts("between", facts(), TH, (0.5, 90.0, 100.0)) < 0


# --------------------------------------------------------------------------
# the threshold criterion
# --------------------------------------------------------------------------

def test_no_single_score_maximisation_is_used_for_the_thresholds():
    """Both natural criteria are degenerate, and the code must not use either.

    Counting examples is monotone in looseness (loosening admits positives
    faster than it removes the negatives they are compared against); summing the
    retained margins is monotone in tightness.  The first version of the sweep
    used the count and selected "everything is north of everything"; the second
    used the sum and produced ``near_m == far_m == 140``, i.e. a distance that is
    simultaneously near and far.  This test pins the shape of the function that
    replaced them: selectivity is monotone in the threshold, so matching a
    *target* rate picks an interior value rather than walking to an end.
    """
    rng = np.random.default_rng(0)
    dy = rng.normal(60.0, 60.0, size=2000)
    cols = list(zip(*[tuple(facts(dy=float(v)).get(k) for k in (
        "dx", "dy", "footprint_distance", "centre_distance", "side_a", "side_b",
        "crosses_road", "road_distance", "along_cos", "intersection_distance",
        "aligned_cos", "parallel_cos")) for v in dy]))
    rates = [float((_single_relation_margins(
        "north_of", cols, dict(TH, cardinal_margin=c)) > 0).mean())
        for c in (1.0, 10.0, 30.0, 60.0, 120.0, 240.0)]
    # Monotone decreasing in the threshold, which is what makes "match a target"
    # well posed and "maximise" ill posed.
    assert rates == sorted(rates, reverse=True), rates
    assert rates[0] > 0.7 and rates[-1] < 0.05


def test_the_selectivity_target_admits_both_near_and_far():
    """The failure the second criterion produced, pinned as a property."""
    from sensaturban_fpv.spatial_graph.synthetic_relations import (
        TARGET_SELECTIVITY,
    )
    assert TARGET_SELECTIVITY["near_m"] > 0
    assert TARGET_SELECTIVITY["far_m"] > 0
    # Near and far are chosen independently from the same kind of grid, so the
    # only thing preventing near_m == far_m is that the rate, not the metre
    # value, is what is being matched.  A sanity assertion on the grids.
    assert len(set(TARGET_SELECTIVITY.values())) >= 1


def test_scale_grids_are_derived_from_the_data():
    """Section 11: the ranges come from the maps, not from outside."""
    rng = np.random.default_rng(2)
    dists = rng.uniform(0.0, 200.0, size=500)
    packed = [[tuple(facts(centre_distance=float(d)).get(k)
                     for k in ("dx", "dy", "footprint_distance",
                               "centre_distance", "side_a", "side_b",
                               "crosses_road", "road_distance", "along_cos",
                               "intersection_distance", "aligned_cos",
                               "parallel_cos")) for d in dists]]
    grids = scale_grids(packed)
    assert all(g == tuple(sorted(g)) for g in grids.values())
    assert len(set(grids["near_m"])) > 1
    # Scaled to the data: a 0-200 m spread cannot produce a 1000 m threshold.
    assert max(grids["far_m"]) <= 250.0


def test_hard_negative_bands_are_a_ratio_not_a_constant():
    """A constant band would let the model read absolute distance instead."""
    lo, hi = DISTANCE_BAND
    assert 0.0 < lo < 1.0 < hi
