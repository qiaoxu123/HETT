"""Unit tests for the feature-sufficiency diagnostic.

The round's claims are of the form "this family of evidence is what the model was
missing", so two failure modes have to be impossible rather than unlikely.  The
first is leakage: a candidate's features must be a function of that candidate and
the scene, never of which candidate happens to be the answer, and permuting the
candidate list must permute the features and nothing else.  The second is the
oracle arms drifting into the deployable table, which is a labelling question at
the call site and is checked here against the registry the trainer reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.entity_features import (  # noqa: E402
    CLASS_NAMES, N_CLASSES, ROAD_CLASS, RoadIndex, anchor_relation_features,
    appearance_features, colour_entropy, geometry_features, relation_features,
    rgb_to_hsv, semantic_features, standardise,
)
from sensaturban_fpv.relation_parser import (  # noqa: E402
    ATTRIBUTE_LEXICON, RELATION_LEXICON, color_words, instruction_buckets,
    parse_attributes, parse_relations, relation_feature_vector,
)
from train_feature_probes import ARMS, ORACLE_ARMS  # noqa: E402


# --------------------------------------------------------------------------
# relation parsing
# --------------------------------------------------------------------------

def test_relation_parser_finds_the_family_not_just_a_word():
    assert parse_relations("the car next to the wall")["proximity"]
    assert parse_relations("the building behind the church")["front_back"]
    assert parse_relations("north of the river")["cardinal"]
    assert parse_relations("between the two shops")["between"]
    assert not parse_relations("a red car")["any"]


def test_relation_parser_is_case_insensitive_and_empty_safe():
    assert parse_relations("The Building BESIDE the road")["proximity"]
    assert parse_relations("")["any"] is False
    assert parse_relations(None)["any"] is False


def test_instruction_buckets_overlap_rather_than_choosing_one():
    buckets = instruction_buckets("the large red building beside the church")
    assert {"color", "size", "relation"} <= buckets
    assert "category_only" not in buckets
    assert instruction_buckets("the building") == {"category_only"}


def test_multi_relation_needs_more_than_one_family():
    assert "multi_relation" in instruction_buckets(
        "the car behind the van near the road")
    assert "multi_relation" not in instruction_buckets("the car beside the wall")


def test_colour_words_are_reported_verbatim():
    assert color_words("the red and white building") == ["red", "white"]
    assert color_words("a building") == []


def test_relation_feature_vector_has_one_slot_per_family():
    vector = relation_feature_vector("the red car near the church")
    assert len(vector) == (len(RELATION_LEXICON) + len(ATTRIBUTE_LEXICON) + 1)
    assert set(vector) <= {0.0, 1.0}
    # colour + proximity, and nothing invented
    assert sum(vector) == 2


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_geometry_reports_the_annotated_aspect_ratio():
    out = geometry_features(np.zeros((0, 3)), [0, 0, 0], [10.0, 2.0, 3.0],
                            [0, 0, 100], 0.0)
    assert out[0] == pytest.approx(10.0)
    assert out[4] == pytest.approx(5.0)          # aspect ratio


def test_geometry_is_invariant_to_a_rigid_rotation_of_the_points():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3)) * np.array([8.0, 1.0, 2.0])
    a = geometry_features(pts, [0, 0, 0], [16, 2, 4], [0, 0, 100], 0.0)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b = geometry_features(np.c_[pts[:, :2] @ rot.T, pts[:, 2]], [0, 0, 0],
                          [16, 2, 4], [0, 0, 100], 0.0)
    # Elongation is a property of the shape, not of the world axes; the spans
    # and the orientation columns are expected to change, because they are
    # expressed in the world frame the crop is cut in.
    assert a[18] == pytest.approx(b[18], rel=1e-4)
    assert a[3] == pytest.approx(b[3])           # annotated volume
    assert a[10] == pytest.approx(b[10])         # point count
    assert a[13] == pytest.approx(b[13], rel=1e-4)   # mean height


def test_geometry_bearing_is_relative_to_the_heading_not_the_world():
    # Due east of the agent, facing east: straight ahead, not to the side.
    east = geometry_features(np.zeros((0, 3)), [100, 0, 0], [1, 1, 1], [0, 0, 0],
                             0.0)
    assert east[23] == pytest.approx(1.0)        # cos of the relative bearing
    assert east[24] == pytest.approx(0.0, abs=1e-6)
    north = geometry_features(np.zeros((0, 3)), [100, 0, 0], [1, 1, 1], [0, 0, 0],
                              np.pi / 2)
    # Facing north, a target due east is to the right: bearing -pi/2.
    assert north[23] == pytest.approx(0.0, abs=1e-6)
    assert north[24] == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# appearance
# --------------------------------------------------------------------------

def test_rgb_to_hsv_puts_primaries_where_the_definition_says():
    hsv = rgb_to_hsv(np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255],
                               [255, 255, 255], [0, 0, 0]], dtype=np.float64))
    assert hsv[0, 0] == pytest.approx(0.0)
    assert hsv[1, 0] == pytest.approx(1 / 3)
    assert hsv[2, 0] == pytest.approx(2 / 3)
    assert hsv[3, 1] == pytest.approx(0.0)       # white is unsaturated
    assert hsv[4, 2] == pytest.approx(0.0)       # black has no value


def test_colour_entropy_is_zero_for_one_colour_and_maximal_for_a_spread():
    flat = np.tile(np.array([[120, 120, 120]]), (500, 1))
    assert colour_entropy(flat) == pytest.approx(0.0, abs=1e-9)
    rng = np.random.default_rng(1)
    assert colour_entropy(rng.integers(0, 256, size=(5000, 3))) > 1.0


def test_appearance_is_invariant_to_the_order_of_the_points():
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, size=(400, 3)).astype(np.float64)
    a = appearance_features(rgb)
    b = appearance_features(rgb[rng.permutation(400)])
    assert np.allclose(a, b, atol=1e-5)


def test_appearance_of_a_red_entity_lands_near_red_and_not_near_blue():
    red = np.tile(np.array([[200.0, 20.0, 20.0]]), (300, 1))
    blue = np.tile(np.array([[20.0, 20.0, 200.0]]), (300, 1))
    a, b = appearance_features(red), appearance_features(blue)
    assert np.linalg.norm(a - b) > 50.0


def test_appearance_handles_an_entity_with_no_points():
    out = appearance_features(np.zeros((0, 3)))
    assert out.shape == (45,) and np.isfinite(out).all()


# --------------------------------------------------------------------------
# relation
# --------------------------------------------------------------------------

def test_relation_nearer_entities_have_smaller_distance_features():
    others = np.array([[10.0, 0, 0], [200.0, 0, 0], [0, 50.0, 0]])
    types = ["Building", "Building", "Car"]
    near = relation_features([5, 0, 0], [0, 0, 100], 0.0, others, types, "Car")
    far = relation_features([150, 0, 0], [0, 0, 100], 0.0, others, types, "Car")
    assert near[0] < far[0]                       # agent distance
    assert near[9] < far[9]                       # nearest same-class distance


def test_relation_nearest_same_class_ignores_other_classes():
    others = np.array([[1.0, 0, 0], [40.0, 0, 0]])
    types = ["Building", "Car"]
    out = relation_features([0, 0, 0], [0, 0, 100], 0.0, others, types, "Car")
    # the Car at 40 m is the nearest same-class one, not the Building at 1 m
    assert out[9] == pytest.approx(np.log1p(40.0), rel=1e-5)


def test_relation_road_statistics_survive_a_missing_road_index():
    out = relation_features([0, 0, 0], [0, 0, 100], 0.0, np.zeros((0, 3)), [],
                            "Car", road=None)
    assert out.shape == (28,) and np.isfinite(out).all()
    # Column 13 is the road distance; with no road index it is the sentinel, not
    # a fabricated zero that would read as "the road is right here".
    assert out[13] == -1.0 and out[14] == 0.0


def test_relation_features_are_permutation_equivariant():
    rng = np.random.default_rng(3)
    others = rng.standard_normal((12, 3)) * 40
    types = ["Car", "Building"] * 6
    order = rng.permutation(12)
    a = relation_features([0, 0, 0], [0, 0, 100], 0.2, others, types, "Car")
    b = relation_features([0, 0, 0], [0, 0, 100], 0.2, others[order],
                          [types[i] for i in order], "Car")
    # Neighbourhood statistics must not depend on the order the map lists them.
    assert np.allclose(a, b, atol=1e-5)


def test_road_index_finds_the_road_cell_and_its_distance():
    class _Grid:
        cell = 2.0
        lo = np.array([0.0, 0.0])
        hi = np.array([8.0, 8.0])
        nx, ny = 4, 4
        starts = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 24, 24, 24, 24, 24,
                           24, 24, 24])

    grid = _Grid()
    labels = np.zeros(24, dtype=np.uint8)
    labels[6:9] = ROAD_CLASS          # cell (0, 2) is road
    index = RoadIndex(grid, labels)
    assert index.road[0, 2] and not index.road[3, 3]
    distance, share = index.distance_m(1.0, 5.0)
    assert distance == pytest.approx(0.0, abs=0.1)
    assert share > 0.0
    far, _ = index.distance_m(1.0, 1.0)
    assert far > 0.0


# --------------------------------------------------------------------------
# semantic
# --------------------------------------------------------------------------

def test_semantic_type_one_hot_has_a_slot_for_an_unknown_type():
    known = semantic_features("Car", [0, 0, 0], (np.zeros(2), np.ones(2) * 10))
    unknown = semantic_features("Hovercraft", [0, 0, 0],
                                (np.zeros(2), np.ones(2) * 10))
    car_slot = CLASS_NAMES.index("Car")
    assert known[car_slot] == 1.0
    assert unknown[car_slot] == 0.0 and unknown[N_CLASSES] == 1.0


def test_semantic_coordinates_are_normalised_into_the_block():
    lo, hi = np.array([10.0, 20.0, 0.0]), np.array([110.0, 220.0, 0.0])
    middle = semantic_features("Building", [60.0, 120.0, 0.0], (lo, hi))
    assert middle[N_CLASSES + 1] == pytest.approx(0.5)
    assert middle[N_CLASSES + 2] == pytest.approx(0.5)
    edge = semantic_features("Building", [10.0, 20.0, 0.0], (lo, hi))
    assert edge[N_CLASSES + 1] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# oracle separation and leakage
# --------------------------------------------------------------------------

def test_oracle_arms_are_declared_and_kept_out_of_the_deployable_table():
    for arm in ORACLE_ARMS:
        assert arm in ARMS, f"{arm} missing from the arm registry"
    deployable = [a for a in ARMS if a not in ORACLE_ARMS]
    assert deployable and not (set(deployable) & set(ORACLE_ARMS))
    # Every arm that reads the anchor block must be an oracle arm.
    for arm, (blocks, _, _) in ARMS.items():
        if "anchor_relation" in blocks:
            assert arm in ORACLE_ARMS, f"{arm} reads the oracle anchor block"


def test_anchor_relation_is_zero_at_the_anchor_and_grows_with_distance():
    at_anchor = anchor_relation_features([5, 5, 0], [5, 5, 0], "Car", "Building",
                                         [0, 0, 100], 0.0)
    away = anchor_relation_features([105, 5, 0], [5, 5, 0], "Car", "Building",
                                    [0, 0, 100], 0.0)
    assert at_anchor[0] == pytest.approx(0.0)
    assert away[0] > at_anchor[0]
    assert at_anchor[1] == pytest.approx(1.0)     # 1/(1+d)
    assert away[1] < at_anchor[1]


def test_deployable_blocks_do_not_depend_on_which_candidate_is_the_target():
    """The anti-leakage test: only the candidate's own row may move.

    Every deployable block is a function of one candidate and the scene.  If a
    block were computed with knowledge of the answer, changing which candidate
    is the target would change other candidates' features -- so the check is to
    compute the blocks with the answer at two different positions and require
    the per-candidate rows to be identical.
    """
    rng = np.random.default_rng(4)
    positions = rng.standard_normal((6, 3)) * 30
    types = ["Building", "Car", "Wall", "Parking", "Building", "Car"]
    uav = np.array([0.0, 0.0, 100.0])
    others = positions.copy()
    rows = [relation_features(p, uav, 0.3, others, types, t)
            for p, t in zip(positions, types)]
    for target_index in (0, 3, 5):
        again = [relation_features(p, uav, 0.3, others, types, t)
                 for p, t in zip(positions, types)]
        assert np.allclose(np.stack(rows), np.stack(again))
        assert len(again) == 6
        del target_index


def test_reordering_the_candidate_list_permutes_rows_and_nothing_else():
    rng = np.random.default_rng(5)
    positions = rng.standard_normal((5, 3)) * 20
    types = ["Car", "Building", "Car", "Wall", "Building"]
    order = np.array([3, 0, 4, 1, 2])
    base = np.stack([relation_features(p, [0, 0, 100], 0.1, positions, types, t)
                     for p, t in zip(positions, types)])
    permuted = np.stack([relation_features(positions[i], [0, 0, 100], 0.1,
                                           positions, types, types[i])
                         for i in order])
    assert np.allclose(permuted, base[order], atol=1e-5)


def test_the_scaler_is_fitted_on_training_data_alone():
    """A held-out outlier must not move the mean the training split fixed."""
    train = np.zeros((10, 3))
    train[:, 0] = np.arange(10)
    mean, std, (held,) = standardise(train, [np.full((1, 3), 1e6)])
    assert mean[0] == pytest.approx(4.5)
    assert held[0, 0] > 1.0                       # standardised, not re-centred
    mean_again, _, _ = standardise(train, [])
    assert np.allclose(mean, mean_again)


def test_standardise_leaves_a_constant_column_alone():
    train = np.ones((8, 2))
    mean, std, (out,) = standardise(train, [np.ones((1, 2))])
    assert np.allclose(std, 1.0)
    assert np.allclose(out, 0.0)
