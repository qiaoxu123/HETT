"""The map graph: node partition, road regions, and the edge schema.

Two of these matter more than the rest.  The first is that a road *name* becomes
one node even though it is many annotated segments, because that is the premise
the whole round rests on.  The second is that a ``Node`` has nowhere to record
which entity is the answer: the field simply does not exist, so no later arm can
read it by accident and no test has to police it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.spatial_graph import node_types  # noqa: E402
from sensaturban_fpv.spatial_graph.edge_features import (  # noqa: E402
    EDGE_DIM, EDGE_FEATURES, EDGE_INDEX, MISSING_ROAD_M, EdgeContext,
    edge_vector, footprint_distance, road_frame,
)
from sensaturban_fpv.spatial_graph.graph_builder import classify  # noqa: E402
from sensaturban_fpv.spatial_graph.node_types import Node, NodeKind  # noqa: E402
from sensaturban_fpv.spatial_graph.road_region import (  # noqa: E402
    build_regions, connected_components,
)


def square(x, y, half=2.0):
    return np.array([[x - half, y - half], [x + half, y - half],
                     [x + half, y + half], [x - half, y + half]])


def node(nid, x, y, name="", kind=NodeKind.OBJECT, otype="Car", half=2.0,
         axis=(1.0, 0.0), height=1.0):
    return Node(node_id=nid, kind=kind, entity_type=otype, name=name,
                center=np.array([x, y, height / 2]),
                dimension=np.array([half * 2, half * 2, height]),
                footprint=square(x, y, half),
                major_axis=np.array(axis, dtype=np.float64), height=height)


# --------------------------------------------------------------------------
# the partition
# --------------------------------------------------------------------------

def test_every_entity_lands_in_exactly_one_kind():
    assert classify("Building", "") is NodeKind.BUILDING
    assert classify("Building", "One Stop") is NodeKind.BUILDING
    assert classify("TrafficRoad", "Walsall Road") is NodeKind.ROAD_SEGMENT
    assert classify("TrafficRoad", "") is NodeKind.ROAD_SEGMENT
    assert classify("Car", "some car") is NodeKind.LANDMARK
    assert classify("Car", "") is NodeKind.OBJECT
    # A footpath with a name is not a landmark: binding language to it would
    # compete with the road it runs beside.
    assert classify("Footpath", "NCN 5") is NodeKind.OBJECT


def test_a_node_has_nowhere_to_record_the_answer():
    """The leak the last three rounds each had to police, removed structurally."""
    fields = set(Node.__dataclass_fields__)
    for forbidden in ("is_target", "target", "candidate_index", "rank",
                      "gt_rank", "referenced", "is_referenced", "answer"):
        assert forbidden not in fields


# --------------------------------------------------------------------------
# road regions
# --------------------------------------------------------------------------

def test_segments_sharing_a_name_become_one_region():
    segs = [node(1, 0, 0, "Aldridge Road", NodeKind.ROAD_SEGMENT),
            node(2, 8, 0, "Aldridge Road", NodeKind.ROAD_SEGMENT),
            node(3, 16, 0, "Aldridge Road", NodeKind.ROAD_SEGMENT)]
    regions = build_regions(segs)
    assert len(regions) == 1
    assert len(regions[0].member_ids) == 3
    assert regions[0].kind is NodeKind.ROAD_REGION


def test_a_name_on_two_separate_roads_becomes_two_regions():
    """Otherwise the region would span a gap no road crosses."""
    segs = [node(1, 0, 0, "High Street", NodeKind.ROAD_SEGMENT),
            node(2, 6, 0, "High Street", NodeKind.ROAD_SEGMENT),
            node(3, 400, 0, "High Street", NodeKind.ROAD_SEGMENT),
            node(4, 406, 0, "High Street", NodeKind.ROAD_SEGMENT)]
    regions = build_regions(segs)
    assert len(regions) == 2
    assert sorted(len(r.member_ids) for r in regions) == [2, 2]


def test_different_names_never_merge():
    segs = [node(1, 0, 0, "Alpha Road", NodeKind.ROAD_SEGMENT),
            node(2, 6, 0, "Beta Road", NodeKind.ROAD_SEGMENT)]
    regions = build_regions(segs)
    assert len(regions) == 2
    assert {r.norm_name for r in regions} == {"alpha road", "beta road"}


def test_unnamed_segments_do_not_become_a_region():
    """Adjacency alone would merge a block's whole unnamed network into one blob."""
    segs = [node(i, i * 5.0, 0, "", NodeKind.ROAD_SEGMENT) for i in range(10)]
    assert build_regions(segs) == []


def test_connected_components_are_transitive():
    segs = [node(1, 0, 0, "", NodeKind.ROAD_SEGMENT),
            node(2, 10, 0, "", NodeKind.ROAD_SEGMENT),
            node(3, 20, 0, "", NodeKind.ROAD_SEGMENT),
            node(4, 500, 0, "", NodeKind.ROAD_SEGMENT)]
    groups = sorted(len(g) for g in connected_components(segs, gap=6.0))
    assert groups == [1, 3]


# --------------------------------------------------------------------------
# the edge schema
# --------------------------------------------------------------------------

def test_the_schema_is_pinned_and_its_columns_are_unique():
    names = [n for n, _ in EDGE_FEATURES]
    assert len(names) == len(set(names)) == EDGE_DIM
    assert set(EDGE_INDEX) == set(names)
    for required in ("dx", "dy", "footprint_distance", "same_side_of_road",
                     "opposite_side_of_road", "road_tangent_coord",
                     "anchor_major_coord", "agent_ahead"):
        assert required in EDGE_INDEX


def test_the_edge_vector_has_a_fixed_width_and_no_infinities():
    a, b = node(1, 0, 0), node(2, 30, 40)
    v = edge_vector(a, b, EdgeContext())
    assert v.shape == (EDGE_DIM,)
    assert np.isfinite(v).all()


def test_a_node_with_no_road_gets_a_finite_distance():
    """An infinity here would propagate through log1p into the whole batch."""
    v = edge_vector(node(1, 0, 0), node(2, 10, 0), EdgeContext())
    assert v[EDGE_INDEX["nearest_road_distance"]] == pytest.approx(
        MISSING_ROAD_M / 200.0)
    assert np.isfinite(v).all()


def test_the_edge_is_directional():
    """A symmetric vector could not express "north of" at all."""
    a, b = node(1, 0, 0), node(2, 30, 40)
    ab, ba = edge_vector(a, b, EdgeContext()), edge_vector(b, a, EdgeContext())
    assert ab[EDGE_INDEX["dx"]] == pytest.approx(-ba[EDGE_INDEX["dx"]])
    assert ab[EDGE_INDEX["dy"]] == pytest.approx(-ba[EDGE_INDEX["dy"]])
    # And a pure east-west pair is separable by the global axes alone.
    east = node(3, 50, 0)
    assert edge_vector(a, east, EdgeContext())[EDGE_INDEX["dx"]] > 0
    assert edge_vector(a, east, EdgeContext())[EDGE_INDEX["dy"]] == pytest.approx(0)


def test_the_anchor_frame_rotates_with_the_anchor():
    a = node(1, 0, 0, axis=(1.0, 0.0))
    b = node(2, 0, 40)
    along_x = edge_vector(a, b, EdgeContext())[EDGE_INDEX["anchor_major_coord"]]
    turned = node(1, 0, 0, axis=(0.0, 1.0))
    along_y = edge_vector(turned, b, EdgeContext())[EDGE_INDEX["anchor_major_coord"]]
    assert abs(along_x) < 1e-6 and along_y > 0


def test_the_agent_frame_follows_the_heading_only():
    a, b = node(1, 0, 0), node(2, 40, 0)
    facing_east = edge_vector(a, b, EdgeContext(agent_yaw=0.0))
    facing_west = edge_vector(a, b, EdgeContext(agent_yaw=np.pi))
    assert facing_east[EDGE_INDEX["agent_ahead"]] > 0
    assert facing_west[EDGE_INDEX["agent_ahead"]] < 0
    # The global columns must not move.
    assert facing_east[EDGE_INDEX["dx"]] == pytest.approx(
        facing_west[EDGE_INDEX["dx"]])


def test_footprint_distance_beats_centre_distance_when_boxes_are_wide():
    a = node(1, 0, 0, half=10.0)
    b = node(2, 12, 0, half=10.0)
    v = edge_vector(a, b, EdgeContext())
    assert v[EDGE_INDEX["center_distance"]] > v[EDGE_INDEX["footprint_distance"]]
    assert footprint_distance(a, b) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# road frames
# --------------------------------------------------------------------------

def _road_segment(nid, x, y, name="Test Road"):
    return node(nid, x, y, name, NodeKind.ROAD_SEGMENT, "TrafficRoad",
                half=3.0, axis=(1.0, 0.0))


def test_side_of_road_is_a_sign_against_the_roads_own_normal():
    """The region's direction, not the map's: this is what "across the road" needs."""
    road = _road_segment(1, 0, 0)
    region = Node(node_id=99, kind=NodeKind.ROAD_REGION, entity_type="RoadRegion",
                  name="Test Road", center=np.array([0.0, 0.0, 0.0]),
                  dimension=np.array([30.0, 6.0, 0.2]),
                  footprint=square(0, 0, 3.0), major_axis=np.array([1.0, 0.0]),
                  member_ids=(1,), height=0.2)
    north = node(2, 0, 30)
    south = node(3, 0, -30)
    f_north = road_frame(north, region, [road])
    f_south = road_frame(south, region, [road])
    assert f_north.side > 0 > f_south.side
    # Being on the road means being ~0 from it and having no meaningful side.
    f_on = road_frame(node(4, 0, 0), region, [road])
    assert f_on.distance < f_north.distance


def test_same_and_opposite_side_are_computed_from_the_road_frame():
    road = _road_segment(1, 0, 0)
    region = Node(node_id=99, kind=NodeKind.ROAD_REGION, entity_type="RoadRegion",
                  name="Test Road", center=np.array([0.0, 0.0, 0.0]),
                  dimension=np.array([30.0, 6.0, 0.2]),
                  footprint=square(0, 0, 3.0), major_axis=np.array([1.0, 0.0]),
                  member_ids=(1,), height=0.2)
    ctx = EdgeContext(road_regions=[region], road_members={0: [road]})
    anchor = node(2, 0, 20)
    same = node(3, 5, 25)
    across = node(4, 5, -25)
    a_same = edge_vector(anchor, same, ctx)
    a_across = edge_vector(anchor, across, ctx)
    assert a_same[EDGE_INDEX["same_side_of_road"]] == 1.0
    assert a_same[EDGE_INDEX["opposite_side_of_road"]] == 0.0
    assert a_across[EDGE_INDEX["opposite_side_of_road"]] == 1.0
    assert a_across[EDGE_INDEX["same_side_of_road"]] == 0.0
