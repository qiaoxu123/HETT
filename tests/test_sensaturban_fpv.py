"""Unit tests for the SensatUrban FPV rendering package.

Every test that can be settled analytically is settled analytically and then
checked against the implementation, so a sign error in the projection, the yaw
convention or the coordinate transform fails here rather than silently
producing a plausible-looking image.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sensaturban_fpv.coordinate_diagnostic import bbox_overlap  # noqa: E402
from sensaturban_fpv.fit_coordinate_transform import (  # noqa: E402
    DIHEDRAL, CoordinateTransform, occupancy_score, trajectory_in_bounds_ratio,
)
from sensaturban_fpv.pointcloud_renderer import (  # noqa: E402
    Camera, depth_histogram, estimate_shift, project_points, render,
    render_cloud_region, reprojection_shift, rasterize_topdown, select_lod, unproject,
)
from sensaturban_fpv.project_landmarks import (  # noqa: E402
    landmark_visibility, project_centre, summarise_visibility,
)


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def test_point_on_axis_projects_to_centre():
    cam = Camera(position=[0, 0, 10], yaw=0.0, pitch=0.0, width=512, height=512,
                 hfov_deg=90.0, far=1000.0)
    proj = project_points(np.array([[100.0, 0.0, 10.0]]), cam)
    assert proj["u"][0] == pytest.approx(256.0, abs=1e-9)
    assert proj["v"][0] == pytest.approx(256.0, abs=1e-9)
    assert proj["z_cam"][0] == pytest.approx(100.0)
    assert proj["metric"][0] == pytest.approx(100.0)


def test_bearing_maps_to_horizontal_pixel():
    """u = W/2 + focal * tan(bearing), with the bearing taken to the camera's right.

    Facing +x with +z up, "+y" is to the camera's *left*, so a world azimuth of
    +30 degrees must land left of centre.
    """
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=512, height=512,
                 hfov_deg=90.0, near=1e-3, far=1e6)
    focal = cam.focal
    for az_deg, expected in ((30.0, 256 - 256 * np.tan(np.deg2rad(30.0))),
                             (-30.0, 256 + 256 * np.tan(np.deg2rad(30.0))),
                             (0.0, 256.0),
                             (44.0, 256 - 256 * np.tan(np.deg2rad(44.0)))):
        a = np.deg2rad(az_deg)
        proj = project_points(np.array([[100 * np.cos(a), 100 * np.sin(a), 0.0]]), cam)
        assert proj["u"][0] == pytest.approx(expected, abs=1e-6), az_deg


def test_hfov_edges_and_half_open_clipping():
    """Half-open [0, width) clipping, checked just either side of the border.

    Points exactly on the border are deliberately not used: ``focal * x / z`` at
    exactly 45 degrees lands on ``u = 0`` or ``u = width`` only up to a rounding
    error, so the boundary case is measure-zero and not worth pinning.
    """
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=512, height=512,
                 hfov_deg=90.0, near=1e-3, far=1e6)

    inside = np.array([[100.0, 99.0, 0.0], [100.0, -99.0, 0.0]])
    proj = project_points(inside, cam)
    assert proj["u"].size == 2
    assert proj["u"].min() > 0.0 and proj["u"].max() < 512.0

    outside = np.array([[100.0, 101.0, 0.0], [100.0, -101.0, 0.0]])
    proj_out = project_points(outside, cam)
    assert proj_out["u"].size == 0

    # Exactly 45 degrees off-axis is the edge of a 90 degree hFOV.
    assert cam.focal * np.tan(np.deg2rad(45.0)) == pytest.approx(256.0)


def test_focal_matches_hfov_definition():
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=640, height=480, hfov_deg=60.0)
    assert cam.focal == pytest.approx(640 / (2 * np.tan(np.deg2rad(30.0))))
    assert cam.vfov_deg == pytest.approx(
        np.rad2deg(2 * np.arctan((480 / 2) / cam.focal)))


def test_right_handed_camera_basis_is_orthonormal():
    for yaw in (0.0, 0.7, -2.3, 3.0):
        for pitch in (0.0, -0.5, 0.3):
            cam = Camera(position=[1, 2, 3], yaw=yaw, pitch=pitch)
            right, up, forward = cam.basis()
            for v in (right, up, forward):
                assert np.linalg.norm(v) == pytest.approx(1.0)
            assert right @ up == pytest.approx(0.0, abs=1e-12)
            assert right @ forward == pytest.approx(0.0, abs=1e-12)
            assert up @ forward == pytest.approx(0.0, abs=1e-12)
            # right x up == -forward only if the frame were left-handed; the
            # image convention here is u right, v down, so check the actual
            # identity that the projection relies on.
            assert np.allclose(np.cross(right, up), -forward, atol=1e-12)


def test_east_facing_camera_has_south_on_its_right():
    """z-up right-handed world: facing +x means -y is to the right."""
    cam = Camera(position=[0, 0, 0], yaw=0.0, pitch=0.0, far=1e6)
    right, up, forward = cam.basis()
    assert np.allclose(right, [0, -1, 0], atol=1e-12)
    assert np.allclose(up, [0, 0, 1], atol=1e-12)

    # A point to the world-south must therefore land on the right of the image.
    proj = project_points(np.array([[50.0, -10.0, 0.0]]), cam)
    assert proj["u"][0] > 256.0


def test_fov_clipping_drops_outside_points():
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=512, height=512,
                 hfov_deg=90.0, near=0.5, far=100.0)
    pts = np.array([
        [50.0, 0.0, 0.0],      # dead ahead, inside
        [-50.0, 0.0, 0.0],     # behind
        [0.2, 0.0, 0.0],       # inside the near plane
        [500.0, 0.0, 0.0],     # past the far plane
        [50.0, 500.0, 0.0],    # far off to the side
    ])
    proj = project_points(pts, cam)
    assert proj["u"].size == 1
    assert proj["u"][0] == pytest.approx(256.0)


def test_yaw_increase_turns_left_so_content_moves_right():
    """CityNav yaw is atan2(dy, dx), counter-clockwise positive.

    Turning the camera to the left makes the world sweep to the *right* of the
    image, so u grows with yaw.  Getting this backwards would flip every yaw
    probe in gate 4, so it is pinned here against the closed form.
    """
    # Level camera, so the closed form u = W/2 + focal*tan(bearing) is exact.
    cam0 = Camera(position=[0, 0, 50], yaw=0.0, pitch=0.0, far=1e6)
    cam1 = Camera(position=[0, 0, 50], yaw=np.deg2rad(10.0), pitch=0.0, far=1e6)
    pt = np.array([[100.0, 0.0, 0.0]])
    u0 = project_points(pt, cam0)["u"][0]
    u1 = project_points(pt, cam1)["u"][0]
    assert u0 == pytest.approx(256.0, abs=1e-6)
    assert u1 > u0

    # Horizontally, the point's bearing relative to the camera goes from 0 to
    # -10 degrees in world azimuth, i.e. +10 degrees to the camera's right.
    focal = cam0.focal
    expected = 256.0 + focal * np.tan(np.deg2rad(10.0))
    assert u1 == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------
# z-buffer and depth
# --------------------------------------------------------------------------

def test_zbuffer_keeps_the_nearer_point():
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    xyz = np.array([[10.0, 0, 0], [30.0, 0, 0]])
    rgb = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    res = render(xyz, rgb, cam, splat_radius=0, lod=None)
    assert res.depth[32, 32] == pytest.approx(10.0)
    assert tuple(res.rgb[32, 32]) == (255, 0, 0)
    assert res.point_id[32, 32] == 0
    assert res.density[32, 32] == 2


def test_zbuffer_is_order_independent():
    """Reversing the input order must not change which point wins."""
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    xyz = np.array([[30.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
    rgb = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.uint8)
    a = render(xyz, rgb, cam, lod=None)
    b = render(xyz[::-1], rgb[::-1], cam, ids=np.array([2, 1, 0]), lod=None)
    assert a.depth[32, 32] == pytest.approx(10.0)
    assert a.depth[32, 32] == pytest.approx(b.depth[32, 32])
    assert tuple(a.rgb[32, 32]) == tuple(b.rgb[32, 32]) == (2, 2, 2)


def test_depth_is_metric_range_not_forward_component():
    """A point 30 degrees off-axis at forward distance d has metric range d/cos30."""
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=512, height=512,
                 hfov_deg=90.0, near=1e-3, far=1e6)
    d = 10.0
    a = np.deg2rad(30.0)
    pt = np.array([[d, -d * np.tan(a), 0.0]])
    proj = project_points(pt, cam)
    assert proj["z_cam"][0] == pytest.approx(d)
    assert proj["metric"][0] == pytest.approx(d / np.cos(a))
    assert proj["u"][0] == pytest.approx(256.0 + 256.0 * np.tan(a))


def test_holes_are_masked_not_filled():
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    # A single point covers exactly one pixel; everything else must be invalid.
    res = render(np.array([[10.0, 0, 0]]), np.array([[9, 9, 9]], np.uint8), cam, lod=None)
    assert res.valid.sum() == 1
    assert np.isnan(res.depth[0, 0])
    assert res.valid_ratio == pytest.approx(1.0 / (64 * 64))


def test_splat_expands_coverage_but_keeps_nearest_depth():
    """Splatting fills neighbouring pixels and still resolves them by depth."""
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    # Two points two pixels apart in u, the farther one nearer the centre.
    xyz = np.array([[10.0, 0.0, 0], [8.0, 0.05, 0]])
    rgb = np.array([[10, 10, 10], [200, 200, 200]], dtype=np.uint8)
    raw = render(xyz, rgb, cam, splat_radius=0, lod=None)
    splat = render(xyz, rgb, cam, splat_radius=2, lod=None)

    # Raw: one pixel each, and the far point owns the exact centre.
    assert raw.valid.sum() == 2
    assert raw.depth[32, 32] == pytest.approx(10.0)
    # Splat: both points now reach the centre pixel, and the nearer one wins.
    assert splat.valid.sum() > raw.valid.sum()
    assert splat.depth[32, 32] == pytest.approx(np.hypot(8.0, 0.05))
    assert tuple(splat.rgb[32, 32]) == (200, 200, 200)
    # No splatted pixel is nearer than the nearest real point.
    assert np.nanmin(splat.depth) >= np.nanmin(raw.depth) - 1e-6


def test_removing_far_points_cannot_bring_depth_closer():
    """The monotonicity property a real z-buffer must satisfy."""
    rng = np.random.default_rng(7)
    cam = Camera(position=[0, 0, 20], yaw=0.3, pitch=-0.4, width=96, height=96,
                 hfov_deg=90.0, near=0.5, far=120.0)
    xyz = np.column_stack([
        rng.uniform(-60, 60, 4000), rng.uniform(-60, 60, 4000), rng.uniform(0, 25, 4000)])
    rgb = rng.integers(0, 255, (4000, 3), dtype=np.uint8)
    full = render(xyz, rgb, cam, lod=None)
    dist = np.linalg.norm(xyz - np.asarray(cam.position), axis=1)
    keep = dist <= 80.0
    reduced = render(xyz[keep], rgb[keep], cam, lod=None)
    both = full.valid & reduced.valid
    assert both.any()
    delta = reduced.depth[both].astype(float) - full.depth[both].astype(float)
    assert (delta >= -1e-6).all(), "a truncated cloud pulled some pixel nearer"


def test_unproject_inverts_projection():
    cam = Camera(position=[3, -4, 12], yaw=1.1, pitch=-0.35, width=64, height=48,
                 hfov_deg=80.0, near=0.5, far=500.0)
    # Projecting an integer pixel's own ray endpoint must come back to exactly
    # that pixel and that range: this is the property the depth map relies on.
    for u0, v0, value in ((20, 15, 20.0), (5, 40, 35.0), (63, 47, 8.0)):
        depth = np.full((48, 64), np.nan, dtype=np.float32)
        depth[v0, u0] = value
        world = unproject(depth, cam)
        assert np.isfinite(world[v0, u0]).all()

        proj = project_points(world[v0, u0][None, :], cam)
        assert proj["u"].size == 1
        assert proj["u"][0] == pytest.approx(u0, abs=1e-6)
        assert proj["v"][0] == pytest.approx(v0, abs=1e-6)
        assert proj["metric"][0] == pytest.approx(value, rel=1e-6)


def test_depth_histogram_flags_constant_depth():
    flat = np.full((32, 32), 5.0)
    hist = depth_histogram(flat)
    assert hist["std"] == pytest.approx(0.0)
    assert hist["unique_rounded"] == 1
    varied = np.linspace(1, 100, 32 * 32).reshape(32, 32)
    assert depth_histogram(varied)["unique_rounded"] > 20


def test_lod_keep_is_deterministic_and_bounded():
    ids = np.arange(100000)
    dist = np.linspace(0, 200, 100000)
    keep = select_lod(ids, dist)
    assert np.array_equal(keep, select_lod(ids, dist))
    near = (dist <= 40) & keep
    assert near.sum() == (dist <= 40).sum()  # near shell is untouched


# --------------------------------------------------------------------------
# shift estimation (the convention gates 3 and 4 depend on)
# --------------------------------------------------------------------------

def test_estimate_shift_recovers_a_known_translation():
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (96, 96, 3)).astype(np.uint8)
    for du, dv in ((5, 0), (-7, 3), (0, -4), (11, -6)):
        shifted = np.roll(np.roll(base, du, axis=1), dv, axis=0)
        est = estimate_shift(base, shifted)
        assert est["du"] == du and est["dv"] == dv, (du, dv, est)


def _centre_only_depth(shape, value):
    """A depth map with one valid pixel, so the median is that pixel's shift."""
    depth = np.full(shape, np.nan, dtype=np.float32)
    depth[shape[0] // 2, shape[1] // 2] = value
    return depth


def test_reprojection_shift_matches_pure_yaw_rotation():
    """A yaw change of D moves the centre pixel by focal * tan(D) to the right."""
    cam0 = Camera(position=[0, 0, 40], yaw=0.0, pitch=0.0, width=512, height=512,
                  hfov_deg=90.0, near=0.5, far=1000.0)
    depth = _centre_only_depth((512, 512), 100.0)
    for deg in (5.0, 10.0, -7.0):
        cam1 = Camera(position=[0, 0, 40], yaw=np.deg2rad(deg), pitch=0.0,
                      width=512, height=512, hfov_deg=90.0, near=0.5, far=1000.0)
        shift = reprojection_shift(cam0, cam1, depth)
        assert shift["du"] == pytest.approx(cam0.focal * np.tan(np.deg2rad(deg)),
                                            rel=1e-6)
        assert shift["dv"] == pytest.approx(0.0, abs=1e-6)


def test_reprojection_shift_matches_pure_translation():
    """Closed-form check for a strafe and for a forward step."""
    W, D = 256, 50.0
    cam0 = Camera(position=[0, 0, 0], yaw=0.0, pitch=0.0, width=W, height=W,
                  hfov_deg=90.0, near=0.5, far=1e6)
    depth = _centre_only_depth((W, W), D)
    focal = cam0.focal

    # Camera steps +y, which is to its left, so content slides right: u grows.
    side = Camera(position=[0, 4.0, 0], yaw=0.0, pitch=0.0, width=W, height=W,
                  hfov_deg=90.0, near=0.5, far=1e6)
    shift = reprojection_shift(cam0, side, depth)
    assert shift["du"] == pytest.approx(focal * 4.0 / D, rel=1e-6)
    assert shift["dv"] == pytest.approx(0.0, abs=1e-6)

    # Camera steps +x (towards the surface): the ray through the centre still
    # hits the same world point, so the centre pixel does not move.
    forward = Camera(position=[5.0, 0, 0], yaw=0.0, pitch=0.0, width=W, height=W,
                     hfov_deg=90.0, near=0.5, far=1e6)
    shift_fwd = reprojection_shift(cam0, forward, depth)
    assert shift_fwd["du"] == pytest.approx(0.0, abs=1e-6)
    assert shift_fwd["dv"] == pytest.approx(0.0, abs=1e-6)

    # Camera rises; a point straight ahead is now below the axis, so v grows.
    up = Camera(position=[0, 0, 3.0], yaw=0.0, pitch=0.0, width=W, height=W,
                hfov_deg=90.0, near=0.5, far=1e6)
    shift_up = reprojection_shift(cam0, up, depth)
    assert shift_up["dv"] == pytest.approx(focal * 3.0 / D, rel=1e-6)


# --------------------------------------------------------------------------
# coordinate transform
# --------------------------------------------------------------------------

def test_identity_transform_changes_nothing():
    t = CoordinateTransform.identity()
    xyz = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, -6.0]])
    assert np.allclose(t.apply_xyz(xyz), xyz)
    assert np.allclose(t.inverse_xyz(xyz), xyz)
    assert t.yaw_delta_rad() == pytest.approx(0.0)


def test_every_dihedral_is_a_signed_permutation_and_invertible():
    for name, a, b, c, d in DIHEDRAL:
        m = np.array([[a, b], [c, d]], dtype=np.float64)
        assert sorted(np.abs(m).ravel().tolist()) == [0.0, 0.0, 1.0, 1.0], name
        t = CoordinateTransform(m, 2.5, np.array([10.0, -3.0]), name)
        xyz = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, -6.0], [0.0, 0.0, 0.0]])
        assert np.allclose(t.inverse_xyz(t.apply_xyz(xyz)), xyz, atol=1e-9)


def test_swap_xy_orientation_swaps_coordinates():
    t = CoordinateTransform(np.array([[0.0, 1.0], [1.0, 0.0]]), 1.0,
                            np.array([0.0, 0.0]), "swap_xy")
    out = t.apply_xyz(np.array([[3.0, 7.0, 5.0]]))
    assert out[0, 0] == pytest.approx(7.0)
    assert out[0, 1] == pytest.approx(3.0)
    assert out[0, 2] == pytest.approx(5.0)  # z is never touched


def test_yaw_delta_reports_the_heading_rotation():
    assert CoordinateTransform(np.array([[1.0, 0.0], [0.0, 1.0]]), 1, np.zeros(2)) \
        .yaw_delta_rad() == pytest.approx(0.0)
    rot90 = CoordinateTransform(np.array([[0.0, -1.0], [1.0, 0.0]]), 1, np.zeros(2))
    assert rot90.yaw_delta_rad() == pytest.approx(np.pi / 2)
    flip = CoordinateTransform(np.array([[-1.0, 0.0], [0.0, 1.0]]), 1, np.zeros(2))
    assert abs(flip.yaw_delta_rad()) == pytest.approx(np.pi)


def test_transform_json_round_trip(tmp_path):
    t = CoordinateTransform(np.array([[0.0, 1.0], [1.0, 0.0]]), 1.0,
                            np.array([12.5, -7.25]), "swap_xy")
    path = tmp_path / "t.json"
    path.write_text(json.dumps(t.to_json()))
    from sensaturban_fpv.fit_coordinate_transform import load_transform
    back = load_transform(path)
    assert np.allclose(back.m, t.m)
    assert back.scale == t.scale
    assert np.allclose(back.t, t.t)


class _FakeGrid:
    """Bucket-count stub: points only inside the unit square at the origin."""

    def __init__(self, points, cell=1.0):
        self.points = np.asarray(points, dtype=np.float64)
        self.cell = cell

    def count_radius_approx(self, x, y, radius):
        d = np.hypot(self.points[:, 0] - x, self.points[:, 1] - y)
        return int((d <= radius).sum() * 10)


def test_occupancy_score_prefers_the_aligned_placement():
    rng = np.random.default_rng(11)
    cluster = rng.normal(0, 1.0, (4000, 2))
    grid = _FakeGrid(cluster)
    aligned = occupancy_score(np.zeros((5, 2)), grid, radius=5.0)
    shifted = occupancy_score(np.full((5, 2), 500.0), grid, radius=5.0)
    assert aligned["mean_points"] > shifted["mean_points"]
    assert aligned["nonempty_ratio"] == 1.0
    assert shifted["nonempty_ratio"] == 0.0


def test_trajectory_in_bounds_ratio():
    traj = np.array([[0.0, 0.0], [5.0, 5.0], [100.0, 100.0]])
    lo, hi = np.array([0.0, 0.0]), np.array([10.0, 10.0])
    assert trajectory_in_bounds_ratio(traj, lo, hi) == pytest.approx(2 / 3)


def test_fit_transform_recovers_identity_from_identity_data():
    """Structure, not a uniform field: only the true placement is well sampled.

    Landmarks sit on a few dense clusters at known spots.  Any other orientation
    or a translation away from zero puts them in the empty space between
    clusters, so identity has to win.
    """
    from sensaturban_fpv.fit_coordinate_transform import fit_transform
    rng = np.random.default_rng(5)

    # Clusters deliberately asymmetric in x and y, so no dihedral maps one onto
    # another (a symmetric layout would make several orientations tie).
    centres = np.array([[30.0, 40.0], [150.0, 60.0], [90.0, 170.0]])
    cloud = np.vstack([rng.normal(c, 1.5, (4000, 2)) for c in centres])
    grid = _FakeGrid(cloud, cell=1.0)

    # Landmarks sit on the cluster centres, so only a correctly placed transform
    # puts them in dense space.
    landmarks = centres
    traj = rng.uniform(20, 180, (50, 2))

    lo = cloud.min(axis=0)
    hi = cloud.max(axis=0)
    result = fit_transform(
        np.column_stack([traj, np.zeros(len(traj))]),
        np.column_stack([landmarks, np.zeros(len(landmarks))]),
        np.array([lo[0], lo[1], 0.0]), np.array([hi[0], hi[1], 50.0]), grid,
        radius=4.0, scales=(1.0,),
        coarse_extent=40.0, coarse_step=8.0, fine_extent=6.0, fine_step=1.0,
    )
    assert result["best"]["orientation"] == "identity"
    assert result["best"]["scale"] == pytest.approx(1.0)
    assert np.allclose(result["best"]["transform"].t, 0.0, atol=1.0), \
        result["best"]["transform"].t
    # Every landmark is genuinely backed by points there, and each reaches the
    # capped occupancy, so the score saturates at 1.0.
    assert result["best"]["nonempty_ratio"] == 1.0
    assert result["best"]["score"] == pytest.approx(1.0)
    # The wrong orientations must not also saturate.
    others = [r for r in result["all"] if r["orientation"] != "identity"]
    assert all(r["score"] < 1.0 for r in others)


# --------------------------------------------------------------------------
# landmarks
# --------------------------------------------------------------------------

def test_project_centre_returns_none_behind_the_camera():
    cam = Camera(position=[0, 0, 0], yaw=0.0, far=1e6)
    assert project_centre(np.array([-10.0, 0.0, 0.0]), cam) is None
    assert project_centre(np.array([10.0, 0.0, 0.0]), cam) is not None


def test_project_centre_places_a_known_point():
    cam = Camera(position=[0, 0, 10], yaw=0.0, width=512, height=512,
                 hfov_deg=90.0, far=1e6)
    p = project_centre(np.array([10.0, -10.0, 10.0]), cam)
    # Bearing -45 deg, exactly on the right border.
    assert p["u"] == pytest.approx(512.0)
    assert p["v"] == pytest.approx(256.0)
    assert p["inside"] is False


def test_landmark_visibility_flags_occlusion():
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    wall = np.array([[10.0, dy, dz] for dy in np.linspace(-20, 20, 200)
                     for dz in np.linspace(-20, 20, 200)])
    res = render(wall, np.full((len(wall), 3), 100, np.uint8), cam, lod=None)
    landmark = (1, "Tower", "Building", (50.0, 0.0, 0.0), (5.0, 5.0, 5.0))
    vis = landmark_visibility(landmark, cam, res)
    assert vis["in_fov"] is True
    assert vis["occluded_at_centre"] is True
    assert vis["visible"] is False


class _StubCloud:
    def __init__(self, xyz):
        self._xyz = np.asarray(xyz, dtype=np.float64)

    def xyz(self, index=None):
        return self._xyz if index is None else self._xyz[index]


class _StubGrid:
    def __init__(self, xyz):
        self._xyz = np.asarray(xyz, dtype=np.float64)

    def query_radius(self, x, y, radius):
        d = np.hypot(self._xyz[:, 0] - x, self._xyz[:, 1] - y)
        return np.flatnonzero(d <= radius)


def test_landmark_visibility_unoccluded_and_summary():
    """A landmark backed by measured points and nothing in front is visible."""
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    surface = np.array([[50.0, 0.0, 0.0]])
    res = render(surface, np.array([[200, 100, 50]], np.uint8), cam, lod=None)
    landmark = (7, "Hall", "Building", (50.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    cloud, grid = _StubCloud(surface), _StubGrid(surface)

    vis = landmark_visibility(landmark, cam, res, cloud, grid)
    assert vis["in_fov"] is True
    assert vis["occluded_at_centre"] is False
    assert vis["observed_points"] == 1
    assert vis["approx_visible_ratio"] > 0.0
    assert vis["visible"] is True

    summary = summarise_visibility([vis], referenced_ids=[7])
    assert summary["referenced_total"] == 1
    assert summary["referenced_visible"] == 1
    assert summary["referenced_in_fov"] == 1


def test_landmark_with_no_local_points_is_not_visible():
    """Centre in frame but nothing measured near it must not count as visible."""
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    surface = np.array([[50.0, 0.0, 0.0]])
    res = render(surface, np.array([[200, 100, 50]], np.uint8), cam, lod=None)
    landmark = (9, "Empty", "Building", (50.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    empty = np.zeros((0, 3))
    vis = landmark_visibility(landmark, cam, res, _StubCloud(empty), _StubGrid(empty))
    assert vis["in_fov"] is True
    assert vis["observed_points"] == 0
    assert vis["visible"] is False


def test_summarise_visibility_separates_referenced_from_other():
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    res = render(np.array([[50.0, 0.0, 0.0]]), np.array([[1, 1, 1]], np.uint8),
                 cam, lod=None)
    ref = (1, "Ref", "Building", (50.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    other = (2, "Other", "Building", (50.0, 1.0, 0.0), (2.0, 2.0, 2.0))
    a = landmark_visibility(ref, cam, res)
    b = landmark_visibility(other, cam, res)
    s = summarise_visibility([a, b], referenced_ids=[1])
    assert s["referenced_total"] == 1
    assert s["landmarks_total"] == 2


# --------------------------------------------------------------------------
# coordinate diagnostic helpers
# --------------------------------------------------------------------------

def test_bbox_overlap_identical_boxes():
    lo, hi = [0.0, 0.0], [10.0, 20.0]
    r = bbox_overlap(lo, hi, lo, hi)
    assert r["iou"] == pytest.approx(1.0)
    assert r["overlap_frac_of_a"] == pytest.approx(1.0)


def test_bbox_overlap_disjoint_boxes():
    r = bbox_overlap([0, 0], [10, 10], [100, 100], [110, 110])
    assert r["iou"] == pytest.approx(0.0)
    assert r["overlap_frac_of_a"] == pytest.approx(0.0)


def test_bbox_overlap_partial():
    r = bbox_overlap([0, 0], [10, 10], [5, 0], [15, 10])
    assert r["overlap_frac_of_a"] == pytest.approx(0.5)
    assert r["overlap_frac_of_b"] == pytest.approx(0.5)
    assert r["iou"] == pytest.approx(50.0 / 150.0)


# --------------------------------------------------------------------------
# top-down rasterisation
# --------------------------------------------------------------------------

def test_rasterize_topdown_takes_max_z_per_cell():
    """``origin`` is the top-left corner, so rows grow as y falls."""
    xyz = np.array([[0.5, 0.5, 1.0], [0.6, 0.6, 5.0], [1.5, 0.5, 3.0]])
    rgb = np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0]], dtype=np.uint8)
    out = rasterize_topdown(xyz, rgb, origin=(0.0, 2.0), resolution=1.0, shape=(2, 2))
    # y in (0, 1] -> row 1; x in [0, 1) -> col 0, x in [1, 2) -> col 1.
    assert out["height"][1, 0] == pytest.approx(5.0)   # max of 1.0 and 5.0
    assert out["counts"][1, 0] == 2
    assert out["rgb"][1, 0, 0] == pytest.approx(15.0)  # mean of 10 and 20
    assert out["height"][1, 1] == pytest.approx(3.0)
    assert np.isnan(out["height"][0, 0])               # nothing at y > 1


def test_rasterize_topdown_marks_empty_cells_as_nodata():
    out = rasterize_topdown(np.array([[0.5, 0.5, 2.0]]), None,
                            origin=(0.0, 2.0), resolution=1.0, shape=(2, 2))
    assert np.isfinite(out["height"][1, 0])
    assert np.isnan(out["height"][0, 0])


# --------------------------------------------------------------------------
# empty-render detection
# --------------------------------------------------------------------------

def test_empty_render_is_detected():
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=32, height=32,
                 hfov_deg=90.0, near=0.5, far=100.0)
    res = render(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), cam, lod=None)
    assert res.valid.sum() == 0
    assert res.valid_ratio == 0.0
    assert np.isnan(res.depth).all()
    assert depth_histogram(res.depth)["count"] == 0


def test_points_out_of_range_render_empty():
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=32, height=32,
                 hfov_deg=90.0, near=0.5, far=10.0)
    res = render(np.array([[1000.0, 0, 0]]), np.array([[1, 1, 1]], np.uint8), cam, lod=None)
    assert res.valid.sum() == 0


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def test_gate1_threshold_behaviour():
    from sensaturban_fpv.validate_rendering import gate1_valid_pixels

    class R:
        def __init__(self, ratio, below=None):
            self.valid_ratio = ratio
            self.stats = {"below_horizon_valid_ratio": below,
                          "sky_pixel_ratio": 0.0}

    assert gate1_valid_pixels([R(0.9), R(0.8), R(0.7)])["passed"] is True
    assert gate1_valid_pixels([R(0.1), R(0.2), R(0.9)])["passed"] is False
    assert gate1_valid_pixels([])["passed"] is False

    # A sky-dominated frame fails the literal test but the below-horizon figure
    # still shows the geometry is fully covered.
    sky_pose = [R(0.2, below=0.95), R(0.2, below=0.95)]
    g = gate1_valid_pixels(sky_pose)
    assert g["passed"] is False
    assert g["below_horizon_passed"] is True


def test_gate2_rejects_constant_depth():
    from sensaturban_fpv.validate_rendering import gate2_depth

    class R:
        def __init__(self, depth):
            self.depth = depth

    flat = R(np.full((16, 16), 5.0, dtype=np.float32))
    assert gate2_depth([flat])["passed"] is False
    varied = R(np.linspace(1, 100, 256).reshape(16, 16).astype(np.float32))
    assert gate2_depth([varied])["passed"] is True


def test_gate6_detects_out_of_bounds_poses():
    from sensaturban_fpv.validate_rendering import gate6_topdown

    lo, hi = np.array([0.0, 0.0]), np.array([10.0, 10.0])
    inside = gate6_topdown([[1, 1], [2, 2]], lo, hi, [5, 5])
    assert inside["passed"] is True
    outside = gate6_topdown([[1, 1], [50, 2]], lo, hi, [5, 5])
    assert outside["passed"] is False


# --------------------------------------------------------------------------
# cross-backend agreement (CPU numpy backends and, where present, the device)
# --------------------------------------------------------------------------

CPU_BACKENDS = ("lexsort", "argsort", "minimum_at")


def _torch_backend_available() -> bool:
    try:
        import torch  # noqa: F401
        from sensaturban_fpv.torch_backend import render_points_torch  # noqa: F401
    except Exception:
        return False
    return True


def _run_backends(xyz, rgb, cam):
    """Same scene through every backend, as {name: (rgb, depth, valid)}."""
    out = {}
    for name in CPU_BACKENDS:
        res = render(np.asarray(xyz, float), np.asarray(rgb, np.uint8), cam,
                     splat_radius=0, lod=None, zbuffer=name)
        out[name] = res
    if _torch_backend_available():
        from sensaturban_fpv.torch_backend import render_points_torch
        res = render_points_torch(xyz, rgb, cam, device="cpu")
        out["torch_cpu"] = res
    return out


def _assert_backends_agree(backend_results, pixel, expect_rgb, msg=""):
    for name, res in backend_results.items():
        if name == "torch_cpu":
            got = tuple(res["rgb"][pixel[0], pixel[1]].tolist())
            depth = float(res["depth"][pixel[0], pixel[1]])
        else:
            got = tuple(res.rgb[pixel[0], pixel[1]].tolist())
            depth = float(res.depth[pixel[0], pixel[1]])
        assert got == expect_rgb, (name, got, expect_rgb, msg)
        assert np.isfinite(depth), (name, msg)
    return depth


def test_all_backends_agree_on_nearer_point_wins():
    """The canonical case: two points on one pixel, the nearer one must win."""
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    xyz = np.array([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    rgb = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    results = _run_backends(xyz, rgb, cam)
    depth = _assert_backends_agree(results, (32, 32), (255, 0, 0), "nearer wins")
    assert depth == pytest.approx(5.0, abs=1e-4)

    # Reversed input order must not change the outcome.
    results_rev = _run_backends(xyz[::-1], rgb[::-1], cam)
    _assert_backends_agree(results_rev, (32, 32), (255, 0, 0), "order independent")


def test_all_backends_agree_on_distinct_pixels():
    """Pixels are floor-assigned: u = 38.4 and 25.6 land on columns 38 and 25."""
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=0.1, far=1000.0)
    xyz = np.array([[10.0, -2.0, 0.0], [10.0, 2.0, 0.0]])
    rgb = np.array([[10, 20, 30], [200, 210, 220]], dtype=np.uint8)
    results = _run_backends(xyz, rgb, cam)
    _assert_backends_agree(results, (32, 38), (10, 20, 30), "right of centre")
    _assert_backends_agree(results, (32, 25), (200, 210, 220), "left of centre")
    for name, res in results.items():
        valid = res["valid"] if name == "torch_cpu" else res.valid
        assert int(valid.sum()) == 2, name


def test_all_backends_agree_on_clipping():
    cam = Camera(position=[0, 0, 0], yaw=0.0, width=64, height=64,
                 hfov_deg=90.0, near=2.0, far=20.0)
    xyz = np.array([
        [10.0, 0.0, 0.0],    # visible
        [-10.0, 0.0, 0.0],   # behind the camera
        [1.0, 0.0, 0.0],     # inside the near plane
        [500.0, 0.0, 0.0],   # past the far plane
        [10.0, 60.0, 0.0],   # outside the horizontal FOV
    ])
    rgb = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
                   dtype=np.uint8)
    results = _run_backends(xyz, rgb, cam)
    for name, res in results.items():
        valid = res["valid"] if name == "torch_cpu" else res.valid
        assert int(valid.sum()) == 1, (name, int(valid.sum()))
    _assert_backends_agree(results, (32, 32), (1, 2, 3), "only the visible point")


def test_all_backends_agree_under_yaw_rotation():
    """Yaw changes must land on the analytically predicted pixel in every backend.

    A 90 degree turn takes this point out of a 90 degree frame entirely, so the
    test also pins that the backends agree on *disappearance*, not just on
    placement.
    """
    width = 128
    xyz = np.array([[30.0, 10.0, 0.0]])
    rgb = np.array([[90, 120, 150]], dtype=np.uint8)

    def expected_u(yaw_rad):
        cam = Camera(position=[0, 0, 0], yaw=yaw_rad, pitch=0.0, width=width,
                     height=width, hfov_deg=90.0, near=0.1, far=1000.0)
        right, _, forward = cam.basis()
        rel = xyz[0]
        z_cam = float(rel @ forward)
        x_cam = float(rel @ right)
        return cam.width / 2.0 + cam.focal * x_cam / z_cam, cam

    for yaw_deg in (0.0, -20.0, 20.0):
        u, cam = expected_u(np.deg2rad(yaw_deg))
        results = _run_backends(xyz, rgb, cam)
        for name, res in results.items():
            valid = res["valid"] if name == "torch_cpu" else res.valid
            assert int(valid.sum()) == 1, (name, yaw_deg)
            got = (res["rgb"] if name == "torch_cpu" else res.rgb)
            depth = (res["depth"] if name == "torch_cpu" else res.depth)
            row = int(cam.height / 2.0)
            assert tuple(got[row, int(u)].tolist()) == (90, 120, 150), (name, yaw_deg)
            assert float(depth[row, int(u)]) == pytest.approx(
                np.linalg.norm(xyz[0]), abs=1e-3), (name, yaw_deg)

    # Turning 90 degrees takes the point clean out of the frame, in all backends.
    _, turned = expected_u(np.pi / 2)
    results = _run_backends(xyz, rgb, turned)
    for name, res in results.items():
        valid = res["valid"] if name == "torch_cpu" else res.valid
        assert int(valid.sum()) == 0, (name, "point should be out of frame")


def test_cpu_backends_agree_exactly_on_a_dense_random_scene():
    """All three CPU z-buffers must agree pixel for pixel on a real-ish cloud."""
    rng = np.random.default_rng(23)
    cam = Camera(position=[0, 0, 30], yaw=0.7, pitch=-0.4, width=128, height=128,
                 hfov_deg=90.0, near=0.5, far=200.0)
    xyz = np.column_stack([
        rng.uniform(-80, 80, 60000),
        rng.uniform(-80, 80, 60000),
        rng.uniform(-10, 20, 60000),
    ])
    rgb = rng.integers(0, 255, (60000, 3), dtype=np.uint8)
    results = _run_backends(xyz, rgb, cam)

    ref = results["lexsort"]
    for name in ("argsort", "minimum_at"):
        other = results[name]
        assert np.array_equal(ref.valid, other.valid), name
        d = other.depth[ref.valid].astype(np.float64) - ref.depth[ref.valid].astype(np.float64)
        assert np.abs(d).max() < 1e-6, (name, np.abs(d).max())
        assert np.array_equal(ref.rgb[ref.valid], other.rgb[ref.valid]), name


# --------------------------------------------------------------------------
# the bucket-ordered cache must never serve a half-written file
# --------------------------------------------------------------------------

def _write_tiny_ply(path, points):
    """A minimal binary little-endian PLY with colour, in the real layout."""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float32 x\nproperty float32 y\nproperty float32 z\n"
        "property uint8 red\nproperty uint8 green\nproperty uint8 blue\n"
        "property uint8 class\nend_header\n"
    ).encode("ascii")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("class", "u1")])
    records = np.zeros(len(points), dtype=dtype)
    for i, (x, y, z) in enumerate(points):
        records[i] = (x, y, z, 10, 20, 30, 2)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(records.tobytes())


def test_sorted_cache_is_built_and_marked(tmp_path):
    from sensaturban_fpv.plyio import PlyCloud, XyGridIndex

    rng = np.random.default_rng(4)
    points = rng.uniform(1.0, 50.0, (5000, 3))
    ply = tmp_path / "tiny.ply"
    _write_tiny_ply(ply, points)

    cloud = PlyCloud(ply)
    grid = XyGridIndex(cloud, cell=4.0, cache_dir=tmp_path).build()
    grid.ensure_sorted()

    xyz, rgb = grid.sorted_arrays()
    assert grid.sorted_marker.exists()
    assert xyz.shape == (5000, 3) and rgb.shape == (5000, 3)

    # The permuted copy must be the same points, and in bucket order.
    original = cloud.xyz().astype(np.float64)
    assert np.allclose(np.asarray(xyz), original[grid.order], atol=1e-5)
    cell_of = grid.cell_ids(np.asarray(xyz)[:, 0], np.asarray(xyz)[:, 1], clip=True)
    assert np.all(np.diff(cell_of) >= 0), "bucket-ordered array is not ordered"


def test_truncated_sorted_cache_is_rebuilt(tmp_path, monkeypatch):
    """A cache left by an interrupted build must be rebuilt, not trusted.

    The failure this guards against is silent: ``open_memmap`` allocates the
    whole file immediately, so a killed build leaves a full-size file whose tail
    is zeros, and zeros are valid coordinates.  Rendering from one produced
    empty frames with no error at all.
    """
    from sensaturban_fpv.plyio import PlyCloud, XyGridIndex

    rng = np.random.default_rng(9)
    points = rng.uniform(1.0, 50.0, (4000, 3))
    ply = tmp_path / "tiny.ply"
    _write_tiny_ply(ply, points)

    cloud = PlyCloud(ply)
    grid = XyGridIndex(cloud, cell=4.0, cache_dir=tmp_path).build()
    grid.ensure_sorted()
    xyz_path, rgb_path = grid.sorted_paths

    # Simulate the interrupted build: full-size files, zeros in the tail, and
    # no completion marker.
    grid.sorted_marker.unlink()
    with xyz_path.open("r+b") as handle:
        handle.seek(xyz_path.stat().st_size // 2)
        handle.write(b"\x00" * min(4096, xyz_path.stat().st_size // 4))
    assert float((np.asarray(np.load(xyz_path, mmap_mode="r")[::10])[:, 0]
                  == 0).mean()) > 0.0

    grid.ensure_sorted()
    assert grid.sorted_marker.exists(), "rebuild did not mark the cache complete"
    xyz, _ = grid.sorted_arrays()
    assert float((np.asarray(xyz)[:, 0] == 0).mean()) == 0.0, "corrupt tail survived"
    assert np.allclose(np.asarray(xyz), cloud.xyz().astype(np.float64)[grid.order],
                       atol=1e-5)


def test_read_sorted_matches_direct_gather(tmp_path):
    """The span-read optimisation must return exactly what a plain gather would."""
    from sensaturban_fpv.plyio import PlyCloud, XyGridIndex

    rng = np.random.default_rng(13)
    points = rng.uniform(1.0, 60.0, (9000, 3))
    ply = tmp_path / "tiny.ply"
    _write_tiny_ply(ply, points)
    cloud = PlyCloud(ply)
    grid = XyGridIndex(cloud, cell=4.0, cache_dir=tmp_path).build()
    grid.ensure_sorted()
    xyz, _ = grid.sorted_arrays()

    slots = grid.query_radius_positions(30.0, 30.0, 25.0, ((40.0, 1), (60.0, 2)))
    assert slots.size
    assert np.all(np.diff(slots) > 0), "positions must be ascending for the span read"
    fast = grid.read_sorted(xyz, slots)
    slow = np.asarray(xyz)[slots]
    assert np.array_equal(fast, slow)


def test_slot_positions_and_vertex_ids_describe_the_same_points(tmp_path):
    """``order[slot]`` must be the vertex whose data sits at that slot.

    Note that ``query_radius_positions`` is not the same query as
    ``query_radius``: the former keeps any cell whose nearest edge is in range,
    so it returns a superset that includes points outside the radius.  Renderers
    distance-filter afterwards, but the two are not interchangeable and this
    test asserts the actual invariant rather than an assumed equality.
    """
    from sensaturban_fpv.plyio import PlyCloud, XyGridIndex

    rng = np.random.default_rng(21)
    points = rng.uniform(1.0, 60.0, (6000, 3))
    ply = tmp_path / "tiny.ply"
    _write_tiny_ply(ply, points)
    cloud = PlyCloud(ply)
    grid = XyGridIndex(cloud, cell=4.0, cache_dir=tmp_path).build()
    grid.ensure_sorted()
    xyz, rgb = grid.sorted_arrays()

    slots = grid.query_radius_positions(30.0, 30.0, 15.0, ((40.0, 1),))
    assert slots.size
    vertices = grid.order[slots]
    assert np.allclose(np.asarray(xyz)[slots], cloud.xyz(vertices), atol=1e-5)
    assert np.array_equal(np.asarray(rgb)[slots], cloud.rgb(vertices))

    # And the decimated query must return a subset of the undecimated one.
    full = grid.query_radius_positions(30.0, 30.0, 15.0, ((np.inf, 1),))
    assert set(slots.tolist()).issubset(set(full.tolist()))


def test_overall_verdict_ignores_undecided_gates():
    from sensaturban_fpv.validate_rendering import overall

    v = overall([{"gate": "a", "passed": True}, {"gate": "b", "passed": None}])
    assert v["all_passed"] is True
    assert v["undecided"] == ["b"]
    assert overall([{"gate": "a", "passed": False}])["all_passed"] is False
