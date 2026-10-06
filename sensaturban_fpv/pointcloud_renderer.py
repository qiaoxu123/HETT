"""Perspective RGB-D rendering of a raw coloured point cloud, with a z-buffer.

This is deliberately not a neural renderer and not a mesh: every output pixel is
the nearest *measured* point that projects into it.  No colour is invented, and
a pixel that no point reaches is reported as invalid rather than filled.

Camera convention (matches ``gsamllavanav.space.Pose4D``)
--------------------------------------------------------
CityNav stores a pose as ``(x, y, z, yaw)`` with ``yaw = arctan2(dy, dx)``, i.e.
the heading is measured counter-clockwise from ``+x`` in the world XY plane, and
``+z`` is up.  A camera at that pose looks along

    forward = (cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch))

with ``right = normalize(forward x world_up)`` and ``up = right x forward``.
For a camera facing ``+x`` with ``+z`` up this gives ``right = -y``, which is the
correct "an east-facing observer has south on their right" answer for a
right-handed z-up frame.  Image ``u`` grows along ``right`` and ``v`` grows
downward, so the projection is a plain pinhole with square pixels, no mirroring.

Two index spaces appear throughout
----------------------------------
``pos`` addresses a row of the arrays the caller handed to :func:`render`.
``id`` is the caller's own identifier for that point (the SensatUrban vertex
index, when rendering through :func:`render_cloud_region`).  Colours are looked
up by ``pos`` -- mixing the two silently renders the wrong pixels whenever the
cloud is a subset of a larger block, which it always is here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])

# Default level-of-detail shells, as (max_distance_m, keep_one_in_n).
# Density falls off as 1/d^2, so decimating the far shells still leaves them
# better sampled per pixel than the near shell; this only bounds the sort cost.
DEFAULT_LOD = ((40.0, 1), (80.0, 2), (150.0, 4))


def _hash_keep(index: np.ndarray, stride: int) -> np.ndarray:
    """Deterministic, spatially unbiased 1-in-``stride`` selection by point id."""
    if stride <= 1:
        return np.ones(index.shape, dtype=bool)
    mixed = (index.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(16)
    return (mixed % np.uint64(stride)) == 0


@dataclass
class Camera:
    position: np.ndarray
    yaw: float
    pitch: float = 0.0
    width: int = 512
    height: int = 512
    hfov_deg: float = 90.0
    near: float = 0.5
    far: float = 150.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)

    @property
    def focal(self) -> float:
        return self.width / (2.0 * np.tan(np.deg2rad(self.hfov_deg) / 2.0))

    @property
    def vfov_deg(self) -> float:
        return float(np.rad2deg(2.0 * np.arctan((self.height / 2.0) / self.focal)))

    def basis(self) -> tuple:
        """Camera axes as ``(right, up, forward)``, all unit and right-handed."""
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        forward = np.array([np.cos(self.yaw) * cp, np.sin(self.yaw) * cp, sp])
        right = np.cross(forward, WORLD_UP)
        norm = float(np.linalg.norm(right))
        if norm < 1e-6:
            # Looking straight up or down: pick a stable right vector.
            right = np.array([np.sin(self.yaw), -np.cos(self.yaw), 0.0])
        else:
            right = right / norm
        up = np.cross(right, forward)
        up = up / max(float(np.linalg.norm(up)), 1e-12)
        return right, up, forward

    def to_json(self) -> dict:
        return {
            "position": self.position.tolist(),
            "yaw_rad": float(self.yaw),
            "yaw_deg": float(np.rad2deg(self.yaw)),
            "pitch_deg": float(np.rad2deg(self.pitch)),
            "width": int(self.width), "height": int(self.height),
            "hfov_deg": float(self.hfov_deg), "vfov_deg": self.vfov_deg,
            "near": float(self.near), "far": float(self.far),
        }


@dataclass
class RenderResult:
    rgb: np.ndarray             # (H, W, 3) uint8, background where invalid
    depth: np.ndarray           # (H, W) float32 metric range in metres, nan if invalid
    valid: np.ndarray           # (H, W) bool
    density: np.ndarray         # (H, W) int32, projected points per pixel
    point_id: np.ndarray        # (H, W) caller id of the winning point, -1 if invalid
    stats: dict = field(default_factory=dict)

    @property
    def valid_ratio(self) -> float:
        return float(self.valid.mean())


def select_lod(index: np.ndarray, distance: np.ndarray, lod=DEFAULT_LOD) -> np.ndarray:
    """Boolean mask keeping near points at full rate and far ones decimated.

    Whatever lies beyond the last shell keeps that shell's stride; the caller
    separately rejects anything past ``camera.far``.
    """
    keep = np.zeros(index.shape, dtype=bool)
    prev = -np.inf  # a point at distance 0 still belongs to the first shell
    for limit, stride in lod:
        in_shell = (distance > prev) & (distance <= limit)
        if in_shell.any():
            keep |= in_shell & _hash_keep(index, stride)
        prev = limit
    beyond = distance > prev
    if beyond.any():
        keep |= beyond & _hash_keep(index, lod[-1][1] if lod else 1)
    return keep


def project_points(xyz: np.ndarray, camera: Camera, ids: np.ndarray | None = None,
                   positions: np.ndarray | None = None) -> dict:
    """Project world points into the camera.

    Returns pixel columns/rows plus both range measures; pixels outside the
    image, behind the near plane, or past the far plane are dropped.  This is
    the single place the pinhole model lives, so the unit tests can pin it down
    independently of any rendering policy.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    if ids is None:
        ids = np.arange(n, dtype=np.int64)
    if positions is None:
        positions = np.arange(n, dtype=np.int64)

    right, up, forward = camera.basis()
    rel = xyz - camera.position[None, :]
    # Explicit component arithmetic rather than ``rel @ vector``: numpy routes a
    # (N,3) @ (3,) product through BLAS, which for millions of rows costs several
    # times what three multiply-adds over contiguous columns cost.
    z_cam = rel[:, 0] * forward[0] + rel[:, 1] * forward[1] + rel[:, 2] * forward[2]
    x_cam = rel[:, 0] * right[0] + rel[:, 1] * right[1] + rel[:, 2] * right[2]
    y_cam = rel[:, 0] * up[0] + rel[:, 1] * up[1] + rel[:, 2] * up[2]
    metric2 = rel[:, 0] ** 2 + rel[:, 1] ** 2 + rel[:, 2] ** 2

    # Cheap cone rejection before the reciprocal: dividing every point and then
    # testing the pixel is the expensive way to do this, while
    # ``|x_cam| <= z_cam * tan(hfov/2)`` is the same test as a multiply and a
    # comparison.  Tested with a margin so the survivors are a superset of the
    # truly in-frame points -- the exact test below still runs on them.
    tan_h = np.tan(np.deg2rad(camera.hfov_deg) / 2.0) * 1.02
    tan_v = np.tan(np.deg2rad(camera.vfov_deg) / 2.0) * 1.02
    in_front = z_cam > camera.near
    candidate = (
        in_front
        & (metric2 <= camera.far * camera.far)
        & (np.abs(x_cam) <= z_cam * tan_h)
        & (np.abs(y_cam) <= z_cam * tan_v)
    )
    if not candidate.all():
        z_cam = z_cam[candidate]
        x_cam = x_cam[candidate]
        y_cam = y_cam[candidate]
        metric2 = metric2[candidate]
        positions = positions[candidate]
        ids = ids[candidate]

    metric = np.sqrt(metric2)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_z = 1.0 / np.maximum(z_cam, 1e-9)
        focal = camera.focal
        u = camera.width / 2.0 + focal * x_cam * inv_z
        v = camera.height / 2.0 - focal * y_cam * inv_z

    inside = (
        (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    )
    return {
        "u": u[inside], "v": v[inside],
        "z_cam": z_cam[inside], "metric": metric[inside],
        "pos": positions[inside], "id": ids[inside],
        "N_candidate": int(candidate.sum()) if candidate.ndim else int(candidate),
    }


class StageTimer:
    """Per-stage wall clock and point counts for one render call."""

    def __init__(self):
        self.times = {}
        self.counts = {}
        self._t0 = time.perf_counter()

    def stage(self, name: str):
        return _Stage(self, name)

    def count(self, name: str, value) -> None:
        self.counts[name] = int(value)

    def summary(self) -> dict:
        return {
            "timings_s": {k: round(v, 4) for k, v in self.times.items()},
            "counts": dict(self.counts),
            "T_total": round(time.perf_counter() - self._t0, 4),
        }


class _Stage:
    def __init__(self, timer: StageTimer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timer.times[self.name] = self.timer.times.get(self.name, 0.0) + (
            time.perf_counter() - self._t)
        return False


def cell_frustum_filter(camera: Camera, z_lo: float, z_hi: float, margin: float = 1.05):
    """Build a predicate that culls whole cells the camera frustum cannot reach.

    A cell is dropped only when every one of its eight corners lies outside the
    same side plane (or entirely beyond the far plane / behind the near plane),
    so this is conservative: it never removes geometry that could have been
    visible.  ``margin`` widens the cone slightly to absorb the corner test
    being done at cell granularity.
    """
    right, up, forward = camera.basis()
    tan_h = np.tan(np.deg2rad(camera.hfov_deg) / 2.0) * margin
    tan_v = np.tan(np.deg2rad(camera.vfov_deg) / 2.0) * margin
    origin = camera.position
    far2 = camera.far * camera.far
    near = camera.near

    def predicate(x_lo, y_lo, x_hi, y_hi) -> np.ndarray:
        xs = np.stack([x_lo, x_hi], axis=1)          # (C, 2)
        ys = np.stack([y_lo, y_hi], axis=1)
        zs = np.array([z_lo, z_hi])
        cx = xs[:, :, None, None]                     # (C, 2, 2, 2)
        cy = ys[:, None, :, None]
        cz = zs[None, None, None, :]
        dx = cx - origin[0]
        dy = cy - origin[1]
        dz = cz - origin[2]

        x_cam = dx * right[0] + dy * right[1] + dz * right[2]
        y_cam = dx * up[0] + dy * up[1] + dz * up[2]
        z_cam = dx * forward[0] + dy * forward[1] + dz * forward[2]
        d2 = dx * dx + dy * dy + dz * dz

        flat = (slice(None), slice(None), slice(None), slice(None))
        out_of = [
            np.all(z_cam <= near, axis=(1, 2, 3)),                 # entirely behind
            np.all(d2 > far2, axis=(1, 2, 3)),                     # entirely past far
            np.all(x_cam > z_cam * tan_h, axis=(1, 2, 3)),         # right of the cone
            np.all(x_cam < -z_cam * tan_h, axis=(1, 2, 3)),        # left of the cone
            np.all(y_cam > z_cam * tan_v, axis=(1, 2, 3)),         # above the cone
            np.all(y_cam < -z_cam * tan_v, axis=(1, 2, 3)),        # below the cone
        ]
        return ~np.any(np.stack(out_of, axis=0), axis=0)

    return predicate


def _zbuffer_lexsort(pix, metric, pos, h, w):
    """Legacy z-buffer: one two-key lexsort keyed on (pixel, range).

    Kept as the reference implementation.  It is the only one whose per-pixel
    winner is defined by an explicit sort order, so the cheaper backends are
    validated against it.
    """
    depth = np.full(h * w, np.inf, dtype=np.float64)
    winner = np.full(h * w, -1, dtype=np.int64)
    if pix.size == 0:
        return depth, winner
    order = np.lexsort((metric, pix))
    sorted_pix = pix[order]
    first = np.empty(sorted_pix.size, dtype=bool)
    first[0] = True
    np.not_equal(sorted_pix[1:], sorted_pix[:-1], out=first[1:])
    sel = order[first]
    depth[pix[sel]] = metric[sel]
    winner[pix[sel]] = pos[sel]
    return depth, winner


def _zbuffer_argsort(pix, metric, pos, h, w):
    """Single-key sort; visiting far-to-near makes the last write the nearest.

    Fancy-index assignment keeps the final value for a repeated index, so
    descending range order yields the nearest point without a second key.
    """
    depth = np.full(h * w, np.inf, dtype=np.float64)
    winner = np.full(h * w, -1, dtype=np.int64)
    if pix.size == 0:
        return depth, winner
    order = np.argsort(metric)[::-1]
    depth[pix[order]] = metric[order]
    winner[pix[order]] = pos[order]
    return depth, winner


def _zbuffer_minimum_at(pix, metric, pos, h, w):
    """Sort-free scatter reduce: ``np.minimum.at`` into the depth buffer.

    Then a second pass keeps every point sitting at its pixel's minimum range,
    and the last such write wins, which resolves ties the same way the sorted
    backends do up to point order.
    """
    depth = np.full(h * w, np.inf, dtype=np.float64)
    winner = np.full(h * w, -1, dtype=np.int64)
    if pix.size == 0:
        return depth, winner
    np.minimum.at(depth, pix, metric)
    at_min = metric <= depth[pix]
    sel = np.flatnonzero(at_min)
    winner[pix[sel]] = pos[sel]
    return depth, winner


ZBUCKET_BACKENDS = {
    "lexsort": _zbuffer_lexsort,
    "argsort": _zbuffer_argsort,
    "minimum_at": _zbuffer_minimum_at,
}


def render(
    xyz: np.ndarray,
    rgb: np.ndarray,
    camera: Camera,
    splat_radius: int = 0,
    ids: np.ndarray | None = None,
    lod=DEFAULT_LOD,
    background: tuple = (24, 26, 30),
    tile_bounds=None,
    zbuffer: str = "argsort",
    timer: StageTimer | None = None,
    id_space: str = "index",
) -> RenderResult:
    """Render ``xyz``/``rgb`` from ``camera`` with a z-buffer.

    ``splat_radius`` is the pixel radius each point is splatted over.  Radius 0
    is the raw render: one point, one pixel, so holes stay visible.  A larger
    radius only lets a point cover its immediate neighbourhood; the nearest
    point still wins every pixel it reaches, so this is still a z-buffer using
    measured geometry only.
    """
    timer = timer or StageTimer()
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    n = len(xyz)
    if ids is None:
        ids = np.arange(n, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)

    h, w = camera.height, camera.width
    bg = np.array(background, dtype=np.uint8)
    timer.count("N_supplied", n)

    with timer.stage("T_range_filter"):
        rel_dist = xyz - camera.position[None, :]
        distance = np.sqrt(
            rel_dist[:, 0] ** 2 + rel_dist[:, 1] ** 2 + rel_dist[:, 2] ** 2)
        del rel_dist
        in_range = distance <= camera.far
        if lod:
            in_range &= select_lod(ids, distance, lod)
        keep_pos = np.flatnonzero(in_range)
    timer.count("N_in_range", len(keep_pos))

    with timer.stage("T_project"):
        # project_points rejects on the cone before dividing, so points outside
        # the frame cost three dot products and no reciprocal.
        proj = project_points(
            xyz[keep_pos], camera, ids=ids[keep_pos], positions=keep_pos
        )
    timer.count("N_projected", proj["u"].size)
    timer.count("N_in_fov", proj["u"].size)

    with timer.stage("T_zbuffer"):
        # Pixel assignment truncates the float column/row.  u and v are already
        # non-negative here, so truncation is floor: pixel (r, c) owns
        # [c, c+1) x [r, r+1).  The device backend uses the same rule.
        pix = (proj["v"].astype(np.int64) * w + proj["u"].astype(np.int64)) \
            if proj["u"].size else np.empty(0, dtype=np.int64)
        depth_flat, winner_pos = ZBUCKET_BACKENDS[zbuffer](
            pix, proj["metric"], proj["pos"], h, w)
    timer.count("N_unique_pixels", int(np.isfinite(depth_flat).sum()))

    with timer.stage("T_density"):
        # bincount rather than np.add.at: both count points per pixel, but
        # add.at is an unbuffered ufunc loop and roughly an order slower.
        if pix.size:
            density = np.bincount(pix, minlength=h * w).astype(np.int32)
        else:
            density = np.zeros(h * w, dtype=np.int32)

    if splat_radius > 0 and proj["u"].size:
        with timer.stage("T_splat"):
            depth_flat, winner_pos = _splat(
                proj, rgb, depth_flat, winner_pos, splat_radius, h, w
            )

    with timer.stage("T_finalize"):
        valid_flat = winner_pos >= 0
        out_rgb = np.broadcast_to(bg, (h * w, 3)).copy()
        if valid_flat.any():
            out_rgb[valid_flat] = rgb[winner_pos[valid_flat]]

        id_flat = np.full(h * w, -1, dtype=np.int64)
        if valid_flat.any():
            id_flat[valid_flat] = ids[winner_pos[valid_flat]]
    timer.count("N_visible_points", int(valid_flat.sum()))

    valid2d = valid_flat.reshape(h, w)
    horizon_row = horizon_row_of(camera)
    # Above the horizon there is nothing to hit, so a low whole-frame valid
    # ratio from a high-altitude pose says "sky", not "sparse cloud".  The
    # below-horizon figure is the one that measures geometric coverage.
    below = valid2d[int(np.clip(np.ceil(horizon_row), 0, h)):, :]
    below_ratio = float(below.mean()) if below.size else 0.0

    # And a ray that has left the tile cannot be filled by this block at all.
    in_tile_window = None
    in_tile_ratio = None
    if tile_bounds is not None:
        exit_t = tile_exit_distance(camera, tile_bounds[0], tile_bounds[1])
        in_tile_window = exit_t >= camera.far
        if in_tile_window.any():
            in_tile_ratio = float(valid2d[in_tile_window].mean())

    result = RenderResult(
        rgb=out_rgb.reshape(h, w, 3),
        depth=np.where(valid_flat, depth_flat, np.nan).astype(np.float32).reshape(h, w),
        valid=valid2d,
        density=density.reshape(h, w),
        point_id=id_flat.reshape(h, w),
        stats={
            "camera": camera.to_json(),
            "points_supplied": int(n),
            "points_in_range": int(in_range.sum()),
            "points_projected": int(proj["u"].size),
            "points_rendered": int(valid_flat.sum()),
            "splat_radius_px": int(splat_radius),
            "valid_pixel_ratio": float(valid_flat.mean()),
            "horizon_row": horizon_row,
            "sky_pixel_ratio": float(np.clip(horizon_row, 0, h) / h),
            "below_horizon_valid_ratio": below_ratio,
            "in_tile_pixel_ratio": (float(in_tile_window.mean())
                                    if in_tile_window is not None else None),
            "in_tile_valid_ratio": in_tile_ratio,
            "zbuffer_backend": zbuffer,
            "id_space": id_space,
            **timer.summary(),
        },
    )
    return result


def tile_exit_distance(camera: Camera, lo, hi) -> np.ndarray:
    """Per-pixel distance at which the view ray leaves the block's XY footprint.

    A SensatUrban block is a finite tile roughly 400 m across.  A pixel whose ray
    has already left the tile before reaching ``camera.far`` could not have been
    filled by this block no matter how dense it is, so emptiness there is a
    property of the data extent rather than of the renderer.
    """
    h, w = camera.height, camera.width
    right, up, forward = camera.basis()
    cols = np.arange(w, dtype=np.float64)[None, :]
    rows = np.arange(h, dtype=np.float64)[:, None]
    focal = camera.focal
    x_cam = (cols - w / 2.0) / focal
    y_cam = -(rows - h / 2.0) / focal
    dirs = (x_cam[..., None] * right[None, None, :]
            + y_cam[..., None] * up[None, None, :]
            + forward[None, None, :])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    origin = camera.position
    exit_t = np.full((h, w), np.inf)
    for axis in (0, 1):
        d = dirs[..., axis]
        o = origin[axis]
        lo_a, hi_a = float(lo[axis]), float(hi[axis])
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = (lo_a - o) / d
            t_hi = (hi_a - o) / d
        # Leaving a slab means crossing the far plane of that slab.
        t_leave = np.where(d > 0, t_hi, np.where(d < 0, t_lo, np.inf))
        exit_t = np.minimum(exit_t, np.where(t_leave > 0, t_leave, 0.0))
    return exit_t


def horizon_row_of(camera: Camera) -> float:
    """Image row of the true horizon: where a ray at zero elevation lands.

    A direction at elevation 0 has ``y_cam = -sin(pitch)`` and
    ``z_cam = cos(pitch)``, so ``v = H/2 + focal * tan(pitch)``.  Rows below it
    can contain ground or buildings; rows above it are sky.
    """
    return camera.height / 2.0 + camera.focal * np.tan(camera.pitch)


def _splat(proj, rgb, depth_flat, winner_pos, radius, h, w):
    """Expand projected points over a pixel disk, keeping the nearest depth.

    Every projected point splats, not only the per-pixel winners: a near point
    that lost its own pixel to an even nearer neighbour may still be the nearest
    evidence for an adjacent pixel.
    """
    u, v, metric, pos = proj["u"], proj["v"], proj["metric"], proj["pos"]
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            if du * du + dv * dv > radius * radius:
                continue
            uu = np.rint(u + du).astype(np.int64)
            vv = np.rint(v + dv).astype(np.int64)
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            if not ok.any():
                continue
            p = vv[ok] * w + uu[ok]
            d = metric[ok]
            better = d < depth_flat[p]
            if better.any():
                p_b = p[better]
                depth_flat[p_b] = d[better]
                winner_pos[p_b] = pos[ok][better]
    return depth_flat, winner_pos


def render_cloud_region(
    cloud,
    grid,
    camera: Camera,
    splat_radius: int = 0,
    lod=DEFAULT_LOD,
    pad: float = 2.0,
    zbuffer: str = "argsort",
    timer: StageTimer | None = None,
    cull_frustum: bool = True,
) -> RenderResult:
    """Fetch the camera's neighbourhood through the grid index and render it.

    Decimation happens once, at cell granularity, while gathering -- so the
    points are never read off disk in the first place.  ``render`` is therefore
    called with its own LOD disabled; applying both would decimate twice.
    """
    timer = timer or StageTimer()
    reach = camera.far + pad
    keep_cell = None
    if cull_frustum:
        z_lo, z_hi = grid.exact_bounds()[0][2], grid.exact_bounds()[1][2]
        keep_cell = cell_frustum_filter(camera, z_lo, z_hi)

    with timer.stage("T_query"):
        slots = grid.query_radius_positions(
            camera.position[0], camera.position[1], reach, lod, keep_cell=keep_cell)
    timer.count("N_query", len(slots))

    with timer.stage("T_load"):
        xyz_sorted, rgb_sorted = grid.sorted_arrays()
        xyz = grid.read_sorted(xyz_sorted, slots)
        if cloud.has_rgb:
            rgb = grid.read_sorted(rgb_sorted, slots)
        else:
            rgb = np.full((len(slots), 3), 128, np.uint8)
    timer.count("N_loaded", len(slots))

    # ``point_id`` is a bucket-slot index, not a PLY vertex id; convert with
    # ``grid.order[point_id]`` if the original vertex is needed.  The
    # ``id_space`` marker tells downstream code which of the two it is holding.
    return render(xyz, rgb, camera, splat_radius=splat_radius, ids=slots, lod=None,
                  tile_bounds=(grid.lo[:2], grid.hi[:2]), zbuffer=zbuffer,
                  timer=timer, id_space="slot")


def unproject(depth: np.ndarray, camera: Camera) -> np.ndarray:
    """Back-project a depth map to world points; invalid pixels become ``nan``.

    ``depth`` holds metric range, so the ray direction is scaled to that range
    rather than to the forward component.
    """
    h, w = depth.shape
    right, up, forward = camera.basis()
    focal = camera.focal
    cols = np.arange(w, dtype=np.float64)[None, :]
    rows = np.arange(h, dtype=np.float64)[:, None]
    x_cam = (cols - w / 2.0) / focal
    y_cam = -(rows - h / 2.0) / focal
    # Ray in camera axes, then rotated into the world frame.
    rays = (x_cam[..., None] * right[None, None, :]
            + y_cam[..., None] * up[None, None, :]
            + forward[None, None, :])
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    world = camera.position[None, None, :] + rays * np.asarray(depth)[..., None]
    world[~np.isfinite(depth)] = np.nan
    return world


def reprojection_shift(camera0: Camera, camera1: Camera, depth: np.ndarray) -> dict:
    """Predicted image displacement of the visible surface between two cameras.

    The rendered depth of the first view is treated as the scene, carried into
    the second camera, and the median pixel displacement is returned.  Used as
    the independent prediction that the measured frame-to-frame shift (and the
    yaw-rotation shift) has to agree with.
    """
    world = unproject(depth, camera0)
    h, w = depth.shape
    ok = np.isfinite(world).all(axis=-1)
    if not ok.any():
        return {"du": float("nan"), "dv": float("nan"), "n": 0}

    rows, cols = np.nonzero(ok)
    pts = world[ok]
    proj = project_points(pts, camera1)
    if proj["u"].size == 0:
        return {"du": float("nan"), "dv": float("nan"), "n": 0}

    # project_points already dropped everything outside the second camera's
    # frustum; `pos` maps each survivor back to its row in `pts`, and hence to
    # the pixel it came from in the first view.
    du = proj["u"] - cols[proj["pos"]]
    dv = proj["v"] - rows[proj["pos"]]
    return {
        "du": float(np.median(du)), "dv": float(np.median(dv)),
        "du_mean": float(np.mean(du)), "dv_mean": float(np.mean(dv)),
        "n": int(len(du)),
    }


def estimate_shift(a: np.ndarray, b: np.ndarray, max_shift: int | None = None) -> dict:
    """Phase-correlation shift, signed so that ``b(u, v) ~= a(u - du, v - dv)``.

    The convention is pinned by a unit test against a synthetically shifted
    image; every caller compares its output against :func:`reprojection_shift`,
    so a sign error cannot pass silently.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=-1)
        b = b.mean(axis=-1)
    a = a - a.mean()
    b = b - b.mean()
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2(a * window)
    fb = np.fft.fft2(b * window)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifft2(cross).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dv, du = peak
    if dv > a.shape[0] // 2:
        dv -= a.shape[0]
    if du > a.shape[1] // 2:
        du -= a.shape[1]
    # corr[k] = sum_n a[n] * b[n - k], so for b = shift(a, +d) the peak sits at
    # k = -d.  Negating makes the returned value the displacement itself.
    return {"du": float(-du), "dv": float(-dv), "peak": float(corr[peak])}


def rasterize_topdown(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    origin: tuple,
    resolution: float,
    shape: tuple,
    z_mode: str = "max",
) -> dict:
    """Orthographic XY binning into a north-up grid, PDAL ``writers.gdal`` style.

    ``origin`` is the grid's top-left (x, y) corner and ``shape`` is (rows, cols).
    Rows increase as ``y`` decreases, matching the rasters in ``data/rgbd``.
    Returns the height map (nodata ``nan``), the mean RGB, and per-cell counts.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    rows, cols = shape
    ox, oy = float(origin[0]), float(origin[1])

    col = np.floor((xyz[:, 0] - ox) / resolution).astype(np.int64)
    row = np.floor((oy - xyz[:, 1]) / resolution).astype(np.int64)
    ok = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    flat = (row[ok] * cols + col[ok])

    height = np.full(rows * cols, np.nan)
    if flat.size:
        if z_mode == "max":
            # lexsort's last key is primary: group by cell, largest z last.
            order = np.lexsort((xyz[ok, 2], flat))
            sorted_flat = flat[order]
            last = np.empty(sorted_flat.size, dtype=bool)
            last[-1] = True
            np.not_equal(sorted_flat[1:], sorted_flat[:-1], out=last[:-1])
            sel = order[last]
        elif z_mode == "min":
            order = np.lexsort((-xyz[ok, 2], flat))
            sorted_flat = flat[order]
            last = np.empty(sorted_flat.size, dtype=bool)
            last[-1] = True
            np.not_equal(sorted_flat[1:], sorted_flat[:-1], out=last[:-1])
            sel = order[last]
        else:
            raise ValueError(f"unsupported z_mode {z_mode!r}")
        height[flat[sel]] = xyz[ok, 2][sel]

    counts = np.bincount(flat, minlength=rows * cols).reshape(rows, cols)
    colour = np.zeros((rows * cols, 3), dtype=np.float64)
    if rgb is not None and flat.size:
        rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)[ok]
        for c in range(3):
            colour[:, c] = np.bincount(flat, weights=rgb[:, c], minlength=rows * cols)
        nz = counts.reshape(-1) > 0
        colour[nz] /= counts.reshape(-1)[nz, None]

    return {
        "height": height.reshape(rows, cols).astype(np.float32),
        "rgb": np.clip(colour.reshape(rows, cols, 3), 0, 255).astype(np.uint8),
        "counts": counts,
        "origin": (ox, oy),
        "resolution": float(resolution),
        "shape": (rows, cols),
    }


def shade_height(height: np.ndarray) -> np.ndarray:
    """Grey height map with nodata left black, for quick visual inspection."""
    out = np.zeros(height.shape + (3,), dtype=np.uint8)
    finite = np.isfinite(height)
    if finite.any():
        lo, hi = np.percentile(height[finite], [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((height - lo) / (hi - lo), 0, 1)
        grey = (scaled * 255).astype(np.uint8)
        out[finite] = np.stack([grey[finite]] * 3, axis=-1)
    return out


def depth_histogram(depth: np.ndarray, bins: int = 32) -> dict:
    """Summary statistics used by the depth-degeneracy gate."""
    depth = np.asarray(depth)
    finite = np.isfinite(depth)
    values = depth[finite]
    if values.size == 0:
        return {"count": 0, "nonfinite_fraction": 1.0}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "std": float(values.std()),
        "unique_rounded": int(np.unique(np.round(values, 2)).size),
        "hist_counts": counts.tolist(),
        "hist_edges": edges.tolist(),
        "nonfinite_fraction": float(1.0 - finite.mean()),
    }
