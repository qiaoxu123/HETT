"""World-coordinate alignment of the top-down and oblique views of one landmark.

Both views are pictures of the same block of measured points, so a landmark's
own points are the correspondence: project them through each camera and the pair
of pixels a point lands on *is* the alignment.  Nothing here is estimated from
image content, so a wrong correspondence would be a coordinate bug, not a
modelling failure.

Three products per candidate landmark:

``td_mask`` / ``o_mask``
    Which pixels of the top-down crop and of the oblique crop show that
    landmark.  Pixels whose surface belongs to something else are excluded, so
    the masks are occlusion-correct rather than bounding boxes.

``correspondence``
    A sparse ``(td_patch, o_patch, weight)`` table over the encoder's patch
    grid, counting how many of the landmark's 3D points each patch pair shares.
    This is what lets patch fusion pair the right patches instead of all of them.

Two conventions matter and are pinned by tests:

* The top-down crop is axis-aligned in XY and centred on the landmark; the
  oblique crop is an image-space square centred on the landmark's projection.
  Both span the same physical extent, so the landmark's share of the crop is the
  same in both views by construction.
* A point is visible in the oblique view when the pixel it lands on was won by a
  point inside the landmark's own box, or when the rendered depth there is
  farther than the point's range.  The first test is exact but loses points the
  level-of-detail decimation skipped; the second recovers them and is only
  applied with a tight tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .pointcloud_renderer import Camera, project_points

# A patch is a fixed square of the encoder input; the grid size is derived from
# the token count rather than assumed, because the processor's output resolution
# is a property of the checkpoint.
CROP_EXTENT_MIN_M = 24.0
CROP_EXTENT_MAX_M = 150.0
# Points per landmark are capped: a large warehouse on a dense block holds
# millions, and the masks and the correspondence table are both statistics that
# are already converged at a fraction of that.
MAX_LANDMARK_POINTS = 1_500_000


def crop_square(image, centre_u, centre_v, side_px, border: str = "replicate"):
    """Crop centred on a pixel, clamped to the image, padded if it runs off.

    ``border='constant'`` pads with black -- the value the rasters already use
    for nodata -- rather than smearing the edge row into the padding, which
    would invent texture that was never measured.  Every crop in this round uses
    the constant border, so the two views are padded the same way.
    """
    import cv2

    h, w = image.shape[:2]
    side = int(np.clip(side_px, 16, min(h, w)))
    half = side // 2
    cu, cv_ = int(round(centre_u)), int(round(centre_v))
    c0, r0 = cu - half, cv_ - half
    c1, r1 = c0 + side, r0 + side

    pad_l, pad_t = max(0, -c0), max(0, -r0)
    pad_r, pad_b = max(0, c1 - w), max(0, r1 - h)
    mode = cv2.BORDER_REPLICATE if border == "replicate" else cv2.BORDER_CONSTANT
    canvas = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, mode)
    return canvas[r0 + pad_t:r1 + pad_t, c0 + pad_l:c1 + pad_l]


def landmark_point_set(grid, sorted_xyz, position, dimension, footprint=None,
                       margin: float = 1.0,
                       max_points: int = MAX_LANDMARK_POINTS) -> np.ndarray:
    """The landmark's own measured points, as an ``(N, 3)`` array.

    Box first (which the bucket index can answer cheaply), then CityRefer's
    footprint polygon when there is one -- see :func:`polygon_membership` for
    why the two are intersected rather than either being trusted alone.
    """
    lo, hi = landmark_bounds(position, dimension, margin)
    pts = grid.query_box_slots(lo, hi, sorted_xyz, max_points=max_points)
    if footprint is not None and len(pts):
        pts = pts[polygon_membership(pts[:, :2], footprint)]
    return pts


def candidate_crops(raster_info, camera, render_rgb, position, extent_m) -> dict:
    """The top-down and oblique crops for one candidate, on the same ground extent.

    Returns ``None`` for either view when that view cannot show the landmark --
    a landmark whose oblique crop is centred far outside the frame has no
    oblique appearance to encode, and saying so is not the same as encoding an
    empty crop.
    """
    out = {"td": None, "oblique": None, "td_window": None, "oblique_window": None}
    if raster_info is not None:
        win = topdown_window(raster_info[1], raster_info[2], position, extent_m)
        out["td_window"] = win
        out["td"] = crop_square(raster_info[0], win.centre[0], win.centre[1],
                                win.side, border="constant")
    win = oblique_window(camera, position, extent_m)
    if win is not None:
        out["oblique_window"] = win
        out["oblique"] = crop_square(render_rgb, win.centre[0], win.centre[1],
                                     win.side, border="constant")
    return out


def crop_extent_m(dimension) -> float:
    """Physical side of the square crop, identical for both views.

    Matches the extent rule the earlier resolution and fusion rounds used, so a
    crop here covers the same ground as the crops those rounds scored.
    """
    return float(np.clip(2.5 * max(dimension[0], dimension[1]),
                         CROP_EXTENT_MIN_M, CROP_EXTENT_MAX_M))


def landmark_bounds(position, dimension, margin: float = 0.0) -> tuple:
    """Axis-aligned box of a CityRefer object as ``(lo, hi)`` 3-vectors."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    half = np.asarray(dimension, dtype=np.float64).reshape(3) / 2.0 + margin
    return pos - half, pos + half


def box_membership(xyz: np.ndarray, lo, hi) -> np.ndarray:
    xyz = np.asarray(xyz)
    return np.all((xyz >= lo[None, :]) & (xyz <= hi[None, :]), axis=1)


def polygon_membership(xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for a batch of XY points.

    CityRefer carries the object's footprint as a polygon, but it is not a
    strict refinement of the bounding box: for objects under a few metres it is
    padded to a minimum size and is *larger* than the box, while for a long
    diagonal terrace it is markedly tighter.  Intersecting the two -- box test
    and polygon test both required -- therefore takes the tight one wherever the
    polygon is informative and falls back to the box where it is not, without
    having to decide between them per object.
    """
    xy = np.asarray(xy, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return np.ones(len(xy), dtype=bool)
    if xy.size == 0:
        return np.zeros(0, dtype=bool)
    x1 = poly[:, 0][None, :]
    y1 = poly[:, 1][None, :]
    x2 = np.roll(poly[:, 0], -1)[None, :]
    y2 = np.roll(poly[:, 1], -1)[None, :]

    def test(chunk: np.ndarray) -> np.ndarray:
        x = chunk[:, 0][:, None]
        y = chunk[:, 1][:, None]
        # The half-open rule on y keeps a vertex from being counted twice.
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (y - y1) * (x2 - x1) / np.where(crosses, y2 - y1, 1.0)
        return (np.sum(crosses & (x < x_at), axis=1) % 2) == 1

    # Chunked because every temporary here is (points x vertices) and a large
    # landmark on a dense block supplies millions of points.
    if len(xy) <= 250_000:
        return test(xy)
    return np.concatenate([test(xy[i:i + 250_000])
                           for i in range(0, len(xy), 250_000)])


@dataclass
class CropWindow:
    """Where a landmark's crop sits in a source image, and how it maps to patches.

    ``centre`` is the crop's centre in source pixels, ``side`` its side in source
    pixels.  Patch ``(pr, pc)`` covers source pixels
    ``[centre - side/2 + pc*side/P, ... + (pc+1)*side/P)`` on each axis, which is
    where the encoder's own resize lands that patch: the processor scales the
    crop to the encoder input, and patch ``p`` of that input is a ``16 x 16``
    block, so patch index is proportional to the position inside the crop.
    """

    centre: tuple
    side: float
    image_shape: tuple
    patches: int

    @property
    def _half(self) -> float:
        return self.side / 2.0

    def patch_of_uv(self, u: np.ndarray, v: np.ndarray) -> tuple:
        """Patch indices for source pixels, or ``None`` for those outside the crop."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        cu, cv = self.centre
        with np.errstate(invalid="ignore"):
            fx = (u - (cu - self._half)) / self.side
            fy = (v - (cv - self._half)) / self.side
        ok = (fx >= 0.0) & (fx < 1.0) & (fy >= 0.0) & (fy < 1.0)
        pc = np.clip((fx * self.patches).astype(np.int64), 0, self.patches - 1)
        pr = np.clip((fy * self.patches).astype(np.int64), 0, self.patches - 1)
        return pr, pc, ok

    def patch_centres_uv(self) -> tuple:
        """Source-pixel coordinates of every patch centre, as ``(U, V)`` grids."""
        cu, cv = self.centre
        step = self.side / self.patches
        xs = cu - self._half + (np.arange(self.patches) + 0.5) * step
        ys = cv - self._half + (np.arange(self.patches) + 0.5) * step
        return np.meshgrid(xs, ys)


@dataclass
class CandidateGeometry:
    """Everything the fusion needs about one candidate's appearance in both views."""

    candidate_id: int
    extent_m: float
    td_patches: np.ndarray                 # (P, P) bool, landmark footprint patches
    o_patches: np.ndarray                  # (P, P) bool, visible-landmark patches
    td_patch_count: int
    o_patch_count: int
    td_point_count: int
    o_visible_point_count: int
    td_pixels: int
    o_pixels: int
    corr_td: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    corr_o: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    corr_weight: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))

    @property
    def has_correspondence(self) -> bool:
        return self.corr_td.size > 0

    def summary(self) -> dict:
        return {
            "candidate_id": int(self.candidate_id),
            "crop_extent_m": float(self.extent_m),
            "td_patches": int(self.td_patch_count),
            "o_patches": int(self.o_patch_count),
            "td_point_count": int(self.td_point_count),
            "o_visible_point_count": int(self.o_visible_point_count),
            "td_pixels": int(self.td_pixels),
            "o_pixels": int(self.o_pixels),
            "correspondence_pairs": int(self.corr_td.size),
            "correspondence_shared_points": float(self.corr_weight.sum())
            if self.corr_weight.size else 0.0,
        }


def topdown_window(raster_origin, raster_resolution, position, extent_m) -> CropWindow:
    """Crop window for the shipped orthophoto, in its own pixel frame."""
    ox, oy = float(raster_origin[0]), float(raster_origin[1])
    res = float(raster_resolution)
    col = (float(position[0]) - ox) / res
    row = (oy - float(position[1])) / res
    return CropWindow(centre=(col, row), side=extent_m / res, image_shape=None,
                      patches=0)


def oblique_window(camera: Camera, position, extent_m) -> CropWindow | None:
    """Crop window for a perspective frame, sized by the landmark's angular extent.

    ``None`` when the landmark centre is behind the camera or off-frame far
    enough that no part of the crop can show it.
    """
    proj = project_point(np.asarray(position, dtype=np.float64), camera)
    if proj is None:
        return None
    metric = max(proj["metric"], 1e-3)
    side = camera.focal * extent_m / metric
    cu, cv = proj["u"], proj["v"]
    margin = 2.0 * side
    if not (-margin <= cu <= camera.width + margin
            and -margin <= cv <= camera.height + margin):
        return None
    return CropWindow(centre=(cu, cv), side=side,
                      image_shape=(camera.height, camera.width), patches=0)


def project_point(position, camera: Camera) -> dict | None:
    """Project one world point; ``None`` when it is not in front of the camera."""
    right, up, forward = camera.basis()
    rel = np.asarray(position, dtype=np.float64) - camera.position
    z_cam = float(rel @ forward)
    if z_cam <= camera.near or z_cam > camera.far:
        return None
    metric = float(np.linalg.norm(rel))
    focal = camera.focal
    u = camera.width / 2.0 + focal * float(rel @ right) / z_cam
    v = camera.height / 2.0 - focal * float(rel @ up) / z_cam
    return {"u": float(u), "v": float(v), "z_cam": z_cam, "metric": metric}


def visible_slots(render_result, box_lo, box_hi, sorted_xyz) -> np.ndarray:
    """Slots whose winning pixel the landmark owns -- the occlusion-exact set.

    ``render_result.point_id`` holds bucket slots for a grid render, so the
    winning world position is one gather into the bucket-ordered array.  A pixel
    won by a point inside the landmark's box is unambiguously landmark surface.
    """
    slots = render_result.point_id
    valid = slots >= 0
    if not valid.any():
        return np.empty(0, dtype=np.int64)
    win = slots[valid].astype(np.int64)
    world = np.asarray(sorted_xyz[win])
    inside = box_membership(world, box_lo, box_hi)
    return np.unique(win[inside])


def _render_depth_at(render_result, rows, cols) -> np.ndarray:
    h, w = render_result.depth.shape
    rows = np.clip(rows, 0, h - 1)
    cols = np.clip(cols, 0, w - 1)
    return render_result.depth[rows, cols]


def candidate_geometry(
    candidate_id: int,
    position,
    dimension,
    cloud_xyz: np.ndarray,
    raster_info,
    camera: Camera,
    render_result,
    sorted_xyz,
    patch_grid: int,
    margin: float = 1.0,
    occlusion_tolerance: float = 0.75,
    footprint: np.ndarray | None = None,
) -> CandidateGeometry | None:
    """Masks and patch correspondence for one candidate in both views.

    ``cloud_xyz`` is the landmark's own measured points (already restricted to
    its box by the caller).  ``footprint`` is CityRefer's XY polygon, intersected
    with the box when given.  Returns ``None`` when either view cannot show the
    landmark at all -- no correspondence to build is the honest answer there,
    not an empty mask that a downstream model would read as "unusual object".
    """
    lo, hi = landmark_bounds(position, dimension, margin)

    extent = crop_extent_m(dimension)
    o_win = oblique_window(camera, position, extent)
    if o_win is None or raster_info is None:
        return None
    td_win = topdown_window(raster_info[1], raster_info[2], position, extent)
    o_win.patches = patch_grid
    td_win.patches = patch_grid
    raster = raster_info[0]

    pts = np.asarray(cloud_xyz, dtype=np.float64)
    if pts.size == 0:
        return None
    if footprint is not None:
        pts = pts[polygon_membership(pts[:, :2], footprint)]
        if pts.size == 0:
            return None
    grid = patch_grid

    # ---- top-down: orthographic, one point one pixel, no occlusion to resolve
    ox, oy = float(raster_info[1][0]), float(raster_info[1][1])
    res = float(raster_info[2])
    col = (pts[:, 0] - ox) / res
    row = (oy - pts[:, 1]) / res
    td_pr, td_pc, td_ok = td_win.patch_of_uv(col, row)
    td_mask = np.zeros((grid, grid), dtype=bool)
    if td_ok.any():
        td_mask[td_pr[td_ok], td_pc[td_ok]] = True

    # ---- oblique: project, then decide visibility against the z-buffer
    proj = project_points(pts, camera)
    o_mask = np.zeros((grid, grid), dtype=bool)
    o_vis = np.zeros(len(pts), dtype=bool)
    if proj["u"].size:
        src = proj["pos"]                               # row index into `pts`
        rows = proj["v"].astype(np.int64)
        cols = proj["u"].astype(np.int64)
        depth = _render_depth_at(render_result, rows, cols)
        winner_slot = render_result.point_id[
            np.clip(rows, 0, camera.height - 1), np.clip(cols, 0, camera.width - 1)]
        win_world = np.full((proj["u"].size, 3), np.nan)
        has_winner = winner_slot >= 0
        if has_winner.any():
            win_world[has_winner] = np.asarray(
                sorted_xyz[winner_slot[has_winner].astype(np.int64)])
        won_by_landmark = has_winner & box_membership(
            np.nan_to_num(win_world, nan=np.inf), lo, hi)
        in_front = np.isfinite(depth) & (proj["metric"] <= depth + occlusion_tolerance)
        seen = won_by_landmark | in_front
        o_pr, o_pc, o_in_crop = o_win.patch_of_uv(proj["u"], proj["v"])
        keep = o_in_crop & seen
        if keep.any():
            o_mask[o_pr[keep], o_pc[keep]] = True
        vis_rows = np.zeros(len(pts), dtype=bool)
        vis_rows[src[keep]] = True
        o_vis = vis_rows

    # ---- correspondence: one 3D point contributes one patch pair each side
    corr_td = corr_o = corr_w = np.empty(0)
    if proj["u"].size:
        # Everything from here is in projected-point space: `src` maps each
        # projected point back to its row in `pts`, so the top-down patch index
        # has to be taken through it rather than indexed by the projected mask.
        src = proj["pos"]
        shared = td_ok[src] & o_in_crop & seen
        if shared.any():
            a = td_pr[src][shared].astype(np.int64) * grid + td_pc[src][shared]
            b = o_pr[shared].astype(np.int64) * grid + o_pc[shared]
            flat = np.bincount(a * (grid * grid) + b, minlength=grid ** 4)
            nz = np.flatnonzero(flat)
            corr_td, corr_o = nz // (grid * grid), nz % (grid * grid)
            corr_w = flat[nz].astype(np.float32)

    return CandidateGeometry(
        candidate_id=int(candidate_id),
        extent_m=float(extent),
        td_patches=td_mask,
        o_patches=o_mask,
        td_patch_count=int(td_mask.sum()),
        o_patch_count=int(o_mask.sum()),
        td_point_count=int(td_ok.sum()),
        o_visible_point_count=int(o_vis.sum()),
        td_pixels=int(len(pts)),
        o_pixels=int(proj["u"].size),
        corr_td=corr_td.astype(np.int64),
        corr_o=corr_o.astype(np.int64),
        corr_weight=corr_w,
    )


def correspondence_matrix(geom: CandidateGeometry, patch_grid: int,
                          symmetric: bool = True) -> np.ndarray:
    """Dense row-normalised patch correspondence as ``(grid**2, grid**2)``.

    ``W[p, q]`` is the share of patch ``p``'s shared points that also land on
    patch ``q``.  Rows sum to one where the patch has any correspondence at all
    and are zero elsewhere, which is what makes the fusion a weighted gather
    rather than a learnable all-to-all.
    """
    n = patch_grid * patch_grid
    W = np.zeros((n, n), dtype=np.float32)
    if geom.corr_td.size:
        np.add.at(W, (geom.corr_td, geom.corr_o), geom.corr_weight)
    if symmetric and W.any():
        W = W + W.T
    row_sum = W.sum(axis=1, keepdims=True)
    np.divide(W, row_sum, out=W, where=row_sum > 0)
    return W


def shuffled_correspondence(geom: CandidateGeometry, patch_grid: int,
                            seed: int) -> CandidateGeometry:
    """The same geometry with the patch pairing permuted: the random control.

    The permutation is confined to the oblique patches the landmark actually
    occupies, so the shuffled pairing still draws only from landmark patches and
    still uses every weight unchanged.  What is destroyed is only *which*
    top-down patch goes with which oblique patch.  A fusion whose score does not
    move under this permutation never used the world coordinates the pairing
    came from.
    """
    rng = np.random.default_rng(seed)
    n = patch_grid * patch_grid
    pool = np.flatnonzero(np.asarray(geom.o_patches).ravel())
    if pool.size < 2 or geom.corr_o.size == 0:
        return geom
    perm = np.arange(n, dtype=np.int64)
    perm[pool] = rng.permutation(pool)
    return replace(geom, corr_o=perm[geom.corr_o])


def draw_mask_overlay(image: np.ndarray, mask: np.ndarray,
                      colour=(255, 60, 60), alpha: float = 0.45) -> np.ndarray:
    """Tint the boundary patches of a mask over the crop they were built for.

    ``image`` is the crop itself, so the mask's ``(P, P)`` cells map onto it
    uniformly -- the crop is by construction the exact square the window named.
    Only the boundary of the mask is tinted: flooding the interior would hide
    the very appearance the mask is supposed to be selecting.
    """
    out = np.asarray(image).copy()
    p = np.asarray(mask).astype(np.float32)
    if p.size == 0 or p.ndim != 2:
        return out
    grid = p.shape[0]
    er = p.copy()
    er[1:, :] = np.minimum(er[1:, :], p[:-1, :])
    er[:-1, :] = np.minimum(er[:-1, :], p[1:, :])
    er[:, 1:] = np.minimum(er[:, 1:], p[:, :-1])
    er[:, :-1] = np.minimum(er[:, :-1], p[:, 1:])
    rows, cols = np.nonzero((p - er) > 0.5)

    h, w = out.shape[:2]
    step_y, step_x = h / grid, w / grid
    fill = np.asarray(colour, np.float32)
    for r, c in zip(rows.tolist(), cols.tolist()):
        r0, r1 = int(round(r * step_y)), int(round((r + 1) * step_y))
        c0, c1 = int(round(c * step_x)), int(round((c + 1) * step_x))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, h), min(c1, w)
        if r1 <= r0 or c1 <= c0:
            continue
        patch = out[r0:r1, c0:c1].astype(np.float32)
        out[r0:r1, c0:c1] = np.clip(
            patch * (1 - alpha) + fill * alpha, 0, 255).astype(np.uint8)
    return out


def correspondence_coherence(geom: CandidateGeometry, patch_grid: int) -> float:
    """Mean patch-grid distance between a patch's target and its neighbours'.

    A correspondence that came from projecting the same 3D points through two
    cameras is locally smooth: adjacent top-down patches of one surface land on
    adjacent oblique patches.  The value is reported alongside the same number
    for the shuffled pairing, so a claim that the alignment carries information
    is checkable rather than asserted.
    """
    if geom.corr_td.size == 0:
        return float("nan")
    W = correspondence_matrix(geom, patch_grid, symmetric=False)
    targets = np.full((patch_grid * patch_grid, 2), np.nan)
    rows = np.flatnonzero(W.sum(axis=1) > 0)
    if rows.size < 2:
        return float("nan")
    idx = np.arange(patch_grid * patch_grid).reshape(patch_grid, patch_grid)
    for r in rows.tolist():
        q = int(np.argmax(W[r]))
        targets[r] = (q // patch_grid, q % patch_grid)
    # Neighbours are taken on the top-down patch grid, including diagonals, so
    # a diagonal surface does not read as incoherent.
    rr, cc = np.divmod(rows, patch_grid)
    diffs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = rr + dr, cc + dc
            ok = (nr >= 0) & (nr < patch_grid) & (nc >= 0) & (nc < patch_grid)
            if not ok.any():
                continue
            a = targets[rows[ok]]
            b = targets[idx[nr[ok], nc[ok]]]
            good = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            if good.any():
                diffs.append(np.linalg.norm(a[good] - b[good], axis=1))
    if not diffs:
        return float("nan")
    return float(np.concatenate(diffs).mean())
