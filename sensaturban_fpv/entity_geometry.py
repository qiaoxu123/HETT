"""World-coordinate support, multi-scale crops and cross-view masks for a *target entity*.

The unit here is a target entity of any CityRefer type -- a building, a car, a
wall, a parking area, a piece of street furniture.  Nothing in this module
branches on the type: what varies with type is the entity's *physical size*,
and that enters only through the measured extent of its own point support.  The
one type-keyed table is ``CLASS_IDS``, a dataset dictionary from CityRefer's
object-type names to SensatUrban's semantic class ids, and it is data rather
than behaviour -- it says which points are candidates for belonging to the
entity, not what to do with an entity once it has points.

Two things make the module general rather than building-shaped:

**Support is built by a recorded ladder.**  An annotated box intersected with
the annotated footprint and the matching semantic class is the preferred
support; when that yields too few points the ladder falls through to wider
boxes and then to a radius around the annotated centre.  Which rung was used is
returned, never inferred, so a claim about cars can be checked against how many
of them actually had a car-shaped support.

**Two scales, both derived from the entity's own size.**  A tight crop sized to
the entity answers "what does it look like"; a context crop around it answers
"what is it next to".  A car and a warehouse cannot share a crop, and with a
size-derived rule neither has to.

As before, both views are pictures of the same measured points, so a landmark's
own points *are* the correspondence: project them through each camera and the
pair of pixels a point lands on is the alignment.  A wrong correspondence would
be a coordinate bug, not a modelling failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .pointcloud_renderer import Camera, project_points

# The dataset's own type->class table (SensatUrban's 13 classes).  This is the
# only place an object type is named, it is pure data, and it exists to decide
# which measured points may belong to an entity -- not to change behaviour per
# type anywhere downstream.
CLASS_IDS = {
    "Ground": 0, "HighVegetation": 1, "Building": 2, "Wall": 3, "Bridge": 4,
    "Parking": 5, "Rail": 6, "TrafficRoad": 7, "StreetFurniture": 8, "Car": 9,
    "Footpath": 10, "Bike": 11, "Water": 12,
}
# Reporting and sampling groups.  A round that only measured buildings could
# claim a general result from a building-only test, so the sampling quotas and
# every reported table are per group, and the groups are named here once.
TYPE_GROUPS = {
    "building": ("Building",),
    "vehicle": ("Car", "Bike"),
    "other": ("StreetFurniture", "Wall", "Parking", "TrafficRoad", "Footpath",
              "HighVegetation", "Ground", "Water", "Bridge", "Rail"),
}
GROUP_OF = {t: g for g, types in TYPE_GROUPS.items() for t in types}


def group_of(object_type: str) -> str:
    return GROUP_OF.get(object_type, "other")


# A small object's box on a dense block is mostly ground; these are the classes
# that are worth keeping alongside the object's own when the support would
# otherwise be a patch of road.
MIN_SUPPORT_POINTS = 64
# The tolerance added to the annotated box before looking for the entity's
# points.  It is a fraction of the entity's own size rather than a constant,
# because a fixed metre around a car reaches its neighbours in a parking row:
# the class filter cannot separate two cars, so the box has to.
SUPPORT_MARGIN_FRACTION = 0.05
SUPPORT_MARGIN_MIN_M = 0.1
SUPPORT_MARGIN_MAX_M = 1.0

# CROP_EXTENT_* describe the *context* crop, which the earlier rounds used for
# every candidate; TIGHT_* describe the entity-sized crop added here.
CROP_EXTENT_MIN_M = 24.0
CROP_EXTENT_MAX_M = 150.0
TIGHT_EXTENT_MIN_M = 6.0
# The tight crop is capped at the context cap, not below it: a cap smaller than
# an entity would crop the entity out of its own tight view, which is the one
# place it has to be whole.
TIGHT_EXTENT_MAX_M = 150.0
TIGHT_SIZE_FACTOR = 1.15
CONTEXT_SIZE_FACTOR = 2.5
MAX_SUPPORT_POINTS = 1_500_000


def context_extent_m(dimension) -> float:
    """Side of the context crop: the entity plus a fixed multiple of its size."""
    return float(np.clip(CONTEXT_SIZE_FACTOR * max(dimension[0], dimension[1]),
                         CROP_EXTENT_MIN_M, CROP_EXTENT_MAX_M))


def tight_extent_m(dimension) -> float:
    """Side of the tight crop: the entity itself, floored at a resolvable size.

    The floor is what a vehicle needs.  A 4.5 m car has no appearance left at
    the 24 m floor the context crop uses, and a 150 m crop for a warehouse would
    spend most of its patches on the neighbours.
    """
    return float(np.clip(TIGHT_SIZE_FACTOR * max(dimension[0], dimension[1]),
                         TIGHT_EXTENT_MIN_M, TIGHT_EXTENT_MAX_M))


def entity_scales(dimension) -> dict:
    return {"tight": tight_extent_m(dimension),
            "context": context_extent_m(dimension)}


def entity_bounds(position, dimension, margin: float = 0.0) -> tuple:
    """Axis-aligned box of an annotated entity as ``(lo, hi)`` 3-vectors."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    half = np.asarray(dimension, dtype=np.float64).reshape(3) / 2.0 + margin
    return pos - half, pos + half


def box_membership(xyz: np.ndarray, lo, hi) -> np.ndarray:
    xyz = np.asarray(xyz)
    return np.all((xyz >= lo[None, :]) & (xyz <= hi[None, :]), axis=1)


def polygon_membership(xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for a batch of XY points.

    CityRefer carries some entities' footprint as a polygon, but it is not a
    strict refinement of the bounding box: for objects under a few metres it is
    padded to a minimum size and is *larger* than the box, while for a long
    diagonal terrace it is markedly tighter.  Requiring both -- box test and
    polygon test -- takes the tight one wherever the polygon is informative and
    falls back to the box where it is not, without deciding per object.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if polygon is None:
        return np.ones(len(xy), dtype=bool)
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        return np.ones(len(xy), dtype=bool)
    poly = poly[:, :2]
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

    # Chunked because every temporary here is (points x vertices) and an entity
    # on a dense block supplies millions of points.
    if len(xy) <= 250_000:
        return test(xy)
    return np.concatenate([test(xy[i:i + 250_000])
                           for i in range(0, len(xy), 250_000)])


@dataclass
class EntitySupport:
    """The measured points attributed to one entity, and how they were chosen."""

    points: np.ndarray
    source: str
    object_type: str
    class_id: int | None
    box_points: int
    class_points: int
    footprint_points: int
    radius_m: float | None = None

    def __len__(self) -> int:
        return int(len(self.points))

    def summary(self) -> dict:
        return {
            "support_source": self.source,
            "support_points": int(len(self.points)),
            "support_box_points": int(self.box_points),
            "support_class_points": int(self.class_points),
            "support_footprint_points": int(self.footprint_points),
            "support_radius_m": (None if self.radius_m is None
                                 else float(self.radius_m)),
        }


def support_margin(dimension) -> float:
    """Box tolerance for the support query, scaled to the entity's own size."""
    size = float(max(dimension[0], dimension[1]))
    return float(np.clip(SUPPORT_MARGIN_FRACTION * size,
                         SUPPORT_MARGIN_MIN_M, SUPPORT_MARGIN_MAX_M))


def entity_support(grid, sorted_xyz, position, dimension, object_type=None,
                   footprint=None, labels=None, margin: float | None = None,
                   min_points: int = MIN_SUPPORT_POINTS,
                   max_points: int = MAX_SUPPORT_POINTS) -> EntitySupport:
    """Build an entity's 3D support, falling through a recorded ladder.

    Each rung is tried in order and the first with enough points wins; the rung
    is returned.  A silent fallback would make "this method works for cars"
    unfalsifiable, because a car whose support is really a patch of road would
    be indistinguishable from one that was genuinely localised.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    dim = np.asarray(dimension, dtype=np.float64).reshape(3)
    if margin is None:
        margin = support_margin(dim)
    lo, hi = entity_bounds(pos, dim, margin)
    class_id = CLASS_IDS.get(object_type) if object_type else None
    label_sets = ((labels, np.array([class_id])) if
                  (labels is not None and class_id is not None) else None)

    stats = {"box_points": 0, "class_points": 0, "footprint_points": 0}
    attempts = []
    if class_id is not None and labels is None:
        attempts.append("class_label_unavailable")
    if footprint is None:
        attempts.append("no_footprint")

    # Rung 1: box, class, footprint.
    class_pts = (grid.query_box_slots(lo, hi, sorted_xyz, label_sets=label_sets)
                 if label_sets is not None
                 else grid.query_box_slots(lo, hi, sorted_xyz))
    stats["box_points"] = int(len(class_pts))
    if class_id is None:
        stats["class_points"] = int(len(class_pts))
    else:
        stats["class_points"] = int(len(class_pts))
    chosen, source = None, None
    if footprint is not None and len(class_pts):
        keep = polygon_membership(class_pts[:, :2], footprint)
        stats["footprint_points"] = int(keep.sum())
        if keep.sum() >= min_points:
            chosen, source = class_pts[keep], (
                "box+footprint+class" if label_sets is not None else "box+footprint")
    if chosen is None and len(class_pts) >= min_points:
        chosen, source = class_pts, (
            "box+class" if label_sets is not None else "box")

    # Rung 2: no class constraint, box (and footprint if it helps).
    if chosen is None:
        plain = grid.query_box_slots(lo, hi, sorted_xyz)
        if footprint is not None and len(plain):
            keep = polygon_membership(plain[:, :2], footprint)
            if keep.sum() >= min_points:
                chosen, source = plain[keep], "box+footprint"
        if chosen is None and len(plain) >= min_points:
            chosen, source = plain, "box"

    # Rung 3: radius around the annotated centre, class-restricted when we can.
    radius = float(max(2.0, 1.5 * float(max(dim[0], dim[1]))))
    while chosen is None and radius <= 64.0:
        r_lo = np.array([pos[0] - radius, pos[1] - radius, pos[2] - dim[2] / 2 - 1.0])
        r_hi = np.array([pos[0] + radius, pos[1] + radius, pos[2] + dim[2] / 2 + 1.0])
        radius_pts = (grid.query_box_slots(r_lo, r_hi, sorted_xyz, label_sets=label_sets)
                      if label_sets is not None
                      else grid.query_box_slots(r_lo, r_hi, sorted_xyz))
        if len(radius_pts) >= min_points:
            chosen, source = radius_pts, "radius+class" if label_sets is not None \
                else "radius"
            break
        radius *= 2.0
        attempts.append("radius_widened")

    # Rung 4: whatever is nearest the annotated centre.
    if chosen is None:
        span = 4.0
        near = grid.query_box_slots(
            pos - span, pos + span, sorted_xyz)
        if len(near):
            order = np.argsort(np.linalg.norm(near - pos[None, :], axis=1))
            chosen, source = near[order][:max(min_points, 256)], "centre_neighbourhood"
            radius = float(span)
        else:
            chosen, source = np.empty((0, 3)), "empty"

    if chosen is None:
        chosen, source = np.empty((0, 3)), "empty"
    if len(chosen) > max_points:
        stride = int(np.ceil(len(chosen) / max_points))
        chosen = chosen[::stride]

    return EntitySupport(points=np.asarray(chosen, dtype=np.float64),
                         source=source, object_type=object_type or "",
                         class_id=class_id, box_points=stats["box_points"],
                         class_points=stats["class_points"],
                         footprint_points=stats["footprint_points"],
                         radius_m=(radius if source and source.startswith("radius")
                                   else None))


# --------------------------------------------------------------------------
# crops, masks, correspondence
# --------------------------------------------------------------------------

def crop_square(image, centre_u, centre_v, side_px, border: str = "replicate"):
    """Crop centred on a pixel, clamped to the image, padded if it runs off.

    ``border='constant'`` pads with black -- the value the rasters already use
    for nodata -- rather than smearing the edge row into the padding, which
    would invent texture that was never measured.
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


@dataclass
class CropWindow:
    """Where an entity's crop sits in a source image, and how it maps to patches."""

    centre: tuple
    side: float
    patches: int = 0

    @property
    def _half(self) -> float:
        return self.side / 2.0

    def patch_of_uv(self, u, v) -> tuple:
        """Patch indices for source pixels, plus which of them fall in the crop."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        cu, cv = self.centre
        with np.errstate(invalid="ignore"):
            fx = (u - (cu - self._half)) / self.side
            fy = (v - (cv - self._half)) / self.side
        ok = (fx >= 0.0) & (fx < 1.0) & (fy >= 0.0) & (fy < 1.0)
        pc = np.clip((np.nan_to_num(fx) * self.patches).astype(np.int64),
                     0, self.patches - 1)
        pr = np.clip((np.nan_to_num(fy) * self.patches).astype(np.int64),
                     0, self.patches - 1)
        return pr, pc, ok


def topdown_window(raster_origin, raster_resolution, position, extent_m) -> CropWindow:
    ox, oy = float(raster_origin[0]), float(raster_origin[1])
    res = float(raster_resolution)
    col = (float(position[0]) - ox) / res
    row = (oy - float(position[1])) / res
    return CropWindow(centre=(col, row), side=extent_m / res)


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


def oblique_window(camera: Camera, position, extent_m) -> CropWindow | None:
    """Crop window for a perspective frame, sized by the entity's angular extent."""
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
    return CropWindow(centre=(cu, cv), side=side)


@dataclass
class ScaleView:
    """One entity, one view, one scale: window, mask, and how much was seen."""

    name: str
    extent_m: float
    window: CropWindow
    patch_mask: np.ndarray
    patch_count: int
    visible_points: int
    projected_pixels: int

    def summary(self) -> dict:
        return {"scale": self.name, "extent_m": float(self.extent_m),
                "patches": int(self.patch_count),
                "visible_points": int(self.visible_points),
                "projected_pixels": int(self.projected_pixels)}


@dataclass
class EntityView:
    """Everything about one entity that the fusion consumes."""

    entity_id: int
    object_type: str
    support_source: str
    support_points: int
    scales: dict                                  # {"td": {name: ScaleView}, ...}
    correspondence: dict = field(default_factory=dict)
    geometry: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    visible_point_ratio: float = 0.0
    occlusion_ratio: float = 0.0

    @property
    def has_correspondence(self) -> bool:
        return self.correspondence.get("weight", np.empty(0)).size > 0

    # A view can resolve one scale and not the other -- the two windows have
    # different sizes, so the off-frame test that rejects the context crop can
    # accept the tight one -- and an entity the oblique view never reaches has
    # no scale there at all.  These accessors keep that a zero mask rather than
    # a missing key, so every entity stays in the candidate set every method is
    # scored on.
    def mask_of(self, view: str, scale: str, patch_grid: int) -> np.ndarray:
        entry = self.scales.get(view, {}).get(scale)
        return (entry.patch_mask if entry is not None
                else np.zeros((patch_grid, patch_grid), dtype=bool))

    def count_of(self, view: str, scale: str) -> int:
        entry = self.scales.get(view, {}).get(scale)
        return int(entry.patch_count) if entry is not None else 0

    def window_of(self, view: str, scale: str) -> CropWindow | None:
        entry = self.scales.get(view, {}).get(scale)
        return entry.window if entry is not None else None

    def summary(self) -> dict:
        out = {"entity_id": int(self.entity_id), "object_type": self.object_type,
               "support_source": self.support_source,
               "support_points": int(self.support_points),
               "visible_point_ratio": float(self.visible_point_ratio),
               "occlusion_ratio": float(self.occlusion_ratio),
               "correspondence_pairs": int(
                   self.correspondence.get("weight", np.empty(0)).size),
               "geometry": [float(v) for v in self.geometry]}
        for view, scales in self.scales.items():
            for name, scale in scales.items():
                out[f"{view}_{name}"] = scale.summary()
        return out


def scale_view(name: str, extent_m: float, window: CropWindow,
               rows, cols, ok, patch_grid: int, visible) -> ScaleView:
    mask = np.zeros((patch_grid, patch_grid), dtype=bool)
    keep = ok & visible
    if keep.any():
        mask[rows[keep], cols[keep]] = True
    return ScaleView(name=name, extent_m=float(extent_m), window=window,
                     patch_mask=mask, patch_count=int(mask.sum()),
                     visible_points=int(keep.sum()), projected_pixels=int(ok.size))


def entity_view(entity_id: int, support: EntitySupport, position, dimension,
                raster_info, camera: Camera, render_result, sorted_xyz,
                patch_grid: int, footprint=None, occlusion_tolerance: float = 0.75,
                scales=("tight", "context")) -> EntityView | None:
    """Project one entity's support into both views at every requested scale.

    Returns ``None`` only when the entity has no points at all.  A view that
    cannot show the entity is reported as an empty mask with the reason carried
    in ``visible_point_ratio``, because "the oblique view does not see this car"
    is a result about observability, not a missing record.
    """
    pts = np.asarray(support.points, dtype=np.float64)
    if pts.size == 0 or raster_info is None:
        return None
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    extents = {"tight": tight_extent_m(dimension),
               "context": context_extent_m(dimension)}
    out = EntityView(entity_id=int(entity_id), object_type=support.object_type,
                     support_source=support.source,
                     support_points=int(len(pts)), scales={"td": {}, "oblique": {}})

    # ---- top-down: orthographic, one point one pixel, no occlusion to resolve
    ox, oy = float(raster_info[1][0]), float(raster_info[1][1])
    res = float(raster_info[2])
    td_col = (pts[:, 0] - ox) / res
    td_row = (oy - pts[:, 1]) / res
    raster = raster_info[0]
    in_raster = ((td_row >= 0) & (td_row < raster.shape[0])
                 & (td_col >= 0) & (td_col < raster.shape[1]))

    # ---- oblique: project, then decide visibility against the z-buffer
    proj = project_points(pts, camera)
    n_proj = proj["u"].size
    seen_proj = np.zeros(n_proj, dtype=bool)
    if n_proj:
        src = proj["pos"]
        rows_c = np.clip(proj["v"].astype(np.int64), 0, camera.height - 1)
        cols_c = np.clip(proj["u"].astype(np.int64), 0, camera.width - 1)
        depth = render_result.depth[rows_c, cols_c]
        winner_slot = render_result.point_id[rows_c, cols_c]
        lo, hi = entity_bounds(pos, dimension, 1.0)
        has_winner = winner_slot >= 0
        won_by_entity = np.zeros(n_proj, dtype=bool)
        if has_winner.any():
            world = np.asarray(sorted_xyz[winner_slot[has_winner].astype(np.int64)])
            won_by_entity[has_winner] = box_membership(world, lo, hi)
        # Being the pixel's winner is exact but loses points the level-of-detail
        # decimation skipped; a point clearly in front of whatever was drawn is
        # the same statement with a tolerance, and recovers them.
        in_front = np.isfinite(depth) & (proj["metric"] <= depth + occlusion_tolerance)
        seen_proj = won_by_entity | in_front
        out.visible_point_ratio = float(seen_proj.mean())
        out.occlusion_ratio = float(1.0 - seen_proj.mean())
    else:
        out.visible_point_ratio = 0.0
        out.occlusion_ratio = 1.0

    # Visibility carried back to point space, so both views' masks are indexed
    # by the same point and can be intersected to build the correspondence.
    seen_pts = np.zeros(len(pts), dtype=bool)
    if n_proj:
        seen_pts[proj["pos"][seen_proj]] = True

    for name in scales:
        extent = extents[name]
        td_win = topdown_window(raster_info[1], raster_info[2], pos, extent)
        td_win.patches = patch_grid
        r, c, ok = td_win.patch_of_uv(td_col, td_row)
        out.scales["td"][name] = scale_view(name, extent, td_win, r, c, ok,
                                            patch_grid, in_raster)
        o_win = oblique_window(camera, pos, extent)
        if o_win is None:
            continue
        o_win.patches = patch_grid
        r, c, ok = o_win.patch_of_uv(proj["u"], proj["v"])
        out.scales["oblique"][name] = scale_view(name, extent, o_win, r, c, ok,
                                                 patch_grid, seen_proj)

    # ---- correspondence at the context scale: one 3D point, one patch each side
    out.correspondence = {"td": np.empty(0, np.int64), "o": np.empty(0, np.int64),
                          "weight": np.empty(0, np.float32)}
    ctx_td = out.scales["td"].get("context")
    ctx_o = out.scales["oblique"].get("context")
    if ctx_td is not None and ctx_o is not None and n_proj:
        td_r, td_c, td_ok = ctx_td.window.patch_of_uv(td_col, td_row)
        o_r, o_c, o_ok = ctx_o.window.patch_of_uv(proj["u"], proj["v"])
        o_ok_pts = np.zeros(len(pts), dtype=bool)
        o_ok_pts[proj["pos"]] = o_ok & seen_proj
        shared = td_ok & in_raster & o_ok_pts
        if shared.any():
            o_r_pts = np.zeros(len(pts), dtype=np.int64)
            o_c_pts = np.zeros(len(pts), dtype=np.int64)
            o_r_pts[proj["pos"]] = o_r
            o_c_pts[proj["pos"]] = o_c
            a = td_r[shared] * patch_grid + td_c[shared]
            b = o_r_pts[shared] * patch_grid + o_c_pts[shared]
            counts = np.bincount(a * (patch_grid * patch_grid) + b,
                                 minlength=patch_grid ** 4)
            nz = np.flatnonzero(counts)
            out.correspondence = {
                "td": (nz // (patch_grid * patch_grid)).astype(np.int64),
                "o": (nz % (patch_grid * patch_grid)).astype(np.int64),
                "weight": counts[nz].astype(np.float32)}

    out.geometry = geometry_features(support, pos, dimension, extents, out)
    return out


def geometry_features(support: EntitySupport, position, dimension, extents,
                      view: EntityView) -> np.ndarray:
    """A fixed-length description of the entity's own measured geometry.

    Size, shape, height and how much of it the oblique view actually shows.  No
    category identity is included: an entity's type name would let a model read
    the answer off the candidate list rather than off the evidence.
    """
    pts = np.asarray(support.points, dtype=np.float64)
    dim = np.asarray(dimension, dtype=np.float64).reshape(3)
    if len(pts):
        span = pts.max(axis=0) - pts.min(axis=0)
        z = pts[:, 2]
        height = float(np.percentile(z, 95) - np.percentile(z, 5))
        footprint = float(span[0] * span[1])
        density = float(len(pts) / max(footprint, 1e-3))
    else:
        span = np.zeros(3)
        height, footprint, density = 0.0, 0.0, 0.0
    ctx = view.scales["td"].get("context")
    tight = view.scales["td"].get("tight")
    o_ctx = view.scales["oblique"].get("context")
    o_tight = view.scales["oblique"].get("tight")
    oblique_shown = 1.0 if o_ctx is not None else 0.0
    return np.array([
        float(dim[0]), float(dim[1]), float(dim[2]),
        float(span[0]), float(span[1]), height,
        float(np.log1p(max(len(pts), 0))), float(np.log1p(footprint)),
        float(np.log1p(max(density, 0.0))),
        float(extents["tight"]), float(extents["context"]),
        float(view.visible_point_ratio), float(view.occlusion_ratio),
        float(tight.patch_count if tight else 0) / 1e3,
        float(ctx.patch_count if ctx else 0) / 1e3,
        float(o_tight.patch_count if o_tight else 0) / 1e3,
        float(o_ctx.patch_count if o_ctx else 0) / 1e3,
        float(view.support_points) / 1e5,
        # Whether the oblique view resolves this entity at all.  It is a fact
        # about observability, not about identity, and a method that cannot see
        # an entity should be able to say so rather than read a blank crop.
        oblique_shown,
    ], dtype=np.float32)


@dataclass
class EntitySpec:
    """The annotation for one candidate, independent of which loader produced it."""

    entity_id: int
    object_type: str
    position: tuple
    dimension: tuple
    contour: object = None


def entity_crops(view: EntityView, raster_info, render_rgb,
                 blank_size: int = 512) -> dict:
    """The crop images for every (view, scale) of one entity.

    A view that cannot show the entity -- it is behind the camera, or off the
    frame -- still gets a crop, because the entity is a candidate either way and
    dropping it would remove it from the candidate set that every method is
    scored on.  The substitute is a blank crop: its masks are empty, so no patch
    of it can enter a masked feature, and a whole-crop feature over it is a
    constant that identifies nothing.  The batch shapes stay rectangular, which
    is what makes the four crops encodable in one pass.
    """
    blank = np.zeros((blank_size, blank_size, 3), dtype=np.uint8)
    out = {"td": {}, "oblique": {}}
    for name in ("tight", "context"):
        scale = view.scales["td"].get(name)
        out["td"][name] = (
            blank if (scale is None or raster_info is None) else crop_square(
                raster_info[0], scale.window.centre[0], scale.window.centre[1],
                scale.window.side, border="constant"))
        scale = view.scales["oblique"].get(name)
        out["oblique"][name] = (
            blank if scale is None else crop_square(
                render_rgb, scale.window.centre[0], scale.window.centre[1],
                scale.window.side, border="constant"))
    return out


def build_entity_payload(grid, sorted_xyz, labels, specs, raster_info, camera,
                         render_result, patch_grid, max_support_points=MAX_SUPPORT_POINTS):
    """Support, views and crops for a list of candidate entities.

    Returns ``(payload, skipped)`` where ``skipped`` counts why an entity was
    dropped.  An entity with no measurable points at all cannot be represented,
    and saying so explicitly keeps it out of the denominators that would
    otherwise read as a failure of the method.
    """
    payload, skipped = [], {}
    for spec in specs:
        support = entity_support(grid, sorted_xyz, spec.position, spec.dimension,
                                 object_type=spec.object_type,
                                 footprint=spec.contour, labels=labels,
                                 max_points=max_support_points)
        if len(support) == 0:
            skipped["no_support_points"] = skipped.get("no_support_points", 0) + 1
            continue
        view = entity_view(spec.entity_id, support, spec.position, spec.dimension,
                           raster_info, camera, render_result, sorted_xyz, patch_grid,
                           footprint=spec.contour)
        if view is None:
            skipped["no_view"] = skipped.get("no_view", 0) + 1
            continue
        crops = entity_crops(view, raster_info, render_result.rgb)
        payload.append({"spec": spec, "support": support, "view": view,
                        "crops": crops})
    return payload, skipped


def correspondence_matrix(entity: EntityView, patch_grid: int,
                          symmetric: bool = True) -> np.ndarray:
    """Row-normalised patch correspondence for the context scale."""
    n = patch_grid * patch_grid
    W = np.zeros((n, n), dtype=np.float32)
    td = entity.correspondence.get("td", np.empty(0))
    if td.size:
        np.add.at(W, (td, entity.correspondence["o"]),
                  entity.correspondence["weight"])
    if symmetric and W.any():
        W = W + W.T
    row_sum = W.sum(axis=1, keepdims=True)
    np.divide(W, row_sum, out=W, where=row_sum > 0)
    return W


def correspondence_coherence(entity: EntityView, patch_grid: int) -> float:
    """Mean patch-grid distance between a patch's target and its neighbours'.

    A correspondence that came from projecting the same 3D points through two
    cameras is locally smooth: adjacent top-down patches of one surface land on
    adjacent oblique patches.  Reported next to the same number for the shuffled
    pairing, so "the alignment carries information" is checkable rather than
    asserted.
    """
    td = entity.correspondence.get("td", np.empty(0))
    if td.size == 0:
        return float("nan")
    W = correspondence_matrix(entity, patch_grid, symmetric=False)
    rows = np.flatnonzero(W.sum(axis=1) > 0)
    if rows.size < 2:
        return float("nan")
    targets = np.full((patch_grid * patch_grid, 2), np.nan)
    for r in rows.tolist():
        q = int(np.argmax(W[r]))
        targets[r] = (q // patch_grid, q % patch_grid)
    idx = np.arange(patch_grid * patch_grid).reshape(patch_grid, patch_grid)
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


def shuffled_correspondence(entity: EntityView, patch_grid: int,
                            seed: int) -> EntityView:
    """The same geometry with the patch pairing permuted: the random control.

    The permutation is confined to the oblique patches the entity occupies, so
    the shuffled pairing still draws only from entity patches and uses every
    weight unchanged.  Only *which* top-down patch goes with which oblique patch
    is destroyed.
    """
    corr = entity.correspondence
    if corr.get("o", np.empty(0)).size == 0:
        return entity
    n = patch_grid * patch_grid
    pool = np.flatnonzero(entity.mask_of("oblique", "context", patch_grid).ravel())
    if pool.size < 2:
        return entity
    rng = np.random.default_rng(seed)
    perm = np.arange(n, dtype=np.int64)
    perm[pool] = rng.permutation(pool)
    new = dict(corr)
    new["o"] = perm[corr["o"]]
    return replace(entity, correspondence=new)


def draw_mask_overlay(image: np.ndarray, mask: np.ndarray,
                      colour=(255, 60, 60), alpha: float = 0.45) -> np.ndarray:
    """Tint the boundary patches of a mask over the crop they were built for."""
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
