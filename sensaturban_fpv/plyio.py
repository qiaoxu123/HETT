"""SensatUrban PLY access: header parsing, memmapped reads, and an XY grid index.

The SensatUrban blocks are 20M-80M vertex binary PLYs.  Nothing here loads a
whole block into RAM: the header is parsed for the real property order, the body
is memmapped, and every consumer reads through :class:`PlyCloud` or the cached
:class:`XyGridIndex`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# PLY scalar type names -> numpy dtype characters.  SensatUrban uses the
# `float32`/`uint8` spelling, but accept the short PLY aliases too so a
# re-exported block still reads.
PLY_SCALARS = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_ENDIAN = {"binary_little_endian": "<", "binary_big_endian": ">"}


@dataclass(frozen=True)
class PlyHeader:
    fmt: str
    count: int
    names: tuple
    dtype: np.dtype
    data_offset: int


def read_ply_header(path: Path) -> PlyHeader:
    """Parse the ASCII header and return the exact element dtype and offset."""
    names: list = []
    types: list = []
    fmt = None
    count = None
    with Path(path).open("rb") as handle:
        first = handle.readline().strip()
        if first != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY header has no end_header: {path}")
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                count = int(line.split()[2])
                # every property after this belongs to the vertex element
            elif line.startswith("element ") and count is not None:
                raise ValueError(
                    f"{path}: the vertex element is not the first element; "
                    "this reader assumes a single vertex element"
                )
            elif line.startswith("property "):
                if count is None:
                    raise ValueError(f"{path}: property before element vertex")
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties are unsupported")
                names.append(parts[2])
                types.append(PLY_SCALARS[parts[1]])
            elif line == "end_header":
                offset = handle.tell()
                break

    if fmt not in _ENDIAN:
        raise ValueError(f"{path}: unsupported PLY format {fmt!r}")
    if count is None:
        raise ValueError(f"{path}: header declares no vertex element")
    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError(f"{path}: vertex element has no {required!r} property")
    dtype = np.dtype([(n, _ENDIAN[fmt] + t) for n, t in zip(names, types)])
    return PlyHeader(fmt, count, tuple(names), dtype, offset)


class PlyCloud:
    """A memmapped SensatUrban block."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.header = read_ply_header(self.path)
        self._data = np.memmap(
            self.path, mode="r", dtype=self.header.dtype,
            offset=self.header.data_offset, shape=(self.header.count,),
        )

    def __len__(self) -> int:
        return self.header.count

    @property
    def has_rgb(self) -> bool:
        return all(c in self.header.names for c in ("red", "green", "blue"))

    def xyz(self, index: np.ndarray | None = None) -> np.ndarray:
        return self._fields(("x", "y", "z"), np.float32, index)

    def rgb(self, index: np.ndarray | None = None) -> np.ndarray:
        return self._fields(("red", "green", "blue"), np.uint8, index)

    def _fields(self, names, dtype, index) -> np.ndarray:
        """Read a contiguous run of records.

        ``None`` and a plain ``slice`` take np.memmap's fast slice path; an
        integer array would go through numpy's per-element fancy-index gather,
        which on a structured dtype is roughly two orders of magnitude slower.
        Callers that walk the whole block must therefore pass slices.
        """
        if index is None:
            d = self._data
        elif isinstance(index, slice):
            d = self._data[index]
        else:
            d = self._data[index]
        out = np.empty((len(d), len(names)), dtype=dtype)
        for i, name in enumerate(names):
            out[:, i] = d[name]
        return out

    def xyz_range(self, start: int, stop: int) -> np.ndarray:
        return self._fields(("x", "y", "z"), np.float32, slice(start, stop))

    def rgb_range(self, start: int, stop: int) -> np.ndarray:
        return self._fields(("red", "green", "blue"), np.uint8, slice(start, stop))

    def labels(self, index: np.ndarray | None = None) -> np.ndarray | None:
        if "class" not in self.header.names:
            return None
        d = self._data if index is None else self._data[index]
        return np.asarray(d["class"])

    def bounds(self) -> tuple:
        """Exact per-axis min/max, evaluated in bounded-memory chunks."""
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for start in range(0, len(self), 8_000_000):
            chunk = self.xyz_range(start, min(start + 8_000_000, len(self)))
            lo = np.minimum(lo, chunk.min(axis=0))
            hi = np.maximum(hi, chunk.max(axis=0))
        return lo, hi


class XyGridIndex:
    """Uniform XY bucket index over a block, persisted so it is built once.

    Querying a square window is then a gather over a contiguous run of buckets
    instead of a scan over 20M-80M vertices, which is what makes per-pose
    perspective rendering affordable.
    """

    def __init__(self, cloud: PlyCloud, cell: float = 4.0, cache_dir: Path | None = None):
        self.cloud = cloud
        self.cell = float(cell)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.lo = None
        self.hi = None
        self.nx = 0
        self.ny = 0
        self.order = None
        self.starts = None
        self._counts = None
        # Exact 3D bounds of the source cloud.  Kept here because building the
        # index already entails a full pass; re-deriving them later would be a
        # second full read of a multi-gigabyte PLY.
        self.cloud_lo = None
        self.cloud_hi = None

    @property
    def cache_path(self) -> Path:
        """Cached under the artifact directory by default.

        The source PLY tree is a shared dataset, so derived index files are
        written to the experiment's own cache rather than beside the blocks.
        """
        name = f"{self.cloud.path.stem}.grid{self.cell:g}.npz"
        if self.cache_dir:
            return self.cache_dir / name
        return self.cloud.path.with_suffix(f".grid{self.cell:g}.npz")

    def build(self, force: bool = False, cache: bool = True) -> "XyGridIndex":
        path = self.cache_path
        if cache and path.exists() and not force:
            with np.load(path) as z:
                self.lo = z["lo"]
                self.hi = z["hi"]
                self.nx = int(z["nx"])
                self.ny = int(z["ny"])
                self.order = z["order"]
                self.starts = z["starts"]
                if "cloud_lo" in z.files:
                    self.cloud_lo = z["cloud_lo"]
                    self.cloud_hi = z["cloud_hi"]
            self._counts = None
            if self.cloud_lo is None:
                # Index written before the exact bounds were cached; recover
                # them once and upgrade the file in place.
                self.cloud_lo, self.cloud_hi = self.cloud.bounds()
                _atomic_savez(
                    path, lo=self.lo, hi=self.hi,
                    nx=np.int64(self.nx), ny=np.int64(self.ny),
                    order=self.order, starts=self.starts,
                    cloud_lo=self.cloud_lo, cloud_hi=self.cloud_hi,
                )
            return self

        lo, hi = self.cloud.bounds()
        self.cloud_lo, self.cloud_hi = lo, hi
        # Grow the extent by one cell so both boundary faces bucket cleanly.
        self.lo = lo[:2] - self.cell
        self.hi = hi[:2] + self.cell
        self.nx = max(int(np.ceil((self.hi[0] - self.lo[0]) / self.cell)), 1)
        self.ny = max(int(np.ceil((self.hi[1] - self.lo[1]) / self.cell)), 1)
        ncells = self.nx * self.ny

        # int32 cells keep the sort's working set small; a 400 m block at 2 m
        # resolution is 40k buckets, nowhere near the int32 range.
        cell_of_point = np.empty(len(self.cloud), dtype=np.int32)
        counts = np.zeros(ncells, dtype=np.int64)
        for start in range(0, len(self.cloud), 8_000_000):
            stop = min(start + 8_000_000, len(self.cloud))
            xyz = self.cloud.xyz_range(start, stop)
            cell_id = self.cell_ids(xyz[:, 0], xyz[:, 1], clip=True).astype(np.int32)
            cell_of_point[start:stop] = cell_id
            counts += np.bincount(cell_id, minlength=ncells)

        # Stable sort by cell keeps the within-cell point order in file order,
        # so a window query returns a deterministic subset across runs.
        self.order = np.argsort(cell_of_point, kind="stable").astype(np.int32)
        self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_savez(
                path, lo=self.lo, hi=self.hi,
                nx=np.int64(self.nx), ny=np.int64(self.ny),
                order=self.order, starts=self.starts,
                cloud_lo=self.cloud_lo, cloud_hi=self.cloud_hi,
            )
        return self

    @property
    def sorted_paths(self) -> tuple:
        name = f"{self.cloud.path.stem}.sorted"
        base = self.cache_dir if self.cache_dir else self.cloud.path.parent
        return base / f"{name}.xyz.npy", base / f"{name}.rgb.npy"

    @property
    def sorted_marker(self) -> Path:
        xyz_path, _ = self.sorted_paths
        return xyz_path.with_suffix(".done")

    def _cache_is_complete(self) -> bool:
        """Trust the cache only when its completion marker is present and agrees.

        A size check is not enough.  ``open_memmap`` creates the whole-length
        file up front, so a build killed part way leaves a file of exactly the
        right *size* whose tail is a sparse hole of zeros -- and zeros are
        perfectly plausible coordinates, so the corruption reads back as real
        geometry.  The marker is written only after both renames succeed.
        """
        xyz_path, rgb_path = self.sorted_paths
        marker = self.sorted_marker
        if not (xyz_path.exists() and rgb_path.exists() and marker.exists()):
            return False
        n = len(self.cloud)
        xyz_expected = 128 + n * 12  # .npy header + body
        rgb_expected = 128 + n * 3
        if (xyz_path.stat().st_size < xyz_expected
                or rgb_path.stat().st_size < rgb_expected):
            return False
        try:
            return int(marker.read_text().strip()) == xyz_expected
        except (ValueError, OSError):
            return False

    def ensure_sorted(self) -> None:
        """Materialise the cloud in bucket order as two contiguous arrays.

        A radius query then reads from a bounded, ascending run of addresses
        instead of touching the source PLY in bucket order, which is effectively
        random.  Renderers read tens of millions of points per frame and the
        difference is the difference between seconds and minutes.

        Both arrays are built under temporary names and renamed on completion,
        and a size check guards against a cache left behind by an interrupted
        run -- reading one of those silently yields zeros for the untouched
        region, which is indistinguishable from a real hole in the scene.
        """
        xyz_path, rgb_path = self.sorted_paths
        n = len(self.cloud)
        if self._cache_is_complete():
            return
        if self.order is None:
            raise RuntimeError("build() the index before materialising it")

        tmp_xyz = xyz_path.with_name(f"{xyz_path.stem}.tmp{os.getpid()}.npy")
        tmp_rgb = rgb_path.with_name(f"{rgb_path.stem}.tmp{os.getpid()}.npy")
        xyz_out = np.lib.format.open_memmap(tmp_xyz, mode="w+", dtype=np.float32,
                                            shape=(n, 3))
        rgb_out = np.lib.format.open_memmap(tmp_rgb, mode="w+", dtype=np.uint8,
                                            shape=(n, 3))

        # Read the block into RAM once, then permute there. The alternative --
        # scattering a sequential read into slots through an integer index --
        # costs a per-element gather on every write and is roughly an order of
        # magnitude slower; a gather inside RAM does not.
        chunk = 16_000_000
        xyz_all = np.empty((n, 3), dtype=np.float32)
        rgb_all = np.empty((n, 3), dtype=np.uint8)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            xyz_all[start:stop] = self.cloud.xyz_range(start, stop)
            rgb_all[start:stop] = self.cloud.rgb_range(start, stop)

        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            take = self.order[start:stop].astype(np.int64)
            xyz_out[start:stop] = xyz_all[take]
            rgb_out[start:stop] = rgb_all[take]
        xyz_out.flush()
        rgb_out.flush()
        del xyz_out, rgb_out, xyz_all, rgb_all

        os.replace(tmp_xyz, xyz_path)
        os.replace(tmp_rgb, rgb_path)
        # Written last, and only after both renames: its presence is the claim
        # that the two files under their final names hold complete data.
        self.sorted_marker.write_text(str(128 + n * 12) + "\n")

    def sorted_arrays(self):
        """Memmapped bucket-ordered ``(xyz, rgb)``, building the cache if needed."""
        self.ensure_sorted()
        xyz_path, rgb_path = self.sorted_paths
        return (
            np.load(xyz_path, mmap_mode="r"),
            np.load(rgb_path, mmap_mode="r"),
        )

    def exact_bounds(self):
        """Exact 3D bounds of the source cloud, from the index rather than a rescan."""
        if self.cloud_lo is None:
            self.cloud_lo, self.cloud_hi = self.cloud.bounds()
        return self.cloud_lo, self.cloud_hi

    def cell_ids(self, x: np.ndarray, y: np.ndarray, clip: bool = False) -> np.ndarray:
        ix = np.floor((x - self.lo[0]) / self.cell).astype(np.int64)
        iy = np.floor((y - self.lo[1]) / self.cell).astype(np.int64)
        if clip:
            np.clip(ix, 0, self.nx - 1, out=ix)
            np.clip(iy, 0, self.ny - 1, out=iy)
        return ix * self.ny + iy

    def query_rect(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Indices of every vertex whose XY falls in the half-open rectangle.

        Only cells genuinely inside the rectangle are visited: the cell ids of a
        rectangle are not a contiguous run, so a slice between its two corner
        cells would return a block-shaped superset.
        """
        ix0 = max(int(np.floor((x0 - self.lo[0]) / self.cell)), 0)
        ix1 = min(int(np.floor((x1 - self.lo[0]) / self.cell)), self.nx - 1)
        iy0 = max(int(np.floor((y0 - self.lo[1]) / self.cell)), 0)
        iy1 = min(int(np.floor((y1 - self.lo[1]) / self.cell)), self.ny - 1)
        if ix0 > ix1 or iy0 > iy1:
            return np.empty(0, dtype=np.int64)

        cell_ids = (
            np.arange(ix0, ix1 + 1, dtype=np.int64)[:, None] * self.ny
            + np.arange(iy0, iy1 + 1, dtype=np.int64)[None, :]
        ).ravel()
        return _gather_cells(self.order, self.starts, cell_ids)

    def query_radius(self, x: float, y: float, radius: float) -> np.ndarray:
        return self.query_rect(x - radius, y - radius, x + radius, y + radius)

    def query_rect_cell_ranges(self, x0: float, y0: float, x1: float, y1: float) -> tuple:
        """Slot ranges ``(starts, ends)`` of every cell inside the rectangle.

        A rectangle query in the bucket-ordered cache is not a contiguous run of
        slots -- the cell ids of a rectangle form a wide block, and the slots
        between two consecutive cells of the rectangle belong to cells outside
        it.  Returning the per-cell ranges lets a caller read exactly the
        rectangle's points, one short sequential read per cell, instead of
        dragging in the block-shaped superset.  For a 50 m building on a 100M
        point block that is a factor of roughly eight in bytes read.
        """
        ix0 = max(int(np.floor((x0 - self.lo[0]) / self.cell)), 0)
        ix1 = min(int(np.floor((x1 - self.lo[0]) / self.cell)), self.nx - 1)
        iy0 = max(int(np.floor((y0 - self.lo[1]) / self.cell)), 0)
        iy1 = min(int(np.floor((y1 - self.lo[1]) / self.cell)), self.ny - 1)
        if ix0 > ix1 or iy0 > iy1:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        cell_ids = (
            np.arange(ix0, ix1 + 1, dtype=np.int64)[:, None] * self.ny
            + np.arange(iy0, iy1 + 1, dtype=np.int64)[None, :]
        ).ravel()
        starts = self.starts[cell_ids]
        ends = self.starts[cell_ids + 1]
        keep = ends > starts
        return starts[keep], ends[keep]

    def read_cell_ranges(self, arrays, starts: np.ndarray, ends: np.ndarray,
                         chunk_points: int = 4_000_000) -> np.ndarray:
        """Concatenate the points of the given slot ranges into one array.

        One memmap slice per cell; slices are accumulated into a buffer and
        flushed once it holds ``chunk_points`` so the destination is allocated a
        bounded number of times.
        """
        if starts.size == 0:
            return np.empty((0,) + arrays.shape[1:], dtype=arrays.dtype)
        out, buf, total = [], [], 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            buf.append(np.asarray(arrays[s:e]))
            total += e - s
            if total >= chunk_points:
                out.append(np.concatenate(buf))
                buf, total = [], 0
        if buf:
            out.append(np.concatenate(buf))
        return out[0] if len(out) == 1 else np.concatenate(out)

    def query_box_slots(self, lo, hi, arrays, max_points: int | None = None) -> np.ndarray:
        """Points of an axis-aligned 3D box, as bucket slots.

        The XY rectangle comes from the bucket index; the z filter runs on the
        gathered points afterwards, because buckets are XY-only.  ``lo``/``hi``
        are 3-vectors.
        """
        starts, ends = self.query_rect_cell_ranges(lo[0], lo[1], hi[0], hi[1])
        if starts.size == 0:
            return np.empty((0, 3), dtype=arrays.dtype)
        pts = self.read_cell_ranges(arrays, starts, ends)
        inside = np.all((pts >= lo[None, :]) & (pts <= hi[None, :]), axis=1)
        pts = pts[inside]
        if max_points is not None and len(pts) > max_points:
            stride = int(np.ceil(len(pts) / max_points))
            pts = pts[::stride]
        return pts

    def cell_distance(self, x: float, y: float, ix0: int, ix1: int,
                      iy0: int, iy1: int) -> np.ndarray:
        """Distance from ``(x, y)`` to each cell's rectangle, shape (nx, ny).

        Distance to the rectangle, not to its centre: a cell is only decimated
        when *every* point in it is already far away.
        """
        xa = self.lo[0] + np.arange(ix0, ix1 + 1) * self.cell
        ya = self.lo[1] + np.arange(iy0, iy1 + 1) * self.cell
        dx = np.maximum(np.maximum(xa - x, x - (xa + self.cell)), 0.0)
        dy = np.maximum(np.maximum(ya - y, y - (ya + self.cell)), 0.0)
        return np.sqrt(dx[:, None] ** 2 + dy[None, :] ** 2)

    def query_radius_lod(self, x: float, y: float, radius: float, lod) -> np.ndarray:
        """Radius query that decimates whole cells whose nearest edge is far away.

        Decimating at the cell rather than at the point means a distant region is
        never even read out of the file.  This is what makes a 400 m far plane
        affordable on a 100M-point block: only the near shells are gathered at
        full rate.
        """
        if not lod:
            return self.query_radius(x, y, radius)
        ix0 = max(int(np.floor((x - radius - self.lo[0]) / self.cell)), 0)
        ix1 = min(int(np.floor((x + radius - self.lo[0]) / self.cell)), self.nx - 1)
        iy0 = max(int(np.floor((y - radius - self.lo[1]) / self.cell)), 0)
        iy1 = min(int(np.floor((y + radius - self.lo[1]) / self.cell)), self.ny - 1)
        if ix0 > ix1 or iy0 > iy1:
            return np.empty(0, dtype=np.int64)

        dist = self.cell_distance(x, y, ix0, ix1, iy0, iy1)
        stride = np.full(dist.shape, lod[-1][1], dtype=np.int64)
        prev = -np.inf
        for limit, s in lod:
            stride[(dist > prev) & (dist <= limit)] = s
            prev = limit

        in_range = dist <= radius
        starts = self.starts
        order = self.order
        cells = (
            np.arange(ix0, ix1 + 1, dtype=np.int64)[:, None] * self.ny
            + np.arange(iy0, iy1 + 1, dtype=np.int64)[None, :]
        )
        sel_cells = cells[in_range]
        sel_stride = stride[in_range]
        lo_idx = starts[sel_cells]
        hi_idx = starts[sel_cells + 1]

        chunks = []
        for s, e, st in zip(lo_idx.tolist(), hi_idx.tolist(), sel_stride.tolist()):
            if e > s:
                chunks.append(order[s:e:st])
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks).astype(np.int64)

    def cell_bounds(self, ix: np.ndarray, iy: np.ndarray) -> tuple:
        """XY bounds of the given cell indices, as four arrays."""
        x_lo = self.lo[0] + ix * self.cell
        y_lo = self.lo[1] + iy * self.cell
        return x_lo, y_lo, x_lo + self.cell, y_lo + self.cell

    def query_radius_positions(self, x: float, y: float, radius: float, lod,
                               keep_cell=None) -> np.ndarray:
        """As :meth:`query_radius_lod`, but in bucket-slot rather than vertex ids.

        Slot positions index :meth:`sorted_arrays` directly, so the renderer can
        read the block in ascending address order.  Map back with
        ``order[positions]`` when the original vertex id is needed.

        The result is a *superset* of the exact radius query: whole cells are
        kept whenever their nearest edge is within ``radius``, and every point
        of such a cell comes along, including points beyond the radius.  The
        renderer distance-filters afterwards, so this is a deliberate trade of
        exactness for one contiguous read; it is not interchangeable with
        :meth:`query_radius`.
        """
        # Stride 1 everywhere is the same answer as an undecimated query, and it
        # keeps the result ascending, which callers rely on to read the sorted
        # cache as one contiguous span.
        return self._query_lod_slots(x, y, radius, lod or ((np.inf, 1),),
                                     keep_cell=keep_cell)

    def read_sorted(self, arrays, slots: np.ndarray) -> np.ndarray:
        """Gather ``slots`` from bucket-ordered arrays, reading one span.

        The whole range from the smallest to the largest slot is read
        sequentially and the selection then happens in RAM.  Indexing a memmap
        with an integer array is a per-element gather, and it dominated
        rendering: tens of seconds per frame against roughly one for the span
        read.

        The result is correct for slots in any order.  The read is cheap only
        when they are clustered, which they are for both callers -- a radius
        query returns an ascending run, and a rendered frame's pixels address a
        spatially coherent region.
        """
        if slots.size == 0:
            return np.empty((0,) + arrays.shape[1:], dtype=arrays.dtype)
        lo = int(slots.min())
        hi = int(slots.max()) + 1
        span = np.array(arrays[lo:hi])
        return span[slots - lo]

    def _query_lod_slots(self, x: float, y: float, radius: float, lod,
                         keep_cell=None) -> np.ndarray:
        ix0 = max(int(np.floor((x - radius - self.lo[0]) / self.cell)), 0)
        ix1 = min(int(np.floor((x + radius - self.lo[0]) / self.cell)), self.nx - 1)
        iy0 = max(int(np.floor((y - radius - self.lo[1]) / self.cell)), 0)
        iy1 = min(int(np.floor((y + radius - self.lo[1]) / self.cell)), self.ny - 1)
        if ix0 > ix1 or iy0 > iy1:
            return np.empty(0, dtype=np.int64)

        dist = self.cell_distance(x, y, ix0, ix1, iy0, iy1)
        stride = np.full(dist.shape, lod[-1][1], dtype=np.int64)
        prev = -np.inf
        for limit, s in lod:
            stride[(dist > prev) & (dist <= limit)] = s
            prev = limit

        in_range = dist <= radius
        cells = (
            np.arange(ix0, ix1 + 1, dtype=np.int64)[:, None] * self.ny
            + np.arange(iy0, iy1 + 1, dtype=np.int64)[None, :]
        )
        # Cell-level culling runs before any point is touched.  A camera sees a
        # 90-degree cone, but the radius query returns a disc: dropping cells
        # the frustum cannot reach is what keeps tens of millions of points out
        # of the projection and the sort.
        keep = in_range
        if keep_cell is not None:
            sel_ix, sel_iy = np.nonzero(in_range)
            keep = np.zeros_like(in_range)
            if sel_ix.size:
                x_lo, y_lo, x_hi, y_hi = self.cell_bounds(
                    ix0 + sel_ix.astype(np.int64), iy0 + sel_iy.astype(np.int64))
                keep[sel_ix, sel_iy] = keep_cell(x_lo, y_lo, x_hi, y_hi)

        sel_cells = cells[keep]
        sel_stride = stride[keep]
        lo_idx = self.starts[sel_cells]
        hi_idx = self.starts[sel_cells + 1]

        chunks = []
        for s, e, st in zip(lo_idx.tolist(), hi_idx.tolist(), sel_stride.tolist()):
            if e > s:
                chunks.append(np.arange(s, e, st, dtype=np.int64))
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)

    @property
    def counts(self) -> np.ndarray:
        if getattr(self, "_counts", None) is None:
            self._counts = np.diff(self.starts)
        return self._counts

    def count_radius_approx(self, x: float, y: float, radius: float) -> int:
        """Bucket-sum occupancy estimate: sum whole cells whose centre is in range.

        The transform search scores thousands of candidate placements, and
        materialising every point for each is far too slow.  Summing per-cell
        counts is exact when ``radius`` is large next to the cell size and is a
        stable, deterministic ranking signal everywhere else; the winner is
        always re-scored exactly with :meth:`query_radius`.
        """
        counts = self.counts
        cx0 = np.floor((x - radius - self.lo[0]) / self.cell).astype(int)
        cx1 = np.floor((x + radius - self.lo[0]) / self.cell).astype(int)
        cy0 = np.floor((y - radius - self.lo[1]) / self.cell).astype(int)
        cy1 = np.floor((y + radius - self.lo[1]) / self.cell).astype(int)
        cx0, cx1 = max(cx0, 0), min(cx1, self.nx - 1)
        cy0, cy1 = max(cy0, 0), min(cy1, self.ny - 1)
        if cx0 > cx1 or cy0 > cy1:
            return 0

        xs = self.lo[0] + (np.arange(cx0, cx1 + 1) + 0.5) * self.cell
        ys = self.lo[1] + (np.arange(cy0, cy1 + 1) + 0.5) * self.cell
        inside = ((xs[:, None] - x) ** 2 + (ys[None, :] - y) ** 2) <= radius * radius
        # The cell ids of a rectangle are not a contiguous run, so index the
        # block explicitly rather than slicing between its two corners.
        cell_ids = (
            np.arange(cx0, cx1 + 1)[:, None] * self.ny
            + np.arange(cy0, cy1 + 1)[None, :]
        )
        return int(counts[cell_ids][inside].sum())


def _atomic_savez(path: Path, **arrays) -> None:
    """Write the cache via a temp file and rename.

    Several stages may be running against the same cache directory, and a
    half-written index would be read back as a valid one.
    """
    # np.savez appends ".npz" when the name does not already end with it, so the
    # temporary has to keep that suffix or the rename below targets nothing.
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _gather_cells(order: np.ndarray, starts: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    """Concatenate the given buckets into one allocation, in bucket order."""
    lo = starts[cell_ids]
    hi = starts[cell_ids + 1]
    out = np.empty(int((hi - lo).sum()), dtype=np.int64)
    pos = 0
    for s, e in zip(lo.tolist(), hi.tolist()):
        if e > s:
            out[pos:pos + (e - s)] = order[s:e]
            pos += e - s
    return out


def grid_index(ply_path: Path, cell: float = 4.0, cache: bool = True) -> XyGridIndex:
    return XyGridIndex(PlyCloud(ply_path), cell=cell).build(cache=cache)


def local_ground_height(cloud_xyz: np.ndarray, percentile: float = 2.0) -> float:
    """Robust local ground estimate: a low percentile of the local Z values.

    Used by the Z-consistency diagnostic, where the question is whether the UAV
    sits above whatever the point cloud says the surface is underneath it.
    """
    if cloud_xyz.size == 0:
        return float("nan")
    return float(np.percentile(cloud_xyz[:, 2], percentile))


def dumps_bounds(summary: dict) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)
