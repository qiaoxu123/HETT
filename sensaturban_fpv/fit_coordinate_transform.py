"""Fit ``p_ply = s * R @ p_citynav + t`` under a deliberately tiny hypothesis class.

Only evidence decides.  ``R`` is restricted to the eight signed permutations of
the XY plane (the dihedral group: four rotations, each optionally reflected),
``z`` is carried through untouched, ``s`` is searched over a small explicit grid,
and ``t`` is a translation.  No general affine fit: a 12-parameter fit can
always make a wrong frame look plausible, and the point of this experiment is to
find out whether a *defensible* correspondence exists at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# (name, x' = a*x + b*y, y' = c*x + d*y)
DIHEDRAL = (
    ("identity", 1, 0, 0, 1),
    ("rot90_ccw", 0, -1, 1, 0),
    ("rot180", -1, 0, 0, -1),
    ("rot270_ccw", 0, 1, -1, 0),
    ("swap_xy", 0, 1, 1, 0),
    ("swap_neg_xy", 0, -1, -1, 0),
    ("flip_x", -1, 0, 0, 1),
    ("flip_y", 1, 0, 0, -1),
)


@dataclass(frozen=True)
class CoordinateTransform:
    """``xy_ply = scale * m @ xy_citynav + t``, with ``z`` passed through."""

    m: np.ndarray
    scale: float
    t: np.ndarray
    name: str = "custom"

    def __post_init__(self):
        object.__setattr__(self, "m", np.asarray(self.m, dtype=np.float64).reshape(2, 2))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64).reshape(2))

    @classmethod
    def identity(cls) -> "CoordinateTransform":
        return cls(np.eye(2), 1.0, np.zeros(2), "identity")

    @classmethod
    def from_orientation(cls, name: str, a, b, c, d, scale: float,
                         t) -> "CoordinateTransform":
        return cls(np.array([[a, b], [c, d]], dtype=np.float64), scale,
                   np.asarray(t, dtype=np.float64), name)

    def apply_xy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return self.scale * (xy @ self.m.T) + self.t[None, :]

    def apply_xyz(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        out = xyz.copy()
        out[:, :2] = self.apply_xy(xyz[:, :2])
        return out

    def inverse_xyz(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        out = xyz.copy()
        out[:, :2] = ((xyz[:, :2] - self.t[None, :]) @ self.m) / self.scale
        return out

    def yaw_delta_rad(self) -> float:
        """Heading rotation this transform imposes, in the CityNav yaw sense."""
        # CityNav heading f = (cos yaw, sin yaw); the transform sends it to
        # scale * m @ f, whose heading differs from yaw by this constant.
        f = self.m @ np.array([1.0, 0.0])
        return float(np.arctan2(f[1], f[0]))

    @property
    def swaps_xy(self) -> bool:
        return abs(self.m[0, 0]) < 1e-9

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "matrix_2x2": self.m.tolist(),
            "scale": float(self.scale),
            "translation": self.t.tolist(),
            "swaps_xy": bool(self.swaps_xy),
            "yaw_offset_deg": float(np.rad2deg(self.yaw_delta_rad())),
            "formula": "xy_ply = scale * matrix @ xy_citynav + translation; z_ply = z_citynav",
        }


def occupancy_score(landmarks_xy: np.ndarray, grid, radius: float,
                    target_points: float = 50.0) -> dict:
    """How much measured geometry sits at each landmark, capped at ``target``."""
    counts = np.array(
        [grid.count_radius_approx(float(x), float(y), radius) for x, y in landmarks_xy],
        dtype=np.float64,
    )
    frac = np.clip(counts / target_points, 0.0, 1.0)
    return {
        "mean_capped_occupancy": float(frac.mean()) if frac.size else 0.0,
        "mean_points": float(counts.mean()) if counts.size else 0.0,
        "median_points": float(np.median(counts)) if counts.size else 0.0,
        "nonempty_ratio": float((counts > 0).mean()) if counts.size else 0.0,
        "counts": counts.tolist(),
    }


def trajectory_in_bounds_ratio(traj_xy: np.ndarray, cloud_lo, cloud_hi, margin: float = 0.0) -> float:
    inside = (
        (traj_xy[:, 0] >= cloud_lo[0] - margin) & (traj_xy[:, 0] <= cloud_hi[0] + margin)
        & (traj_xy[:, 1] >= cloud_lo[1] - margin) & (traj_xy[:, 1] <= cloud_hi[1] + margin)
    )
    return float(inside.mean()) if len(traj_xy) else 0.0


def fit_transform(
    trajectory_xyz: np.ndarray,
    landmark_xyz: np.ndarray,
    cloud_lo: np.ndarray,
    cloud_hi: np.ndarray,
    grid,
    radius: float = 5.0,
    scales=(1.0,),
    coarse_extent: float = 40.0,
    coarse_step: float = 8.0,
    fine_extent: float = 8.0,
    fine_step: float = 1.0,
    target_points: float = 50.0,
) -> dict:
    """Search orientation x scale x translation for the best landmark occupancy.

    The translation is seeded by matching the landmark bounding-box centre to the
    point-cloud bounding-box centre and then refined on a two-stage grid, never
    by hand.  Every orientation/scale pair gets the same treatment, so a winner
    is a result rather than a tuning artefact.
    """
    landmark_xy = np.asarray(landmark_xyz, dtype=np.float64)[:, :2]
    traj_xy = np.asarray(trajectory_xyz, dtype=np.float64)[:, :2]
    # Only the horizontal extent takes part: the transform never moves z, so
    # including it here would make the seed depend on the z datum.
    cloud_lo_xy = np.asarray(cloud_lo, dtype=np.float64)[:2]
    cloud_hi_xy = np.asarray(cloud_hi, dtype=np.float64)[:2]
    cloud_centre = (cloud_lo_xy + cloud_hi_xy) / 2.0

    results = []
    for name, a, b, c, d in DIHEDRAL:
        for scale in scales:
            m = np.array([[a, b], [c, d]], dtype=np.float64)
            base = CoordinateTransform(m, scale, np.zeros(2), name)
            if scale <= 0:
                continue
            lm0 = base.apply_xy(landmark_xy)
            seed = cloud_centre - lm0.mean(axis=0)

            # The seed comes from bounding-box centre matching, so a true
            # translation much smaller than the seed offset can fall outside the
            # refinement grid.  The zero translation is therefore always a
            # candidate in its own right: "no translation at all" has to be
            # testable, not merely reachable by luck.
            best = None
            for extent, step in ((coarse_extent, coarse_step), (fine_extent, fine_step)):
                offsets = np.arange(-extent, extent + 1e-9, step)
                candidates = [seed + np.array([dx, dy])
                              for dx in offsets for dy in offsets]
                if best is None:
                    candidates.append(np.zeros(2))
                for t in candidates:
                        cand = CoordinateTransform(m, scale, t, name)
                        lm = cand.apply_xy(landmark_xy)
                        occ = occupancy_score(lm, grid, radius, target_points)
                        score = occ["mean_capped_occupancy"]
                        # Occupancy is a capped, saturating count, so a plateau
                        # of placements can tie at the top.  Break ties towards
                        # the smallest translation: when the identity placement
                        # explains the data just as well, report identity rather
                        # than an arbitrary offset that merely scores the same.
                        tie = float(np.linalg.norm(t))
                        if best is None or (score, -tie) > (best["score"], -best["tie"]):
                            best = {
                                "score": score,
                                "tie": tie,
                                "transform": cand,
                                "occupancy": occ,
                                "seed": seed.tolist(),
                            }
                seed = np.asarray(best["transform"].t, dtype=np.float64)

            cand = best["transform"]
            lm = cand.apply_xy(landmark_xy)
            tr = cand.apply_xy(traj_xy)
            results.append({
                "orientation": name,
                "scale": float(scale),
                "score": float(best["score"]),
                "in_bounds_ratio": trajectory_in_bounds_ratio(
                    tr, cloud_lo_xy, cloud_hi_xy),
                "nonempty_ratio": best["occupancy"]["nonempty_ratio"],
                "median_points": best["occupancy"]["median_points"],
                "transform": cand,
                "seed_translation": best["seed"],
            })

    results.sort(key=lambda r: (-r["score"], -r["in_bounds_ratio"]))
    best = results[0]
    identity = next(r for r in results if r["orientation"] == "identity" and
                    abs(r["scale"] - 1.0) < 1e-12)

    return {
        "best": best,
        "identity": identity,
        "all": results,
        "search": {
            "radius_m": radius,
            "scales": list(scales),
            "coarse_extent_m": coarse_extent, "coarse_step_m": coarse_step,
            "fine_extent_m": fine_extent, "fine_step_m": fine_step,
            "target_points_per_landmark": target_points,
            "landmarks": int(len(landmark_xy)),
            "note": (
                "translation search is seeded from bounding-box centre matching and "
                "refined on a grid; no hand-tuned offset enters the search"
            ),
        },
    }


def write_transform(path: Path, transform: CoordinateTransform, extra: dict | None = None) -> None:
    payload = transform.to_json()
    if extra:
        payload["evidence"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_transform(path: Path) -> CoordinateTransform:
    payload = json.loads(Path(path).read_text())
    return CoordinateTransform(
        np.asarray(payload["matrix_2x2"], dtype=np.float64),
        float(payload["scale"]),
        np.asarray(payload["translation"], dtype=np.float64),
        payload.get("name", "loaded"),
    )
