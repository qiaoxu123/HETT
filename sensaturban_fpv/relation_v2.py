"""A relation reasoner that is fitted, and the counterfactuals that test it.

The previous round's rule function was written by hand and the oracle showed it
was wrong -- handing it the *true* anchor made it score worse than handing it a
guess.  This module replaces it with a small fitted scorer, and, more
importantly, with a set of tests that can fail.

Three things are deliberately separated.

**The geometry is raw.**  :data:`GEOM_DIM` numbers in three frames at once, none
of them pre-combined into a per-relation scalar.  The audit found that no single
axis separates any relation well, so a scorer that received one axis per
relation would have had its answer chosen before it saw the data.

**The relation enters as an embedding, and the loss forces it to matter.**  A
listwise ranking loss alone is happy to ignore the relation word and learn the
trajectory prior instead -- that prior is worth AUC 0.561 on its own, so it is
the cheapest thing to fit.  The counterfactual margin term is what makes the
word carry weight: the same target against the same anchor must score higher
under the relation the sentence used than under its opposite.

**The self-test does not involve the visual channel at all.**  It asks whether
the reasoner can rank the target given the right anchor and the right relation.
If it cannot, the round stops there; a fusion number computed on top of a
reasoner that fails this would only measure the visual baseline it was added to.
"""

from __future__ import annotations

import numpy as np

GEOM_DIM = 18
REL_DIM = 16

# Phrases that name the same relation, so that "the same word" is not confused
# with "the same string".
OPPOSITES = {
    "north of": "south of", "south of": "north of",
    "east of": "west of", "west of": "east of",
    "left of": "right of", "right of": "left of",
    "to the left": "to the right", "to the right": "to the left",
    "behind": "in front of", "in front of": "behind", "ahead of": "behind",
    "near": "far from", "far from": "near",
    "beside": "far from", "next to": "far from", "close to": "far from",
    "adjacent to": "far from", "alongside": "far from", "by the": "far from",
    "away from": "near",
    "on": "far from",
    "above": "below", "below": "above", "under": "above",
    "on top of": "below", "overlooking": "below",
}

# Relations with no meaningful opposite; the counterfactual for these is a
# different relation from the same family wherever one exists, and otherwise a
# random draw, which is the honest fallback and is reported as such.
NO_OPPOSITE = {"between", "across from", "facing", "opposite", "surrounding",
               "bordering"}


# Column meanings of the geometry vector.  Named because the whole round turns
# on which of these a word like "behind" is read off, and an off-by-one here
# would be invisible in a score but would silently answer the wrong question.
GEOM_COLUMNS = (
    "global_x", "global_y", "log_distance", "distance",
    "sin_bearing", "cos_bearing", "agent_ahead", "agent_lateral",
    "agent_distance", "agent_distance_gap", "anchor_along", "anchor_perp",
    "anchor_along_rel", "anchor_perp_rel", "cand_log_area", "anchor_log_area",
    "cand_height", "anchor_log_half_span",
)
AXIS_COLUMNS = {
    "global_x": 0, "global_y": 1,
    "agent_ahead": 6, "agent_lateral": 7,
    "anchor_along": 10, "anchor_perp": 11,
}
assert len(GEOM_COLUMNS) == GEOM_DIM


def principal_axis(contour) -> np.ndarray:
    """Unit vector along the anchor's longest footprint direction, or (1, 0)."""
    pts = np.asarray([[float(p[0]), float(p[1])] for p in contour], dtype=np.float64)
    if pts.shape[0] < 2:
        return np.array([1.0, 0.0])
    centred = pts - pts.mean(axis=0)
    cov = centred.T @ centred / max(pts.shape[0], 1)
    try:
        values, vectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.array([1.0, 0.0])
    axis = vectors[:, int(np.argmax(values))]
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-9 else np.array([1.0, 0.0])


def geometry(cand_xy, anchor_xy, anchor_axis, uav_xy, yaw, cand_dim,
             anchor_dim) -> np.ndarray:
    """The numbers a relation is read off, in three frames at once.

    Kept as raw geometry rather than as a per-relation scalar so that the frame
    comparison and the learned scorer see exactly the same input; a scorer whose
    features had been pre-combined by one frame's rule could not be compared
    against another frame fairly.
    """
    c = np.asarray(cand_xy, dtype=np.float64)[:2]
    a = np.asarray(anchor_xy, dtype=np.float64)[:2]
    u = np.asarray(uav_xy, dtype=np.float64)[:2]
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([np.sin(yaw), -np.cos(yaw)])
    delta = c - a
    dist = float(np.linalg.norm(delta))
    axis = np.asarray(anchor_axis, dtype=np.float64)[:2]
    perp = np.array([-axis[1], axis[0]])
    half_span = 0.5 * float(max(anchor_dim[0], anchor_dim[1]))
    return np.array([
        delta[0] / 100.0, delta[1] / 100.0,                 # global frame
        float(np.log1p(dist)), dist / 200.0,
        float(np.sin(np.arctan2(delta[1], delta[0]))),
        float(np.cos(np.arctan2(delta[1], delta[0]))),
        float(delta @ heading) / 100.0,                     # agent frame
        float(delta @ right) / 100.0,
        float(np.linalg.norm(c - u)) / 200.0,
        (float(np.linalg.norm(a - u)) - float(np.linalg.norm(c - u))) / 100.0,
        float(delta @ axis) / 100.0,                        # anchor frame
        float(delta @ perp) / 100.0,
        float(delta @ axis) / max(half_span, 1.0),          # anchor half-widths
        float(delta @ perp) / max(half_span, 1.0),
        float(np.log1p(max(cand_dim[0] * cand_dim[1], 0.0))) / 5.0,
        float(np.log1p(max(anchor_dim[0] * anchor_dim[1], 0.0))) / 5.0,
        float(cand_dim[2]) / 20.0, float(np.log1p(half_span)) / 5.0,
    ], dtype=np.float32)


def between_geometry(candidate_xy, anchor_a_xy, anchor_b_xy,
                     tolerance_m: float = 20.0) -> tuple:
    """Projection ratio and perpendicular distance for a "between" relation.

    Returns the two quantities the word asserts: where the candidate falls along
    the segment joining the anchors (0 at one end, 1 at the other), and how far
    off that line it sits.  The ratio is what separates "between the church and
    the road" from "past the church", which a distance alone cannot.
    """
    a = np.asarray(anchor_a_xy, dtype=np.float64)[:2]
    b = np.asarray(anchor_b_xy, dtype=np.float64)[:2]
    c = np.asarray(candidate_xy, dtype=np.float64)[:2]
    ab = b - a
    length = float(np.linalg.norm(ab))
    if length < 1e-6:
        return 0.0, float("inf")
    t = float((c - a) @ ab) / (length ** 2)
    perp = float(np.linalg.norm((c - a) - t * ab))
    return t, perp


def build_relation_vocab(phrases) -> tuple:
    """A stable phrase -> index map with 0 reserved for "no relation"."""
    ordered = sorted(set(phrases))
    vocab = {p: i + 1 for i, p in enumerate(ordered)}
    return vocab, {i: p for p, i in vocab.items()}


def opposite_id(rel_id: int, vocab: dict, inv: dict, rng=None) -> int:
    """The id of the relation that means the opposite, or a fallback.

    Falls back to a random draw for the symmetric relations and to ``0`` for an
    unknown id, so the counterfactual arm is always defined and never silently
    becomes "the correct relation".
    """
    phrase = inv.get(int(rel_id))
    if phrase is None:
        return 0
    other = OPPOSITES.get(phrase)
    if other is not None and other in vocab:
        return vocab[other]
    if phrase in NO_OPPOSITE:
        choices = [i for p, i in vocab.items() if p != phrase]
        if rng is not None and choices:
            return int(choices[int(rng.integers(len(choices)))])
    return 0


def shuffled_ids(rel_ids: np.ndarray, rng) -> np.ndarray:
    """The same relation words, dealt out to different samples."""
    out = np.array(rel_ids, copy=True)
    rng.shuffle(out)
    return out


def softmax_logsumexp(values: np.ndarray, weights: np.ndarray,
                      axis: tuple = (1, 2)) -> np.ndarray:
    """``logsumexp_j [ log P(a_j) + score_j ]`` over anchor hypotheses."""
    scaled = values + weights
    top = scaled.max(axis=axis, keepdims=True)
    return (top + np.log(np.exp(scaled - top).sum(axis=axis, keepdims=True))
            ).squeeze(axis)


def rank_metrics(scores: np.ndarray, target: int, topk=(1, 4)) -> dict:
    """Rank of the target in one sample's score vector, and the derived metrics."""
    order = np.argsort(-scores, kind="stable")
    rank = int(np.flatnonzero(order == target)[0]) + 1
    out = {"rank": rank, "mrr": 1.0 / rank}
    for k in topk:
        out[f"top{k}"] = float(rank <= k)
    # Margin against the best wrong answer, which is what a reordering has to
    # overcome; the mean alone hides whether the target is first or fifth.
    other = np.delete(scores, target)
    out["margin"] = float(scores[target] - (other.max() if other.size else 0.0))
    return out


def aggregate(rows: list) -> dict:
    """Mean of the per-sample metrics, with a Wilson interval on Top-1."""
    import math
    out = {}
    if not rows:
        return {"n": 0}
    for key in ("top1", "top4", "mrr"):
        out[key] = float(np.mean([r[key] for r in rows]))
    out["median_margin"] = float(np.median([r["margin"] for r in rows]))
    n = len(rows)
    p = out["top1"]
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    out["top1_ci"] = [max(0.0, centre - half), min(1.0, centre + half)]
    out["n"] = n
    return out


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact-ish paired test over which of two arms got each sample right."""
    from scipy.stats import binomtest
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n": 0, "p_value": 1.0}
    return {"a_only": a_only, "b_only": b_only, "n": n,
            "p_value": float(binomtest(a_only, n, 0.5).pvalue)}
