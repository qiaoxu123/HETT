"""A distribution over relation, frame and anchor, rather than one reading.

Every round before this one committed the sentence to a single relation, and
Gate E priced that commitment: the relation is demonstrably read — correct beats
opposite on the true target, 23 of 32, p = 0.020 — and yet it carries no
information the candidate's distance to its anchor does not already carry. The
hypothesis this round tests is that the loss happens at the commitment itself.
``behind the church`` is not one geometric question; it is at least four, and
which one the annotator meant may genuinely be undetermined by the sentence.

So the language side now returns a **distribution** over
``(relation, frame, anchors)``, and the geometry side is evaluated under each
hypothesis separately, and the two are combined by marginalisation::

    S(c) = logsumexp_k [ log P(h_k | language) + geometry(c | h_k) ]

**The teacher is not retrained.** It was trained on the fifteen relations whose
geometry is unambiguous and it never saw ``behind``, ``front``, ``left`` or
``right`` — those were excluded on purpose, because the audit found no frame the
corpus agrees on. That exclusion is exactly what this round revisits, and the
answer is not to train the viewpoint words into the teacher but to keep the
*frames* separate: a frame-dependent word is scored by an explicit geometric rule
under each frame the language allows, and the frames compete probabilistically.

**Anchors stay uncertain too.** A name that reaches several nodes keeps all of
them with a weight, and the marginalisation runs over anchor, relation and frame
together. Nothing here resolves an anchor by its distance to the answer.
"""

from __future__ import annotations

import numpy as np

# Words that need a frame, and the frames worth trying for each.  The brief
# requires all four to be carried and none to be fixed in advance.
FRAME_RELATIONS = {
    "behind": ("agent", "anchor", "road", "global"),
    "front": ("agent", "anchor", "road", "global"),
    "left": ("agent", "anchor", "road", "global"),
    "right": ("agent", "anchor", "road", "global"),
}
FRAMES = ("agent", "anchor", "road", "global", "none")

# Relations the *teacher* can score, since it was trained on exactly these.
TEACHER_RELATIONS = (
    "north_of", "south_of", "east_of", "west_of", "near", "far", "between",
    "same_side_of_road", "opposite_side_of_road", "across_road", "along_road",
    "near_intersection", "aligned_with", "parallel_to", "perpendicular_to",
)
# Everything the language side may emit.
HYPOTHESIS_RELATIONS = TEACHER_RELATIONS + tuple(FRAME_RELATIONS) + (
    "associated_with", "UNKNOWN",
)
NON_GEOMETRIC = ("associated_with", "UNKNOWN")


def normalised_entropy(probabilities) -> float:
    """Entropy of a distribution, divided by its maximum, so it lands in [0, 1].

    Dividing by ``log n`` rather than by a constant is what makes a two-hypothesis
    answer and a five-hypothesis answer comparable: a model that is certain with
    two options and one that is certain with five should both read as confident.
    """
    p = np.asarray([v for v in probabilities if v > 0], dtype=np.float64)
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(p.size))


def hypothesis_confidence(hypotheses) -> float:
    """``1 - normalised entropy`` over the hypothesis distribution."""
    return 1.0 - normalised_entropy([h.get("probability", 0.0)
                                     for h in hypotheses])


def clean_hypotheses(raw, top_k: int = 5) -> list:
    """Validated, renormalised, sorted hypotheses from a parser answer.

    A model answer is data.  Unknown relations are dropped rather than mapped,
    probabilities are clipped and renormalised so a sloppy sum cannot silently
    reweight an arm, and the list is truncated to ``top_k`` *after* sorting so
    that truncation drops the least likely rather than the last written.
    """
    out = []
    for h in raw or []:
        if not isinstance(h, dict):
            continue
        relation = h.get("relation")
        if relation not in HYPOTHESIS_RELATIONS:
            continue
        frame = h.get("frame") or "none"
        if relation in FRAME_RELATIONS:
            if frame not in FRAME_RELATIONS[relation]:
                frame = "agent"
        else:
            frame = "none"
        anchors = h.get("anchors")
        if isinstance(anchors, int):
            anchors = [anchors]
        anchors = [int(a) for a in (anchors or []) if isinstance(a, (int, float))]
        if relation not in NON_GEOMETRIC and not anchors:
            continue
        try:
            probability = float(h.get("probability", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(probability) or probability <= 0:
            continue
        out.append({"relation": relation, "frame": frame,
                    "anchors": anchors, "probability": probability})
    if not out:
        return []
    out.sort(key=lambda h: -h["probability"])
    out = out[:max(top_k, 1)]
    total = sum(h["probability"] for h in out)
    for h in out:
        h["probability"] = h["probability"] / total
    return out


def frame_axis(frame: str, anchor, candidate, context) -> tuple:
    """``(along, lateral)`` for a frame-dependent word, in metres.

    Each frame is a pair of directions and nothing more:

    * ``agent`` -- the UAV's heading and its right-hand axis
    * ``anchor`` -- the anchor's own footprint axes
    * ``road`` -- the nearest road region's tangent and normal
    * ``global`` -- the world x and y axes

    Returned as two signed numbers so that ``behind``, ``front``, ``left`` and
    ``right`` are four readings of one vector rather than four functions.
    """
    delta = np.asarray(candidate.center[:2], dtype=np.float64) - \
        np.asarray(anchor.center[:2], dtype=np.float64)
    if frame == "global":
        return float(delta[0]), float(delta[1])
    if frame == "agent":
        yaw = float(getattr(context, "agent_yaw", 0.0))
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        right = np.array([np.sin(yaw), -np.cos(yaw)])
        return float(delta @ heading), float(delta @ right)
    if frame == "anchor":
        axis = np.asarray(anchor.major_axis, dtype=np.float64)[:2]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return 0.0, 0.0
        axis = axis / norm
        return float(delta @ axis), float(delta @ np.array([-axis[1], axis[0]]))
    if frame == "road":
        region = getattr(context, "road", None)
        members = getattr(context, "road_members", ())
        if region is None or not members:
            return 0.0, 0.0
        from .edge_features import road_frame
        rf = road_frame(anchor, region, list(members))
        tangent = rf.tangent
        return float(delta @ tangent), float(delta @ np.array([-tangent[1],
                                                               tangent[0]]))
    return 0.0, 0.0


def frame_score(relation: str, along: float, lateral: float,
                scale_m: float = 30.0) -> float:
    """Signed geometry score for a frame-dependent word, in [-1, 1].

    ``behind`` means further along the frame's forward direction than the
    anchor, ``front`` the reverse; ``left`` and ``right`` read the lateral axis.
    All four are the *same two numbers* negated or swapped, which is what makes
    the frames comparable and the marginalisation meaningful: a hypothesis is a
    claim about which axes to read, not about a different function.
    """
    def squash(x):
        return float(np.tanh(x / max(scale_m, 1e-6)))

    if relation == "behind":
        return squash(along)
    if relation == "front":
        return -squash(along)
    if relation in ("left", "right"):
        # Facing +along, the right-hand axis is the frame's lateral direction.
        return -squash(lateral) if relation == "left" else squash(lateral)
    raise KeyError(relation)


def marginalise(log_probabilities, scores, temperature: float = 1.0) -> float:
    """``logsumexp_k [ log P(h_k) + score_k ]`` with the scores in hand."""
    log_p = np.asarray(log_probabilities, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-6)
    if log_p.size == 0:
        return 0.0
    values = log_p + s
    top = float(values.max())
    return float(top + np.log(np.exp(values - top).sum()))


def reliability_table() -> dict:
    """``rho`` per relation, measured on val_seen -- see the fitted artefact.

    Section 6 asks for a per-relation reliability weighting estimated on
    train/val only.  It is *not* hard-coded here: the values live in
    ``artifacts/spatial_graph/relation_reliability.json``, fitted by
    ``scripts/fit_relation_reliability.py`` on ``val_seen``, and this returns an
    empty mapping when that file does not exist so that an arm cannot silently
    use weights nobody measured.
    """
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "artifacts" / "spatial_graph" \
        / "relation_reliability.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())["rho"]
