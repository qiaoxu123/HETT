"""Ground an instruction's anchors in a block, then score candidates by relation.

Two things have to be true for "the white building beside the church" to be
resolved.  First the word *church* has to be attached to one entity in the block
-- the anchor grounder.  Then a candidate's position relative to that entity has
to be checked against *beside* -- the relation reasoner.  The previous round
showed the second is worth a great deal if the first is given, and that nothing
in the visual channel is worth anything at all; this module is the first half of
that chain.

**Anchors are grounded against the whole block.**  Every entity in the map is a
candidate anchor, not the ten target candidates.  Measuring anything over the
candidate list is the trap the feature round fell into: that list is built by
proximity to the answer, so a statistic over it describes the sampling procedure.

**The frame of a relation is written down.**  "left of" means one thing from the
agent's heading and another from the map's axes, and "behind" means one thing
from the heading and another from the anchor's own orientation.  Each relation
declares its frame in :mod:`sensaturban_fpv.anchor_parser`, the reasoner reads
that declaration, and the choice is reported rather than left to be inferred.

Everything here is computed from the instruction, the annotations and the
agent's pose -- all of which exist at test time.  The one function that needs to
know the answer, :func:`gt_anchor_for`, exists only to define the evaluation
subset and is never called on the prediction path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .anchor_parser import TYPE_WORDS, normalise

TYPE_NAMES = list(TYPE_WORDS)
TYPE_INDEX = {t: i for i, t in enumerate(TYPE_NAMES)}


@dataclass
class EntityNode:
    """One entity of a block, as the grounder sees it."""

    entity_id: int
    object_type: str
    name: str
    position: np.ndarray
    dimension: np.ndarray

    @property
    def norm_name(self) -> str:
        return normalise(self.name)


def block_nodes(objects) -> list:
    """Every entity of a block, as ``EntityNode``s."""
    return [EntityNode(int(o.id), o.object_type, (o.name or "").strip(),
                       np.asarray(o.position, dtype=np.float64),
                       np.asarray(o.dimension, dtype=np.float64))
            for o in objects.values()]


# --------------------------------------------------------------------------
# anchor grounding
# --------------------------------------------------------------------------

def _tokens(text: str) -> set:
    return {w for w in text.split() if len(w) > 2}


def lexical_anchor_scores(phrase: str, nodes, phrase_type: str | None = None,
                          name_weight: float = 2.0, type_weight: float = 0.6,
                          token_weight: float = 0.3) -> np.ndarray:
    """A0: match by name substring and by type word, with no learned component.

    This is the sanity baseline the brief asks for, and on this dataset it is
    strong: 93% of instructions contain the exact name of an entity in their own
    block.  Name matching is deliberately asymmetric -- the instruction says "St
    Teresa", the map says "St. Teresa of the Child Jesus Catholic Church" -- so a
    prefix or substring match counts, scaled by how much of the name it covers.
    """
    phrase = normalise(phrase)
    scores = np.zeros(len(nodes), dtype=np.float64)
    if not phrase:
        return scores
    for i, node in enumerate(nodes):
        score = 0.0
        name = node.norm_name
        if name:
            if name == phrase:
                score += name_weight
            elif phrase in name or name in phrase:
                # A short phrase inside a long map name is a weaker claim than a
                # long phrase inside a short one.
                cover = min(len(phrase), len(name)) / max(len(phrase), len(name))
                score += name_weight * (0.4 + 0.6 * cover)
            else:
                shared = _tokens(phrase) & _tokens(name)
                if shared:
                    score += token_weight * len(shared) / max(len(_tokens(phrase)), 1)
        if phrase_type and node.object_type == phrase_type:
            score += type_weight
        elif phrase_type is None:
            words = set(phrase.split())
            if any(w in TYPE_WORDS.get(node.object_type, ()) for w in words):
                score += type_weight
        scores[i] = score
    return scores


def semantic_anchor_scores(phrase_embedding, type_embeddings, name_embeddings,
                           nodes) -> np.ndarray:
    """A1: frozen text-encoder similarity between the anchor phrase and an entity.

    An entity is described to the encoder by its map name when it has one and by
    its type word when it does not.  Names are sparse -- a quarter of buildings
    and no vehicles carry one -- so on most entities this arm reduces to the same
    evidence as the lexical baseline, which is itself worth knowing.
    """
    scores = np.zeros(len(nodes), dtype=np.float64)
    for i, node in enumerate(nodes):
        name_vec = name_embeddings.get(node.norm_name)
        type_vec = type_embeddings.get(node.object_type)
        best = -1.0
        if name_vec is not None:
            best = max(best, float(phrase_embedding @ name_vec))
        if type_vec is not None:
            best = max(best, float(phrase_embedding @ type_vec) - 0.05)
        scores[i] = best
    return scores


def anchor_log_probabilities(scores: np.ndarray, temperature: float = 0.25,
                             top_k: int = 10) -> tuple:
    """Top-K anchors and their log-probabilities, from any score vector.

    Returning K rather than one is the point: the reasoner marginalises over the
    hypotheses, so a grounder that is 60% right still contributes, instead of
    collapsing the whole chain at the first mistake.
    """
    if scores.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    scaled = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-6)
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= max(probs.sum(), 1e-12)
    order = np.argsort(-probs)[:max(top_k, 1)]
    return order, np.log(np.maximum(probs[order], 1e-12))


def gt_anchor_for(instruction_anchors, nodes, target_id: int,
                  min_name_len: int = 5):
    """The anchor entity an instruction names, for *evaluation only*.

    Returns ``(node_index, anchor_row)`` when the instruction contains an entity
    name that matches exactly one entity of the block other than the target, and
    ``None`` otherwise.  The target's own name is excluded so that a sentence
    which only names its target does not masquerade as having an anchor.

    This uses the target's identity, which is why it defines an evaluation subset
    and is never called while predicting.
    """
    best = None
    for row, anchor in enumerate(instruction_anchors):
        phrase = normalise(anchor["phrase"])
        if len(phrase) < min_name_len:
            continue
        matches = [i for i, node in enumerate(nodes)
                   if node.norm_name and node.entity_id != target_id
                   and (phrase in node.norm_name or node.norm_name in phrase)]
        if len(matches) == 1:
            if best is None or len(phrase) > best[2]:
                best = (matches[0], row, len(phrase))
    return None if best is None else (best[0], best[1])


# --------------------------------------------------------------------------
# spatial relations
# --------------------------------------------------------------------------

def agent_frame(uav_position, uav_yaw: float) -> tuple:
    """Heading and right-hand axes of the agent, in world XY."""
    heading = np.array([np.cos(uav_yaw), np.sin(uav_yaw)])
    right = np.array([np.sin(uav_yaw), -np.cos(uav_yaw)])
    return heading, right


def relation_geometry(candidate_xy, anchor_xy, uav_position, uav_yaw: float) -> dict:
    """The geometry a relation is read off, in both frames and at both scales."""
    candidate = np.asarray(candidate_xy, dtype=np.float64)[:2]
    anchor = np.asarray(anchor_xy, dtype=np.float64)[:2]
    uav = np.asarray(uav_position, dtype=np.float64)[:2]
    heading, right = agent_frame(uav_position, uav_yaw)
    delta = candidate - anchor
    distance = float(np.linalg.norm(delta))
    return {
        "dx": float(delta[0]), "dy": float(delta[1]),
        "distance": distance,
        "log_distance": float(np.log1p(distance)),
        "bearing": float(np.arctan2(delta[1], delta[0])),
        "sin_bearing": float(np.sin(np.arctan2(delta[1], delta[0]))),
        "cos_bearing": float(np.cos(np.arctan2(delta[1], delta[0]))),
        # Agent frame: how much further along the heading, and how far to the side.
        "ahead_gap": float(delta @ heading),
        "lateral_gap": float(delta @ right),
        "agent_distance": float(np.linalg.norm(candidate - uav)),
        "anchor_agent_distance": float(np.linalg.norm(anchor - uav)),
    }


def rule_relation_score(geometry: dict, family: str, frame: str,
                        scale_m: float = 25.0) -> float:
    """How well a candidate sits relative to an anchor under a named relation.

    Signed and roughly in [-1, 1], so a relation that is satisfied scores high and
    the opposite of that relation scores low.  Symmetric families ignore the
    frame; directional ones read it from the parser's declaration.
    """
    def squash(x, s=1.0):
        return float(np.tanh(x / max(s, 1e-6)))

    if family in ("proximity", "on_surface", "across"):
        return -squash(geometry["distance"], scale_m)
    if family == "distance":
        return squash(geometry["distance"], scale_m * 2.0)
    if family == "between":
        # Handled by the caller, which has both anchors.
        return -squash(geometry["distance"], scale_m)
    if family == "vertical":
        return squash(geometry["dy"], scale_m * 0.5)
    if family == "front_back":
        if frame == "agent":
            return squash(geometry["ahead_gap"], scale_m * 1.5)
        return squash(geometry["ahead_gap"], scale_m * 1.5)
    if family == "left_right":
        if frame == "agent":
            return squash(geometry["lateral_gap"], scale_m)
        return squash(geometry["dx"], scale_m)
    if family == "cardinal":
        # The phrase names the direction from the anchor to the candidate, so
        # "north of" wants a positive dy and "south of" a negative one; the
        # caller inverts for the southern and western words.
        return squash(geometry["dy"], scale_m)
    return 0.0


# (axis, sign): axis 1 is the world y (north), axis 0 the world x (east).  The
# table had these transposed, so "north of" was read off the east-west
# coordinate and every compass relation scored as noise.
CARDINAL_SIGN = {"north": (1, 1.0), "south": (1, -1.0),
                 "east": (0, 1.0), "west": (0, -1.0)}
FRONT_BACK_SIGN = {"behind": 1.0, "in front of": -1.0, "ahead of": -1.0}
LEFT_RIGHT_SIGN = {"left of": -1.0, "to the left": -1.0,
                   "right of": 1.0, "to the right": 1.0}


def oriented_relation_score(geometry: dict, relation: dict,
                            scale_m: float = 25.0) -> float:
    """Rule score for a relation, oriented by its own surface form.

    ``relation_geometry`` reports a signed gap in each frame; which sign counts
    as satisfying the relation is decided here, from the words, so that "left of"
    and "right of" cannot both be satisfied by the same configuration.
    """
    family, frame = relation["family"], relation["frame"]
    phrase = relation["phrase"]
    base = rule_relation_score(geometry, family, frame, scale_m)
    if family == "cardinal":
        axis, sign = CARDINAL_SIGN.get(phrase.split()[0], (0, 1))
        value = geometry["dy"] if axis == 1 else geometry["dx"]
        return float(np.tanh(sign * value / max(scale_m, 1e-6)))
    if family == "front_back":
        return FRONT_BACK_SIGN.get(phrase, 1.0) * base
    if family == "left_right":
        return LEFT_RIGHT_SIGN.get(phrase, 1.0) * base
    return base


def between_score(candidate_xy, anchor_a_xy, anchor_b_xy, uav_position,
                  uav_yaw: float, tolerance_m: float = 20.0) -> float:
    """How well a candidate sits *between* two anchors.

    Combines the two things "between" asserts: the candidate is near the line
    joining them, and its projection falls inside the segment rather than beyond
    either end.  The second is what separates "between the church and the road"
    from "past the church".
    """
    a = np.asarray(anchor_a_xy, dtype=np.float64)[:2]
    b = np.asarray(anchor_b_xy, dtype=np.float64)[:2]
    c = np.asarray(candidate_xy, dtype=np.float64)[:2]
    ab = b - a
    length = float(np.linalg.norm(ab))
    if length < 1e-6:
        return -1.0
    t = float((c - a) @ ab) / (length ** 2)
    perp = float(np.linalg.norm((c - a) - t * ab))
    inside = 1.0 if 0.0 <= t <= 1.0 else -min(abs(t), abs(t - 1.0), 1.0)
    return float(np.tanh(inside * 1.5) * (1.0 - np.tanh(perp / tolerance_m)))


def relation_features(geometry: dict, relation: dict, candidate_dim,
                      anchor_dim) -> np.ndarray:
    """Fixed-length input for the learned reasoner."""
    family, frame = relation["family"], relation["frame"]
    family_onehot = [1.0 if family == f else 0.0 for f in
                     ("proximity", "distance", "front_back", "left_right",
                      "cardinal", "between", "across", "on_surface", "vertical")]
    return np.array([
        geometry["dx"] / 100.0, geometry["dy"] / 100.0,
        geometry["log_distance"], geometry["distance"] / 200.0,
        geometry["sin_bearing"], geometry["cos_bearing"],
        geometry["ahead_gap"] / 100.0, geometry["lateral_gap"] / 100.0,
        geometry["agent_distance"] / 200.0,
        (geometry["anchor_agent_distance"] - geometry["agent_distance"]) / 100.0,
        float(np.log1p(max(candidate_dim[0] * candidate_dim[1], 0.0))) / 5.0,
        float(np.log1p(max(anchor_dim[0] * anchor_dim[1], 0.0))) / 5.0,
        float(candidate_dim[2]) / 20.0, float(anchor_dim[2]) / 20.0,
        1.0 if frame == "agent" else 0.0,
        1.0 if frame == "global" else 0.0,
    ] + family_onehot, dtype=np.float32)


RELATION_FEATURE_DIM = 16 + 9


def marginalise(anchor_log_probs: np.ndarray, compatibilities: np.ndarray,
                temperature: float = 1.0) -> float:
    """``logsumexp_j [ log P(a_j) + R(t, a_j) ]`` for one candidate.

    Marginalising rather than committing to the best anchor is what keeps a
    single grounding mistake from ending the reasoning: a candidate that is the
    right distance from the second-best anchor still scores.
    """
    if anchor_log_probs.size == 0:
        return 0.0
    values = anchor_log_probs + np.asarray(compatibilities) / max(temperature, 1e-6)
    top = float(values.max())
    return float(top + np.log(np.exp(values - top).sum()))
