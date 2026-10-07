"""Phase 3: a teacher that learns the ontology, and the gate that judges it.

If a relation-conditioned model cannot learn relations whose labels *are*
geometry, then either the graph is wrong or the model cannot express the
question -- and both of those are bugs to be found here rather than carried into
the language stage, where they would be indistinguishable from the corpus's own
inconsistency.  That is the whole reason the gate exists.

**The four controls, and why each one is here.**

* *shuffled relation* -- the same geometry under a relation drawn at random.  If
  this scores as well as the true relation, the model never read the relation.
* *opposite relation* -- the same geometry under the word's opposite.  This is
  the counterfactual the previous round could not pass: there, an opposite
  relation scored *higher* than the correct one.
* *zero relation* -- the embedding replaced by a constant, so the model sees
  geometry alone.  This measures how much of the score is the relation at all.
* *shuffled anchor* -- another example's geometry under this example's relation.
  A control that silently no-ops reads as a pass, so the code asserts the arms
  actually move the scores before reporting them; two earlier attempts at this
  control in this project were no-ops that looked like successes.

The edge vector already encodes the anchor implicitly, so "shuffling the anchor"
is done by donating another example's geometry rather than by permuting a slot.
"""

from __future__ import annotations

import numpy as np

from .edge_features import EDGE_DIM
from .relation_geometry import OPPOSITE, RELATION_NAMES

NO_RELATION = 0


def relation_vocab(names=RELATION_NAMES) -> dict:
    """Stable phrase -> id, with 0 reserved for "no relation"."""
    return {name: i + 1 for i, name in enumerate(sorted(names))}


def build_model(torch, vocab_size: int, hidden: int = 128, rel_dim: int = 48):
    nn = torch.nn

    class RelationTeacher(nn.Module):
        """An MLP over the edge vector, conditioned on a relation embedding.

        Not a transformer, and not a graph network: at this stage the graph's
        contribution is the *edge features*, and adding message passing would
        make a failure impossible to attribute.  Phase 6 adds the graph once
        this one is known to work.
        """

        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size + 1, rel_dim)
            self.geometry = nn.Sequential(
                nn.Linear(EDGE_DIM, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU())
            self.head = nn.Sequential(
                nn.Linear(hidden + rel_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 1))

        def forward(self, edge, rel):
            geo = self.geometry(edge)
            emb = self.embed(rel.clamp(min=0))
            return self.head(torch.cat([geo, emb], dim=-1)).squeeze(-1)

    return RelationTeacher()


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

def arm_relations(arm: str, rel_ids: np.ndarray, vocab: dict, seed: int = 0,
                  rng=None) -> np.ndarray:
    """The relation ids one gate arm uses, given the true ones."""
    rng = rng or np.random.default_rng(seed)
    inv = {i: name for name, i in vocab.items()}
    if arm == "correct":
        return rel_ids
    if arm == "zero_relation":
        return np.full_like(rel_ids, NO_RELATION)
    if arm == "shuffled_relation":
        out = rel_ids.copy()
        rng.shuffle(out)
        return out
    if arm == "opposite":
        out = np.empty_like(rel_ids)
        choices = [i for i in vocab.values()]
        for k, rid in enumerate(rel_ids):
            name = inv.get(int(rid))
            other = OPPOSITE.get(name) if name else None
            if other is not None and other in vocab:
                out[k] = vocab[other]
            else:
                # No opposite exists for this relation (between, aligned_with,
                # ...), so the counterfactual is a different applicable
                # relation.  Reported as such rather than hidden.
                out[k] = int(rng.choice([c for c in choices if c != rid]))
        return out
    raise KeyError(arm)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def rank_metrics(scores: np.ndarray, positive_at: int = 0) -> dict:
    order = np.argsort(-scores, kind="stable")
    rank = int(np.flatnonzero(order == positive_at)[0]) + 1
    out = {"rank": rank, "mrr": 1.0 / rank}
    for k in (1, 4):
        out[f"top{k}"] = float(rank <= k)
    other = np.delete(scores, positive_at)
    out["margin"] = float(scores[positive_at] - (other.max() if other.size else 0.0))
    return out


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    diff = positive[:, None] - negative[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def aggregate(rows: list) -> dict:
    import math
    if not rows:
        return {"n": 0}
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("top1", "top4", "mrr")}
    out["median_margin"] = float(np.median([r["margin"] for r in rows]))
    n = len(rows)
    p, z = out["top1"], 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    out["top1_ci"] = [max(0.0, centre - half), min(1.0, centre + half)]
    out["n"] = n
    return out


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    from scipy.stats import binomtest
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n": 0, "p_value": 1.0}
    return {"a_only": a_only, "b_only": b_only, "n": n,
            "p_value": float(binomtest(a_only, n, 0.5).pvalue)}
