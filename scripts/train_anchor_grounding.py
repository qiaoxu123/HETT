#!/usr/bin/env python3
"""Does grounding the anchor and reasoning about the relation fix the Top-1?

The feature round left a sharp shape: the appearance of a correctly masked target
is at chance, and an anchor-only oracle nearly doubles Top-1.  This script builds
the deployable version of that oracle and measures how much of it survives when
the anchor has to be found rather than given.

Every arm is scored on the entity round's fixed candidate sets, so a number here
is comparable with its 0.325.  The oracle rows exist to localise a failure --
parser, anchor grounder, relation reasoner or fusion -- and are labelled as such
in every table; they are never averaged into a deployable result.

Split discipline: the relation scorer and the fusion weights are fitted on
``train_seen``, every hyper-parameter is chosen on ``val_seen``, and
``val_unseen`` is read once.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.config import artifact_dir, load_config  # noqa: E402

SPLITS = ("train_seen", "val_seen", "val_unseen")
REL_DIM = 25
HIDDEN = 96
EPOCHS = 30
BATCH = 16
SEEDS = (0, 1, 2)
# Anchor-grounding arms, in the order the report presents them.
ANCHOR_ARMS = ("A0_lexical", "A1_semantic", "A0_plus_A1")
# How many anchors a rule or learned reasoner may use, chosen on val_seen.
K_CHOICES = (1, 3, 5, 10)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class Sample:
    __slots__ = ("phrase_text", "uav_position", "uav_yaw", "gt_anchor_dim",
                 "node_positions", "node_dimensions",
                 "split", "map", "key", "instruction", "target", "candidate_ids",
                 "parsed", "anchor_ids", "anchor_logp", "a0_ids", "anchor_xyz",
                 "rel_feat", "rule_score", "gt_row", "gt_node", "gt_xyz",
                 "group", "visual", "buckets")


def permute_sample(sample, rng):
    """Shuffle the candidate order, target included.

    The candidate list is built as ``[referenced] + distractors``, so the answer
    sits at index 0 in every sample.  That is not merely a leak a model could
    learn -- it corrupts any arm whose score vector can tie, because a tie is
    broken by position and position 0 is always the answer.  An earlier version
    of the anchor-only arm scored 0.830 for exactly that reason.
    """
    n = len(sample.candidate_ids)
    if n < 2:
        return
    order = rng.permutation(n)
    sample.target = int(np.flatnonzero(order == sample.target)[0])
    sample.candidate_ids = [sample.candidate_ids[i] for i in order]
    sample.visual = sample.visual[order]
    if sample.rel_feat.ndim == 4 and sample.rel_feat.shape[0] == n:
        sample.rel_feat = sample.rel_feat[order]
    if sample.rule_score.ndim == 3 and sample.rule_score.shape[0] == n:
        sample.rule_score = sample.rule_score[order]


def load_anchors(path: Path, visual_dir: Path):
    out = {s: [] for s in SPLITS}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample = Sample()
        sample.split, sample.map, sample.key = row["split"], row["map"], row["key"]
        sample.instruction = row["instruction"]
        sample.target = int(row["target_index"])
        sample.candidate_ids = [int(v) for v in row["candidate_ids"]]
        sample.parsed = row["parsed"]
        sample.anchor_ids = np.asarray(row.get("anchor_ids", np.zeros((0, 0))),
                                       dtype=np.int64)
        sample.anchor_logp = np.asarray(row.get("anchor_logp", np.zeros((0, 0))),
                                        dtype=np.float64)
        sample.a0_ids = np.asarray(row.get("anchor_a0_ids", np.zeros((0, 0))),
                                   dtype=np.int64)
        sample.anchor_xyz = np.asarray(row.get("anchor_xyz", np.zeros((0, 0, 3))),
                                       dtype=np.float64)
        sample.rel_feat = np.asarray(row.get("rel_feat", np.zeros((0, 0, 0, REL_DIM))),
                                     dtype=np.float32)
        sample.rule_score = np.asarray(row.get("rule_score", np.zeros((0, 0, 0))),
                                       dtype=np.float64)
        sample.gt_row = int(row.get("gt_anchor_row", -1))
        sample.gt_node = int(row.get("gt_anchor_node", -1))
        sample.gt_xyz = np.asarray(row.get("gt_anchor_xyz", np.zeros(3)),
                                   dtype=np.float64)
        sample.uav_position = np.asarray(row.get("uav_position", np.zeros(3)),
                                         dtype=np.float64)
        sample.uav_yaw = float(row.get("uav_yaw", 0.0))
        sample.gt_anchor_dim = np.asarray(row.get("gt_anchor_dim", np.zeros(3)),
                                          dtype=np.float64)
        sample.node_positions = np.asarray(row.get("candidate_positions",
                                                   np.zeros((0, 3))), dtype=np.float64)
        sample.node_dimensions = np.asarray(row.get("candidate_dimensions",
                                                    np.zeros((0, 3))), dtype=np.float64)
        archive = np.load(visual_dir / row["key"], allow_pickle=True)
        sample.visual = archive["td_context_masked"].astype(np.float32)
        # The entity round's 0.325 baseline scored this feature against the
        # *referenced phrase*, not the whole sentence.  Keeping both is what
        # makes the two rounds comparable and shows what the extra words cost.
        phrase = archive["text_phrase"] if "text_phrase" in archive.files else None
        sample.phrase_text = (np.asarray(phrase, np.float32)
                              if phrase is not None and phrase.size else None)
        types = [str(t) for t in archive["entity_types"]]
        target_type = types[sample.target]
        sample.group = ("building" if target_type == "Building" else
                        "vehicle" if target_type in ("Car", "Bike") else "other")
        sample.buckets = _buckets(sample, row)
        out[sample.split].append(sample)
    rng = np.random.default_rng(20261007)
    for split in SPLITS:
        for sample in out[split]:
            permute_sample(sample, rng)
    return out


def _buckets(sample, row) -> set:
    parsed = sample.parsed
    out = set()
    if sample.gt_row >= 0:
        out.add("named_anchor")
    if any(a.get("type") for a in parsed.get("anchors", [])):
        out.add("category_anchor")
    if not parsed.get("anchors"):
        out.add("no_relation")
    fams = {a["family"] for a in parsed.get("anchors", [])}
    if "proximity" in fams:
        out.add("near_beside")
    if fams & {"front_back"}:
        out.add("behind_front")
    if fams & {"cardinal", "left_right"}:
        out.add("directional")
    if "between" in fams:
        out.add("multi_anchor")
    attrs = parsed.get("attributes", {})
    if attrs.get("color"):
        out.add("color_attribute")
    if attrs.get("size"):
        out.add("size_attribute")
    return out or {"other"}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def rank_metrics(scores, target):
    order = np.argsort(-np.asarray(scores))
    rank = int(np.where(order == target)[0][0]) + 1
    others = np.delete(np.asarray(scores), target)
    best_other = float(others.max()) if others.size else float("nan")
    return {"rank": rank, "top1": bool(rank == 1), "top4": bool(rank <= 4),
            "mrr": 1.0 / rank, "margin": float(scores[target] - best_other),
            "positive_margin": bool(scores[target] > best_other)}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(c - h, 0)), float(min(c + h, 1))]


def aggregate(entries):
    if not entries:
        return {"n": 0, "top1": None, "top4": None, "mrr": None,
                "median_margin": None, "positive_margin_ratio": None,
                "top1_ci": [None, None]}
    top1 = np.array([e["top1"] for e in entries], dtype=float)
    return {"n": len(entries), "top1": float(top1.mean()),
            "top4": float(np.mean([e["top4"] for e in entries])),
            "mrr": float(np.mean([e["mrr"] for e in entries])),
            "median_margin": float(np.median([e["margin"] for e in entries])),
            "positive_margin_ratio": float(np.mean(
                [e["positive_margin"] for e in entries])),
            "top1_ci": wilson(float(top1.sum()), len(entries))}


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return {"a_only": n10, "b_only": n01, "n": 0, "p_value": 1.0}
    k = min(n10, n01)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"a_only": n10, "b_only": n01, "n": n,
            "p_value": float(min(2 * tail, 1.0))}


def normalise_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    span = scores.max() - scores.min()
    return (scores - scores.mean()) / span if span > 1e-9 else scores * 0.0


# --------------------------------------------------------------------------
# anchor grounding evaluation
# --------------------------------------------------------------------------

def anchor_grounding_metrics(samples, arm: str, shuffle_seed: int | None = None):
    """Top-1/3/5/MRR of the predicted anchor, on the evaluable subset."""
    entries, by_kind = [], defaultdict(list)
    rng = np.random.default_rng(shuffle_seed if shuffle_seed is not None else 0)
    for sample in samples:
        if sample.gt_row < 0 or sample.anchor_ids.shape[0] == 0:
            continue
        row = sample.gt_row
        if row >= sample.anchor_ids.shape[0]:
            continue
        ids = sample.anchor_ids[row]
        if arm == "A0_lexical":
            ids = sample.a0_ids[row] if sample.a0_ids.shape[0] > row else ids
        elif arm == "A1_semantic":
            # The semantic arm alone: the ranking from the combined score with
            # the lexical component removed is not stored, so this arm is the
            # combined one reported separately from the lexical baseline.
            ids = sample.anchor_ids[row]
        if shuffle_seed is not None:
            # The control: keep the same number of hypotheses but draw them from
            # the block at random, so a grounded anchor has to beat chance.
            pool = [i for i in sample.anchor_ids.reshape(-1).tolist() if i >= 0]
            if pool:
                ids = rng.permutation(np.asarray(pool))[: len(ids)]
        wanted = sample.gt_node
        rank = None
        for position, eid in enumerate(ids.tolist()):
            if eid == wanted:
                rank = position + 1
                break
        if rank is None:
            continue
        entry = {"rank": rank, "top1": rank == 1, "top3": rank <= 3,
                 "top5": rank <= 5, "mrr": 1.0 / rank}
        entries.append(entry)
        by_kind["all"].append(entry)
        by_kind["named" if sample.gt_row >= 0 else "category"].append(entry)
    def agg(rows):
        if not rows:
            return {"n": 0}
        return {"n": len(rows),
                "top1": float(np.mean([r["top1"] for r in rows])),
                "top3": float(np.mean([r["top3"] for r in rows])),
                "top5": float(np.mean([r["top5"] for r in rows])),
                "mrr": float(np.mean([r["mrr"] for r in rows]))}
    return {"overall": agg(entries),
            "by_kind": {k: agg(v) for k, v in by_kind.items()}}


# --------------------------------------------------------------------------
# relation scoring arms
# --------------------------------------------------------------------------

def rule_relation_scores(sample, k: int) -> np.ndarray:
    """Marginalise the rule relation score over the top-k anchors of each phrase."""
    if sample.rule_score.size == 0:
        return np.zeros(len(sample.candidate_ids))
    scores = sample.rule_score[:, :, :k]                 # (C, A, k)
    weight = sample.anchor_logp[:, :k]                   # (A, k)
    values = scores + weight[None, :, :]
    flat = values.reshape(len(sample.candidate_ids), -1)
    top = flat.max(axis=1)
    return top + np.log(np.exp(flat - top[:, None]).sum(axis=1))


def mask_features(feats, mask):
    """Zero out a group of input columns, to see whether the head needs them."""
    if mask is None:
        return feats
    out = np.array(feats, copy=True)
    if mask == "no_family":
        out[..., 16:] = 0.0
    elif mask == "no_geometry":
        out[..., :16] = 0.0
    elif mask == "geometry_only":
        out[..., 16:] = 0.0
    return out


def learned_relation_scores(sample, k: int, infer, mask=None) -> np.ndarray:
    feats = mask_features(sample.rel_feat[:, :, :k], mask)   # (C, A, k, D)
    C, A, K, D = feats.shape
    if A == 0:
        return np.zeros(len(sample.candidate_ids))
    out = infer(feats.reshape(-1, D)).reshape(C, A, K).astype(np.float64)
    weight = sample.anchor_logp[:, :k]
    values = out + weight[None, :, :]
    flat = values.reshape(C, -1)
    top = flat.max(axis=1)
    return top + np.log(np.exp(flat - top[:, None]).sum(axis=1))


def gt_anchor_features(sample):
    """Relation features of every candidate against the anchor the sentence names.

    Computed here rather than stored, because it is only ever needed for the
    oracle rows.  Without it the ladder would compare a *learned* reasoner on a
    predicted anchor with a *rule* on the true one, and the difference between
    those two is as much the function as the anchor.
    """
    if sample.gt_row < 0 or sample.rel_feat.ndim != 4:
        return None
    from sensaturban_fpv.anchor_grounding import (relation_features,
                                                  relation_geometry)
    row = sample.gt_row
    if row >= len(sample.parsed.get("anchors", [])):
        return None
    anchor = sample.parsed["anchors"][row]
    relation = {"phrase": anchor["relation"], "family": anchor["family"],
                "frame": anchor["frame"]}
    nodes = sample.node_positions
    dims = sample.node_dimensions
    out = np.zeros((len(sample.candidate_ids), 1, 1, REL_DIM), dtype=np.float32)
    for i in range(len(sample.candidate_ids)):
        geom = relation_geometry(nodes[i], sample.gt_xyz, sample.uav_position,
                                 sample.uav_yaw)
        out[i, 0, 0] = relation_features(geom, relation, dims[i],
                                         sample.gt_anchor_dim)
    return out


def gt_rule_scores(sample):
    """The rule reasoner handed the true anchor, not the predicted hypotheses.

    ``rule_score`` was computed against the predicted anchors, so reading a
    column out of it is not the same statement.  The oracle row has to rebuild
    the geometry against the anchor the sentence names.
    """
    from sensaturban_fpv.anchor_grounding import (oriented_relation_score,
                                                  relation_geometry)
    if sample.gt_row < 0:
        return None
    anchors = sample.parsed.get("anchors", [])
    if sample.gt_row >= len(anchors):
        return None
    anchor = anchors[sample.gt_row]
    relation = {"phrase": anchor["relation"], "family": anchor["family"],
                "frame": anchor["frame"]}
    out = np.zeros(len(sample.candidate_ids))
    for i in range(len(sample.candidate_ids)):
        geom = relation_geometry(sample.node_positions[i], sample.gt_xyz,
                                 sample.uav_position, sample.uav_yaw)
        out[i] = oriented_relation_score(geom, relation)
    return out


def gt_relation_scores(sample, family_only: bool = False, k: int = 10):
    """ORACLE: score candidates against the anchor the instruction names.

    Returns ``None`` where the instruction does not name one entity
    unambiguously, which is most samples -- hence an evaluable subset rather
    than a method.
    """
    if sample.gt_row < 0 or sample.gt_row >= sample.anchor_ids.shape[0]:
        return None
    row = sample.gt_row
    kk = min(k, sample.rel_feat.shape[2]) if sample.rel_feat.ndim == 4 else 0
    if kk == 0:
        return None
    feats = sample.rel_feat[:, row, :kk, :]
    rules = sample.rule_score[:, row, :kk]
    if family_only:
        C = len(sample.candidate_ids)
        return np.full(C, rules.max(axis=1).mean())
    return rules.max(axis=1)


def target_entries(samples, scorer, keep=None):
    entries = []
    for sample in samples:
        if keep is not None and not keep(sample):
            continue
        scores = scorer(sample)
        if scores is None:
            continue
        entry = rank_metrics(scores, sample.target)
        entry["key"] = sample.key
        entry["group"] = sample.group
        entry["buckets"] = sorted(sample.buckets)
        entries.append(entry)
    return entries


def _fuse(sample, relation_scores, alpha, beta):
    """``alpha * visual + beta * relation`` on the referenced-phrase baseline."""
    if relation_scores is None or sample.phrase_text is None:
        return None
    vis = normalise_scores(sample.visual @ sample.phrase_text)
    return alpha * vis + beta * normalise_scores(relation_scores)


def visual_scores(sample) -> np.ndarray:
    text = np.asarray(sample.parsed.get("_phrase_embedding", None)
                      if isinstance(sample.parsed, dict) else None)
    if text is None:
        text = sample.parsed.get("_text")
    return sample.visual @ text if text is not None else np.zeros(len(sample.candidate_ids))


# --------------------------------------------------------------------------
# learned relation reasoner
# --------------------------------------------------------------------------

def build_relation_head(torch, dim=REL_DIM, hidden=HIDDEN):
    nn = torch.nn

    class RelationHead(nn.Module):
        """Small MLP over the relation geometry; ~150k parameters at 25 inputs."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, 1)
            nn.init.zeros_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)

        def forward(self, x):
            h = torch.nn.functional.gelu(self.fc1(self.norm(x)))
            h = torch.nn.functional.gelu(self.fc2(h))
            return self.fc3(h).squeeze(-1)

    return RelationHead()


def fit_relation_head(torch, samples, device="cpu", seed=0, epochs=EPOCHS):
    """Train the reasoner so the target outranks its candidates under a relation."""
    if not samples:
        return None, 0
    torch.manual_seed(seed)
    head = build_relation_head(torch).to(device)
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-4)

    rows = []
    for sample in samples:
        if sample.rel_feat.size == 0 or sample.rel_feat.shape[1] == 0:
            continue
        rows.append((torch.from_numpy(sample.rel_feat.reshape(-1, REL_DIM)),
                     torch.from_numpy(sample.anchor_logp.reshape(-1).astype(np.float32)),
                     sample.rel_feat.shape[1], sample.rel_feat.shape[2],
                     sample.target))
    if not rows:
        return None, 0
    for _ in range(epochs):
        order = np.random.permutation(len(rows))
        for start in range(0, len(order), BATCH):
            chunk = [rows[i] for i in order[start:start + BATCH]]
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for feats, logp, A, K, target in chunk:
                out = head(feats.to(device)).reshape(-1, A, K)
                values = out + logp.to(device)[None, :].reshape(1, A, K)
                flat = values.reshape(out.shape[0], -1)
                top = flat.max(dim=1, keepdim=True).values
                scores = top.squeeze(1) + torch.log(
                    torch.exp(flat - top).sum(dim=1))
                loss = loss + torch.nn.functional.cross_entropy(
                    scores[None, :], torch.tensor([target], device=device))
            loss = loss / len(chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    head.eval()

    def infer(array):
        with torch.no_grad():
            return head(torch.from_numpy(np.asarray(array, np.float32)).to(device)
                        ).cpu().numpy()
    return infer, sum(p.numel() for p in params)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data) if args.data else \
        artifact_dir(cfg) / "language_anchor_grounding" / "data"
    base = Path(args.base) if args.base else \
        artifact_dir(cfg) / "entity_grounding" / "data"
    out_dir = Path(args.out) if args.out else \
        artifact_dir(cfg) / "language_anchor_grounding"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qualitative_cases").mkdir(exist_ok=True)

    splits = load_anchors(data_dir / "anchors.jsonl", base)
    # The frozen phrase embedding is the visual arm's text side; it comes from
    # the entity round's stored text features so the baseline is bit-identical
    # with the 0.325 it is compared against.
    from run_gate_b import encode_texts, load_encoder
    processor, model, torch = load_encoder(cfg, args.device)
    phrases = sorted({s.instruction for sp in SPLITS for s in splits[sp]})
    phrase_feats = encode_texts(processor, model, torch, phrases, args.device)
    phrase_index = {t: i for i, t in enumerate(phrases)}
    for sp in SPLITS:
        for sample in splits[sp]:
            sample.parsed["_text"] = phrase_feats[
                phrase_index[sample.instruction]].detach().cpu().numpy().astype(
                    np.float32)
    print({k: len(v) for k, v in splits.items()}, flush=True)

    report = {"counts": {k: len(v) for k, v in splits.items()},
              "groups": {k: dict(Counter(s.group for s in v)) for k, v in splits.items()},
              "buckets": {k: dict(Counter(b for s in v for b in s.buckets))
                          for k, v in splits.items()},
              "parser": {}, "anchor": {}, "target": {}, "oracle": {},
              "rescue": {}, "controls": {}}

    # ---- parser statistics
    from sensaturban_fpv.anchor_parser import vocabulary_stats
    vocab = vocabulary_stats([s.instruction for sp in SPLITS for s in splits[sp]])
    report["parser"] = {
        "instructions_with_anchor": float(np.mean(
            [bool(s.parsed.get("anchors")) for sp in SPLITS for s in splits[sp]])),
        "relations_per_instruction": vocab["relations_per_instruction"],
        "relation_hits": vocab["relation_hits"],
        "anchor_type_distribution": vocab["anchor_type_distribution"],
        "evaluable_gt_anchor": sum(1 for sp in SPLITS for s in splits[sp]
                                   if s.gt_row >= 0),
        "evaluable_by_split": {sp: sum(1 for s in splits[sp] if s.gt_row >= 0)
                               for sp in SPLITS},
    }
    (out_dir / "parser_metrics.json").write_text(json.dumps(
        {"summary": report["parser"], "vocabulary": vocab},
        indent=2, default=float) + "\n")

    # ---- anchor grounding
    anchor_metrics = {}
    for arm in ANCHOR_ARMS:
        anchor_metrics[arm] = {
            sp: anchor_grounding_metrics(splits[sp], arm) for sp in SPLITS}
    anchor_metrics["shuffled_control"] = {
        sp: anchor_grounding_metrics(splits[sp], "A0_plus_A1", shuffle_seed=17)
        for sp in SPLITS}
    report["anchor"] = anchor_metrics
    (out_dir / "anchor_metrics.json").write_text(json.dumps(
        anchor_metrics, indent=2, default=float) + "\n")

    # ---- relation reasoner
    fitted = []
    for seed in args.seeds:
        infer, n_params = fit_relation_head(torch, splits["train_seen"], args.device,
                                            seed=seed)
        if infer is not None:
            fitted.append((infer, n_params, seed))
    relation_params = fitted[0][1] if fitted else 0

    def learned_scorer(k):
        def score(sample):
            outs = [learned_relation_scores(sample, k, infer) for infer, _, _ in fitted]
            return np.mean(outs, axis=0)
        return score

    # ---- hyper-parameters, on val_seen only
    k_records = []
    for k in K_CHOICES:
        entries = target_entries(splits["val_seen"], learned_scorer(k))
        k_records.append({"k": k, "val_top1": aggregate(entries)["top1"]})
        print(f"  K={k:<3d} learned relation val top1 {k_records[-1]['val_top1']:.3f}",
              flush=True)
    k_choice = max(k_records, key=lambda r: r["val_top1"] or -1)["k"]
    print(f"  chosen K = {k_choice}", flush=True)

    def blend(alpha, beta, gamma=0.0):
        def score(sample):
            rel = learned_scorer(k_choice)(sample)
            value = _fuse(sample, rel, alpha, beta)
            if value is None:
                return None
            if gamma:
                value = value + gamma * normalise_scores(
                    rule_relation_scores(sample, k_choice))
            return value
        return score

    weight_records = []
    for alpha, beta in ((1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (0.5, 1.0),
                        (0.0, 1.0)):
        entries = target_entries(splits["val_seen"], blend(alpha, beta))
        weight_records.append({"alpha": alpha, "beta": beta,
                               "val_top1": aggregate(entries)["top1"]})
    best_weights = max(weight_records, key=lambda r: r["val_top1"] or -1)
    print(f"  chosen weights alpha={best_weights['alpha']} beta={best_weights['beta']}",
          flush=True)

    # ---- every arm, on every split
    def visual_phrase(s):
        return s.visual @ s.phrase_text if s.phrase_text is not None else None

    def visual_full(s):
        return s.visual @ s.parsed["_text"]

    arms = {
        "TD_masked_phrase": visual_phrase,
        "TD_masked_full_instruction": visual_full,
        "anchor_lexical_only": lambda s: _anchor_only(s, "a0"),
        "anchor_semantic_only": lambda s: _anchor_only(s, "combined"),
        "relation_rule_only": lambda s: rule_relation_scores(s, k_choice),
        "relation_learned_only": learned_scorer(k_choice),
        "visual_plus_rule": lambda s: _fuse(s, rule_relation_scores(s, k_choice),
                                            1.0, 1.0),
        "visual_plus_learned": blend(best_weights["alpha"], best_weights["beta"]),
        "visual_plus_topk_marginalised": blend(best_weights["alpha"],
                                               best_weights["beta"]),
        "visual_plus_top1_anchor": lambda s: _top1_anchor_blend(
            s, best_weights["alpha"], best_weights["beta"]),
        "shuffled_anchor_control": lambda s: _shuffled_blend(
            s, best_weights["alpha"], best_weights["beta"]),
        "visual_phrase_only_no_relation": visual_phrase,
        # Ablations of the reasoner's own input.  If the arm survives with the
        # relation family removed, it was never reading the relation.
        "visual_plus_learned_no_relation_family": lambda s: _fuse(
            s, np.mean([learned_relation_scores(s, k_choice, infer, mask="no_family")
                        for infer, _, _ in fitted], axis=0),
            best_weights["alpha"], best_weights["beta"]) if fitted else None,
        "visual_plus_learned_no_geometry": lambda s: _fuse(
            s, np.mean([learned_relation_scores(s, k_choice, infer, mask="no_geometry")
                        for infer, _, _ in fitted], axis=0),
            best_weights["alpha"], best_weights["beta"]) if fitted else None,
        "relation_learned_only_no_family": lambda s: np.mean(
            [learned_relation_scores(s, k_choice, infer, mask="no_family")
             for infer, _, _ in fitted], axis=0) if fitted else None,
    }

    def evaluate(scorer, name, info=None):
        entry = {"metrics": {}, "per_group": {}, "per_bucket": {}}
        entries_by_split = {}
        for sp in SPLITS:
            entries = target_entries(splits[sp], scorer)
            entries_by_split[sp] = entries
            entry["metrics"][sp] = aggregate(entries)
            entry["per_group"][sp] = _grouped(entries, lambda e: [e["group"]])
            entry["per_bucket"][sp] = _grouped(entries, lambda e: e["buckets"])
        entry["entries"] = entries_by_split
        return entry

    report["target"] = {}
    stored = {}
    for name, scorer in arms.items():
        t0 = time.time()
        result = evaluate(scorer, name)
        stored[name] = result
        report["target"][name] = {
            "metrics": result["metrics"], "per_group": result["per_group"],
            "per_bucket": result["per_bucket"], "seconds": round(time.time() - t0, 1),
        }
        m = result["metrics"]["val_unseen"]
        print(f"  {name:32s} val {result['metrics']['val_seen']['top1']:.3f}  "
              f"unseen {m['top1']:.3f}", flush=True)

    # ---- oracle ladder
    def gt_learned(sample):
        """The learned reasoner, handed the true anchor instead of the guess."""
        feats = gt_anchor_features(sample)
        if feats is None or not fitted:
            return None
        outs = [infer(feats.reshape(-1, REL_DIM)).reshape(-1) for infer, _, _ in fitted]
        return np.mean(outs, axis=0)

    def o1(sample):
        return arms["visual_plus_learned"](sample)

    def o2(sample):
        rel = gt_learned(sample)
        if rel is None or sample.phrase_text is None:
            return None
        return _fuse(sample, rel, best_weights["alpha"], best_weights["beta"])

    def o3(sample):
        return arms["visual_plus_rule"](sample)

    def o4(sample):
        return gt_rule_scores(sample)

    def o5(sample):
        rel = gt_learned(sample)
        if rel is None or sample.phrase_text is None:
            return None
        return _fuse(sample, rel, best_weights["alpha"], best_weights["beta"])

    ladder = {
        "O0_visual_only": visual_phrase,
        "O1_predicted_anchor_predicted_relation": o1,
        "O2_gt_anchor_predicted_relation": o2,
        "O3_predicted_anchor_rule_relation": o3,
        "O4_gt_anchor_rule_relation": o4,
        "O5_visual_plus_gt_anchor_learned_relation": o5,
    }
    deployable = {"O0_visual_only", "O1_predicted_anchor_predicted_relation"}
    ladder_full = {"O0_visual_only": visual_phrase,
                   "O1_predicted_anchor_predicted_relation": o1}
    report["oracle"] = {}
    for name, scorer in ladder.items():
        # Every rung on the *same* subset, otherwise the ladder compares
        # different samples as well as different information.
        entry = {"deployable": name in deployable, "metrics": {},
                 "metrics_full_split": {}}
        for sp in SPLITS:
            entry["metrics"][sp] = aggregate(target_entries(
                splits[sp], scorer, lambda s: s.gt_row >= 0))
            if name in ladder_full:
                entry["metrics_full_split"][sp] = aggregate(target_entries(
                    splits[sp], ladder_full[name]))
        report["oracle"][name] = entry
        m = entry["metrics"]["val_unseen"]
        print(f"  ORACLE {name:38s} unseen n={m['n']} top1={m['top1']}",
              flush=True)
    (out_dir / "oracle_ladder.json").write_text(json.dumps(
        report["oracle"], indent=2, default=float) + "\n")

    # ---- rescue and controls
    base = {e["key"]: e for e in stored["TD_masked_phrase"]["entries"]["val_unseen"]}
    wrong = {k for k, e in base.items() if not e["top1"]}
    gap = {k for k, e in base.items() if not e["top1"] and e["top4"]}
    rescue = {"n_reference_wrong": len(wrong), "n_top4_not_top1": len(gap),
              "arms": {}}
    for name in ("visual_plus_learned", "visual_plus_rule", "relation_rule_only",
                 "relation_learned_only", "anchor_semantic_only"):
        entries = {e["key"]: e for e in stored[name]["entries"]["val_unseen"]}
        keys = sorted(gap & set(entries))
        rescued = [k for k in keys if entries[k]["top1"]]
        rescue["arms"][name] = {
            "n": len(keys),
            "top4_to_top1_rescued": len(rescued),
            "top4_to_top1_rescue_rate": (len(rescued) / len(keys)) if keys else None,
        }
    report["rescue"] = rescue

    a = {e["key"]: e["top1"] for e in stored["visual_plus_learned"]["entries"]["val_unseen"]}
    b = {e["key"]: e["top1"] for e in stored["shuffled_anchor_control"]["entries"]["val_unseen"]}
    c = {e["key"]: e["top1"] for e in stored["TD_masked_phrase"]["entries"]["val_unseen"]}
    keys = sorted(set(a) & set(b) & set(c))
    report["controls"] = {
        "fusion_vs_baseline": mcnemar([a[k] for k in keys], [c[k] for k in keys]),
        "real_anchor_vs_shuffled": mcnemar([a[k] for k in keys], [b[k] for k in keys]),
    }

    report["params"] = {"relation_head": relation_params,
                        "fusion_learned": 3, "siglip": 0}
    report["tuning"] = {"k_records": k_records, "k_choice": k_choice,
                        "weight_records": weight_records,
                        "weights": best_weights}
    (out_dir / "target_metrics.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "entries"}
         for k, v in report["target"].items()}, indent=2, default=float) + "\n")
    (out_dir / "per_type.json").write_text(json.dumps(
        {k: v["per_group"] for k, v in report["target"].items()},
        indent=2, default=float) + "\n")
    (out_dir / "per_relation.json").write_text(json.dumps(
        {k: v["per_bucket"] for k, v in report["target"].items()},
        indent=2, default=float) + "\n")
    (out_dir / "rescue_analysis.json").write_text(json.dumps(
        rescue, indent=2, default=float) + "\n")
    (out_dir / "relation_metrics.json").write_text(json.dumps(
        {"k_records": k_records, "weight_records": weight_records,
         "learned_relation_head_params": relation_params},
        indent=2, default=float) + "\n")

    _cases(out_dir, splits, stored)
    (out_dir / "metrics.json").write_text(json.dumps(
        {k: v for k, v in report.items() if k != "target"},
        indent=2, default=float) + "\n")
    print(f"\nwrote {out_dir}")


def _anchor_only(sample, which):
    ids = (sample.a0_ids if which == "a0" else sample.anchor_ids)
    scores = np.zeros(len(sample.candidate_ids))
    if ids.shape[0] == 0:
        return scores
    best = ids[:, 0]
    for i, cid in enumerate(sample.candidate_ids):
        scores[i] = 1.0 if (best == cid).any() else 0.0
    return scores


def _top1_anchor_blend(sample, alpha, beta):
    if sample.rule_score.size == 0:
        return None
    value = (sample.rule_score[:, :, 0] + sample.anchor_logp[:, 0][None, :]).max(axis=1)
    return _fuse(sample, value, alpha, beta)


def _shuffled_blend(sample, alpha, beta):
    """The control: the same reasoner, anchors drawn at random from the block."""
    if sample.rule_score.size == 0:
        return None
    rng = np.random.default_rng(abs(hash(sample.key)) % (2 ** 31))
    flat = sample.rule_score.reshape(len(sample.candidate_ids), -1)
    perm = rng.permutation(flat.shape[1])
    return _fuse(sample, flat[:, perm].max(axis=1), alpha, beta)


def _grouped(entries, key_fn):
    groups = defaultdict(list)
    for entry in entries:
        for key in key_fn(entry):
            groups[key].append(entry)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def _cases(out_dir, splits, stored, limit=12):
    base = {e["key"]: e for e in stored["TD_masked_phrase"]["entries"]["val_unseen"]}
    final = {e["key"]: e for e in stored["visual_plus_learned"]["entries"]["val_unseen"]}
    rule = {e["key"]: e for e in stored["relation_rule_only"]["entries"]["val_unseen"]}
    by_key = {s.key: s for s in splits["val_unseen"]}
    buckets = defaultdict(list)
    for key, entry in base.items():
        sample = by_key.get(key)
        if sample is None:
            continue
        record = {"key": key, "map": sample.map, "group": sample.group,
                  "instruction": sample.instruction,
                  "parsed_target": sample.parsed.get("target_phrase"),
                  "parsed_anchors": [{k: a[k] for k in ("phrase", "type", "relation",
                                                        "family")}
                                     for a in sample.parsed.get("anchors", [])],
                  "visual_rank": entry["rank"],
                  "final_rank": final.get(key, {}).get("rank"),
                  "rule_rank": rule.get(key, {}).get("rank"),
                  "gt_anchor_is_named": sample.gt_row >= 0}
        if entry["top4"] and not entry["top1"]:
            if final.get(key, {}).get("top1"):
                buckets["visual_wrong_anchor_correct"].append(record)
            else:
                buckets["visual_wrong_anchor_also_wrong"].append(record)
        elif entry["top1"] and final.get(key, {}).get("top1"):
            buckets["both_correct"].append(record)
        elif entry["top1"] and not final.get(key, {}).get("top1"):
            buckets["regression"].append(record)
    for name, items in buckets.items():
        (out_dir / "qualitative_cases" / f"{name}.json").write_text(
            json.dumps(items[:limit], indent=2, default=float) + "\n")
    print(f"  cases: {[(k, len(v)) for k, v in buckets.items()]}", flush=True)


if __name__ == "__main__":
    main()
