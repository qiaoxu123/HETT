#!/usr/bin/env python3
"""Fit the relation reasoner, then try to falsify it before using it.

The round's first deliverable is not a target-grounding number.  It is the
answer to one question: **given the correct anchor and the correct relation,
does the reasoner rank the target more highly than it does under a wrong
relation?**  Everything here is arranged so that question is answered before any
visual score is added, and so that a failure cannot hide inside a fusion.

Order of work:

1. fit the scorer on ``train_seen`` with two losses -- a listwise ranking loss
   over candidates, and a margin term putting the sentence's own relation above
   its opposite for the same target and anchor;
2. **the self-test**: the scorer's rank of the target under the correct
   relation, against the opposite, a shuffled relation, a zeroed relation, a
   shuffled anchor and no relation word at all;
3. **the gate**: five conditions, all of which must hold before fusion runs;
4. only if the gate passes, the visual fusion of section 16 onward.

Hyper-parameters are chosen on ``val_seen``.  ``val_unseen`` is read once, at the
end, and reports the choice made without it.

Everything is padded into ``(N, C, A, K, D)`` tensors and run in chunks.  The
per-sample loop this replaces spent more time in Python dispatch than in
arithmetic -- a minute per epoch, for a model with 12k parameters -- and a round
that costs an hour per iteration does not get iterated on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.config import artifact_dir, load_config  # noqa: E402
from sensaturban_fpv.relation_v2 import (  # noqa: E402
    GEOM_DIM, REL_DIM, aggregate, build_relation_vocab, mcnemar, opposite_id,
    rank_metrics, shuffled_ids,
)

SPLITS = ("train_seen", "val_seen", "val_unseen")
AMAX = 4          # anchor phrases carried per sample
KMAX = 24         # entities carried per phrase
CHUNK = 64

# The vocabulary id of the "no relation word" item.  Set from the vocabulary
# rather than assumed: ``build_relation_vocab`` sorts its phrases, so "unstated"
# does not land at any particular index.
UNSTATED = 0


class Sample:
    """One instruction: candidates, anchors, relations, and the answer."""

    __slots__ = ("split", "key", "map", "geom", "rel_ids", "rel_codes",
                 "anchor_ok", "target_index", "candidate_ids", "instruction")

    def __init__(self, row):
        self.split = row["split"]
        self.key = row["key"]
        self.map = row["map"]
        self.instruction = row["instruction"]
        self.target_index = int(row["target_index"])
        self.candidate_ids = [int(c) for c in row["candidate_ids"]]
        self.geom = np.asarray(row.get("annotation_rel_geom") or [],
                               dtype=np.float32)
        # One relation per *annotation* anchor phrase.  ``None`` means the
        # parser found no relation word for that phrase, which is carried as its
        # own vocabulary item rather than as a zero embedding: if the corpus's
        # relation words carry no signal, the model should be able to say so by
        # using "unstated" everywhere, and the gate then reads that off.
        self.rel_ids = np.array(
            [((r or {}).get("phrase") if r else "unstated")
             for r in (row.get("annotation_relations") or [])], dtype=object)
        ids = np.asarray(row.get("annotation_anchor_ids") or [], dtype=np.int64)
        self.anchor_ok = ids >= 0
        self.rel_codes = np.zeros(self.rel_ids.size, dtype=np.int64)


def load_samples(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def encode_relations(samples, vocab) -> None:
    for s in samples:
        s.rel_codes = np.array([vocab.get(p, 0) if p else 0 for p in s.rel_ids],
                               dtype=np.int64)


def usable(sample) -> bool:
    """Geometry and at least one anchor: enough to score, not enough to test."""
    return (sample.geom.ndim == 4 and sample.geom.shape[1] > 0
            and sample.geom.shape[2] > 0 and sample.anchor_ok.any())


def has_relation(sample) -> bool:
    """At least one anchor phrase the parser attached a relation word to.

    Only 54.7% of anchor phrases have one.  The rest are co-referenced features
    -- "the building with the Gym, Nike and The Food Warehouse" names landmarks
    standing in no spatial relation to the target -- and a relation self-test run
    over them would be measuring the geometry prior with no word to test.
    """
    return bool(np.any((sample.rel_codes > 0) & (sample.rel_codes != UNSTATED)))


# --------------------------------------------------------------------------
# padding
# --------------------------------------------------------------------------

class Tensors:
    """Every sample padded to one shape, plus the masks that undo the padding."""

    def __init__(self, samples):
        n = len(samples)
        c_max = max(len(s.candidate_ids) for s in samples)
        self.geom = np.zeros((n, c_max, AMAX, KMAX, GEOM_DIM), dtype=np.float32)
        self.valid = np.zeros((n, AMAX, KMAX), dtype=bool)
        self.rel = np.zeros((n, AMAX), dtype=np.int64)
        self.target = np.zeros(n, dtype=np.int64)
        self.keys = [s.key for s in samples]
        self.splits = [s.split for s in samples]
        for i, s in enumerate(samples):
            c, a, k, d = s.geom.shape
            a, k = min(a, AMAX), min(k, KMAX)
            self.geom[i, :c, :a, :k, :d] = s.geom[:c, :a, :k, :d]
            self.valid[i, :a, :k] = s.anchor_ok[:a, :k]
            self.rel[i, :a] = s.rel_codes[:a]
            self.target[i] = s.target_index
        self.c_max = c_max


def attach_counterfactuals(tensors, vocab, inv, margin_seed=7):
    """The (target, anchor, correct word, opposite word) quadruples.

    The anchor slot is chosen by position -- the first phrase carrying a real
    relation word -- rather than by score or by proximity to the answer, so that
    the counterfactual term cannot see what it is being tested on.
    """
    n = tensors.geom.shape[0]
    a_idx = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    pos = np.zeros(n, dtype=np.int64)
    neg = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(margin_seed)
    for i in range(n):
        for a in range(tensors.rel.shape[1]):
            code = int(tensors.rel[i, a])
            if code > 0 and code != UNSTATED:
                a_idx[i], pos[i] = a, code
                other = opposite_id(code, vocab, inv, rng)
                if other > 0 and other != code:
                    neg[i], ok[i] = other, True
                break
    tensors.cf_a, tensors.cf_ok = a_idx, ok
    tensors.cf_pos, tensors.cf_neg = pos, neg
    return tensors


def to_batch(tensors, device, torch):
    """The padded tensors as device tensors, with the counterfactual picks."""
    n = tensors.geom.shape[0]
    batch = {
        "geom": torch.as_tensor(tensors.geom, dtype=torch.float32, device=device),
        "valid": torch.as_tensor(tensors.valid, dtype=torch.bool, device=device),
        "rel": torch.as_tensor(tensors.rel, dtype=torch.long, device=device),
        "target": torch.as_tensor(tensors.target, dtype=torch.long, device=device),
        "keys": tensors.keys, "splits": tensors.splits,
    }
    if hasattr(tensors, "cf_ok"):
        rows = torch.arange(n, device=device)
        batch["cf_geom"] = batch["geom"][rows, batch["target"],
                                        torch.as_tensor(tensors.cf_a,
                                                        device=device), 0]
        batch["cf_ok"] = torch.as_tensor(tensors.cf_ok, dtype=torch.bool,
                                         device=device)
        batch["cf_pos"] = torch.as_tensor(tensors.cf_pos, dtype=torch.long,
                                          device=device)
        batch["cf_neg"] = torch.as_tensor(tensors.cf_neg, dtype=torch.long,
                                          device=device)
    return batch


def build_model(torch, geom_dim=GEOM_DIM, rel_vocab=64, rel_dim=REL_DIM,
                hidden=96):
    nn = torch.nn

    class RelationScorer(nn.Module):
        """A relation embedding concatenated with raw geometry, then an MLP.

        Small on purpose.  The audit found the relation signal in this corpus is
        barely above a trajectory prior, so capacity is not the constraint and a
        larger model would only fit the prior harder.
        """

        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(rel_vocab + 1, rel_dim)
            self.mlp = nn.Sequential(
                nn.Linear(geom_dim + rel_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, geom, rel):
            return self.mlp(torch.cat([geom, self.embed(rel.clamp(min=0))],
                                      dim=-1)).squeeze(-1)

    return RelationScorer()


# --------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------

def candidate_scores(torch, model, batch, rel_override=None, donor=None,
                     chunk=CHUNK):
    """``logsumexp`` over anchor hypotheses, per candidate, for a whole batch.

    Marginalised rather than committed: 41.6% of anchor phrases name more than
    one entity, so a hard argmax over the hypotheses would discard exactly the
    disambiguation the reference audit found to be the binding problem.
    """
    geom, valid = batch["geom"], batch["valid"]
    rel = batch["rel"] if rel_override is None else rel_override
    if donor is not None:
        # Anchor shuffling takes each sample's anchor *phrases* from another
        # sample -- geometry, validity and relation together -- while leaving
        # its own candidates alone.  Three earlier versions of this control
        # were no-ops and each looked like a passing test:
        #   * permuting the anchor slots within a sample only reorders the
        #     terms of the logsumexp that marginalises them, which is invariant;
        #   * permuting the batch axis moves the candidates with the anchors,
        #     so every sample is simply relabelled and the aggregates are
        #     preserved to the last decimal.
        # The anchors live on dim 2, so that is the axis a donor index has to
        # address.
        # donor indexes *samples*, and there are only ever AMAX anchor slots,
        # so the swap is written as one assignment per slot rather than as a
        # gather over a sample-valued index.
        g2, v2, r2 = torch.empty_like(geom), torch.empty_like(valid), \
            torch.empty_like(rel)
        for slot in range(geom.shape[2]):
            # The anchor's geometry is the same for every candidate of a
            # sample, so the donor's slot is read from candidate row 0 and
            # broadcast; reading `geom[donor, :, slot]` instead would carry the
            # donor's *candidates* across as well and turn the whole arm into a
            # relabelling of samples, which preserves every aggregate exactly
            # and reads as a control that passes.
            g2[:, :, slot] = geom[donor, 0, slot][:, None, :, :]
            v2[:, slot] = valid[donor, slot]
            r2[:, slot] = rel[donor, slot]
        geom, valid, rel = g2, v2, r2
    n, c, a, k, d = geom.shape
    out = torch.empty((n, c), dtype=torch.float32, device=geom.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        g, v = geom[start:stop], valid[start:stop]
        # The padding is sized for the widest sample, but the mean anchor set is
        # 2.8 entities, so slicing each chunk to its own width is most of the
        # arithmetic.  The training order is sorted by width to make the slices
        # tight rather than ragged.
        live = v.any(dim=1)
        k_use = int(live.sum(dim=0).nonzero().max().item()) + 1 if bool(live.any()) else 1
        g = g[:, :, :, :k_use]
        v = v[:, :, :k_use]
        m = g.shape[0]
        r = rel[start:stop][:, None, :, None].expand(m, g.shape[1], g.shape[2],
                                                     g.shape[3])
        logits = model(g.reshape(-1, d), r.reshape(-1))
        logits = logits.reshape(m, g.shape[1], g.shape[2], g.shape[3])
        logits = logits.masked_fill(~v[:, None, :, :], -1e9)
        flat = logits.reshape(m, g.shape[1], -1)
        top = flat.max(dim=-1, keepdim=True).values
        out[start:stop] = (top + torch.logsumexp(flat - top, dim=-1,
                                                 keepdim=True)).squeeze(-1)
    return out


def evaluate(torch, model, batch, **kwargs):
    """Per-sample rank of the target under one arm."""
    with torch.no_grad():
        scores = candidate_scores(torch, model, batch, **kwargs).cpu().numpy()
    target = batch["target"].cpu().numpy()
    rows = []
    for i in range(scores.shape[0]):
        row = rank_metrics(scores[i], int(target[i]))
        row["key"] = batch["keys"][i]
        row["split"] = batch["splits"][i]
        rows.append(row)
    return rows


def train(torch, model, batch, epochs, lr, margin, cf_weight, seed, chunk=CHUNK):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    geom, valid, rel, target = (batch["geom"], batch["valid"], batch["rel"],
                                batch["target"])
    n = geom.shape[0]
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        widths = (~batch["valid"]).all(dim=2).sum(dim=1).cpu().numpy()
        order_list = []
        for width in np.unique(widths):
            group = np.flatnonzero(widths == width)
            rng.shuffle(group)
            order_list.append(group)
        order = torch.as_tensor(np.concatenate(order_list))
        total, total_cf, steps = 0.0, 0.0, 0
        for start in range(0, n, chunk):
            idx = order[start:start + chunk]
            sub = {"geom": geom[idx], "valid": valid[idx], "rel": rel[idx],
                   "target": target[idx]}
            scores = candidate_scores(torch, model, sub)
            loss = torch.nn.functional.cross_entropy(scores, sub["target"])
            if cf_weight > 0 and "cf_geom" in batch:
                keep = batch["cf_ok"][idx]
                if bool(keep.any()):
                    g = batch["cf_geom"][idx][keep]
                    pos = model(g, batch["cf_pos"][idx][keep])
                    neg = model(g, batch["cf_neg"][idx][keep])
                    # The same target, the same anchor, two words: the one the
                    # sentence used has to come out on top.  Without this term
                    # the listwise loss is free to ignore the word entirely and
                    # fit the trajectory prior instead, which is what the
                    # previous round's reasoner did.
                    cf_loss = torch.clamp(margin - pos + neg, min=0.0).mean()
                    loss = loss + cf_weight * cf_loss
                    total_cf += float(cf_loss.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            steps += 1
        history.append({"epoch": epoch, "loss": total / max(steps, 1),
                        "cf": total_cf / max(steps, 1)})
        print(f"    epoch {epoch} loss {history[-1]['loss']:.4f} "
              f"cf {history[-1]['cf']:.4f}", flush=True)
    return history


def main() -> None:
    global UNSTATED
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tune-epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "relation_v2"
    data = Path(args.data) if args.data else out_dir / "relation_samples.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    samples = [Sample(r) for r in load_samples(data)]
    vocab, inv = build_relation_vocab(list(RELATION_PHRASES))
    UNSTATED = vocab["unstated"]
    encode_relations(samples, vocab)
    by_split = {s: [x for x in samples if x.split == s] for s in SPLITS}
    print(f"{len(samples)} samples; usable "
          f"{ {s: sum(usable(x) for x in v) for s, v in by_split.items()} }; "
          f"with a relation word "
          f"{ {s: sum(has_relation(x) for x in v) for s, v in by_split.items()} }",
          flush=True)

    # Train on everything with geometry and an anchor: the listwise loss is over
    # candidates, and a sample whose anchor carries no relation word still
    # teaches what an anchor's position is worth.  The self-test then runs only
    # where there *is* a word, which is where it can fail.
    train_t = Tensors([s for s in by_split["train_seen"] if usable(s)])
    eval_t = {split: Tensors([s for s in by_split[split]
                              if usable(s) and has_relation(s)])
              for split in SPLITS}
    attach_counterfactuals(train_t, vocab, inv)
    for split in SPLITS:
        attach_counterfactuals(eval_t[split], vocab, inv)

    # One fixed relabelling, reused by the shuffled-relation arm, so the control
    # is a property of the data rather than of a draw.
    flat = train_t.rel.reshape(-1)
    live = (flat > 0) & (flat != UNSTATED)
    drawn = shuffled_ids(flat[live], np.random.default_rng(12345))
    shuffle_map = {int(a): int(b) for a, b in zip(flat[live], drawn)}

    def arm_kwargs(name, tensors):
        rel = tensors.rel
        if name == "correct":
            return {}
        if name == "opposite":
            mapped = np.vectorize(lambda v: opposite_id(int(v), vocab, inv),
                                  otypes=[np.int64])(rel)
            return {"rel_override": torch.as_tensor(
                mapped, dtype=torch.long, device=device)}
        if name == "shuffled_relation":
            mapped = np.vectorize(lambda v: shuffle_map.get(int(v), int(v)),
                                  otypes=[np.int64])(rel)
            return {"rel_override": torch.as_tensor(
                mapped, dtype=torch.long, device=device)}
        if name == "no_relation_word":
            mapped = np.where((rel > 0) & (rel != UNSTATED), UNSTATED, 0)
            return {"rel_override": torch.as_tensor(
                mapped, dtype=torch.long, device=device)}
        if name == "zero_relation":
            return {"rel_override": torch.zeros(
                tensors.geom.shape[0], tensors.rel.shape[1], dtype=torch.long,
                device=device)}
        if name == "shuffled_anchor":
            rng = np.random.default_rng(args.seed)
            n = tensors.geom.shape[0]
            donor = rng.permutation(n)
            # A permutation can leave a sample pointing at itself; forcing a
            # derangement keeps every sample genuinely given someone else's
            # anchors.  A self-donor would quietly re-measure the correct arm.
            for i in np.flatnonzero(donor == np.arange(n)):
                j = (i + 1) % n
                donor[i], donor[j] = donor[j], donor[i]
            return {"donor": torch.as_tensor(donor, dtype=torch.long,
                                             device=device)}
        return {}

    # ---- hyper-parameters on val_seen only
    tuning, best = [], None
    for lr in (3e-3, 1e-2):
        for cf_weight in (0.0, 1.0, 3.0):
            if lr == 1e-2 and cf_weight in (0.0, 3.0):
                continue
            torch.manual_seed(args.seed)
            model = build_model(torch, rel_vocab=len(vocab)).to(device)
            print(f"  lr={lr} cf={cf_weight}", flush=True)
            train(torch, model, to_batch(train_t, device, torch),
                  args.tune_epochs, lr, 0.5, cf_weight, args.seed)
            rows = evaluate(torch, model, to_batch(eval_t["val_seen"], device,
                                                   torch),
                            **arm_kwargs("correct", eval_t["val_seen"]))
            top1 = aggregate(rows)["top1"]
            tuning.append({"lr": lr, "cf_weight": cf_weight, "val_top1": top1})
            print(f"  -> lr={lr} cf={cf_weight} val_seen top1 {top1:.4f}",
                  flush=True)
            if best is None or top1 > best[0]:
                best = (top1, lr, cf_weight)
    _, lr, cf_weight = best
    print(f"  chosen lr={lr} cf_weight={cf_weight}", flush=True)

    # ---- final fit
    torch.manual_seed(args.seed)
    model = build_model(torch, rel_vocab=len(vocab)).to(device)
    history = train(torch, model, to_batch(train_t, device, torch), args.epochs,
                    lr, 0.5, cf_weight, args.seed)
    params = int(sum(p.numel() for p in model.parameters()))

    # ---- the self-test
    arms = ("correct", "opposite", "shuffled_relation", "no_relation_word",
            "zero_relation", "shuffled_anchor")
    per_arm, self_test = {}, {}
    for arm in arms:
        per_arm[arm] = {}
        for split in SPLITS:
            per_arm[arm][split] = evaluate(
                torch, model, to_batch(eval_t[split], device, torch),
                **arm_kwargs(arm, eval_t[split]))
        self_test[arm] = {split: aggregate(per_arm[arm][split])
                          for split in SPLITS}

    # A control that silently no-ops reads as a pass, so prove it moved the
    # scores before reporting what it did.
    probe = to_batch(eval_t["val_unseen"], device, torch)
    with torch.no_grad():
        base_scores = candidate_scores(torch, model, probe)
        donor_scores = candidate_scores(
            torch, model, probe,
            **arm_kwargs("shuffled_anchor", eval_t["val_unseen"]))
    moved = float((base_scores - donor_scores).abs().max())
    print(f"  shuffled-anchor control moves the scores by {moved:.4f}", flush=True)
    if moved < 1e-6:
        raise SystemExit("the shuffled-anchor control is a no-op; refusing to "
                         "report it as a passing control")

    print(f"\n{'arm':20s} {'val_seen':>10s} {'unseen':>9s} {'unseen mrr':>11s} "
          f"{'unseen top4':>12s}")
    for arm in arms:
        print(f"{arm:20s} {self_test[arm]['val_seen']['top1']:10.4f} "
              f"{self_test[arm]['val_unseen']['top1']:9.4f} "
              f"{self_test[arm]['val_unseen']['mrr']:11.4f} "
              f"{self_test[arm]['val_unseen']['top4']:12.4f}", flush=True)

    # ---- the gate
    unseen = "val_unseen"
    correct = per_arm["correct"][unseen]
    gate = {"correct_top1": self_test["correct"][unseen]["top1"]}
    for arm in arms[1:]:
        a = np.array([r["top1"] for r in correct], dtype=bool)
        b_map = {r["key"]: r["top1"] for r in per_arm[arm][unseen]}
        b = np.array([bool(b_map.get(r["key"], 0)) for r in correct], dtype=bool)
        gate[f"correct_vs_{arm}"] = mcnemar(a, b)
        gate[f"{arm}_top1"] = self_test[arm][unseen]["top1"]

    conditions = {
        "C1_correct_beats_shuffled_relation": (
            gate["correct_vs_shuffled_relation"]["p_value"] < 0.05
            and gate["correct_top1"] > gate["shuffled_relation_top1"]),
        "C2_correct_beats_opposite_relation": (
            gate["correct_vs_opposite"]["p_value"] < 0.05
            and gate["correct_top1"] > gate["opposite_top1"]),
        "C3_dropping_the_relation_hurts": (
            gate["correct_top1"] - max(gate["no_relation_word_top1"],
                                       gate["zero_relation_top1"]) > 0.01),
        "C4_shuffling_the_anchor_hurts": (
            gate["correct_vs_shuffled_anchor"]["p_value"] < 0.05
            and gate["correct_top1"] > gate["shuffled_anchor_top1"]),
        "C5_the_reasoner_ranks_the_target_above_chance": (
            gate["correct_top1"] > 0.10),
    }
    gate["conditions"] = conditions
    gate["passed"] = all(conditions.values())

    report = {
        "params": {"relation_scorer": params, "siglip": 0, "fusion": 0},
        "vocab_size": len(vocab), "tuning": tuning,
        "chosen": {"lr": lr, "cf_weight": cf_weight, "epochs": args.epochs,
                   "margin": 0.5},
        "history": history, "self_test": self_test, "gate": gate,
        "n_usable": {s: int(sum(usable(x) for x in by_split[s])) for s in SPLITS},
        "n_with_relation": {s: int(sum(has_relation(x) for x in by_split[s]))
                            for s in SPLITS},
    }
    (out_dir / "relation_self_test.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps(conditions, indent=1), flush=True)
    print(f"GATE {'PASS' if gate['passed'] else 'FAIL'}", flush=True)
    print(f"-> {out_dir / 'relation_self_test.json'}", flush=True)


RELATION_PHRASES = [
    # The parser found no relation word for this anchor, so the anchor still
    # supplies geometry and the model gets to decide what that is worth.
    "unstated",
    "north of", "south of", "east of", "west of", "left of", "right of",
    "to the left", "to the right", "behind", "in front of", "ahead of",
    "near", "far from", "beside", "next to", "close to", "adjacent to",
    "alongside", "by the", "away from", "on", "above", "below", "under",
    "on top of", "overlooking", "between", "across from", "facing",
    "opposite", "surrounding", "bordering",
]

if __name__ == "__main__":
    main()
