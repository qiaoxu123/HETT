#!/usr/bin/env python3
"""How does this corpus actually use ``behind``, ``beside``, ``north of``?

The previous round hard-coded a sign convention per relation family and the
oracle then showed it was wrong: handing the rule reasoner the *true* anchor made
it score worse, which can only happen if the function does not mean what the
corpus means.  This script replaces that assumption with a measurement.

For every relation phrase it asks one question: **relative to the anchor, along
which direction does the target sit?**  Six candidate directions are available
from the extractor -- the two global axes, the agent's heading and right-hand
axis, and the anchor's own footprint long and short axes -- and each is tested
with both signs.  The one that best separates the true target from same-class
distractors is the corpus's convention, and it is chosen on
``train_seen`` + ``val_seen`` only.

The separation is an AUC over paired comparisons: within one sample, does the
true target score above each same-class distractor?  Paired rather than pooled,
because pooling would let a per-sample offset in the anchor's position be read
as signal.

Two conditions are reported for each phrase, because they answer different
questions:

* **nearest** -- the anchor entity nearest the true target.  This isolates the
  *semantics* of the word from the disambiguation of which road segment is
  meant, and it is the one the frame is chosen on.
* **marginalised** -- a logsumexp over every entity carrying the anchor name.
  This is what a deployable reasoner actually has.

The second is reported for the record, not used for selection, and neither is
allowed to read ``val_unseen``.

Outputs ``artifacts/relation_v2/relation_distributions.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.anchor_parser import RELATIONS  # noqa: E402
from sensaturban_fpv.config import artifact_dir, load_config, load_landmarks  # noqa: E402
from sensaturban_fpv.entity_geometry import group_of  # noqa: E402

# The six directions, as columns of the 18-dim geometry vector written by
# run_relation_v2_data.py.
AXES = {
    "global_x": 0, "global_y": 1,
    "agent_ahead": 6, "agent_lateral": 7,
    "anchor_along": 10, "anchor_perp": 11,
}
DIST_COL, LOGDIST_COL = 3, 2
SELECT_SPLITS = ("train_seen", "val_seen")
# Low enough that the compass words are reported rather than silently dropped:
# the brief asks about north/south specifically, and "n = 17, no effect" is a
# more useful answer than an omission.  Anything under RELIABLE_PAIRS is flagged.
MIN_PAIRS = 15
RELIABLE_PAIRS = 40


def load_samples(path: Path) -> list:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def paired_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(true target scores above a same-class distractor), ties at 0.5."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def iterate_pairs(rows, splits, types_by_map, condition="nearest"):
    """Yield (split, relation phrase, family, positive, negatives) per sample."""
    for row in rows:
        if row["split"] not in splits:
            continue
        ids = row.get("annotation_anchor_ids") or []
        geom = np.asarray(row.get("annotation_rel_geom") or [], dtype=np.float32)
        if geom.ndim != 4 or geom.shape[1] == 0:
            continue
        # One relation per *annotation* phrase, matched by phrase.  The
        # parser's own list is a different length and pairing them by position
        # attributes one phrase's words to another phrase's geometry.
        relations = row.get("annotation_relations") or []
        t_idx = row["target_index"]
        cand_ids = row["candidate_ids"]
        types = types_by_map.get(row["map"], {})
        target_group = group_of(types.get(row["target_id"], ""))
        rel_idx = np.array([i for i, c in enumerate(cand_ids)
                            if group_of(types.get(c, "")) == target_group
                            and i != t_idx], dtype=np.int64)
        if rel_idx.size == 0:
            continue
        for a_i in range(min(geom.shape[1], len(relations))):
            relation = relations[a_i]
            if not relation:
                # The phrase stands in no relation to the target -- a
                # co-referenced feature like "the building with the Gym" -- so
                # there is no word to measure.
                continue
            phrase = relation["phrase"]
            valid = [k for k in range(geom.shape[2]) if ids[a_i][k] >= 0]
            if not valid:
                continue
            if condition == "nearest":
                # Isolate the word's semantics: pick the entity the sentence most
                # plausibly means, which for a multi-segment road is the one next
                # to the thing being described.
                dists = [np.abs(geom[t_idx, a_i, k, DIST_COL]) for k in valid]
                chosen = [valid[int(np.argmin(dists))]]
            else:
                chosen = valid
            pos_all, neg_all = [], []
            for k in chosen:
                # Positive and negative must be read against the same anchor
                # entity, so each anchor contributes its own comparison.
                for c in rel_idx:
                    pos_all.append(geom[t_idx, a_i, k, :])
                    neg_all.append(geom[c, a_i, k, :])
            if not pos_all:
                continue
            yield (row["split"], phrase, relation["family"],
                   np.stack(pos_all), np.stack(neg_all))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "relation_v2"
    data = Path(args.data) if args.data else out_dir / "relation_samples.jsonl"
    rows = load_samples(data)
    print(f"{len(rows)} samples", flush=True)

    objects_by_map = load_landmarks(cfg)
    types_by_map = {m: {int(o.id): o.object_type for o in objs.values()}
                    for m, objs in objects_by_map.items()}

    collected = defaultdict(list)
    for split, phrase, family, pos, neg in iterate_pairs(
            rows, SELECT_SPLITS + ("val_unseen",), types_by_map, "nearest"):
        collected[(split, phrase, family)].append((pos, neg))
    marginal = defaultdict(list)
    for split, phrase, family, pos, neg in iterate_pairs(
            rows, SELECT_SPLITS + ("val_unseen",), types_by_map, "marginalised"):
        marginal[(split, phrase, family)].append((pos, neg))

    # ---- per-phrase frame selection, on train_seen + val_seen
    phrases = sorted({(p, f) for (s, p, f) in collected})
    records = []
    for phrase, family in phrases:
        train = [x for s, p, f in collected if p == phrase and f == family
                 and s in SELECT_SPLITS for x in collected[(s, p, f)]]
        unseen = [x for (s, p, f) in collected if p == phrase and f == family
                  and s == "val_unseen" for x in collected[(s, p, f)]]
        if len(train) < MIN_PAIRS:
            records.append({"phrase": phrase, "family": family,
                            "n_train": len(train), "n_unseen": len(unseen),
                            "verdict": "too_few_samples"})
            continue
        best = None
        table = {}
        for axis, col in AXES.items():
            for sign in (1.0, -1.0):
                aucs = [paired_auc(sign * pos[:, col], sign * neg[:, col])
                        for pos, neg in train]
                aucs = [a for a in aucs if np.isfinite(a)]
                if not aucs:
                    continue
                value = float(np.mean(aucs))
                table[f"{axis}*{'+' if sign > 0 else '-'}"] = round(value, 4)
                if best is None or value > best[0]:
                    best = (value, axis, sign)
        if best is None:
            continue
        _, axis, sign = best
        unseen_auc = [paired_auc(sign * pos[:, AXES[axis]],
                                 sign * neg[:, AXES[axis]])
                      for pos, neg in unseen]
        unseen_auc = [a for a in unseen_auc if np.isfinite(a)]
        declared = RELATIONS.get(phrase)
        records.append({
            "phrase": phrase, "family": family,
            "n_train": len(train), "n_unseen": len(unseen),
            "best_axis": axis, "best_sign": int(sign),
            "auc_train": round(best[0], 4),
            "auc_unseen": round(float(np.mean(unseen_auc)), 4) if unseen_auc else None,
            "declared_frame": declared[1] if declared else None,
            "low_confidence": len(train) < RELIABLE_PAIRS,
            "table": table,
        })

    # ---- control: is the winning axis the relation, or a trajectory prior?
    #
    # The agent's heading points at the goal, so a candidate sitting "ahead" of
    # the anchor is more likely to be the answer under *every* relation word.
    # If the best axis is really that prior, then shuffling which relation each
    # sample is labelled with will leave the winning AUC where it was.
    observed = [r["auc_train"] for r in records if "best_axis" in r]
    observed_mean = float(np.mean(observed)) if observed else float("nan")

    flat = [(s, p, f, x) for (s, p, f), pool in collected.items()
            for x in pool if s in SELECT_SPLITS]
    rng = np.random.default_rng(20261007)
    shuffle_means = []
    for _ in range(5):
        labels = [f for _, _, f, _ in flat]
        rng.shuffle(labels)
        regrouped = defaultdict(list)
        for (_, phrase, _, sample), family in zip(flat, labels):
            regrouped[(phrase, family)].append(sample)
        values = []
        for phrase, family in phrases:
            pool = regrouped.get((phrase, family), [])
            if len(pool) < MIN_PAIRS:
                continue
            best = 0.0
            for axis, col in AXES.items():
                for sign in (1.0, -1.0):
                    aucs = [paired_auc(sign * p[:, col], sign * n[:, col])
                            for p, n in pool]
                    aucs = [a for a in aucs if np.isfinite(a)]
                    if aucs:
                        best = max(best, float(np.mean(aucs)))
            if best:
                values.append(best)
        if values:
            shuffle_means.append(float(np.mean(values)))

    # The prior on its own: how well does "the answer is ahead of the anchor"
    # do when the relation word is ignored entirely?
    ahead_aucs = []
    for (s, p, f), pool in collected.items():
        if s not in SELECT_SPLITS:
            continue
        for pos, neg in pool:
            value = paired_auc(pos[:, AXES["agent_ahead"]],
                               neg[:, AXES["agent_ahead"]])
            if np.isfinite(value):
                ahead_aucs.append(value)

    control = {
        "observed_best_axis_auc_mean": round(observed_mean, 4),
        "shuffled_relation_auc_mean": round(float(np.mean(shuffle_means)), 4),
        "shuffled_relation_auc_sd": round(float(np.std(shuffle_means)), 4),
        "agent_ahead_prior_auc": round(float(np.mean(ahead_aucs)), 4),
        "n_phrases_scored": len(observed),
        "reading": (
            "A winning axis that survives shuffling the relation label is a "
            "property of the sampling, not of the word."),
    }

    # ---- geometry distributions for the headline families
    distributions = {}
    for phrase, family in phrases:
        pool = [x for s, p, f in collected if p == phrase and f == family
                and s in SELECT_SPLITS for x in collected[(s, p, f)]]
        if len(pool) < MIN_PAIRS:
            continue
        pos = np.concatenate([p for p, _ in pool])
        neg = np.concatenate([n for _, n in pool])
        distributions[phrase] = {
            "family": family, "n": len(pool),
            "positive_xy_mean": [round(float(pos[:, 0].mean() * 100), 2),
                                 round(float(pos[:, 1].mean() * 100), 2)],
            "negative_xy_mean": [round(float(neg[:, 0].mean() * 100), 2),
                                 round(float(neg[:, 1].mean() * 100), 2)],
            "distance_m": {
                "positive_median": round(float(np.median(pos[:, DIST_COL]) * 200), 1),
                "negative_median": round(float(np.median(neg[:, DIST_COL]) * 200), 1)},
            "proximity_auc": round(float(np.mean(
                [paired_auc(-p[:, DIST_COL], -n[:, DIST_COL]) for p, n in pool])), 4),
        }

    # ---- marginalised condition, for the record
    marginal_summary = {}
    for phrase, family in phrases:
        pool = [x for s, p, f in marginal if p == phrase and f == family
                and s in SELECT_SPLITS for x in marginal[(s, p, f)]]
        if len(pool) < MIN_PAIRS:
            continue
        rec = next(r for r in records if r["phrase"] == phrase)
        if "best_axis" not in rec:
            continue
        col = AXES[rec["best_axis"]]
        sign = rec["best_sign"]
        aucs = [paired_auc(sign * p[:, col], sign * n[:, col]) for p, n in pool]
        aucs = [a for a in aucs if np.isfinite(a)]
        marginal_summary[phrase] = {
            "n": len(pool), "auc": round(float(np.mean(aucs)), 4), "axis": rec["best_axis"],
            "sign": sign}

    out = {"per_phrase_frame": records,
           "distributions": distributions,
           "marginalised_condition": marginal_summary,
           "control": control,
           "selection_splits": list(SELECT_SPLITS),
           "min_pairs": MIN_PAIRS, "reliable_pairs": RELIABLE_PAIRS}
    (out_dir / "relation_distributions.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n")

    print(f"{'phrase':16s} {'fam':11s} {'n':>4s} {'best frame':22s} "
          f"{'AUC':>6s} {'unseen':>7s} {'marg':>6s}")
    for rec in sorted(records, key=lambda r: -(r.get("auc_train") or 0)):
        if "best_axis" not in rec:
            print(f"{rec['phrase']:16s} {rec['family']:11s} {rec['n_train']:4d} "
                  f"{'-- too few --':22s}")
            continue
        marg = marginal_summary.get(rec["phrase"], {}).get("auc")
        print(f"{rec['phrase']:16s} {rec['family']:11s} {rec['n_train']:4d} "
              f"{rec['best_axis']+'*'+('+' if rec['best_sign']>0 else '-'):22s} "
              f"{rec['auc_train']:6.3f} {str(rec['auc_unseen']):>7s} "
              f"{marg if marg is None else round(marg,3)}")
    print(json.dumps(control, indent=1), flush=True)
    print(f"-> {out_dir / 'relation_distributions.json'}", flush=True)


if __name__ == "__main__":
    main()
