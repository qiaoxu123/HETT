#!/usr/bin/env python3
"""Which kind of evidence closes the Top-4 to Top-1 gap?

The entity round left a specific shape: Top-4 around 0.76, Top-1 around 0.32, and
a joint encoding of the two views worth nothing over one masked view.  So the
model generally knows which candidates are plausible and cannot say which one it
is.  This script hands a deliberately small probe one family of evidence at a
time -- appearance, geometry, spatial relation, semantic map -- and measures what
each buys over the frozen masked visual feature.

Two things are kept strictly apart.  A **deployable** arm may use the
instruction, the agent's pose, the candidate annotations and the point cloud,
all of which exist at test time.  An **ORACLE** arm uses the target's own
annotation to pick something (an anchor entity, an entity identity) and is an
upper bound on what perfect parsing would buy, never a method.  Oracle rows are
labelled in every table and never averaged into a deployable result.

Sample reuse is verbatim: the same instruction, target, candidate ids in the
same order, same split, episode and step as the entity round, so every number
here is comparable with its 0.320.
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
HIDDEN = 128
EPOCHS = 40
BATCH = 16
SEEDS = (0, 1, 2)
TEXT_DIM = 1152
ID_EMBED = 32

# arm -> (block names, uses instruction conditioning, uses entity identity)
ARMS = {
    "Visual": (("visual",), False, False),
    "Visual_tight": (("visual_tight",), False, False),
    "O_visual": (("visual_oblique",), False, False),
    "Appearance": (("appearance",), False, False),
    "Geometry": (("geometry",), False, False),
    "Relation": (("relation",), False, False),
    "Semantic": (("semantic",), False, False),
    "Relation_flags": (("relation_flags",), False, False),
    "Visual+Appearance": (("visual", "appearance"), False, False),
    "Visual+Geometry": (("visual", "geometry"), False, False),
    "Visual+Relation": (("visual", "relation"), False, False),
    "Visual+Relation_noPose": (("visual", "relation[7:]"), False, False),
    "Relation_noPose": (("relation[7:]",), False, False),
    "Pose_only": (("relation[:7]",), False, False),
    "Visual+Semantic": (("visual", "semantic"), False, False),
    "Full_deployable": (("visual", "visual_oblique", "appearance", "geometry",
                         "relation", "semantic"), False, False),
    "Text:name": (("visual", "relation"), True, False),
    "Text:phrase": (("visual", "relation"), True, False),
    "Text:full": (("visual", "relation"), True, False),
    "Relation_ORACLE": (("anchor_relation",), False, False),
    "Visual+Relation_ORACLE": (("visual", "anchor_relation"), False, False),
    "EntityID_ORACLE": (("visual",), False, True),
}
ORACLE_ARMS = ("Relation_ORACLE", "Visual+Relation_ORACLE", "EntityID_ORACLE")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class Sample:
    __slots__ = ("split", "map", "key", "blocks", "types", "entity_ids", "target",
                 "buckets", "texts", "group", "instruction")


def group_of(object_type: str) -> str:
    if object_type == "Building":
        return "building"
    if object_type in ("Car", "Bike"):
        return "vehicle"
    return "other"


def permute_candidates(sample, rng):
    """Randomise the candidate order, target included.

    The candidate list is built as ``[referenced] + distractors``, so the answer
    sits at index 0 in every sample.  A probe cannot read an index off
    permutation-invariant features, but anything in a block that *does* correlate
    with list position would be learned as the answer, so the order is shuffled
    once, deterministically, before any arm sees it.  This changes nothing about
    which candidates are present.
    """
    n = len(sample.entity_ids)
    order = rng.permutation(n)
    sample.target = int(np.flatnonzero(order == sample.target)[0])
    sample.entity_ids = [sample.entity_ids[i] for i in order]
    sample.types = [sample.types[i] for i in order]
    for name, value in list(sample.blocks.items()):
        sample.blocks[name] = np.asarray(value)[order]


def load_splits(data_dir: Path, base_dir: Path, text_variant: str = "phrase",
                seed: int = 12345):
    records = {}
    for path in sorted(data_dir.glob("features.shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records[(row["split"], row["episode_index"], row["step"])] = row
    out = {s: [] for s in SPLITS}
    for (split, episode, step), row in sorted(records.items()):
        flags = np.asarray(row["is_target"], dtype=bool)
        if not flags.any():
            continue
        archive = np.load(base_dir / row["key"], allow_pickle=True)
        sample = Sample()
        sample.split, sample.map, sample.key = split, row["map"], row["key"]
        sample.types = list(row["entity_types"])
        sample.entity_ids = [int(v) for v in row["entity_ids"]]
        sample.target = int(np.flatnonzero(flags)[0])
        sample.buckets = set(row["buckets"])
        sample.instruction = row["instruction"]
        sample.group = group_of(sample.types[sample.target])
        sample.blocks = {
            "appearance": np.asarray(row["appearance"], np.float32),
            "geometry": np.asarray(row["geometry"], np.float32),
            "relation": np.asarray(row["relation"], np.float32),
            "semantic": np.asarray(row["semantic"], np.float32),
            "anchor_relation": np.asarray(row["anchor_relation"], np.float32),
            "visual": archive["td_context_masked"].astype(np.float32),
            "visual_tight": archive["td_tight_masked"].astype(np.float32),
            "visual_oblique": archive["oblique_context_masked"].astype(np.float32),
        }
        flags_relation = np.asarray(row["relation_flags"], np.float32)
        sample.blocks["relation_flags"] = np.repeat(
            flags_relation[None, :], len(sample.entity_ids), axis=0)
        sample.texts = {"full": np.asarray(row["text_full"], np.float32)}
        for variant in ("phrase", "name"):
            sample.texts[variant] = (np.asarray(row[f"text_{variant}"], np.float32)
                                     if row.get(f"text_{variant}") else None)
        out[split].append(sample)
    rng = np.random.default_rng(seed)
    for split in SPLITS:
        for sample in out[split]:
            permute_candidates(sample, rng)
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def rank_metrics(scores, target):
    order = np.argsort(-scores)
    rank = int(np.where(order == target)[0][0]) + 1
    others = np.delete(scores, target)
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


def aggregate(entries) -> dict:
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


def mcnemar(a, b) -> dict:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return {"a_only": n10, "b_only": n01, "n": 0, "p_value": 1.0}
    k = min(n10, n01)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"a_only": n10, "b_only": n01, "n": n,
            "p_value": float(min(2 * tail, 1.0))}


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def build_probe(torch, dim_in, use_text=False, use_ids=False, id_vocab=1,
                hidden=HIDDEN):
    nn = torch.nn

    class Probe(nn.Module):
        """Two-layer ranking head, identical shape for every arm.

        The comparison is between *features*, so a bigger head for one family
        would be measuring the head.  The optional instruction conditioning is a
        FiLM scale initialised to zero, so an unconditioned arm and a conditioned
        one start from the same function.
        """

        def __init__(self):
            super().__init__()
            # ``hidden=0`` gives a single linear layer: the minimum-capacity
            # ranking probe, and the only one whose visual arm is comparable
            # with the frozen cosine it is measured against.  The two-layer
            # version overfits 1175 samples badly enough to move the visual arm
            # from 0.32 to 0.16, which would flatter every feature family.
            self.hidden = hidden
            if hidden:
                self.fc1 = nn.Linear(dim_in, hidden)
                self.fc2 = nn.Linear(hidden, 1)
            else:
                self.fc1 = nn.Linear(dim_in, 1)
            self.film = nn.Linear(TEXT_DIM, hidden) if (use_text and hidden) else None
            if self.film is not None:
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)
            self.emb = nn.Embedding(id_vocab + 1, ID_EMBED) if use_ids else None
            self.id_fc = nn.Linear(ID_EMBED, hidden or 1) if use_ids else None
            if self.emb is not None:
                nn.init.zeros_(self.emb.weight)
                nn.init.zeros_(self.id_fc.weight)
                nn.init.zeros_(self.id_fc.bias)

        def forward(self, x, text=None, ids=None):
            if not self.hidden:
                return self.fc1(x).squeeze(-1)
            h = self.fc1(x)
            if self.film is not None and text is not None:
                h = h * (1.0 + torch.tanh(self.film(text)))
            if self.emb is not None:
                h = h + self.id_fc(self.emb(ids))
            return self.fc2(torch.nn.functional.gelu(h)).squeeze(-1)

    return Probe()


def split_block(name: str):
    """``"relation[7:]"`` -> ``("relation", slice(7, None))``."""
    if "[" not in name:
        return name, None
    base, spec = name[:-1].split("[")
    lo, hi = spec.split(":")
    return base, slice(int(lo) if lo else None, int(hi) if hi else None)


def _slice(value, selector):
    if selector is None:
        return value
    arr = np.asarray(value)
    if arr.ndim == 1:
        return arr[selector]
    return arr[:, selector]


def fit_scaler(samples, arm):
    """Mean and standard deviation of each block, from ``train_seen`` only."""
    names = ARMS[arm][0]
    scaler = {}
    for name in names:
        base, selector = split_block(name)
        stacked = np.concatenate([_slice(np.asarray(s.blocks[base], np.float64),
                                         selector) for s in samples], axis=0)
        mean = stacked.mean(axis=0)
        std = np.where(stacked.std(axis=0) < 1e-6, 1.0, stacked.std(axis=0))
        scaler[name] = (mean.astype(np.float64), std.astype(np.float64))
    return scaler


def build_rows(samples, arm, scaler, id_map, variant="phrase"):
    """One row per sample: the candidate matrix, the text row, the id row."""
    names, use_text, use_ids = ARMS[arm]
    rows = []
    for sample in samples:
        cols = []
        for name in names:
            base, selector = split_block(name)
            values = _slice(np.asarray(sample.blocks[base], np.float64), selector)
            cols.append(((values - scaler[name][0]) / scaler[name][1]).astype(np.float32))
        text = None
        if use_text:
            text = sample.texts.get(variant)
            if text is None:
                return None
        ids = None
        if use_ids:
            ids = np.array([id_map.get((sample.map, eid), 0)
                            for eid in sample.entity_ids], dtype=np.int64)
        rows.append({"x": np.concatenate(cols, axis=1), "text": text, "ids": ids,
                     "target": sample.target, "sample": sample})
    return rows


def run_arm(torch, arm, rows_by_split, scaler, id_map, seed, device,
            variant="phrase", hidden=HIDDEN):
    names, use_text, use_ids = ARMS[arm]
    train = build_rows(rows_by_split["train_seen"], arm, scaler, id_map, variant)
    if train is None:
        return None
    dim_in = train[0]["x"].shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe = build_probe(torch, dim_in, use_text, use_ids,
                        max(len(id_map), 1), hidden=hidden).to(device)
    params = list(probe.parameters())
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)

    def tensors(rows):
        """Flatten the batch: one row per candidate, with its sample's segment.

        The candidates of one sample are ranked against each other, so the loss
        is per segment; flattening lets the whole batch go through the probe in
        one call, which is what makes 19 arms over three seeds affordable.
        """
        x = torch.from_numpy(np.concatenate([r["x"] for r in rows])).to(device)
        text = None
        if use_text:
            text = torch.from_numpy(np.concatenate(
                [np.repeat(r["text"][None, :], r["x"].shape[0], axis=0)
                 for r in rows])).to(device)
        ids = (torch.from_numpy(np.concatenate([r["ids"] for r in rows])).to(device)
               if use_ids else None)
        segments, offset = [], 0
        for r in rows:
            n = r["x"].shape[0]
            segments.append((offset, n, r["target"]))
            offset += n
        return x, text, ids, segments

    def epoch_pass(rows, train_mode):
        probe.train(train_mode)
        order = np.random.permutation(len(rows)) if train_mode else np.arange(len(rows))
        entries, total, count = [], 0.0, 0
        for start in range(0, len(order), BATCH):
            chunk = [rows[i] for i in order[start:start + BATCH]]
            x, text, ids, segments = tensors(chunk)
            if train_mode:
                opt.zero_grad(set_to_none=True)
            scores = probe(x, text, ids)
            loss = 0.0
            for offset, n, target in segments:
                seg = scores[offset:offset + n]
                loss = loss + torch.nn.functional.cross_entropy(
                    seg[None, :], torch.tensor([target], device=device))
                if not train_mode:
                    entries.append(rank_metrics(seg.detach().cpu().numpy(), target))
            if not train_mode:
                for entry, row in zip(entries[-len(segments):], chunk):
                    entry["key"] = row["sample"].key
                    entry["group"] = row["sample"].group
                    entry["buckets"] = sorted(row["sample"].buckets)
                    entry["target_type"] = row["sample"].types[row["sample"].target]
            loss = loss / len(chunk)
            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
            total += float(loss.detach()) * len(chunk)
            count += len(chunk)
        return total / max(count, 1), entries

    history, best = [], None
    val_rows = build_rows(rows_by_split["val_seen"], arm, scaler, id_map, variant)
    if val_rows is None:
        return None
    for epoch in range(EPOCHS):
        train_loss, _ = epoch_pass(train, True)
        with torch.no_grad():
            _, val_entries = epoch_pass(val_rows, False)
        val = aggregate(val_entries)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_top1": val["top1"], "val_mrr": val["mrr"]})
        key = (val["top1"], val["mrr"])
        if best is None or key > best[0]:
            best = (key, copy.deepcopy(probe.state_dict()))
    probe.load_state_dict(best[1])
    out = {}
    with torch.no_grad():
        probe.eval()
        for split in SPLITS:
            rows = build_rows(rows_by_split[split], arm, scaler, id_map, variant)
            if rows is None:
                out[split] = []
                continue
            _, out[split] = epoch_pass(rows, False)
    return {"params": sum(p.numel() for p in params),
            "best_epoch": int(np.argmax([h["val_top1"] for h in history])),
            "entries": out, "history": history}


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------

def frozen_entries(samples):
    """The entity round's TD_masked cosine, recomputed on exactly these samples."""
    out = []
    for sample in samples:
        text = sample.texts.get("phrase")
        if text is None:
            text = sample.texts["full"]
        entry = rank_metrics(sample.blocks["visual"] @ text, sample.target)
        entry["key"] = sample.key
        entry["group"] = sample.group
        entry["buckets"] = sorted(sample.buckets)
        entry["target_type"] = sample.types[sample.target]
        out.append(entry)
    return out


def group_metrics(entries, key_fn) -> dict:
    groups = defaultdict(list)
    for entry in entries:
        for key in key_fn(entry):
            groups[key].append(entry)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def rescue_analysis(all_entries, reference: str, split="val_unseen") -> dict:
    """For the samples the reference gets wrong, which arm fixes them."""
    base = {e["key"]: e for e in all_entries[reference][split]}
    wrong = {k for k, e in base.items() if not e["top1"]}
    gap = {k for k, e in base.items() if not e["top1"] and e["top4"]}
    out = {"reference": reference, "split": split, "n_reference_wrong": len(wrong),
           "n_top4_not_top1": len(gap), "arms": {}}
    for arm, per_split in all_entries.items():
        if arm == reference:
            continue
        entries = {e["key"]: e for e in per_split[split]}
        keys = sorted(wrong & set(entries))
        rescued = [k for k in keys if entries[k]["top1"]]
        gap_keys = sorted(gap & set(entries))
        rescued_gap = [k for k in gap_keys if entries[k]["top1"]]
        ranks = [entries[k]["rank"] for k in gap_keys]
        out["arms"][arm] = {
            "n_scored": len(keys),
            "rescued_from_wrong": len(rescued),
            "rescue_rate": (len(rescued) / len(keys)) if keys else None,
            "top4_to_top1_n": len(gap_keys),
            "top4_to_top1_rescued": len(rescued_gap),
            "top4_to_top1_rescue_rate": ((len(rescued_gap) / len(gap_keys))
                                         if gap_keys else None),
            "median_rank_on_gap": float(np.median(ranks)) if ranks else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--hidden", type=int, default=HIDDEN,
                    help="probe hidden width; 0 gives a single linear layer")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data) if args.data else \
        artifact_dir(cfg) / "feature_sufficiency" / "data"
    base_dir = Path(args.base) if args.base else \
        artifact_dir(cfg) / "entity_grounding" / "data"
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "feature_sufficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qualitative_cases").mkdir(exist_ok=True)

    import torch

    splits = load_splits(data_dir, base_dir)
    print({k: len(v) for k, v in splits.items()}, flush=True)
    if min(len(v) for v in splits.values()) == 0:
        raise SystemExit("a split is empty")

    id_map = {}
    for sample in splits["train_seen"]:
        for eid in sample.entity_ids:
            id_map.setdefault((sample.map, eid), len(id_map) + 1)
    print(f"entity id vocabulary from train_seen: {len(id_map)}", flush=True)

    all_entries = {"TD_masked_frozen": {s: frozen_entries(splits[s]) for s in SPLITS}}
    arms_report = {"TD_masked_frozen": {
        "deployable": True, "params": 0, "seconds": 0.0,
        "metrics": {s: aggregate(all_entries["TD_masked_frozen"][s]) for s in SPLITS}}}

    for arm in args.arms:
        scaler = fit_scaler(splits["train_seen"], arm)
        t0 = time.time()
        runs = []
        for seed in args.seeds:
            result = run_arm(torch, arm, splits, scaler, id_map, seed, args.device,
                             hidden=args.hidden)
            if result is None:
                runs = []
                break
            runs.append(result)
        if not runs:
            print(f"  {arm:26s} skipped (a variant is missing)", flush=True)
            continue
        per_split = {}
        for split in SPLITS:
            merged = defaultdict(lambda: defaultdict(list))
            for run in runs:
                for entry in run["entries"][split]:
                    for k, v in entry.items():
                        merged[entry["key"]][k].append(v)
            entries = []
            for key, values in merged.items():
                entry = {"key": key, "rank": float(np.mean(values["rank"]))}
                entry["top1"] = float(np.mean(values["top1"])) >= 0.5
                entry["top4"] = float(np.mean(values["top4"])) >= 0.5
                entry["mrr"] = float(np.mean(values["mrr"]))
                entry["margin"] = float(np.mean(values["margin"]))
                entry["positive_margin"] = float(np.mean(
                    values["positive_margin"])) >= 0.5
                entry["group"] = values["group"][0]
                entry["buckets"] = values["buckets"][0]
                entry["target_type"] = values["target_type"][0]
                entries.append(entry)
            per_split[split] = entries
        all_entries[arm] = per_split
        arms_report[arm] = {
            "deployable": arm not in ORACLE_ARMS,
            "params": runs[0]["params"],
            "best_epochs": [r["best_epoch"] for r in runs],
            "seconds": round(time.time() - t0, 1),
            "metrics": {s: aggregate(per_split[s]) for s in SPLITS},
            "per_group": {s: group_metrics(per_split[s], lambda e: [e["group"]])
                          for s in SPLITS},
            "per_type": {s: group_metrics(per_split[s], lambda e: [e["target_type"]])
                         for s in SPLITS},
        }
        m = arms_report[arm]["metrics"]
        print(f"  {arm:26s} params {runs[0]['params']:7d}  "
              f"val {m['val_seen']['top1']:.3f}  unseen {m['val_unseen']['top1']:.3f}"
              f"  ({arms_report[arm]['seconds']}s)", flush=True)

    # instruction-bucket breakdown on val_unseen, for every arm
    per_bucket = {}
    for arm, per_split in all_entries.items():
        per_bucket[arm] = {s: group_metrics(per_split[s], lambda e: e["buckets"])
                           for s in SPLITS}

    rescue = rescue_analysis(all_entries, "TD_masked_frozen")
    rescue_by_group = {}
    for group in ("building", "vehicle", "other"):
        subset = {arm: {s: [e for e in per_split[s] if e["group"] == group]
                        for s in SPLITS}
                  for arm, per_split in all_entries.items()}
        rescue_by_group[group] = rescue_analysis(subset, "TD_masked_frozen")

    oracle = {
        "note": ("Oracle arms use the target's own annotation to choose an anchor "
                 "entity or an identity, so they are upper bounds, not methods."),
        "arms": {a: arms_report[a]["metrics"] for a in arms_report
                 if a in ORACLE_ARMS},
        "deployable_best": None,
    }
    deployable = {a: m for a, m in arms_report.items() if m["deployable"]}
    oracle["deployable_best"] = max(
        deployable, key=lambda a: deployable[a]["metrics"]["val_unseen"]["top1"]
        or -1)

    # View independence: are the two views different information at all?
    td, ob = [], []
    for sample in splits["val_unseen"]:
        td.append(sample.blocks["visual"])
        ob.append(sample.blocks["visual_oblique"])
    td, ob = np.concatenate(td), np.concatenate(ob)
    cos = np.sum(td * ob, axis=1) / (np.linalg.norm(td, axis=1)
                                     * np.linalg.norm(ob, axis=1) + 1e-9)
    rank_corr = []
    for sample in splits["val_unseen"]:
        a = sample.blocks["visual"] @ sample.texts["full"]
        b = sample.blocks["visual_oblique"] @ sample.texts["full"]
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        if ra.std() > 0 and rb.std() > 0:
            rank_corr.append(float(np.corrcoef(ra, rb)[0, 1]))
    view = {"n_candidates": int(len(td)),
            "mean_cosine_td_vs_oblique": float(np.mean(cos)),
            "median_cosine": float(np.median(cos)),
            "mean_rank_correlation": float(np.mean(rank_corr)) if rank_corr else None,
            "note": ("Feature-level agreement between the two views of the same "
                     "entity, on candidates that are projections of one point "
                     "cloud. High values mean a viewpoint change does not "
                     "introduce an independent source of information.")}

    report = {
        "counts": {k: len(v) for k, v in splits.items()},
        "groups": {s: dict(Counter(x.group for x in splits[s])) for s in SPLITS},
        "types": {s: dict(Counter(x.types[x.target] for x in splits[s]))
                  for s in SPLITS},
        "buckets": {s: dict(Counter(b for x in splits[s] for b in x.buckets))
                    for s in SPLITS},
        "arms": arms_report, "oracle_analysis": oracle,
        "per_bucket": per_bucket,
        "rescue": rescue, "rescue_by_group": rescue_by_group,
        "view_independence": view,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    (out_dir / f"metrics{suffix}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    (out_dir / f"per_type{suffix}.json").write_text(json.dumps(
        {a: arms_report[a]["per_type"] for a in arms_report
         if "per_type" in arms_report[a]},
        indent=2, sort_keys=True, default=float) + "\n")
    (out_dir / f"per_instruction_type{suffix}.json").write_text(json.dumps(
        per_bucket, indent=2, sort_keys=True, default=float) + "\n")
    (out_dir / f"rescue_analysis{suffix}.json").write_text(json.dumps(
        {"overall": rescue, "by_group": rescue_by_group},
        indent=2, sort_keys=True, default=float) + "\n")
    (out_dir / f"oracle_analysis{suffix}.json").write_text(json.dumps(
        oracle, indent=2, sort_keys=True, default=float) + "\n")

    _write_cases(out_dir, all_entries, splits)

    print("\n--- val_unseen, deployable ---")
    print(f"{'arm':26s} {'params':>8s} {'top1':>6s} {'top4':>6s} {'mrr':>6s}")
    for arm, info in arms_report.items():
        if not info["deployable"]:
            continue
        m = info["metrics"]["val_unseen"]
        print(f"{arm:26s} {info['params']:8d} {m['top1']:6.3f} {m['top4']:6.3f} "
              f"{m['mrr']:6.3f}")
    print("\n--- ORACLE (upper bounds, not methods) ---")
    for arm in ORACLE_ARMS:
        if arm not in arms_report:
            continue
        m = arms_report[arm]["metrics"]["val_unseen"]
        print(f"{arm:26s} {arms_report[arm]['params']:8d} {m['top1']:6.3f} "
              f"{m['top4']:6.3f} {m['mrr']:6.3f}")
    print(f"\nwrote {out_dir}")


def _write_cases(out_dir: Path, all_entries, splits, limit: int = 12) -> None:
    """Cases for the three rescue routes, dumped as data rather than images."""
    base_split = "val_unseen"
    base = {e["key"]: e for e in all_entries["TD_masked_frozen"][base_split]}
    by_key = {s.key: s for s in splits[base_split]}
    buckets = defaultdict(list)
    for key, entry in base.items():
        sample = by_key.get(key)
        if sample is None:
            continue
        record = {"key": key, "map": sample.map, "group": sample.group,
                  "target_type": sample.types[sample.target],
                  "instruction": sample.instruction,
                  "buckets": sorted(sample.buckets),
                  "td_masked_rank": entry["rank"],
                  "td_masked_top4": entry["top4"],
                  "arms": {}}
        for arm, per_split in all_entries.items():
            match = [e for e in per_split[base_split] if e["key"] == key]
            if match:
                record["arms"][arm] = {"rank": match[0]["rank"],
                                       "margin": match[0]["margin"]}
        if entry["top1"] or not entry["top4"]:
            continue
        buckets["top4_not_top1"].append(record)
        for arm in ("Visual+Relation", "Visual+Appearance", "Visual+Geometry",
                    "Visual+Semantic"):
            if arm in record["arms"] and record["arms"][arm]["rank"] == 1:
                buckets[f"rescued_by_{arm}"].append(record)
                break
    for name, items in buckets.items():
        (out_dir / "qualitative_cases" / f"{name}.json").write_text(
            json.dumps(items[:limit], indent=2, default=float) + "\n")
    print(f"  wrote cases: {[(k, len(v)) for k, v in buckets.items()]}", flush=True)


if __name__ == "__main__":
    main()
