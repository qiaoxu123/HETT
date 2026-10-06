#!/usr/bin/env python3
"""Does a world-coordinate-aligned joint encoding beat either view alone?

Every method here ends in the *same* frozen readout: the fused landmark token is
pushed through SigLIP2's own pooling head as a one-token sequence, L2-normalised,
and dotted with the same text embedding the whole-crop baselines use.  Nothing
about the readout differs between methods, so a difference in the table is a
difference of features rather than of how they were read out.

Fusion happens where the geometry is: the two views' patch tokens are paired by
projecting the landmark's own 3D points through both cameras, and only paired
patches are combined.  The control is the identical model trained with that
pairing permuted inside each landmark -- same weights, masks and sparsity, wrong
geometry -- so a real-over-shuffled gap is the value the world coordinates add.

Selection is structural, not conventional: the epoch, the learning rate and the
cross-view weight are chosen with ``train_seen`` / ``val_seen`` only, and
``val_unseen`` is read once, at the end.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.config import artifact_dir, load_config  # noqa: E402
from sensaturban_fpv.siglip_masked import l2norm, readout_tokens  # noqa: E402
from run_gate_b import load_encoder  # noqa: E402

FROZEN_METHODS = ("TD_global", "O_global", "TD_masked", "O_masked")
FROZEN_FEATURE = {"TD_global": "td_global", "O_global": "o_global",
                  "TD_masked": "td_masked", "O_masked": "o_masked"}
LEARNED_METHODS = ("ConcatGlobal", "ConcatMasked", "GeoPairShuffle",
                   "GeoPairFusion", "GeoPairFusionXL")
TEXT_VARIANTS = ("phrase", "name")
HIDDEN = 128
DROPOUT = 0.1
EPOCHS = 30
BATCH_SAMPLES = 4
SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class Sample:
    key: str
    split: str
    map_name: str
    episode_index: int
    step: int
    candidate_ids: np.ndarray
    candidate_types: np.ndarray
    target_index: int
    target_type: str
    target_distance: float
    same_class_candidates: int
    texts: dict
    variant: str = "phrase"
    arrays: dict = field(default_factory=dict)


def load_records(data_dir: Path, split: str, variant: str):
    """Kept records for one split that carry the requested text variant."""
    records, seen = [], set()
    for path in sorted(data_dir.glob(f"{split}*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("kept"):
                continue
            key = (record["episode_index"], record["step"])
            if key in seen or variant not in record["texts"]:
                continue
            seen.add(key)
            records.append(record)
    records.sort(key=lambda r: (r["episode_index"], r["step"]))
    out = []
    for record in records:
        archive = np.load(data_dir / record["file"], allow_pickle=True)
        if archive[f"text_{variant}"].size == 0:
            continue
        target = np.flatnonzero(archive["is_target"])
        if target.size != 1:
            continue
        out.append(Sample(
            key=record["file"], split=split, map_name=record["map"],
            episode_index=record["episode_index"], step=record["step"],
            candidate_ids=archive["candidate_ids"],
            candidate_types=archive["candidate_types"],
            target_index=int(target[0]), target_type=record["target_type"],
            target_distance=float(record["target_distance"]),
            same_class_candidates=int(record["same_class_candidates"]),
            texts=record["texts"], variant=variant,
            arrays={k: archive[k] for k in archive.files},
        ))
    return out


# --------------------------------------------------------------------------
# fusion modules
# --------------------------------------------------------------------------

def make_pair_fusion(torch, dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
    """``z -> delta`` with a zero-initialised output, added to the view mean.

    Starting from the mean of the two views and learning a correction means the
    fused feature cannot fall below that baseline for a reason unrelated to the
    data: the network has to earn any movement away from it.
    """
    nn = torch.nn

    class PairFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4 * dim)
            self.fc1 = nn.Linear(4 * dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)
            self.drop = nn.Dropout(dropout)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, p, q):
            z = self.norm(torch.cat([p, q, (p - q).abs(), p * q], dim=-1))
            delta = self.fc2(self.drop(nn.functional.gelu(self.fc1(z))))
            return 0.5 * (p + q) + delta

    return PairFusion()


def make_concat_head(torch, dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
    """The same shape of module, but over pooled landmark features."""
    nn = torch.nn

    class ConcatHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4 * dim)
            self.fc1 = nn.Linear(4 * dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)
            self.drop = nn.Dropout(dropout)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x, base):
            z = self.norm(x)
            return base + self.fc2(self.drop(nn.functional.gelu(self.fc1(z))))

    return ConcatHead()


def scatter_mean(torch, src, index, n, weights=None):
    d = src.shape[-1]
    if weights is None:
        weights = torch.ones(src.shape[0], dtype=src.dtype, device=src.device)
    acc = torch.zeros((n, d), dtype=src.dtype, device=src.device)
    acc.index_add_(0, index, src * weights[:, None])
    den = torch.zeros(n, dtype=src.dtype, device=src.device)
    den.index_add_(0, index, weights)
    return acc / den.clamp_min(1e-6)[:, None], den


def permute_correspondence(corr_o_row: np.ndarray, corr_cand: np.ndarray, seed: int):
    """The random control: pair each top-down patch with the wrong oblique patch.

    The permutation is applied inside a candidate, so every oblique patch the
    real pairing could reach is still reachable and every weight is unchanged.
    Only the association -- the part the world coordinates determined -- is gone.
    """
    out = corr_o_row.astype(np.int64).copy()
    rng = np.random.default_rng(seed)
    for k in np.unique(corr_cand):
        rows = np.flatnonzero(corr_cand == k)
        if rows.size > 1:
            out[rows] = out[rows][rng.permutation(rows.size)]
    return out


class FusionInputs:
    """Per-sample tensors, in token space, ready for a fusion module.

    Cached on the device: a sample's patch tokens are ~10 MB as float32 and the
    training loop revisits every sample every epoch, so re-uploading them would
    cost more wall clock than the forward and backward passes together.  They are
    held as float16 and widened where they are used.
    """

    _cache: dict = {}

    def __init__(self, torch, sample, device, shuffle_seed=None):
        a = sample.arrays
        self.torch = torch
        self.K = len(a["candidate_ids"])
        self.D = a["td_global"].shape[1]
        self.device = device
        self.td_tokens = to_device(torch, a["td_tokens"], device, torch.float16)
        self.o_tokens = to_device(torch, a["o_tokens"], device, torch.float16)
        self.td_row = to_device(torch, a["td_row"], device, torch.int64)
        self.o_row = to_device(torch, a["o_row"], device, torch.int64)
        self.corr_td_row = to_device(torch, a["corr_td_row"], device, torch.int64)
        corr_o = (a["corr_o_row"] if shuffle_seed is None
                  else permute_correspondence(a["corr_o_row"], a["corr_cand"],
                                              shuffle_seed))
        self.corr_o_row = to_device(torch, corr_o, device, torch.int64)
        self.td_global = to_device(torch, a["td_global"], device, torch.float32)
        self.o_global = to_device(torch, a["o_global"], device, torch.float32)
        self.td_masked = to_device(torch, a["td_masked"], device, torch.float32)
        self.o_masked = to_device(torch, a["o_masked"], device, torch.float32)
        self.corr_w = to_device(torch, a["corr_w"], device, torch.float32)
        self.coverage = float(a["corr_cand"].size) / max(a["td_row"].size, 1)

    def gather(self, tokens, index):
        return tokens[index].float()


def to_device(torch, array, device, dtype):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)


def get_inputs(torch, sample, device, shuffle_seed=None, cache: dict | None = None):
    key = (sample.key, shuffle_seed)
    if cache is None:
        return FusionInputs(torch, sample, device, shuffle_seed)
    if key not in cache:
        cache[key] = FusionInputs(torch, sample, device, shuffle_seed)
    return cache[key]


def run_pair_fusion(torch, module, inputs, with_reverse):
    """Fuse the landmark's paired patches, then pool them per candidate."""
    p = inputs.gather(inputs.td_tokens, inputs.corr_td_row)
    q_raw, den = scatter_mean(torch, inputs.gather(inputs.o_tokens,
                                                   inputs.corr_o_row),
                              inputs.corr_td_row, inputs.td_tokens.shape[0],
                              inputs.corr_w)
    q = torch.where(den[:, None] > 0, q_raw, p)
    fused = scatter_mean(torch, module(p, q), inputs.td_row, inputs.K)[0]

    reverse = None
    if with_reverse:
        p_r = inputs.gather(inputs.o_tokens, inputs.corr_o_row)
        q_r_raw, den_r = scatter_mean(
            torch, inputs.gather(inputs.td_tokens, inputs.corr_td_row),
            inputs.corr_o_row, inputs.o_tokens.shape[0], inputs.corr_w)
        q_r = torch.where(den_r[:, None] > 0, q_r_raw, p_r)
        reverse = scatter_mean(torch, module(p_r, q_r), inputs.o_row, inputs.K)[0]
    return fused, reverse


def forward_scores(model, torch, spec, inputs, text):
    """Landmark features -> frozen readout -> cosine with the text."""
    if spec["kind"] == "concat":
        a = inputs.td_global if spec["which"] == "global" else inputs.td_masked
        b = inputs.o_global if spec["which"] == "global" else inputs.o_masked
        x = torch.cat([a, b, (a - b).abs(), a * b], dim=-1)
        fused = spec["head"](x, 0.5 * (a + b))
        return l2norm(readout_tokens(model, fused)) @ text, None, None, None
    fused, reverse = run_pair_fusion(torch, spec["head"], inputs,
                                     spec["with_reverse"])
    joint = l2norm(readout_tokens(model, fused))
    scores = joint @ text
    if reverse is None:
        return scores, None, joint, None
    joint_rev = l2norm(readout_tokens(model, reverse))
    return scores, joint_rev @ text, joint, joint_rev


def build_spec(torch, device, name, dim):
    if name in ("ConcatGlobal", "ConcatMasked"):
        head = make_concat_head(torch, dim).to(device)
        return {"kind": "concat", "head": head,
                "which": "global" if name == "ConcatGlobal" else "masked",
                "with_reverse": False,
                "params": list(head.parameters())}
    head = make_pair_fusion(torch, dim).to(device)
    return {"kind": "pair", "head": head, "with_reverse": name == "GeoPairFusionXL",
            "params": list(head.parameters())}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def rank_metrics(scores: np.ndarray, target: int) -> dict:
    order = np.argsort(-scores)
    rank = int(np.where(order == target)[0][0]) + 1
    others = np.delete(scores, target)
    best_other = float(others.max()) if others.size else float("nan")
    return {"rank": rank, "top1": bool(rank == 1), "top4": bool(rank <= 4),
            "mrr": 1.0 / rank, "margin": float(scores[target] - best_other),
            "positive_margin": bool(scores[target] > best_other)}


def aggregate(entries) -> dict:
    if not entries:
        return {"n": 0}
    top1 = np.array([e["top1"] for e in entries], dtype=float)
    return {
        "n": len(entries),
        "top1": float(top1.mean()),
        "top4": float(np.mean([e["top4"] for e in entries])),
        "mrr": float(np.mean([e["mrr"] for e in entries])),
        "median_margin": float(np.median([e["margin"] for e in entries])),
        "positive_margin_ratio": float(np.mean([e["positive_margin"] for e in entries])),
        "top1_ci": wilson(float(top1.sum()), len(entries)),
    }


def wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(centre - half, 0.0)), float(min(centre + half, 1.0))]


def mean_over_seeds(metrics) -> dict:
    out = {}
    for key in metrics[0]:
        values = [m.get(key) for m in metrics]
        if key == "n":
            out[key] = values[0]
            continue
        if key == "top1_ci":
            out[key] = [float(np.mean([v[0] for v in values])),
                        float(np.mean([v[1] for v in values]))]
            continue
        numeric = [v for v in values if v is not None]
        out[key] = float(np.mean(numeric)) if numeric else None
        if len(values) > 1 and numeric:
            out[f"{key}_seed_std"] = float(np.std(numeric))
    return out


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_method(torch, model, device, name, train_samples, val_samples, args,
                 lr, cross_weight=0.0, seed=0, cache=None, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dim = train_samples[0].arrays["td_global"].shape[1]
    spec = build_spec(torch, device, name, dim)
    shuffle_seed = (seed * 1000 + 7) if name == "GeoPairShuffle" else None
    logit_scale = torch.nn.Parameter(
        torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(spec["params"] + [logit_scale], lr=lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in spec["params"])

    def text_of(sample):
        return to_device(torch, sample.arrays[f"text_{args.variant}"], device,
                         torch.float32)

    def epoch_pass(samples, train: bool):
        spec["head"].train(train)
        order = np.random.permutation(len(samples)) if train else np.arange(len(samples))
        entries, total, count = [], 0.0, 0
        for start in range(0, len(order), BATCH_SAMPLES):
            chunk = [samples[i] for i in order[start:start + BATCH_SAMPLES]]
            if train:
                opt.zero_grad(set_to_none=True)
            loss = 0.0
            for sample in chunk:
                text = text_of(sample)
                inputs = get_inputs(torch, sample, device, shuffle_seed, cache)
                scores, rev, joint, joint_rev = forward_scores(model, torch, spec,
                                                               inputs, text)
                target = torch.tensor([sample.target_index], device=device)
                loss = loss + torch.nn.functional.cross_entropy(
                    (logit_scale.exp() * scores)[None, :], target)
                if rev is not None:
                    loss = loss + torch.nn.functional.cross_entropy(
                        (logit_scale.exp() * rev)[None, :], target)
                    if cross_weight > 0:
                        logits = joint @ joint_rev.T * logit_scale.exp()
                        labels = torch.arange(joint.shape[0], device=device)
                        loss = loss + cross_weight * 0.5 * (
                            torch.nn.functional.cross_entropy(logits, labels)
                            + torch.nn.functional.cross_entropy(logits.T, labels))
                if not train:
                    entry = rank_metrics(scores.detach().cpu().numpy(),
                                         sample.target_index)
                    entry["coverage"] = inputs.coverage
                    entries.append(entry)
            loss = loss / len(chunk)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(spec["params"] + [logit_scale], 5.0)
                opt.step()
            total += float(loss.detach()) * len(chunk)
            count += len(chunk)
        return total / max(count, 1), (aggregate(entries) if not train else None)

    history, best = [], None
    for epoch in range(EPOCHS):
        train_loss, _ = epoch_pass(train_samples, True)
        with torch.no_grad():
            _, val_metrics = epoch_pass(val_samples, False)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_top1": val_metrics["top1"],
                        "val_mrr": val_metrics["mrr"]})
        key = (val_metrics["top1"], val_metrics["mrr"])
        if best is None or key > best[0]:
            best = (key, copy.deepcopy(spec["head"].state_dict()),
                    logit_scale.detach().clone())
    spec["head"].load_state_dict(best[1])
    with torch.no_grad():
        logit_scale.copy_(best[2])
        spec["head"].eval()
        _, val_metrics = epoch_pass(val_samples, False)
    return {"spec": spec, "params": n_params, "history": history,
            "val_metrics": val_metrics,
            "best_epoch": int(np.argmax([h["val_top1"] for h in history]))}


def score_samples(model, torch, spec, samples, variant, device, cache=None,
                  shuffle_seed=None):
    spec["head"].eval()
    entries = []
    with torch.no_grad():
        for sample in samples:
            text = to_device(torch, sample.arrays[f"text_{variant}"], device,
                             torch.float32)
            inputs = get_inputs(torch, sample, device, shuffle_seed, cache)
            scores, _, _, _ = forward_scores(model, torch, spec, inputs, text)
            entry = rank_metrics(scores.detach().cpu().numpy(), sample.target_index)
            entry["key"] = sample.key
            entry["coverage"] = inputs.coverage
            entries.append(entry)
    return entries


def entry_rows(method, split, seed, samples, entries):
    by_key = {s.key: s for s in samples}
    rows = []
    for entry in entries:
        sample = by_key[entry["key"]]
        rows.append({
            "method": method, "split": split, "seed": seed, "key": entry["key"],
            "map": sample.map_name, "episode_index": sample.episode_index,
            "step": sample.step, "target_type": sample.target_type,
            "target_distance": sample.target_distance,
            "same_class_candidates": sample.same_class_candidates,
            "candidate_count": int(len(sample.candidate_ids)),
            "rank": int(entry["rank"]), "top1": bool(entry["top1"]),
            "top4": bool(entry["top4"]), "mrr": float(entry["mrr"]),
            "margin": float(entry["margin"]),
            "positive_margin": bool(entry["positive_margin"]),
            "coverage": float(entry.get("coverage", 1.0)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant", default="phrase", choices=list(TEXT_VARIANTS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--lr-grid", type=float, nargs="*", default=[1e-3, 3e-4])
    ap.add_argument("--cross-grid", type=float, nargs="*", default=[0.0, 0.1, 0.5, 1.0])
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data) if args.data else artifact_dir(cfg) / "geo_fusion" / "data"
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "geo_fusion" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model, torch = load_encoder(cfg, args.device)
    device = args.device
    cache: dict = {}

    splits = {s: load_records(data_dir, s, args.variant)
              for s in ("train_seen", "val_seen", "val_unseen")}
    print({k: len(v) for k, v in splits.items()}, flush=True)
    if min(len(v) for v in splits.values()) == 0:
        raise SystemExit("a split is empty; nothing to compare")

    report = {
        "variant": args.variant,
        "counts": {k: len(v) for k, v in splits.items()},
        "google/siglip2-so400m-patch16-512": None,
        "candidate_counts": {
            k: {"mean": float(np.mean([len(s.candidate_ids) for s in v])),
                "min": int(min(len(s.candidate_ids) for s in v)),
                "max": int(max(len(s.candidate_ids) for s in v)),
                "histogram": np.bincount(
                    [len(s.candidate_ids) for s in v]).tolist()}
            for k, v in splits.items()},
        "correspondence_coverage": {
            k: float(np.mean([s.arrays["corr_cand"].size
                              / max(s.arrays["td_row"].size, 1) for s in v]))
            for k, v in splits.items()},
        "correspondence_per_candidate": {
            k: float(np.mean([s.arrays["corr_cand"].size / len(s.candidate_ids)
                              for s in v])) for k, v in splits.items()},
        "same_class_candidates_mean": {
            k: float(np.mean([s.same_class_candidates for s in v]))
            for k, v in splits.items()},
        "target_distance_median": {
            k: float(np.median([s.target_distance for s in v]))
            for k, v in splits.items()},
        "methods": {}, "tuning": {},
    }
    del report["google/siglip2-so400m-patch16-512"]
    report["encoder"] = cfg["vision_probe"]["model"]

    all_rows = []

    # ---- frozen, zero-parameter methods
    for name in FROZEN_METHODS:
        feature = FROZEN_FEATURE[name]
        metrics = {}
        for split, samples in splits.items():
            entries = []
            for sample in samples:
                scores = sample.arrays[feature] @ sample.arrays[f"text_{args.variant}"]
                entry = rank_metrics(scores, sample.target_index)
                entry["key"] = sample.key
                entries.append(entry)
            metrics[split] = aggregate(entries)
            all_rows += entry_rows(name, split, -1, samples, entries)
        report["methods"][name] = {"kind": "frozen", "trainable_params": 0,
                                   "fusion_level": "none", "metrics": metrics,
                                   "metrics_per_seed": {s: [metrics[s]] for s in metrics}}
        print(f"  {name:16s} val {metrics['val_seen']['top1']:.3f} "
              f"unseen {metrics['val_unseen']['top1']:.3f}", flush=True)

    # ---- two hyper-parameters, both chosen on val_seen
    lr_records = []
    for lr in args.lr_grid:
        run = train_method(torch, model, device, "GeoPairFusion", splits["train_seen"],
                           splits["val_seen"], args, lr, seed=args.seeds[0], cache=cache)
        lr_records.append({"lr": lr, "val_top1": run["val_metrics"]["top1"],
                           "val_mrr": run["val_metrics"]["mrr"],
                           "best_epoch": run["best_epoch"]})
        print(f"  lr {lr:g} -> val top1 {run['val_metrics']['top1']:.3f}", flush=True)
    lr_choice = max(lr_records, key=lambda r: (r["val_top1"], r["val_mrr"]))["lr"]

    cross_records = []
    for weight in args.cross_grid:
        run = train_method(torch, model, device, "GeoPairFusionXL", splits["train_seen"],
                           splits["val_seen"], args, lr_choice, cross_weight=weight,
                           seed=args.seeds[0], cache=cache)
        cross_records.append({"weight": weight,
                              "val_top1": run["val_metrics"]["top1"],
                              "val_mrr": run["val_metrics"]["mrr"],
                              "best_epoch": run["best_epoch"],
                              "trainable_params": run["params"]})
        print(f"  cross weight {weight:g} -> val top1 "
              f"{run['val_metrics']['top1']:.3f}", flush=True)
    cross_choice = max(cross_records,
                       key=lambda r: (r["val_top1"], r["val_mrr"]))["weight"]
    report["tuning"] = {"lr": lr_records, "cross_weight": cross_records,
                        "chosen": {"lr": lr_choice, "cross_weight": cross_choice}}

    # ---- every learned method, every seed
    for name in LEARNED_METHODS:
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            run = train_method(torch, model, device, name, splits["train_seen"],
                               splits["val_seen"], args, lr_choice,
                               cross_weight=(cross_choice
                                             if name == "GeoPairFusionXL" else 0.0),
                               seed=seed, cache=cache)
            entry = {"seed": seed, "trainable_params": run["params"],
                     "best_epoch": run["best_epoch"],
                     "val_metrics": run["val_metrics"],
                     "seconds": round(time.time() - t0, 1)}
            for split, samples in splits.items():
                entries = score_samples(
                    model, torch, run["spec"], samples, args.variant, device,
                    cache=cache,
                    shuffle_seed=((seed * 1000 + 7) if name == "GeoPairShuffle"
                                  else None))
                entry[f"{split}_metrics"] = aggregate(entries)
                all_rows += entry_rows(name, split, seed, samples, entries)
            runs.append(entry)
            print(f"  {name:16s} seed {seed}: val {entry['val_seen_metrics']['top1']:.3f}"
                  f"  unseen {entry['val_unseen_metrics']['top1']:.3f}"
                  f"  ({entry['seconds']}s, epoch {run['best_epoch']})", flush=True)
        report["methods"][name] = {
            "kind": "learned",
            "trainable_params": runs[0]["trainable_params"],
            "fusion_level": ("landmark feature" if name.startswith("Concat")
                             else "patch"),
            "metrics": {split: mean_over_seeds([r[f"{split}_metrics"] for r in runs])
                        for split in splits},
            "metrics_per_seed": {split: [r[f"{split}_metrics"] for r in runs]
                                 for split in splits},
            "runs": runs,
        }

    with (out_dir / f"metrics_{args.variant}.json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=float)
    with (out_dir / f"persample_{args.variant}.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")

    print("\n--- val_unseen ---")
    for name in FROZEN_METHODS + LEARNED_METHODS:
        m = report["methods"][name]["metrics"]["val_unseen"]
        print(f"  {name:16s} top1={m['top1']:.3f} top4={m['top4']:.3f} "
              f"mrr={m['mrr']:.3f} median_margin={m['median_margin']:+.4f}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
