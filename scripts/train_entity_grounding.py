#!/usr/bin/env python3
"""Does a 3D-anchored multi-view entity representation ground better than one view?

The unit is a target entity of any type.  Its 3D support is measured, projected
into both views at two scales, and turned into a set of pooled components --
tight appearance, context appearance, surroundings, single-view-exclusive
patches, measured geometry -- plus one *learned* component: the fusion of the
two views' patch tokens whose pairing came from the world coordinates.

Every method ends in the same frozen readout: the entity token goes through
SigLIP2's own pooling head as a one-token sequence, is L2-normalised, and is
dotted with the same text embedding the whole-crop baselines use.  Nothing about
the readout differs between methods, so a difference in the table is a
difference of features.

The text is used as the *query* in a single attention layer over the entity's
components, so which component a phrase draws on is learned rather than written
down: there is no branch anywhere in this file on an entity's type.

Selection is structural: the epoch, the learning rate and the cross-view weight
come from ``train_seen`` / ``val_seen`` only, and ``val_unseen`` is read once.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.config import artifact_dir, load_config  # noqa: E402
from sensaturban_fpv.entity_geometry import group_of  # noqa: E402
from sensaturban_fpv.siglip_masked import l2norm, readout_tokens  # noqa: E402
from run_gate_b import load_encoder  # noqa: E402

# Frozen, zero-parameter single-view references.
FROZEN_METHODS = {
    "TD_global": "td_context_global",
    "O_global": "oblique_context_global",
    "TD_masked": "td_context_masked",
    "O_masked": "oblique_context_masked",
    "TD_tight": "td_tight_masked",
    "O_tight": "oblique_tight_masked",
}
FROZEN_VIEW = {"TD_global": "TD", "TD_masked": "TD", "TD_tight": "TD",
               "O_global": "O", "O_masked": "O", "O_tight": "O"}

APPEARANCE = ("td_tight_masked", "oblique_tight_masked",
              "td_context_masked", "oblique_context_masked")
SURROUNDINGS = ("td_context_background", "oblique_context_background")
EXCLUSIVE = ("td_exclusive", "oblique_exclusive")

LEARNED_METHODS = {
    "ConcatGlobal": {"components": ("td_context_global", "oblique_context_global"),
                     "fusion": "concat", "shared": False},
    "ConcatMasked": {"components": ("td_context_masked", "oblique_context_masked"),
                     "fusion": "concat", "shared": False},
    "EntityConcat": {"components": APPEARANCE + SURROUNDINGS + ("geometry",),
                     "fusion": "concat", "shared": False},
    "GeoNoContext": {"components": APPEARANCE + EXCLUSIVE + ("geometry",),
                     "fusion": "attn", "shared": True},
    "GeoAligned": {"components": APPEARANCE + SURROUNDINGS + EXCLUSIVE + ("geometry",),
                   "fusion": "attn", "shared": True},
    "GeoShuffled": {"components": APPEARANCE + SURROUNDINGS + EXCLUSIVE + ("geometry",),
                    "fusion": "attn", "shared": True, "shuffle": True},
    "GeoAlignedXL": {"components": APPEARANCE + SURROUNDINGS + EXCLUSIVE + ("geometry",),
                     "fusion": "attn", "shared": True, "reverse": True},
}
TEXT_VARIANTS = ("phrase", "name")
HIDDEN = 192
ATTN_DIM = 64
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
    entity_ids: np.ndarray
    entity_types: np.ndarray
    target_index: int
    target_type: str
    target_group: str
    target_distance: float
    same_class_candidates: int
    candidate_count: int
    texts: dict
    variant: str = "phrase"
    arrays: dict = field(default_factory=dict)


def load_records(data_dir: Path, split: str, variant: str):
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
            entity_ids=archive["entity_ids"], entity_types=archive["entity_types"],
            target_index=int(target[0]), target_type=record["target_type"],
            target_group=group_of(record["target_type"]),
            target_distance=float(record["target_distance"]),
            same_class_candidates=int(record["same_class_candidates"]),
            candidate_count=int(record["candidate_count"]),
            texts=record["texts"], variant=variant,
            arrays={k: archive[k] for k in archive.files},
        ))
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def make_model(torch, vision_model, dim, components, fusion, shared, geom_dim,
               geom_mean, geom_std, hidden=HIDDEN, attn=ATTN_DIM, dropout=DROPOUT):
    """The fusion and readout module.

    ``vision_model`` is put in the module's ``__dict__`` rather than assigned,
    because assigning an ``nn.Module`` to an attribute registers it as a
    submodule -- and then ``module.parameters()``, which is what the optimizer is
    built from, would contain all 400M parameters of SigLIP2.  The readout has
    to stay frozen, and this is what enforces it rather than a comment saying so.
    """
    nn = torch.nn

    class EntityGrounding(nn.Module):
        def __init__(self):
            super().__init__()
            self.__dict__["vision"] = vision_model
            self.components = list(components)
            self.fusion = fusion
            self.shared = shared
            self.use_reverse = False
            self.register_buffer("geom_mean", torch.tensor(geom_mean, dtype=torch.float32))
            self.register_buffer("geom_std", torch.tensor(geom_std, dtype=torch.float32))
            width = hidden * (len(components) + (1 if shared else 0))

            self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden))
            self.geom_mlp = nn.Sequential(nn.Linear(geom_dim, hidden), nn.GELU(),
                                          nn.Linear(hidden, hidden))
            self.drop = nn.Dropout(dropout)
            if shared:
                self.pair = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(),
                                          nn.Linear(hidden, hidden))
                nn.init.zeros_(self.pair[-1].weight)
                nn.init.zeros_(self.pair[-1].bias)
                self.res = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden))
                nn.init.zeros_(self.res[-1].weight)
                nn.init.zeros_(self.res[-1].bias)
            if fusion == "attn":
                # The text is the query; the components are the keys.  Which
                # component a phrase draws on is therefore learned, and no
                # branch anywhere tests the entity's type.
                self.wq = nn.Linear(dim, attn)
                self.wk = nn.Linear(hidden, attn)
            else:
                # Same residual shape as the patch fusion: start from the mean of
                # the encoded components and learn a correction, with the last
                # layer zeroed so the first step is the mean and nothing else.
                self.concat_head = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                                 nn.Linear(hidden, hidden))
                nn.init.zeros_(self.concat_head[-1].weight)
                nn.init.zeros_(self.concat_head[-1].bias)
            self.dec = nn.Linear(hidden, dim)

        def encode_components(self, arrays, device):
            cols = []
            for name in self.components:
                if name == "geometry":
                    g = arrays["geometry"]
                    g = (g - self.geom_mean) / self.geom_std
                    cols.append(self.geom_mlp(g))
                else:
                    cols.append(self.enc(arrays[name]))
            return cols

        def _pair(self, p, q):
            z = torch.cat([p, q, (p - q).abs(), p * q], dim=-1)
            return 0.5 * (p + q) + self.pair(self.drop(z))

        def pair_fusion(self, inputs):
            """Fuse paired patches, then pool them into one component.

            The gather is over correspondence entries, so the weighted mean of
            the partners has to be taken back in token-row space first and then
            read out at the entries -- pooling in entry space would average each
            entry with itself.
            """
            query = self.enc(inputs.gather(inputs.td_tokens, inputs.corr_td_row))
            partner_rows, den = scatter_mean(
                self.enc(inputs.gather(inputs.o_tokens, inputs.corr_o_row)),
                inputs.corr_td_row, inputs.td_tokens.shape[0], inputs.corr_w)
            partner = torch_where(den[inputs.corr_td_row], partner_rows[inputs.corr_td_row],
                                  query)
            fused = self._pair(query, partner)
            pooled, _ = scatter_mean(fused, inputs.td_row[inputs.corr_td_row],
                                     inputs.K)
            return pooled

        def pair_fusion_reverse(self, inputs):
            """The same fusion with the views swapped: oblique patches query
            top-down ones.  Two independent joint readings of one entity are what
            the cross-view identity loss is built on."""
            query = self.enc(inputs.gather(inputs.o_tokens, inputs.corr_o_row))
            partner_rows, den = scatter_mean(
                self.enc(inputs.gather(inputs.td_tokens, inputs.corr_td_row)),
                inputs.corr_o_row, inputs.o_tokens.shape[0], inputs.corr_w)
            partner = torch_where(den[inputs.corr_o_row], partner_rows[inputs.corr_o_row],
                                  query)
            fused = self._pair(query, partner)
            pooled, _ = scatter_mean(fused, inputs.o_row[inputs.corr_o_row], inputs.K)
            return pooled

        def forward(self, arrays, text, inputs, want_reverse=False):
            cols = self.encode_components(arrays, arrays["geometry"].device)
            if self.shared:
                shared = self.pair_fusion(inputs)
                cols.append(shared + self.res(shared))
            stacked = torch.stack(cols, dim=1)                    # (K, M, hidden)
            if self.fusion == "attn":
                q = self.wq(text)                                 # (attn,)
                k = self.wk(stacked)                              # (K, M, attn)
                a = torch.softmax(
                    (k @ q) / np.sqrt(k.shape[-1]), dim=1)         # (K, M)
                agg = (a[..., None] * stacked).sum(dim=1)
            else:
                agg = stacked.mean(dim=1) + self.concat_head(stacked.flatten(1))
            token = self.dec(self.drop(agg))
            joint = l2norm(readout_tokens(self.vision, token))
            scores = joint @ text
            reverse = None
            if want_reverse and self.shared:
                rev = self.pair_fusion_reverse(inputs)
                rev = rev + self.res(rev)
                agg_r = rev
                if self.fusion == "attn":
                    k = self.wk(agg_r[:, None, :])
                    a = torch.softmax((k @ q) / np.sqrt(k.shape[-1]), dim=1)
                    agg_r = (a * agg_r[:, None, :]).sum(dim=1)
                else:
                    agg_r = agg_r + self.concat_head(agg_r)
                token_r = self.dec(self.drop(agg_r))
                joint_r = l2norm(readout_tokens(self.vision, token_r))
                reverse = joint_r @ text, joint_r
            return scores, reverse, joint

    return EntityGrounding()


def torch_where(den, value, fallback):
    import torch

    return torch.where(den[:, None] > 0, value, fallback)


def scatter_mean(src, index, n, weights=None):
    import torch

    d = src.shape[-1]
    if weights is None:
        weights = torch.ones(src.shape[0], dtype=src.dtype, device=src.device)
    acc = torch.zeros((n, d), dtype=src.dtype, device=src.device)
    acc.index_add_(0, index, src * weights[:, None])
    den = torch.zeros(n, dtype=src.dtype, device=src.device)
    den.index_add_(0, index, weights)
    return acc / den.clamp_min(1e-6)[:, None], den


class Inputs:
    """Per-sample pairing tensors, cached on the device."""

    def __init__(self, torch, sample, device, shuffle_seed=None):
        a = sample.arrays
        self.K = len(a["entity_ids"])
        self.td_tokens = to_device(torch, a["td_tokens"], device, torch.float16)
        self.o_tokens = to_device(torch, a["oblique_tokens"], device, torch.float16)
        self.td_row = to_device(torch, a["td_row"], device, torch.int64)
        self.o_row = to_device(torch, a["oblique_row"], device, torch.int64)
        self.corr_td_row = to_device(torch, a["corr_td_row"], device, torch.int64)
        corr_o = a["corr_o_row"]
        if shuffle_seed is not None and corr_o.size:
            corr_o = permute_correspondence(corr_o, a["corr_cand"], shuffle_seed)
        self.corr_o_row = to_device(torch, corr_o, device, torch.int64)
        self.corr_w = to_device(torch, a["corr_w"], device, torch.float32)
        self.coverage = float(a["corr_cand"].size) / max(a["td_row"].size, 1)

    def gather(self, tokens, index):
        """Tokens are held as float16 to halve the device footprint; the fusion
        itself is float32, so every read widens."""
        return tokens[index].float()


def to_device(torch, array, device, dtype):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)


def permute_correspondence(corr_o_row, corr_cand, seed):
    """The random control: pair each top-down patch with the wrong oblique patch.

    The permutation is applied inside an entity, so every oblique patch the real
    pairing could reach is still reachable and every weight is unchanged; only
    the association the world coordinates determined is gone.
    """
    out = corr_o_row.astype(np.int64).copy()
    rng = np.random.default_rng(seed)
    for k in np.unique(corr_cand):
        rows = np.flatnonzero(corr_cand == k)
        if rows.size > 1:
            out[rows] = out[rows][rng.permutation(rows.size)]
    return out


def get_inputs(torch, sample, device, shuffle_seed, cache):
    key = (sample.key, shuffle_seed)
    if key not in cache:
        cache[key] = Inputs(torch, sample, device, shuffle_seed)
    return cache[key]


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
    return {"n": len(entries), "top1": float(top1.mean()),
            "top4": float(np.mean([e["top4"] for e in entries])),
            "mrr": float(np.mean([e["mrr"] for e in entries])),
            "median_margin": float(np.median([e["margin"] for e in entries])),
            "positive_margin_ratio": float(np.mean(
                [e["positive_margin"] for e in entries])),
            "top1_ci": wilson(float(top1.sum()), len(entries))}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(centre - half, 0)), float(min(centre + half, 1))]


def mean_over_seeds(metrics) -> dict:
    if any(m.get("n", 0) == 0 for m in metrics):
        return {"n": 0}
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

def geom_stats(samples):
    g = np.concatenate([s.arrays["geometry"] for s in samples], axis=0)
    return g.mean(axis=0), np.clip(g.std(axis=0), 1e-3, None)


def train_method(torch, vision_model, device, name, train_samples, val_samples,
                 args, lr, cross_weight=0.0, seed=0, cache=None, stats=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = LEARNED_METHODS[name]
    shuffle_seed = (seed * 1000 + 7) if spec.get("shuffle") else None
    module = make_model(torch, vision_model,
                        train_samples[0].arrays["td_context_masked"].shape[1],
                        spec["components"], spec["fusion"], spec["shared"],
                        train_samples[0].arrays["geometry"].shape[1],
                        stats[0], stats[1]).to(device)
    params = list(module.parameters())
    logit_scale = torch.nn.Parameter(
        torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(params + [logit_scale], lr=lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in params)

    def text_of(sample, device=device):
        return to_device(torch, sample.arrays[f"text_{args.variant}"], device,
                         torch.float32)

    def device_arrays(sample):
        a = sample.arrays
        out = {}
        for key in ("geometry", *spec["components"]):
            if key == "geometry":
                continue
            out[key] = to_device(torch, a[key], device, torch.float32)
        out["geometry"] = to_device(torch, a["geometry"], device, torch.float32)
        return out

    def run_pass(samples, train: bool):
        module.train(train)
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
                scores, reverse, joint = module(device_arrays(sample), text, inputs,
                                                want_reverse=spec.get("reverse", False))
                target = torch.tensor([sample.target_index], device=device)
                loss = loss + torch.nn.functional.cross_entropy(
                    (logit_scale.exp() * scores)[None, :], target)
                if reverse is not None:
                    rev_scores, joint_rev = reverse
                    loss = loss + torch.nn.functional.cross_entropy(
                        (logit_scale.exp() * rev_scores)[None, :], target)
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
                torch.nn.utils.clip_grad_norm_(params + [logit_scale], 5.0)
                opt.step()
            total += float(loss.detach()) * len(chunk)
            count += len(chunk)
        return total / max(count, 1), (aggregate(entries) if not train else None)

    history, best = [], None
    for epoch in range(EPOCHS):
        train_loss, _ = run_pass(train_samples, True)
        with torch.no_grad():
            _, val_metrics = run_pass(val_samples, False)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_top1": val_metrics["top1"], "val_mrr": val_metrics["mrr"]})
        key = (val_metrics["top1"], val_metrics["mrr"])
        if best is None or key > best[0]:
            best = (key, copy.deepcopy(module.state_dict()),
                    logit_scale.detach().clone())
    module.load_state_dict(best[1])
    with torch.no_grad():
        logit_scale.copy_(best[2])
        module.eval()
        _, val_metrics = run_pass(val_samples, False)
    return {"module": module, "spec": spec, "params": n_params, "history": history,
            "val_metrics": val_metrics,
            "best_epoch": int(np.argmax([h["val_top1"] for h in history]))}


def score_samples(torch, device, module, spec, samples, args, cache,
                  shuffle_seed=None):
    module.eval()
    entries = []
    with torch.no_grad():
        for sample in samples:
            text = to_device(torch, sample.arrays[f"text_{args.variant}"], device,
                             torch.float32)
            arrays = {}
            for key in spec["components"]:
                arrays[key] = to_device(torch, sample.arrays[key], device,
                                        torch.float32)
            arrays["geometry"] = to_device(torch, sample.arrays["geometry"], device,
                                           torch.float32)
            inputs = get_inputs(torch, sample, device, shuffle_seed, cache)
            scores, _, _ = module(arrays, text, inputs, want_reverse=False)
            entry = rank_metrics(scores.detach().cpu().numpy(), sample.target_index)
            entry["key"] = sample.key
            entries.append(entry)
    return entries


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

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
    ap.add_argument("--methods", nargs="*", default=list(LEARNED_METHODS))
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data) if args.data else \
        artifact_dir(cfg) / "entity_grounding" / "data"
    out_dir = Path(args.out) if args.out else \
        artifact_dir(cfg) / "entity_grounding" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, vision_model, torch = load_encoder(cfg, args.device)
    device = args.device
    cache: dict = {}

    splits = {s: load_records(data_dir, s, args.variant)
              for s in ("train_seen", "val_seen", "val_unseen")}
    print({k: len(v) for k, v in splits.items()}, flush=True)
    if min(len(v) for v in splits.values()) == 0:
        raise SystemExit("a split is empty; nothing to compare")

    stats = geom_stats(splits["train_seen"])
    report = {
        "encoder": cfg["vision_probe"]["model"], "variant": args.variant,
        "counts": {k: len(v) for k, v in splits.items()},
        "target_groups": {k: dict(Counter(s.target_group for s in v))
                          for k, v in splits.items()},
        "target_types": {k: dict(Counter(s.target_type for s in v))
                         for k, v in splits.items()},
        "candidate_counts": {
            k: {"mean": float(np.mean([s.candidate_count for s in v])),
                "min": int(min(s.candidate_count for s in v)),
                "max": int(max(s.candidate_count for s in v))}
            for k, v in splits.items()},
        "support_sources": {k: dict(Counter(
            str(t) for s in v for t in s.arrays["support_source"]))
            for k, v in splits.items()},
        "correspondence": {
            k: {"mean_pairs": float(np.mean([s.arrays["corr_w"].size for s in v])),
                "samples_with_any": float(np.mean(
                    [s.arrays["corr_w"].size > 0 for s in v]))}
            for k, v in splits.items()},
        "target_distance_median": {k: float(np.median(
            [s.target_distance for s in v])) for k, v in splits.items()},
        "methods": {}, "tuning": {},
    }

    all_rows = []

    def add_rows(method, split, seed, samples, entries):
        by_key = {s.key: s for s in samples}
        for entry in entries:
            sample = by_key[entry["key"]]
            all_rows.append({
                "method": method, "split": split, "seed": seed,
                "key": entry["key"], "map": sample.map_name,
                "episode_index": sample.episode_index, "step": sample.step,
                "target_type": sample.target_type, "group": sample.target_group,
                "target_distance": sample.target_distance,
                "same_class_candidates": sample.same_class_candidates,
                "candidate_count": sample.candidate_count,
                "rank": int(entry["rank"]), "top1": bool(entry["top1"]),
                "top4": bool(entry["top4"]), "mrr": float(entry["mrr"]),
                "margin": float(entry["margin"]),
                "positive_margin": bool(entry["positive_margin"]),
                "coverage": float(entry.get("coverage", 1.0)),
            })

    # ---- frozen single-view references
    for name, feature in FROZEN_METHODS.items():
        metrics, per_group = {}, {}
        for split, samples in splits.items():
            entries = []
            for sample in samples:
                text = sample.arrays[f"text_{args.variant}"]
                scores = sample.arrays[feature] @ text
                entry = rank_metrics(scores, sample.target_index)
                entry["key"] = sample.key
                entries.append(entry)
            metrics[split] = aggregate(entries)
            per_group[split] = {g: aggregate([e for e, s in zip(entries, samples)
                                              if s.target_group == g])
                                for g in ("building", "vehicle", "other")}
            add_rows(name, split, -1, samples, entries)
        report["methods"][name] = {
            "kind": "frozen", "trainable_params": 0, "view": FROZEN_VIEW[name],
            "fusion_level": "none", "metrics": metrics, "metrics_per_group": per_group,
            "metrics_per_seed": {s: [metrics[s]] for s in metrics}}
        print(f"  {name:14s} val {metrics['val_seen']['top1']:.3f} "
              f"unseen {metrics['val_unseen']['top1']:.3f}", flush=True)

    # ---- hyper-parameters, chosen on val_seen only
    lr_records = []
    for lr in args.lr_grid:
        run = train_method(torch, vision_model, device, "GeoAligned", splits["train_seen"],
                           splits["val_seen"], args, lr, seed=args.seeds[0],
                           cache=cache, stats=stats)
        lr_records.append({"lr": lr, "val_top1": run["val_metrics"]["top1"],
                           "val_mrr": run["val_metrics"]["mrr"],
                           "best_epoch": run["best_epoch"]})
        print(f"  lr {lr:g} -> val top1 {run['val_metrics']['top1']:.3f}", flush=True)
    lr_choice = max(lr_records, key=lambda r: (r["val_top1"], r["val_mrr"]))["lr"]

    cross_records = []
    for weight in args.cross_grid:
        run = train_method(torch, vision_model, device, "GeoAlignedXL", splits["train_seen"],
                           splits["val_seen"], args, lr_choice, cross_weight=weight,
                           seed=args.seeds[0], cache=cache, stats=stats)
        cross_records.append({"weight": weight,
                              "val_top1": run["val_metrics"]["top1"],
                              "val_mrr": run["val_metrics"]["mrr"],
                              "best_epoch": run["best_epoch"],
                              "trainable_params": run["params"]})
        print(f"  cross {weight:g} -> val top1 {run['val_metrics']['top1']:.3f}",
              flush=True)
    cross_choice = max(cross_records,
                       key=lambda r: (r["val_top1"], r["val_mrr"]))["weight"]
    report["tuning"] = {"lr": lr_records, "cross_weight": cross_records,
                        "chosen": {"lr": lr_choice, "cross_weight": cross_choice}}

    # ---- learned methods
    for name in args.methods:
        spec = LEARNED_METHODS[name]
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            run = train_method(torch, vision_model, device, name, splits["train_seen"],
                               splits["val_seen"], args, lr_choice,
                               cross_weight=(cross_choice if spec.get("reverse") else 0.0),
                               seed=seed, cache=cache, stats=stats)
            entry = {"seed": seed, "trainable_params": run["params"],
                     "best_epoch": run["best_epoch"],
                     "val_metrics": run["val_metrics"],
                     "seconds": round(time.time() - t0, 1)}
            per_group = {}
            for split, samples in splits.items():
                entries = score_samples(
                    torch, device, run["module"], spec, samples, args, cache,
                    shuffle_seed=((seed * 1000 + 7) if spec.get("shuffle") else None))
                entry[f"{split}_metrics"] = aggregate(entries)
                per_group[split] = {g: aggregate([e for e, s in zip(entries, samples)
                                                  if s.target_group == g])
                                    for g in ("building", "vehicle", "other")}
                add_rows(name, split, seed, samples, entries)
            entry["metrics_per_group"] = per_group
            runs.append(entry)
            print(f"  {name:14s} seed {seed}: val "
                  f"{entry['val_seen_metrics']['top1']:.3f}  unseen "
                  f"{entry['val_unseen_metrics']['top1']:.3f}  "
                  f"({entry['seconds']}s, epoch {run['best_epoch']})", flush=True)
        report["methods"][name] = {
            "kind": "learned", "trainable_params": runs[0]["trainable_params"],
            "components": list(spec["components"]), "fusion": spec["fusion"],
            "shared": spec["shared"], "shuffle": bool(spec.get("shuffle")),
            "reverse": bool(spec.get("reverse")),
            "fusion_level": ("patch" if spec["shared"] else "landmark feature"),
            "metrics": {s: mean_over_seeds([r[f"{s}_metrics"] for r in runs])
                        for s in splits},
            "metrics_per_seed": {s: [r[f"{s}_metrics"] for r in runs] for s in splits},
            "metrics_per_group": {
                s: {g: mean_over_seeds([r["metrics_per_group"][s][g] for r in runs])
                    for g in ("building", "vehicle", "other")} for s in splits},
            "runs": runs,
        }

    with (out_dir / f"metrics_{args.variant}.json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=float)
    with (out_dir / f"persample_{args.variant}.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")

    print("\n--- val_unseen ---")
    for name in list(FROZEN_METHODS) + list(args.methods):
        m = report["methods"][name]["metrics"]["val_unseen"]
        print(f"  {name:14s} top1={m['top1']:.3f} top4={m['top4']:.3f} "
              f"mrr={m['mrr']:.3f} margin={m['median_margin']:+.4f}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
