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
PAIR_DIM = 128
DROPOUT = 0.1
EPOCHS = 30
BATCH_SAMPLES = 16
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
               geom_mean, geom_std, hidden=HIDDEN, attn=ATTN_DIM, dropout=DROPOUT,
               pair_dim=PAIR_DIM):
    """The fusion and readout module.

    Two spaces meet here and the distinction is the whole design.  The pooled
    components loaded from a sample are SigLIP2 *joint-space* vectors -- they are
    already the output of the checkpoint's pooling head.  The patch tokens the
    correspondence pairs are *hidden-space* vectors, the head's input.  So the
    head is applied exactly once, to the patch fusion, and never again to a
    component that has already been through it; applying it twice would put the
    learned methods behind a random projection of the very features they are
    supposed to beat, which is a defect that looks exactly like a negative
    result.

    ``vision_model`` is put in the module's ``__dict__`` rather than assigned:
    assigning an ``nn.Module`` to an attribute registers it as a submodule, and
    then ``module.parameters()`` -- which is what the optimizer is built from --
    would contain all 400M parameters of SigLIP2.

    Fusing a convex combination of the joint-space components and adding a
    zero-initialised residual means the first training step is exactly the mean
    of the components the method was given, and the attention starts uniform
    because its query and key projections start at zero.
    """
    nn = torch.nn

    class EntityGrounding(nn.Module):
        def __init__(self):
            super().__init__()
            self.__dict__["vision"] = vision_model
            self.components = list(components)
            self.fusion = fusion
            self.shared = shared
            width = attn * (len(components) + (1 if shared else 0))
            self.register_buffer("geom_mean", torch.tensor(geom_mean, dtype=torch.float32))
            self.register_buffer("geom_std", torch.tensor(geom_std, dtype=torch.float32))
            self.drop = nn.Dropout(dropout)

            # Score-side projections: a shared low-dimensional view of every
            # component, and a residual that starts at zero.
            self.proj = nn.Linear(dim, attn)
            self.delta = nn.Sequential(nn.Linear(width, attn), nn.GELU(),
                                       nn.Linear(attn, dim))
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
            self.geom_mlp = nn.Sequential(nn.Linear(geom_dim, attn), nn.GELU(),
                                          nn.Linear(attn, dim))
            if fusion == "attn":
                # The text is the query, the components are the keys, and both
                # start at zero so the first step weights every component
                # equally.  Which component a phrase draws on is learned, and no
                # branch anywhere tests the entity's type.
                self.wq = nn.Linear(dim, attn)
                self.wk = nn.Linear(attn, attn)
                nn.init.zeros_(self.wq.weight), nn.init.zeros_(self.wq.bias)
                nn.init.zeros_(self.wk.weight), nn.init.zeros_(self.wk.bias)
            if shared:
                # Patch fusion lives in hidden space and returns to joint space
                # through the frozen head, once.
                self.penc = nn.Linear(dim, pair_dim)
                self.pair = nn.Sequential(nn.Linear(4 * pair_dim, pair_dim), nn.GELU(),
                                          nn.Linear(pair_dim, pair_dim))
                nn.init.zeros_(self.pair[-1].weight)
                nn.init.zeros_(self.pair[-1].bias)
                self.pdec = nn.Linear(pair_dim, dim)

        def pooled_components(self, columns):
            """Every component as a joint-space vector, in the declared order."""
            cols = [columns[name] for name in self.components if name != "geometry"]
            if "geometry" in self.components:
                g = (columns["geometry"] - self.geom_mean) / self.geom_std
                cols.append(self.geom_mlp(g))
            return cols

        def _pair(self, p, q):
            z = torch.cat([p, q, (p - q).abs(), p * q], dim=-1)
            return 0.5 * (p + q) + self.pair(self.drop(z))

        def pair_fusion(self, inputs, reverse=False):
            """Fuse paired patches and pool them into one joint-space component.

            The gather is over correspondence entries, so the weighted mean of
            the partners has to be taken back in token-row space and read out at
            the entries -- pooling in entry space would average each entry with
            itself.
            """
            if reverse:
                query = self.penc(inputs.gather(inputs.o_tokens, inputs.corr_o_row))
                partner_rows, den = scatter_mean(
                    self.penc(inputs.gather(inputs.td_tokens, inputs.corr_td_row)),
                    inputs.corr_o_row, inputs.o_tokens.shape[0], inputs.corr_w)
                index = inputs.corr_o_row
                pool_index = inputs.o_row[inputs.corr_o_row]
            else:
                query = self.penc(inputs.gather(inputs.td_tokens, inputs.corr_td_row))
                partner_rows, den = scatter_mean(
                    self.penc(inputs.gather(inputs.o_tokens, inputs.corr_o_row)),
                    inputs.corr_td_row, inputs.td_tokens.shape[0], inputs.corr_w)
                index = inputs.corr_td_row
                pool_index = inputs.td_row[inputs.corr_td_row]
            partner = torch_where(den[index], partner_rows[index], query)
            fused = self._pair(query, partner)
            pooled, _ = scatter_mean(fused, pool_index, inputs.K)
            return readout_tokens(self.vision, self.pdec(pooled))

        def fuse(self, cols, text):
            stacked = torch.stack(cols, dim=1)                     # (K, M, dim)
            u = self.proj(stacked)                                 # (K, M, attn)
            if self.fusion == "attn":
                q = self.wq(text)                                  # (K, attn)
                k = self.wk(u)                                     # (K, M, attn)
                a = torch.softmax(
                    (k * q[:, None, :]).sum(-1) / np.sqrt(u.shape[-1]), dim=1)
                base = (a[..., None] * stacked).sum(dim=1)
            else:
                base = stacked.mean(dim=1)
            delta = self.delta(self.drop(u.flatten(1)))
            return base + delta

        def forward(self, columns, text, inputs, want_reverse=False):
            cols = self.pooled_components(columns)
            if self.shared:
                cols.append(self.pair_fusion(inputs))
            # One text row per candidate: a batch mixes samples, and a shared row
            # would cross every candidate with every instruction.
            if text.dim() == 1:
                text = text[None, :].expand(cols[0].shape[0], -1)
            joint = l2norm(self.fuse(cols, text))
            scores = (joint * text).sum(dim=-1)

            reverse = None
            if want_reverse and self.shared:
                cols_r = self.pooled_components(columns) + [self.pair_fusion(
                    inputs, reverse=True)]
                joint_r = l2norm(self.fuse(cols_r, text))
                reverse = (joint_r * text).sum(dim=-1), joint_r
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


class BatchInputs:
    """Several samples' pairing tensors concatenated into one flat batch.

    Fusion over one sample at a time is dominated by kernel launch rather than
    arithmetic -- a sample contributes a few thousand rows of width 1152 to a
    GPU that wants tens of thousands -- and the training loop revisits every
    sample every epoch.  Concatenating the samples with row and candidate
    offsets turns that into a handful of large, well-shaped operations, and the
    per-sample losses are then read off the segments of the resulting score
    vector.
    """

    def __init__(self, torch, parts, device):
        parts = list(parts)
        td_off, o_off, k_off = [], [], []
        td_total = o_total = k_total = 0
        for part in parts:
            td_off.append(td_total)
            o_off.append(o_total)
            k_off.append(k_total)
            td_total += part.td_tokens.shape[0]
            o_total += part.o_tokens.shape[0]
            k_total += part.K

        self.K = k_total
        self.segments = []
        for part, to, oo, ko in zip(parts, td_off, o_off, k_off):
            self.segments.append((ko, ko + part.K, part.target_index + ko,
                                  part.coverage))
        self.td_tokens = torch.cat([p.td_tokens for p in parts], dim=0)
        self.o_tokens = torch.cat([p.o_tokens for p in parts], dim=0)
        self.td_row = torch.cat([p.td_row + ko for p, ko in zip(parts, k_off)])
        self.o_row = torch.cat([p.o_row + ko for p, ko in zip(parts, k_off)])
        self.corr_td_row = torch.cat([p.corr_td_row + to
                                      for p, to in zip(parts, td_off)])
        self.corr_o_row = torch.cat([p.corr_o_row + oo
                                     for p, oo in zip(parts, o_off)])
        self.corr_w = torch.cat([p.corr_w for p in parts])
        self.coverage = float(np.mean([p.coverage for p in parts]))
        self.size = len(parts)

    def gather(self, tokens, index):
        return tokens[index].float()


class Inputs:
    """Per-sample pairing tensors, cached on the device."""

    def __init__(self, torch, sample, device, shuffle_seed=None):
        a = sample.arrays
        self.K = len(a["entity_ids"])
        self.target_index = sample.target_index
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

# Every frozen component a method could ask for, cached on the device once per
# sample.  Re-uploading them is the training loop's real cost -- a sample's
# patch tokens are a few megabytes and every epoch would move them again -- and
# holding the whole set costs about a megabyte per sample.
ALL_FEATURES = ("td_tight_masked", "oblique_tight_masked", "td_context_masked",
                "oblique_context_masked", "td_context_background",
                "oblique_context_background", "td_exclusive", "oblique_exclusive",
                "td_context_global", "oblique_context_global", "geometry")
_FEATURE_CACHE: dict = {}
_TEXT_CACHE: dict = {}


def cached_features(torch, sample, device):
    if sample.key not in _FEATURE_CACHE:
        _FEATURE_CACHE[sample.key] = {
            k: to_device(torch, sample.arrays[k], device, torch.float32)
            for k in ALL_FEATURES}
    return _FEATURE_CACHE[sample.key]


def cached_text(torch, sample, device, variant):
    key = (sample.key, variant)
    if key not in _TEXT_CACHE:
        row = to_device(torch, sample.arrays[f"text_{variant}"], device, torch.float32)
        _TEXT_CACHE[key] = row[None, :].expand(len(sample.entity_ids), -1).contiguous()
    return _TEXT_CACHE[key]


def batch_tensors(torch, chunk, device, variant, wanted):
    columns = {k: torch.cat([cached_features(torch, s, device)[k] for s in chunk],
                            dim=0) for k in wanted}
    text = torch.cat([cached_text(torch, s, device, variant) for s in chunk], dim=0)
    return columns, text


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

    wanted = sorted(set(spec["components"]) | {"geometry"})

    def run_pass(samples, train: bool):
        module.train(train)
        order = np.random.permutation(len(samples)) if train else np.arange(len(samples))
        entries, total, count = [], 0.0, 0
        for start in range(0, len(order), BATCH_SAMPLES):
            chunk = [samples[i] for i in order[start:start + BATCH_SAMPLES]]
            inputs = BatchInputs(torch, [get_inputs(torch, s, device, shuffle_seed,
                                                    cache) for s in chunk], device)
            columns, text = batch_tensors(torch, chunk, device, args.variant, wanted)
            if train:
                opt.zero_grad(set_to_none=True)
            scores, reverse, joint = module(columns, text, inputs,
                                            want_reverse=spec.get("reverse", False))
            loss = 0.0
            for a, b, target, _ in inputs.segments:
                loss = loss + torch.nn.functional.cross_entropy(
                    (logit_scale.exp() * scores[a:b])[None, :],
                    torch.tensor([target - a], device=device))
                if reverse is not None:
                    rev_scores, joint_rev = reverse
                    loss = loss + torch.nn.functional.cross_entropy(
                        (logit_scale.exp() * rev_scores[a:b])[None, :],
                        torch.tensor([target - a], device=device))
                    if cross_weight > 0:
                        logits = joint[a:b] @ joint_rev[a:b].T * logit_scale.exp()
                        labels = torch.arange(b - a, device=device)
                        loss = loss + cross_weight * 0.5 * (
                            torch.nn.functional.cross_entropy(logits, labels)
                            + torch.nn.functional.cross_entropy(logits.T, labels))
            loss = loss / len(chunk)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params + [logit_scale], 5.0)
                opt.step()
            total += float(loss.detach()) * len(chunk)
            count += len(chunk)
            if not train:
                flat = scores.detach().cpu().numpy()
                for a, b, target, coverage in inputs.segments:
                    entry = rank_metrics(flat[a:b], target - a)
                    entry["coverage"] = coverage
                    entries.append(entry)
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
                  shuffle_seed=None, batch=BATCH_SAMPLES * 4):
    """Score every sample, batched the same way the training pass is."""
    module.eval()
    entries = []
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start:start + batch]
            inputs = BatchInputs(torch, [get_inputs(torch, s, device, shuffle_seed,
                                                    cache) for s in chunk], device)
            wanted = sorted(set(spec["components"]) | {"geometry"})
            columns, text = batch_tensors(torch, chunk, device, args.variant, wanted)
            scores, _, _ = module(columns, text, inputs, want_reverse=False)
            flat = scores.detach().cpu().numpy()
            # Segments are built in the order the samples were handed in, so the
            # key travels alongside the slice rather than through the tensor.
            for sample, (a, b, target, _) in zip(chunk, inputs.segments):
                entry = rank_metrics(flat[a:b], target - a)
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
