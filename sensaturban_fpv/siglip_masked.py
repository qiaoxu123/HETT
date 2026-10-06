"""Frozen SigLIP2, used at patch-token resolution instead of whole-image only.

SigLIP2 pools with a learned attention probe: ``head.attention(probe, tokens,
tokens)`` followed by a residual MLP.  That probe is a *query*, not a learned
average, so it can be pointed at a subset of the tokens by passing a
``key_padding_mask`` -- the same pooling the checkpoint was trained with, over
the patches a landmark actually occupies, with **no new parameters and no
fine-tuning**.  The result lands in the same embedding space as
``get_image_features``, so the top-down and oblique masked features are directly
comparable to each other and to the plain whole-crop features.

Every method in this round is read out through this one frozen path.  A fused
patch token is pushed through the same head as a one-token sequence, so the
fusion, the masked single views and the whole-image baselines all end in the
identical space and are scored by the identical cosine.  Any difference between
them is therefore a difference of *features*, not of readout.
"""

from __future__ import annotations

import numpy as np


def vision_tokens(model, pixel_values):
    """Post-layernorm patch tokens, the pooled feature, and the patch grid size.

    The grid is derived from the token count rather than assumed: the
    checkpoint's input resolution and patch size decide it, and a hard-coded 32
    would silently mis-map every mask if either changed.
    """
    import torch

    vision = model.vision_model
    with torch.no_grad():
        embeds = vision.embeddings(pixel_values)
        encoded = vision.encoder(embeds).last_hidden_state
        tokens = vision.post_layernorm(encoded)
        pooled = vision.head(tokens)
    n = tokens.shape[1]
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f"{n} patch tokens do not form a square grid")
    return tokens, pooled, side


def encode_crops(processor, model, images, device, batch: int = 8):
    """Batch-encode crops into ``(tokens, pooled, grid)``.

    ``pooled`` is bit-identical to ``get_image_features``: that method returns
    ``vision_outputs[1]``, which is this same head applied to these same tokens.
    Computing both from one forward pass therefore costs nothing extra and
    guarantees the whole-crop baseline and the masked features share a forward.
    """
    import torch

    all_tokens, all_pooled, grid = [], [], None
    for i in range(0, len(images), batch):
        inputs = processor(images=images[i:i + batch], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        tokens, pooled, grid = vision_tokens(model, pixel_values)
        all_tokens.append(tokens.float().cpu())
        all_pooled.append(pooled.float().cpu())
    if not all_tokens:
        raise ValueError("no images to encode")
    return torch.cat(all_tokens, dim=0), torch.cat(all_pooled, dim=0), grid


def masked_pool(model, tokens, keep):
    """Pool patch tokens through the frozen head, attending only to ``keep``.

    ``tokens`` is ``(B, N, D)`` and ``keep`` is a boolean ``(B, N)``.  A row with
    nothing kept falls back to unmasked pooling, and the caller is expected to
    have excluded such candidates already: a masked feature over an empty set is
    not a feature of anything.
    """
    import torch

    head = model.vision_model.head
    tokens = tokens.to(device=head.probe.device, dtype=head.probe.dtype)
    keep = keep.to(tokens.device)
    any_kept = keep.any(dim=1)
    if not bool(any_kept.all()):
        keep = torch.where(any_kept[:, None], keep, torch.ones_like(keep))
    probe = head.probe.repeat(tokens.shape[0], 1, 1)
    pooled = head.attention(probe, tokens, tokens, key_padding_mask=~keep)[0]
    return pooled[:, 0] + head.mlp(head.layernorm(pooled))[:, 0]


def readout_tokens(model, token):
    """Push a single synthetic token through the frozen head, as ``get_image_features`` would.

    Attention over a one-token sequence returns that token unchanged whatever the
    probe is, so the head reduces here to its residual MLP; applying it is what
    puts a fused landmark token in the same space as a pooled image.
    """
    head = model.vision_model.head
    token = token.to(device=head.probe.device, dtype=head.probe.dtype)
    seq = token[:, None, :]
    pooled = head.attention(head.probe.repeat(token.shape[0], 1, 1), seq, seq)[0]
    return pooled[:, 0] + head.mlp(head.layernorm(pooled))[:, 0]


def gather_corresponding(tokens, row, col, weights):
    """Weighted mean of ``tokens`` over the patches each row points at.

    A vectorised gather over a sparse correspondence: ``row``/``col``/``weights``
    describe which source patches feed which destination patch.  Rows with no
    correspondence come back as ``nan`` and are the caller's cue to fall back --
    a patch of the landmark that the other view never saw has nothing to fuse
    with, and inventing a zero vector there would inject a constant into the
    feature.
    """
    import torch

    n_out = int(row.max().item()) + 1 if row.numel() else 0
    d = tokens.shape[-1]
    out = torch.full((n_out, d), float("nan"), dtype=tokens.dtype,
                     device=tokens.device)
    w = weights.to(tokens.dtype).to(tokens.device)
    gathered = tokens[col] * w[:, None]
    acc = torch.zeros((n_out, d), dtype=tokens.dtype, device=tokens.device)
    acc.index_add_(0, row, gathered)
    denom = torch.zeros(n_out, dtype=tokens.dtype, device=tokens.device)
    denom.index_add_(0, row, w)
    ok = denom > 0
    out[ok] = acc[ok] / denom[ok, None]
    return out, ok


def l2norm(x, dim: int = -1, eps: float = 1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def patch_mask_to_tokens(mask: np.ndarray) -> np.ndarray:
    """Flatten a ``(P, P)`` patch mask into the token order the encoder returns."""
    return np.asarray(mask, dtype=bool).reshape(-1)
