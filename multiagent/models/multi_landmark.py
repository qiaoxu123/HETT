"""Per-reference landmark geometry and language-grounded relation scoring.

Inference uses only matched CityRefer landmark contours, instruction text, and
current map features; it NEVER sees target coordinates or labels.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(maxsize=8192)
def _cached_landmark_token_ids(tokenizer, name):
    # Repeated teacher/student rollouts process the same landmark names.
    # Tokenizer vocabulary/normalization must be fixed during an experiment.
    return tuple(tokenizer.encode(name, add_special_tokens=False))


def build_landmark_batch(observations, tokenizer, instruction_ids, *, max_landmarks=16):
    """Align named landmarks to BERT instruction tokens, without extra BERT runs.

    observation['reference_landmarks'] preserves the name/contour pairing from
    LandmarkNavMap.referenced_landmark_map.landmarks (the processed-description
    order). Unmatched names remain geometric anchors but receive only global
    instruction context. Padding is never interpreted as a real anchor.
    """
    if max_landmarks < 1:
        raise ValueError("max_landmarks must be positive")
    if instruction_ids.ndim != 2 or len(observations) != instruction_ids.shape[0]:
        raise ValueError("instruction ids must have shape [batch, sequence]")
    batch, seq_len = instruction_ids.shape
    ids_cpu = instruction_ids.detach().cpu().tolist()
    processed = []
    truncated = 0
    for row, obs in enumerate(observations):
        anchors = []
        sequence = ids_cpu[row]
        for item in obs.get("reference_landmarks", ()):
            name = str(item["name"])
            name_ids = _cached_landmark_token_ids(tokenizer, name)
            start = -1
            if name_ids and len(name_ids) <= seq_len:
                start = next(
                    (i for i in range(seq_len - len(name_ids) + 1)
                     if sequence[i:i + len(name_ids)] == name_ids), -1
                )
            anchors.append((start, name_ids, item))
        # If an instruction mentions more landmarks than the configured cap,
        # keep earliest matched mentions before unmatched names.
        anchors.sort(key=lambda a: a[0] if a[0] >= 0 else seq_len + 1)
        truncated += max(0, len(anchors) - max_landmarks)
        processed.append(anchors[:max_landmarks])
    num = max(1, max((len(row) for row in processed), default=0))
    centers = torch.zeros(batch, num, 2, dtype=torch.float32)
    extents = torch.zeros_like(centers)
    valid = torch.zeros(batch, num, dtype=torch.bool)
    text_mask = torch.zeros(batch, num, seq_len, dtype=torch.bool)
    for b, row in enumerate(processed):
        for m, (start, ids, item) in enumerate(row):
            centers[b, m] = torch.as_tensor(item["center_xy"], dtype=torch.float32)
            extents[b, m] = torch.as_tensor(item["extent_xy"], dtype=torch.float32)
            valid[b, m] = bool(torch.isfinite(centers[b, m]).all())
            if start >= 0:
                text_mask[b, m, start:start + len(ids)] = True
    valid &= torch.isfinite(extents).all(dim=-1)
    centers = torch.nan_to_num(centers)
    extents = torch.nan_to_num(extents).clamp_min(0)
    return {
        "landmark_xy": centers.to(instruction_ids.device),
        "landmark_extent": extents.to(instruction_ids.device),
        "landmark_valid": valid.to(instruction_ids.device),
        "landmark_text_mask": text_mask.to(instruction_ids.device),
        "landmark_truncated": truncated,
    }


def candidate_landmark_geometry(centers, extents, *, field_size):
    """[B, H*W, M, 8], relative east/north bearing and landmark extents.

    The CityNav map is normalized east/south while metric world axes are
    east/north. Features remain scale-normalized to map width.
    """
    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError("landmark centers must be [B,M,2]")
    if extents.shape != centers.shape:
        raise ValueError("landmark extents shape mismatch")
    if field_size < 1:
        raise ValueError("field_size must be positive")
    axis = (torch.arange(field_size, device=centers.device, dtype=centers.dtype) + 0.5) / field_size
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    cell_xy = torch.stack((cols.flatten(), rows.flatten()), dim=-1)
    delta = cell_xy[None, :, None, :] - centers[:, None, :, :]
    dx = delta[..., 0]
    dy_north = -delta[..., 1]
    range_norm = torch.sqrt(dx.square() + dy_north.square())
    safe_dist = range_norm.clamp_min(1e-6)
    sx = extents[:, None, :, 0].expand_as(dx)
    sy = extents[:, None, :, 1].expand_as(dx)
    return torch.stack((
        dx, dy_north, range_norm,
        dy_north / safe_dist, dx / safe_dist,
        sx, sy,
        (dx.abs() <= sx / 2).logical_and(dy_north.abs() <= sy / 2).to(dx.dtype),
    ), dim=-1)


class MultiLandmarkRelationHead(nn.Module):
    """Jointly evaluate all named landmark-candidate relations.

    Contextual BERT embeddings of each matched landmark name distinguish its
    identity, the candidate-relative bearings encode its geometry, and a
    small self-attention layer exchanges evidence among all referenced anchors.
    A masked mean plus differentiable soft minimum discourage single-landmark
    shortcuts (e.g., satisfying "east of A" but not "north of B").
    """

    def __init__(self, *, map_dim=256, language_dim=768, hidden_dim=96,
                 attention_heads=4, dropout=0.1):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must divide attention_heads")
        self.map_proj = nn.Linear(map_dim, hidden_dim)
        self.name_proj = nn.Linear(language_dim, hidden_dim)
        self.instruction_proj = nn.Linear(language_dim, hidden_dim)
        self.geometry_proj = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.cross_landmark = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.relation_score = nn.Linear(hidden_dim, 1)
        self.joint_score = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, spatial_features, language_tokens, landmark_xy,
                landmark_extent, landmark_valid, landmark_text_mask,
                language_mask=None):
        if spatial_features.ndim != 4:
            raise ValueError("spatial_features must be [B,D,H,W]")
        batch, channels, height, width = spatial_features.shape
        if height != width:
            raise ValueError("dense belief field must be square")
        if landmark_xy.ndim != 3 or landmark_xy.shape[:2] != landmark_valid.shape or landmark_xy.shape[-1] != 2:
            raise ValueError("landmark dimensions mismatch")
        if landmark_extent.shape != landmark_xy.shape:
            raise ValueError("landmark extent dimensions mismatch")
        if landmark_text_mask.shape != (batch, landmark_xy.shape[1], language_tokens.shape[1]):
            raise ValueError("landmark token mask dimensions mismatch")
        if language_tokens.shape[0] != batch:
            raise ValueError("language batch mismatch")
        if landmark_valid.dtype != torch.bool:
            landmark_valid = landmark_valid.bool()

        num_cells, num_landmarks = height * width, landmark_xy.shape[1]
        if num_landmarks == 0 or not landmark_valid.any():
            return spatial_features.new_zeros(batch, height, width)

        lang = language_tokens
        if language_mask is None:
            language_mask = torch.ones(lang.shape[:2], device=lang.device, dtype=torch.bool)
        tokens = landmark_text_mask.bool() & language_mask[:, None, :].bool()
        token_weight = tokens.to(lang.dtype)
        name_vectors = torch.einsum("bml,bld->bmd", token_weight, lang)
        name_vectors = name_vectors / token_weight.sum(-1, keepdim=True).clamp_min(1)
        global_weight = language_mask.to(lang.dtype)
        global_context = ((lang * global_weight[..., None]).sum(1)
                          / global_weight.sum(1, keepdim=True).clamp_min(1))
        # Unmatched name gets global context instead of a falsely assigned token.
        name_vectors = torch.where(tokens.any(-1)[..., None],
                                   name_vectors, global_context[:, None, :])
        name_vectors = self.name_proj(name_vectors)
        global_context = self.instruction_proj(global_context)[:, None, None, :]

        geometry = candidate_landmark_geometry(
            landmark_xy.to(spatial_features.dtype),
            landmark_extent.to(spatial_features.dtype),
            field_size=height,
        )
        spatial = spatial_features.flatten(2).transpose(1, 2)
        map_tokens = self.map_proj(spatial)
        evidence = self.norm(
            self.geometry_proj(geometry) + name_vectors[:, None, :, :]
            + map_tokens[:, :, None, :] + global_context
        )
        flat = evidence.reshape(batch * num_cells, num_landmarks, -1)
        present = landmark_valid[:, None, :].expand(batch, num_cells, num_landmarks)
        present_flat = present.reshape(batch * num_cells, num_landmarks)
        # MultiheadAttention cannot process a sequence with all keys masked.
        safe = present_flat.clone()
        safe[~safe.any(dim=-1), 0] = True
        exchanged, _ = self.cross_landmark(
            flat, flat, flat, key_padding_mask=~safe, need_weights=False
        )
        mixed = self.norm(flat + exchanged)
        anchor_scores = self.relation_score(mixed).squeeze(-1)
        active = present_flat.to(anchor_scores.dtype)
        counts = active.sum(-1).clamp_min(1)
        average = (anchor_scores * active).sum(-1) / counts
        temperature = 0.5
        minimum = -temperature * (
            torch.logsumexp(
                (-anchor_scores / temperature).masked_fill(~safe, -torch.inf),
                dim=-1,
            ) - counts.log()
        )
        minimum = torch.where(present_flat.any(-1), minimum,
                              torch.zeros_like(minimum))
        pooled = (mixed * active[..., None]).sum(1) / counts[:, None]
        flat_map = map_tokens.reshape(batch * num_cells, -1)
        combined = torch.cat((flat_map, pooled, average[:, None], minimum[:, None]), -1)
        score = self.joint_score(combined).reshape(batch, height, width)
        return score * landmark_valid.any(dim=1).to(score.dtype)[:, None, None]
