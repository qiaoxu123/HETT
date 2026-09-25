"""Minimal unified language/landmark/BEV belief transformer.

The model deliberately has no target-coordinate argument.  Target positions are
used by the training code only to construct supervision and evaluation masks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_centers(grid_size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    axis = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / grid_size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def target_mask(target_xy: torch.Tensor, grid_size: int, radius_m: float = 20.0,
                map_size_m: float = 410.0) -> torch.Tensor:
    """Rasterize a metric-radius target disk; labels only, never model input."""
    cells = grid_centers(grid_size, device=target_xy.device, dtype=target_xy.dtype)
    distance = torch.linalg.vector_norm(
        cells[None] - target_xy[:, None, None], dim=-1) * map_size_m
    return distance.le(radius_m)


def rasterize_landmarks(boundaries: torch.Tensor, valid: torch.Tensor,
                        grid_size: int) -> torch.Tensor:
    """Rasterize landmark polygons as instance-preserving masks without OpenCV."""
    batch, landmarks, points = boundaries.shape[:3]
    cells = grid_centers(grid_size, device=boundaries.device, dtype=boundaries.dtype)
    x = cells[..., 0][None, None, :, :, None]
    y = cells[..., 1][None, None, :, :, None]
    p1 = boundaries[:, :, None, None]
    p2 = torch.roll(p1, shifts=-1, dims=4)
    x1, y1 = p1[..., 0], p1[..., 1]
    x2, y2 = p2[..., 0], p2[..., 1]
    # Use a signed epsilon only for horizontal edges in the ray-casting test.
    denominator = y2 - y1
    denominator = torch.where(
        denominator.abs() < 1e-7,
        torch.where(denominator < 0, -torch.full_like(denominator, 1e-7),
                    torch.full_like(denominator, 1e-7)),
        denominator)
    crossing = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / denominator + x1)
    inside = (crossing.sum(-1) % 2 == 1)
    return inside & valid[:, :, None, None]


def landmark_shape_features(centers: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    minimum = boundaries.amin(-2)
    maximum = boundaries.amax(-2)
    size = maximum - minimum
    shifted = torch.roll(boundaries, shifts=-1, dims=-2)
    area = 0.5 * torch.abs(
        (boundaries[..., 0] * shifted[..., 1]
         - boundaries[..., 1] * shifted[..., 0]).sum(-1, keepdim=True))
    return torch.cat((centers, minimum, maximum, size, area), dim=-1)


def grid_landmark_geometry(cell_xy: torch.Tensor, centers: torch.Tensor,
                           boundaries: torch.Tensor) -> torch.Tensor:
    """Explicit cell-to-landmark geometry in the audited language frame."""
    delta = cell_xy[None, :, None] - centers[:, None]
    delta = torch.stack((delta[..., 1], -delta[..., 0]), dim=-1)
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    centroid = torch.cat((
        delta,
        (torch.log1p(distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (delta[..., 1] / distance).unsqueeze(-1),
        (delta[..., 0] / distance).unsqueeze(-1),
    ), dim=-1)

    boundary_delta = cell_xy[None, :, None, None] - boundaries[:, None]
    boundary_distance = torch.linalg.vector_norm(boundary_delta, dim=-1)
    nearest = boundary_distance.argmin(-1)
    gather = nearest[..., None, None].expand(-1, -1, -1, 1, 2)
    nearest_delta = boundary_delta.gather(3, gather).squeeze(3)
    nearest_delta = torch.stack(
        (nearest_delta[..., 1], -nearest_delta[..., 0]), dim=-1)
    nearest_distance = torch.linalg.vector_norm(nearest_delta, dim=-1).clamp_min(1e-6)
    boundary = torch.cat((
        nearest_delta,
        (torch.log1p(nearest_distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (nearest_delta[..., 1] / nearest_distance).unsqueeze(-1),
        (nearest_delta[..., 0] / nearest_distance).unsqueeze(-1),
    ), dim=-1)
    return torch.cat((centroid, boundary), dim=-1)


class BeliefTransformer(nn.Module):
    """Fuse per-landmark language/geometry tokens with a BEV token grid."""

    def __init__(self, grid_size: int = 20, relation_dim: int = 768,
                 cue_dim: int = 12, hidden: int = 96, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.relation = nn.Sequential(
            nn.Linear(relation_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.cue = nn.Sequential(
            nn.Linear(cue_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.landmark_geometry = nn.Sequential(
            nn.Linear(9, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.landmark_mask = nn.Sequential(
            nn.Linear(grid_size * grid_size, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.global_language = nn.Sequential(
            nn.Linear(relation_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.map_token = nn.Sequential(
            nn.Linear(4, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.relative_geometry = nn.Sequential(
            nn.Linear(10, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False)
        self.goal_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.pair_score = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        self.relation_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("cell_xy", grid_centers(grid_size).reshape(-1, 2))

    def forward(self, relation_features: torch.Tensor, relation_cues: torch.Tensor,
                landmark_centers: torch.Tensor, landmark_boundaries: torch.Tensor,
                landmark_valid: torch.Tensor, landmark_masks: torch.Tensor,
                map_bev: torch.Tensor) -> torch.Tensor:
        """Return ``[B,H,W]`` goal logits from inference-visible inputs only."""
        batch, landmarks = landmark_valid.shape
        valid_float = landmark_valid[..., None].float()
        count = valid_float.sum(1).clamp_min(1.0)
        global_relation = (relation_features.float() * valid_float).sum(1) / count
        global_token = self.global_language(global_relation)[:, None]

        geometry = landmark_shape_features(
            landmark_centers.float(), landmark_boundaries.float())
        landmark_token = (
            self.relation(relation_features.float())
            + self.cue(relation_cues.float())
            + self.landmark_geometry(geometry)
            + self.landmark_mask(landmark_masks.float().flatten(-2)))

        xy = self.cell_xy.to(map_bev).expand(batch, -1, -1)
        bev = map_bev.float().flatten(2).transpose(1, 2)
        map_token = self.map_token(torch.cat((xy, bev), dim=-1))
        sequence = torch.cat((global_token, landmark_token, map_token), dim=1)
        padding = torch.cat((
            torch.zeros(batch, 1, dtype=torch.bool, device=sequence.device),
            ~landmark_valid,
            torch.zeros(batch, self.grid_size ** 2, dtype=torch.bool,
                        device=sequence.device)), dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        encoded_landmarks = encoded[:, 1:1 + landmarks]
        grid = encoded[:, 1 + landmarks:]
        relative = self.relative_geometry(grid_landmark_geometry(
            self.cell_xy.to(landmark_centers), landmark_centers.float(),
            landmark_boundaries.float()))
        grid_pair = grid[:, :, None].expand(-1, -1, landmarks, -1)
        landmark_pair = encoded_landmarks[:, None].expand(
            -1, self.grid_size ** 2, -1, -1)
        pair = self.pair_score(torch.cat((
            grid_pair, landmark_pair, relative, grid_pair * landmark_pair), dim=-1)
        ).squeeze(-1)
        pair = pair.masked_fill(~landmark_valid[:, None], 0.0)
        relation_score = pair.sum(-1) / landmark_valid.sum(-1, keepdim=True).clamp_min(1)
        logits = (self.goal_head(grid).squeeze(-1)
                  + F.softplus(self.relation_scale) * relation_score)
        return logits.reshape(
            batch, self.grid_size, self.grid_size)


class TemporalBeliefTransformer(nn.Module):
    """Update a static belief with visibility-gated RGB semantic evidence."""

    def __init__(self, static_model: BeliefTransformer, semantic_channels: int = 5,
                 hidden: int = 64, maximum_evidence: float = 4.0):
        super().__init__()
        self.static_model = static_model
        self.maximum_evidence = float(maximum_evidence)
        self.semantic = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1))
        self.state = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1))
        self.language = nn.Sequential(
            nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.cue = nn.Sequential(
            nn.Linear(12, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.target_type = nn.Linear(hidden, 4)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.evidence_head = nn.Conv2d(hidden, 1, 1)
        self.quality_head = nn.Conv2d(hidden, 1, 1)

    def forward(self, static_inputs, semantic_bev: torch.Tensor,
                visibility: torch.Tensor, type_roll: int = 0):
        """Return initial and per-frame belief logits plus signed evidence.

        ``semantic_bev`` is ``[B,T,K,H,W]`` and ``visibility`` is
        ``[B,T,H,W]``. Evidence is exactly zero outside visible cells.
        """
        relation, cue = static_inputs[:2]
        valid = static_inputs[4]
        weight = valid[..., None].float()
        count = weight.sum(1).clamp_min(1.0)
        language = self.language(
            (relation.float() * weight).sum(1) / count)
        cue_weight = valid[..., None].float()
        cue_context = (cue.float() * cue_weight).sum(1) / count
        language = language + self.cue(cue_context)
        context = language[:, :, None, None]
        type_logits = self.target_type(language)
        type_probability = type_logits.softmax(-1)
        if type_roll:
            type_probability = torch.roll(type_probability, shifts=type_roll, dims=-1)

        initial = self.static_model(*static_inputs)
        belief = initial
        beliefs, raw_evidence, applied_evidence, qualities = [], [], [], []
        for step in range(semantic_bev.shape[1]):
            visible = visibility[:, step].float()
            observation = self.semantic(semantic_bev[:, step].float())
            target_match = torch.einsum(
                "bk,bkhw->bhw", type_probability,
                semantic_bev[:, step, 1:5].float())
            state = self.state(torch.stack(
                (belief.sigmoid(), visible, target_match), dim=1))
            feature = self.fuse(observation + state + context)
            raw = torch.tanh(self.evidence_head(feature).squeeze(1)) * self.maximum_evidence
            quality = torch.sigmoid(self.quality_head(feature).squeeze(1))
            applied = raw * quality * visible
            belief = (belief + applied).clamp(-10.0, 10.0)
            beliefs.append(belief)
            raw_evidence.append(raw)
            applied_evidence.append(applied)
            qualities.append(quality)
        return {
            "initial_logits": initial,
            "belief_logits": torch.stack(beliefs, dim=1),
            "raw_evidence": torch.stack(raw_evidence, dim=1),
            "applied_evidence": torch.stack(applied_evidence, dim=1),
            "quality": torch.stack(qualities, dim=1),
            "target_type_logits": type_logits,
        }


def belief_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Distribution loss plus a balanced 20 m region loss."""
    flat_logits = logits.flatten(1)
    flat_labels = labels.float().flatten(1)
    distribution = flat_labels / flat_labels.sum(-1, keepdim=True).clamp_min(1.0)
    listwise = -(distribution * F.log_softmax(flat_logits, dim=-1)).sum(-1).mean()
    positive = flat_labels.sum()
    negative = flat_labels.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).detach().clamp(max=30.0)
    region = F.binary_cross_entropy_with_logits(
        flat_logits, flat_labels, pos_weight=pos_weight)
    return listwise + 0.2 * region


def target_log_probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probability = F.log_softmax(logits.flatten(1), dim=-1)
    weights = labels.float().flatten(1)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0)
    return (probability * weights).sum(-1)


def contrastive_input_loss(correct_logits: torch.Tensor, perturbed_logits: torch.Tensor,
                           labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    correct = target_log_probability(correct_logits, labels)
    perturbed = target_log_probability(perturbed_logits, labels)
    return F.softplus(margin - correct + perturbed).mean()
