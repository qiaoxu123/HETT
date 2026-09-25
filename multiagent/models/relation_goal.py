"""Per-landmark spatial-relation scorer used as a HETT goal-logit branch."""

import math

import torch
import torch.nn as nn


def rotate_270(delta):
    return torch.stack((delta[..., 1], -delta[..., 0]), dim=-1)


def candidate_landmark_geometry(candidates, centers, boundaries, valid):
    """Return independent candidate-to-landmark geometry [B,K,N,10]."""
    centroid_delta = rotate_270(candidates[:, :, None] - centers[:, None])
    centroid_distance = torch.linalg.vector_norm(
        centroid_delta, dim=-1).clamp_min(1e-6)
    centroid = torch.cat((
        centroid_delta,
        (torch.log1p(centroid_distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (centroid_delta[..., 1] / centroid_distance).unsqueeze(-1),
        (centroid_delta[..., 0] / centroid_distance).unsqueeze(-1),
    ), dim=-1)

    boundary_delta = candidates[:, :, None, None] - boundaries[:, None]
    boundary_distance = torch.linalg.vector_norm(boundary_delta, dim=-1)
    nearest = boundary_distance.argmin(-1)
    gather = nearest[..., None, None].expand(-1, -1, -1, 1, 2)
    nearest_delta = rotate_270(boundary_delta.gather(3, gather).squeeze(3))
    nearest_distance = torch.linalg.vector_norm(
        nearest_delta, dim=-1).clamp_min(1e-6)
    boundary = torch.cat((
        nearest_delta,
        (torch.log1p(nearest_distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (nearest_delta[..., 1] / nearest_distance).unsqueeze(-1),
        (nearest_delta[..., 0] / nearest_distance).unsqueeze(-1),
    ), dim=-1)
    return torch.cat((centroid, boundary), dim=-1) * valid[:, None, :, None]


class RelationGoalRanker(nn.Module):
    """Checkpoint-compatible form of the accepted offline relation ranker."""

    def __init__(self, hidden=128, cue_dim=12):
        super().__init__()
        self.relation = nn.Sequential(
            nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.geometry = nn.Sequential(
            nn.Linear(10, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        self.cue = nn.Sequential(
            nn.Linear(cue_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.LayerNorm(hidden), nn.SiLU())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, candidates, centers, boundaries, valid, relation, cue,
                use_relation=True):
        geometry = candidate_landmark_geometry(
            candidates.float(), centers.float(), boundaries.float(), valid)
        batch, candidate_count, landmark_count = geometry.shape[:3]
        if not use_relation:
            relation = torch.zeros_like(relation)
            cue = torch.zeros_like(cue)
        language = self.relation(relation.float()) + self.cue(cue.float())
        language = language[:, None].expand(
            batch, candidate_count, landmark_count, -1)
        spatial = self.geometry(geometry)
        token = self.fuse(torch.cat((
            language, spatial, language * spatial,
            torch.abs(language - spatial)), dim=-1))
        token = token.reshape(batch * candidate_count, landmark_count, -1)
        cls = self.cls.expand(batch * candidate_count, -1, -1)
        sequence = torch.cat((cls, token), dim=1)
        padding = torch.cat((
            torch.zeros(batch, 1, dtype=torch.bool, device=valid.device),
            ~valid.bool()), dim=1)
        padding = padding[:, None].expand(
            batch, candidate_count, landmark_count + 1).reshape(
                batch * candidate_count, landmark_count + 1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return self.score(encoded[:, 0]).reshape(batch, candidate_count)


def rms_normalize(score):
    centered = score - score.mean(-1, keepdim=True)
    return centered / centered.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


def merge_goal_logits(hett_logits, relation_logits, alpha, relation_only=False):
    """Fuse goal scores, or remove the HETT target prior for the strict ablation."""
    if relation_only:
        return relation_logits
    hett_scale = (hett_logits - hett_logits.mean(-1, keepdim=True)).square(
        ).mean(-1, keepdim=True).sqrt().clamp_min(0.1)
    return hett_logits + alpha * hett_scale * relation_logits
