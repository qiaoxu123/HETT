"""Permutation-equivariant verifier for two path-visible goal hypotheses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PathLocalCandidateVerifier(nn.Module):
    """Fuse full-instruction RGB matching with per-landmark relation geometry."""

    def __init__(self, vision_dim: int, relation_dim: int = 768, hidden: int = 96):
        super().__init__()
        self.image = nn.Sequential(
            nn.Linear(vision_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.text = nn.Sequential(
            nn.Linear(vision_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.visual_pair = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.geometry = nn.Sequential(
            nn.Linear(10, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.relation = nn.Sequential(
            nn.Linear(relation_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.relation_pair = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.scalar = nn.Sequential(
            nn.Linear(3, hidden // 2), nn.LayerNorm(hidden // 2), nn.SiLU())
        self.score = nn.Sequential(
            nn.Linear(hidden * 2 + hidden // 2, hidden), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, image, text, geometry, relation, landmark_valid,
                prior, semantic_evidence):
        # No rank embedding or candidate positional encoding: swapping candidate
        # order must swap scores and nothing else.
        image_token = self.image(image.float())
        text_token = self.text(text.float())[:, None].expand_as(image_token)
        visual = self.visual_pair(torch.cat((
            image_token * text_token,
            torch.abs(image_token - text_token)), dim=-1))

        geometry_token = self.geometry(geometry.float())
        relation_token = self.relation(relation.float())[:, None].expand_as(
            geometry_token)
        per_landmark = self.relation_pair(torch.cat((
            geometry_token * relation_token,
            torch.abs(geometry_token - relation_token)), dim=-1))
        valid = landmark_valid[:, None, :, None].float()
        relation_summary = (per_landmark * valid).sum(2) / valid.sum(2).clamp_min(1)

        separation = torch.linalg.vector_norm(
            geometry[:, 0, 0, :2] - geometry[:, 1, 0, :2], dim=-1)
        separation = separation[:, None].expand_as(prior)
        scalar = self.scalar(torch.stack((
            prior.float(), semantic_evidence.float(), separation.float()), -1))
        return self.score(torch.cat((visual, relation_summary, scalar), -1)).squeeze(-1)


class LandmarkAwareCandidateVerifier(nn.Module):
    """Compare each candidate with every independently bound landmark patch."""

    def __init__(self, vision_dim: int, relation_dim: int = 768, hidden: int = 96):
        super().__init__()
        self.candidate_image = nn.Sequential(
            nn.Linear(vision_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.landmark_image = nn.Sequential(
            nn.Linear(vision_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.text = nn.Sequential(
            nn.Linear(vision_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.candidate_text = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.geometry = nn.Sequential(
            nn.Linear(10, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.relation = nn.Sequential(
            nn.Linear(relation_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.landmark_fuse = nn.Sequential(
            nn.Linear(hidden * 6 + 1, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.scalar = nn.Sequential(
            nn.Linear(3, hidden // 2), nn.LayerNorm(hidden // 2), nn.SiLU())
        self.score = nn.Sequential(
            nn.Linear(hidden * 2 + hidden // 2, hidden), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, candidate_image, landmark_image, text, geometry, relation,
                landmark_valid, landmark_visual_valid, prior, semantic_evidence):
        candidate = self.candidate_image(candidate_image.float())
        landmark = self.landmark_image(landmark_image.float())
        instruction = self.text(text.float())[:, None].expand_as(candidate)
        candidate_text = self.candidate_text(torch.cat((
            candidate * instruction, torch.abs(candidate - instruction)), -1))

        geometry_token = self.geometry(geometry.float())
        relation_token = self.relation(relation.float())[:, None].expand_as(
            geometry_token)
        candidate_token = candidate[:, :, None].expand_as(geometry_token)
        landmark_token = landmark[:, None].expand_as(geometry_token)
        visible = landmark_visual_valid[:, None, :, None].float().expand(
            -1, geometry_token.shape[1], -1, -1)
        per_landmark = self.landmark_fuse(torch.cat((
            geometry_token * relation_token,
            torch.abs(geometry_token - relation_token),
            candidate_token * landmark_token,
            torch.abs(candidate_token - landmark_token),
            landmark_token * relation_token,
            torch.abs(landmark_token - relation_token),
            visible), -1))
        valid = landmark_valid[:, None, :, None].float()
        relation_summary = (per_landmark * valid).sum(2) / valid.sum(2).clamp_min(1)

        # Candidate separation is repeated relative to each landmark. Average
        # across valid slots so the scalar cannot depend on landmark order.
        per_landmark_separation = torch.linalg.vector_norm(
            geometry[:, 0, :, :2] - geometry[:, 1, :, :2], dim=-1)
        valid_slots = landmark_valid.float()
        separation = (per_landmark_separation * valid_slots).sum(-1) / (
            valid_slots.sum(-1).clamp_min(1))
        separation = separation[:, None].expand_as(prior)
        scalar = self.scalar(torch.stack((
            prior.float(), semantic_evidence.float(), separation.float()), -1))
        return self.score(torch.cat((candidate_text, relation_summary, scalar), -1)).squeeze(-1)


def soft_distance_ranking_loss(scores, distances_m, temperature_m=20.0):
    target = F.softmax(-distances_m.float() / temperature_m, dim=-1)
    # Ambiguous rows remain useful but should not dominate rows where one mode
    # is clearly inside the 20 m success radius and the other is not.
    success = distances_m.le(20.0)
    discriminative = success[:, 0].logical_xor(success[:, 1]).float()
    weight = 0.25 + 0.75 * discriminative
    loss = -(target * F.log_softmax(scores, dim=-1)).sum(-1)
    return (loss * weight).sum() / weight.sum().clamp_min(1)
