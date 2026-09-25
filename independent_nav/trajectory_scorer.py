"""Target-free scoring of multiple navigation trajectory hypotheses."""

from __future__ import annotations

import torch
from torch import nn


class TrajectoryHypothesisScorer(nn.Module):
    """Score K paths from belief, budget and per-landmark relations.

    No candidate-rank embedding is used.  Candidate permutation therefore
    permutes scores, while landmark permutation leaves scores unchanged.
    """

    def __init__(self, relation_dim=768, cue_dim=12, route_channels=5,
                 belief_dim=4, hidden=96, dropout=0.1):
        super().__init__()
        self.relation = nn.Sequential(
            nn.Linear(relation_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.cue = nn.Sequential(
            nn.Linear(cue_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.geometry = nn.Sequential(
            nn.Linear(10, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.landmark_pair = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.route = nn.Sequential(
            nn.Conv1d(route_channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.belief = nn.Sequential(
            nn.Linear(belief_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.SiLU())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.candidate_context = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False)
        self.score = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))

    def encode(self, relation, cue, geometry, landmark_valid,
               trajectory, belief_features):
        batch, candidates, landmarks = geometry.shape[:3]
        language = (self.relation(relation.float())
                    + self.cue(cue.float()))[:, None].expand(
                        batch, candidates, landmarks, -1)
        spatial = self.geometry(geometry.float())
        pair = self.landmark_pair(torch.cat((
            language, spatial, language * spatial,
            torch.abs(language - spatial)), -1))
        valid = landmark_valid[:, None, :, None].float()
        relation_summary = (pair * valid).sum(2) / valid.sum(2).clamp_min(1)

        sequence = trajectory.float().reshape(
            batch * candidates, trajectory.shape[2], trajectory.shape[3])
        route = self.route(sequence.transpose(1, 2)).transpose(1, 2)
        route_valid = sequence[..., -1:].clamp(0, 1)
        route = (route * route_valid).sum(1) / route_valid.sum(1).clamp_min(1)
        route = route.reshape(batch, candidates, -1)
        belief = self.belief(belief_features.float())
        token = self.fuse(torch.cat((relation_summary, route, belief), -1))
        return self.candidate_context(token)

    def forward(self, relation, cue, geometry, landmark_valid,
                trajectory, belief_features):
        return self.score(self.encode(
            relation, cue, geometry, landmark_valid,
            trajectory, belief_features)).squeeze(-1)


class MultiObjectiveTrajectoryScorer(TrajectoryHypothesisScorer):
    """Separate endpoint correctness from target-free observation utility."""

    def __init__(self, *args, utility_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.score[1].in_features
        self.utility_score = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        self.utility_scale = float(utility_scale)

    def forward(self, relation, cue, geometry, landmark_valid,
                trajectory, belief_features, return_components=False):
        token = self.encode(relation, cue, geometry, landmark_valid,
                            trajectory, belief_features)
        goal = self.score(token).squeeze(-1)
        utility = self.utility_score(token).squeeze(-1)
        final = goal + self.utility_scale * utility
        if return_components:
            return {"final": final, "goal": goal, "utility": utility}
        return final
