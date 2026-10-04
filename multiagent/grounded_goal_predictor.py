"""One-shot landmark-conditioned target localization without navigation state."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class LandmarkRelativeGoalPredictor(nn.Module):
    """Fuse cached language, landmark identity, RGB and geometry once per episode.

    All coordinates are relative to the center of the episode's map, in 100 m
    units. The model returns a relative target location, attention weights and
    weak relation logits. It does not see target coordinates or rollout state.
    """

    def __init__(self, text_dim=768, visual_dim=768, geometry_dim=11,
                 hidden_dim=256, relation_count=10, use_visual=True):
        super().__init__()
        self.use_visual = use_visual
        self.query = nn.Sequential(
            nn.LayerNorm(text_dim * 3),
            nn.Linear(text_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.landmark = nn.Sequential(
            nn.LayerNorm(text_dim + visual_dim + geometry_dim),
            nn.Linear(text_dim + visual_dim + geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.null_landmark = nn.Parameter(torch.zeros(hidden_dim))
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.offset = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.relation = nn.Linear(hidden_dim, relation_count)

    def forward(self, instruction, target_query, attributes, landmark_names,
                landmark_visuals, geometry, positions_relative, reference_mask):
        query = self.query(torch.cat((instruction, target_query, attributes), dim=-1))
        if not self.use_visual:
            landmark_visuals = torch.zeros_like(landmark_visuals)
        tokens = self.landmark(torch.cat(
            (landmark_names, landmark_visuals, geometry), dim=-1
        ))
        batch_size = query.shape[0]
        null_token = self.null_landmark.expand(batch_size, 1, -1)
        tokens = torch.cat((null_token, tokens), dim=1)
        null_position = positions_relative.new_zeros((batch_size, 1, 2))
        positions = torch.cat((null_position, positions_relative), dim=1)
        null_valid = ~reference_mask.any(dim=1, keepdim=True)
        valid = torch.cat((null_valid, reference_mask), dim=1)
        scores = (tokens * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(query.shape[-1])
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        context = (attention.unsqueeze(-1) * tokens).sum(dim=1)
        base = (attention.unsqueeze(-1) * positions).sum(dim=1)
        fused = self.fusion(torch.cat((query, context), dim=-1))
        # A bounded residual permits up to 250 m beyond the attended anchor.
        prediction = base + 2.5 * torch.tanh(self.offset(fused))
        return prediction, attention, self.relation(fused)
