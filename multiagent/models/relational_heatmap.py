"""Language-weighted geometric target field used by experiment 27."""

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.output_dim = hidden_dim * 2

    def forward(self, tokens):
        mask = tokens.ne(0)
        encoded, _ = self.encoder(self.embedding(tokens))
        encoded = encoded.masked_fill(~mask.unsqueeze(-1), -1e4)
        pooled = encoded.max(dim=1).values
        return torch.where(mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))


class RelationalHeatmap(nn.Module):
    """Factor direction and distance, combine them, then refine spatially.

    Args to forward:
      angle_fields: [B, K, A, H, W]
      distance_fields: [B, K, D, H, W]
      ref_tokens: [B, K, L], with the current landmark replaced by <ref>
      global_tokens: [B, L], with all landmark names replaced by <landmark>
      valid: [B, K]
      pair_field: [B, 1, H, W]
    """

    def __init__(self, vocab_size: int, max_landmarks: int,
                 angle_bases: int, distance_bases: int):
        super().__init__()
        self.max_landmarks = max_landmarks
        self.angle_bases = angle_bases
        self.distance_bases = distance_bases
        self.text = TextEncoder(vocab_size)
        dim = self.text.output_dim
        self.angle_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, angle_bases)
        )
        self.distance_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, distance_bases)
        )
        self.pair_head = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1))

        channels = max_landmarks + 3  # per-ref fields, union contour, pair, max prior
        self.conv1 = nn.Conv2d(channels, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 1, 1)
        self.film = nn.Linear(dim, 64)
        self.act = nn.ReLU()

    def forward(self, angle_fields, distance_fields, pair_field,
                ref_tokens, global_tokens, valid):
        batch, refs, _, height, width = angle_fields.shape
        ref_text = self.text(ref_tokens.reshape(batch * refs, -1)).reshape(batch, refs, -1)
        angle_weights = self.angle_head(ref_text).softmax(dim=-1) * valid.unsqueeze(-1)
        distance_weights = self.distance_head(ref_text).softmax(dim=-1) * valid.unsqueeze(-1)
        angle_map = (angle_weights[..., None, None] * angle_fields).sum(dim=2)
        distance_map = (distance_weights[..., None, None] * distance_fields).sum(dim=2)
        fields = angle_map * distance_map

        global_text = self.text(global_tokens)
        pair_gate = self.pair_head(global_text).sigmoid().view(batch, 1, 1, 1)
        pair = pair_field * pair_gate

        union_contour = distance_fields[:, :, 0].amax(dim=1, keepdim=True)
        max_prior = torch.maximum(fields.amax(dim=1, keepdim=True), pair)
        features = torch.cat([fields, union_contour, pair, max_prior], dim=1)
        hidden = self.conv1(features)
        scale, bias = self.film(global_text).chunk(2, dim=-1)
        hidden = hidden * (1 + 0.1 * scale[:, :, None, None]) + 0.1 * bias[:, :, None, None]
        hidden = self.act(hidden)
        hidden = self.act(self.conv2(hidden))
        hidden = self.act(self.conv3(hidden))
        residual = self.out(hidden)
        logits = residual + torch.log(max_prior.clamp_min(1e-4))
        joint_weights = angle_weights.unsqueeze(-1) * distance_weights.unsqueeze(-2)
        return logits, joint_weights, pair_gate.reshape(batch)


class GeometricCandidateSelector(nn.Module):
    """Score explicit landmark x angle x distance candidates from marked text."""

    def __init__(self, vocab_size: int, max_landmarks: int, candidates_per_landmark: int):
        super().__init__()
        self.max_landmarks = max_landmarks
        self.candidates_per_landmark = candidates_per_landmark
        self.text = TextEncoder(vocab_size)
        dim = self.text.output_dim
        self.candidate_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dim, candidates_per_landmark),
        )
        self.landmark_gate = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.pair_head = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, ref_tokens, global_tokens, valid, pair_valid):
        batch, refs, _ = ref_tokens.shape
        encoded = self.text(ref_tokens.reshape(batch * refs, -1)).reshape(batch, refs, -1)
        logits = self.candidate_head(encoded) + self.landmark_gate(encoded)
        logits = logits.masked_fill(~valid.bool().unsqueeze(-1), -1e4)
        logits = logits.reshape(batch, refs * self.candidates_per_landmark)
        pair_logit = self.pair_head(self.text(global_tokens))
        pair_logit = pair_logit.masked_fill(~pair_valid.bool().unsqueeze(-1), -1e4)
        return torch.cat([logits, pair_logit], dim=1)
