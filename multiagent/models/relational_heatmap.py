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
    """Mix fixed landmark fields with language, then refine spatially.

    Args to forward:
      bases: [B, K, R, H, W]
      ref_tokens: [B, K, L], with the current landmark replaced by <ref>
      global_tokens: [B, L], with all landmark names replaced by <landmark>
      valid: [B, K]
      pair_field: [B, 1, H, W]
    """

    def __init__(self, vocab_size: int, max_landmarks: int, relation_bases: int):
        super().__init__()
        self.max_landmarks = max_landmarks
        self.relation_bases = relation_bases
        self.text = TextEncoder(vocab_size)
        dim = self.text.output_dim
        self.relation_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, relation_bases)
        )
        self.pair_head = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1))

        channels = max_landmarks + 3  # per-ref fields, union contour, pair, max prior
        self.conv1 = nn.Conv2d(channels, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 1, 1)
        self.film = nn.Linear(dim, 64)
        self.act = nn.ReLU()

    def forward(self, bases, pair_field, ref_tokens, global_tokens, valid):
        batch, refs, _, height, width = bases.shape
        ref_text = self.text(ref_tokens.reshape(batch * refs, -1)).reshape(batch, refs, -1)
        relation_logits = self.relation_head(ref_text)
        relation_weights = relation_logits.softmax(dim=-1)
        relation_weights = relation_weights * valid.unsqueeze(-1)
        fields = (relation_weights[..., None, None] * bases).sum(dim=2)

        global_text = self.text(global_tokens)
        pair_gate = self.pair_head(global_text).sigmoid().view(batch, 1, 1, 1)
        pair = pair_field * pair_gate

        union_contour = bases[:, :, 0].amax(dim=1, keepdim=True)
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
        return logits, relation_weights, pair_gate.squeeze((1, 2, 3))
