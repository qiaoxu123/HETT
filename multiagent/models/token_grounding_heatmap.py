"""Token-level instruction-to-landmark grounding over geometric heatmap bases."""

import math

import torch
from torch import nn
from transformers import AutoModel


class TokenGroundingHeatmap(nn.Module):
    """Ground each named landmark in instruction tokens before predicting geometry."""

    def __init__(self, bert_path: str, max_landmarks: int,
                 angle_bases: int, distance_bases: int, hidden_dim: int = 256):
        super().__init__()
        self.max_landmarks = max_landmarks
        self.bert = AutoModel.from_pretrained(bert_path, local_files_only=True)
        self.bert.requires_grad_(False)
        bert_dim = self.bert.config.hidden_size
        self.token_key = nn.Linear(bert_dim, hidden_dim)
        self.name_query = nn.Linear(bert_dim, hidden_dim)
        self.value = nn.Linear(bert_dim, hidden_dim)
        self.ground = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.angle_head = nn.Linear(hidden_dim, angle_bases)
        self.distance_head = nn.Linear(hidden_dim, distance_bases)
        self.pair_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))

        channels = max_landmarks + 3
        self.conv1 = nn.Conv2d(channels, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 1, 1)
        self.film = nn.Linear(hidden_dim, 64)
        self.act = nn.ReLU()

    def train(self, mode: bool = True):
        super().train(mode)
        self.bert.eval()
        return self

    def _encode(self, input_ids, attention_mask):
        with torch.no_grad():
            return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(self, angle_fields, distance_fields, pair_field, valid,
                instruction_ids, instruction_mask, name_ids, name_mask):
        batch, refs, _, _, _ = angle_fields.shape
        instruction_tokens = self._encode(instruction_ids, instruction_mask)

        # Landmark names use BERT's unchanged lexical embeddings as instance queries.
        name_word = self.bert.embeddings.word_embeddings(name_ids)
        name_weight = name_mask.unsqueeze(-1).to(name_word.dtype)
        name_mean = (name_word * name_weight).sum(dim=2) / name_weight.sum(dim=2).clamp_min(1.0)
        query = self.name_query(name_mean)
        key = self.token_key(instruction_tokens)
        value = self.value(instruction_tokens)
        scores = torch.einsum("bkh,blh->bkl", query, key) / math.sqrt(key.shape[-1])
        scores = scores.masked_fill(~instruction_mask[:, None].bool(), -1e4)
        attention = scores.softmax(dim=-1)
        context = torch.einsum("bkl,blh->bkh", attention, value)
        grounded = self.ground(torch.cat([query, context, query * context, query - context], dim=-1))

        angle_weights = self.angle_head(grounded).softmax(dim=-1) * valid.unsqueeze(-1)
        distance_weights = self.distance_head(grounded).softmax(dim=-1) * valid.unsqueeze(-1)
        angle_map = (angle_weights[..., None, None] * angle_fields).sum(dim=2)
        distance_map = (distance_weights[..., None, None] * distance_fields).sum(dim=2)
        fields = angle_map * distance_map

        token_weight = instruction_mask.unsqueeze(-1).to(value.dtype)
        global_text = (value * token_weight).sum(dim=1) / token_weight.sum(dim=1).clamp_min(1.0)
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
        logits = self.out(hidden) + torch.log(max_prior.clamp_min(1e-4))
        joint_weights = angle_weights.unsqueeze(-1) * distance_weights.unsqueeze(-2)
        return logits, joint_weights, pair_gate.reshape(batch), attention
