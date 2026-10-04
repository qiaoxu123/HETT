"""Small scene matcher used only after the explicit-evidence Gate."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SceneMatcher(nn.Module):
    def __init__(self, language_dim: int, visual_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.language = nn.Linear(language_dim, hidden_dim)
        self.visual = nn.Linear(visual_dim, hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def embeddings(self, language, visual):
        return F.normalize(self.language(language), dim=-1), F.normalize(self.visual(visual), dim=-1)

    def forward(self, language, visual):
        left, right = self.embeddings(language, visual)
        return self.classifier(torch.cat((left, right, left * right, torch.abs(left - right)), -1)).squeeze(-1)

    def contrastive_logits(self, language, visual, temperature=0.07):
        left, right = self.embeddings(language, visual)
        return left @ right.T / temperature
