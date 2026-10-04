"""Masked multi-task attribute heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiAttributeHead(nn.Module):
    def __init__(self, input_dim, class_counts, hidden_dim=256):
        super().__init__(); self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, count) for name, count in class_counts.items()})

    def forward(self, features):
        shared = self.shared(features); return {name: head(shared) for name, head in self.heads.items()}

    @staticmethod
    def masked_loss(outputs, targets, masks, weights=None):
        losses = []
        for name, logits in outputs.items():
            selected = masks[name].bool()
            if selected.any(): losses.append((weights or {}).get(name, 1.0) * F.cross_entropy(logits[selected], targets[name][selected]))
        return sum(losses) if losses else torch.tensor(0.0, device=next(iter(outputs.values())).device)

