"""Frozen/partially-trainable DINOv2 feature wrapper."""
from __future__ import annotations

import torch
from torch import nn


class DinoV2AttributeEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-small", unfreeze_last_blocks=0):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        if unfreeze_last_blocks:
            for block in self.model.encoder.layer[-unfreeze_last_blocks:]: block.requires_grad_(True)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state[:, 0]

