"""SigLIP/SigLIP2 image and text feature wrapper."""
from __future__ import annotations

import torch
from torch import nn


class SiglipAttributeEncoder(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-256"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name).eval().requires_grad_(False)

    def image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values)

    def text_features(self, **tokens):
        return self.model.get_text_features(**tokens)

