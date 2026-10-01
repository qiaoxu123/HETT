"""Shared frozen SigLIP text/vision backbone for HETT."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor


class FrozenSiglipBackbone(nn.Module):
    """One frozen SigLIP model shared by language and RGB observations."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        *,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def text_dim(self) -> int:
        return int(self.model.config.text_config.hidden_size)

    @property
    def vision_dim(self) -> int:
        return int(self.model.config.vision_config.hidden_size)

    def train(self, mode: bool = True):
        # The shared backbone always stays frozen/eval. HETT heads remain trainable.
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode_text(
        self,
        texts: Sequence[str],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.processor(
            text=list(texts),
            padding=True,
            truncation=True,
            max_length=int(self.model.config.text_config.max_position_embeddings),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            pad_id = int(self.model.config.text_config.pad_token_id)
            attention_mask = input_ids.ne(pad_id)
        else:
            attention_mask = attention_mask.to(device).bool()
        attention_mask[:, 0] = True

        output = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return output.last_hidden_state, output.pooler_output

    def preprocess_images(
        self,
        images: np.ndarray | torch.Tensor,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Vectorized SigLIP preprocessing for NHWC RGB images."""
        pixels = torch.as_tensor(images, device=device)
        if not pixels.is_floating_point():
            pixels = pixels.float().div_(255.0)
        else:
            pixels = pixels.float()
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError("SigLIP images must be NHWC RGB")
        pixels = pixels.permute(0, 3, 1, 2).contiguous()

        image_size = int(self.model.config.vision_config.image_size)
        if pixels.shape[-2:] != (image_size, image_size):
            pixels = F.interpolate(
                pixels,
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        # SigLIP uses image mean/std [0.5, 0.5, 0.5].
        return pixels.mul(2.0).sub(1.0)

    @torch.no_grad()
    def encode_images(
        self,
        images: np.ndarray | torch.Tensor,
        device: torch.device | str,
        *,
        output_grid: int = 7,
    ) -> torch.Tensor:
        """Return [B, C, output_grid^2] spatial features."""
        pixel_values = self.preprocess_images(images, device)
        output = self.model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        tokens = output.last_hidden_state
        token_count = tokens.shape[1]

        side = math.isqrt(token_count)
        if side * side != token_count:
            # Support ViT variants that prepend a single class token.
            side = math.isqrt(token_count - 1)
            if side * side != token_count - 1:
                raise ValueError(
                    f"cannot reshape {token_count} SigLIP vision tokens to a square grid"
                )
            tokens = tokens[:, 1:]

        spatial = tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], side, side
        )
        spatial = F.adaptive_avg_pool2d(spatial, (output_grid, output_grid))
        return spatial.flatten(2)
