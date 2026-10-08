"""Frozen visual feature extraction for SigLIP, SigLIP2 and DINOv2."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


class HFVisionEncoder:
    def __init__(self, model_name: str, device: str = "cuda", model=None, processor=None):
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.processor = processor or AutoProcessor.from_pretrained(model_name)
        self.model = (model or AutoModel.from_pretrained(model_name)).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, images, batch_size=32):
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
            else:
                output = self.model(**inputs)
                features = getattr(output, "pooler_output", None)
                if features is None:
                    features = output.last_hidden_state[:, 0]
            chunks.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(chunks, dim=0).numpy() if chunks else torch.empty((0, 0)).numpy()


def build_encoder(name: str, device="cuda", model_path=None):
    model_ids = {
        "siglip": "google/siglip-base-patch16-224",
        "siglip2": "google/siglip2-so400m-patch16-512",
        "siglip2_partial": "google/siglip2-so400m-patch16-512",
        "dinov2_small": "facebook/dinov2-small",
    }
    if name == "siglip2_partial" and model_path:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        resolved = checkpoint["model_path"]
        encoder = HFVisionEncoder(resolved, device=device)
        result = encoder.model.load_state_dict(checkpoint["state_dict"], strict=False)
        if result.unexpected_keys:
            raise ValueError(f"unexpected partial-tuning checkpoint keys: {result.unexpected_keys[:3]}")
        return encoder
    return HFVisionEncoder(model_path or model_ids[name], device=device)
