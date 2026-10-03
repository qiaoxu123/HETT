from __future__ import annotations

import torch
from torch import nn
from transformers import SiglipModel


class GoalApproachModel(nn.Module):
    def __init__(
        self, model_name: str = "google/siglip-base-patch16-224", frames: int = 4,
        use_language: bool = True, use_pose: bool = False, hidden_dim: int = 256,
        freeze_siglip: bool = True, siglip_model=None,
    ):
        super().__init__()
        self.siglip = siglip_model if siglip_model is not None else SiglipModel.from_pretrained(model_name)
        self.frames = frames
        self.use_language = use_language
        self.use_pose = use_pose
        embed_dim = getattr(
            self.siglip.config, "projection_dim",
            getattr(getattr(self.siglip.config, "text_config", None), "projection_size", None),
        )
        if embed_dim is None:
            raise ValueError("cannot infer SigLIP projection dimension")
        if freeze_siglip:
            self.siglip.requires_grad_(False)
        self.temporal = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.pose_encoder = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 64)) if use_pose else None
        fusion_dim = hidden_dim + (embed_dim if use_language else 0) + (64 if use_pose else 0)
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.heads = nn.ModuleDict({
            "target_match": nn.Linear(hidden_dim, 2),
            "arrival": nn.Linear(hidden_dim, 2),
            "distance_bin": nn.Linear(hidden_dim, 6),
            "bearing_bin": nn.Linear(hidden_dim, 5),
            "phase": nn.Linear(hidden_dim, 4),
        })

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.siglip.parameters()):
            self.siglip.eval()
        return self

    def forward(self, pixel_values, input_ids, attention_mask, pose=None):
        batch_size, frame_count = pixel_values.shape[:2]
        flat = pixel_values.reshape(batch_size * frame_count, *pixel_values.shape[2:])
        grad_enabled = any(parameter.requires_grad for parameter in self.siglip.parameters())
        with torch.set_grad_enabled(grad_enabled):
            image_features = self.siglip.get_image_features(pixel_values=flat)
            image_features = nn.functional.normalize(image_features, dim=-1)
            if self.use_language:
                text_features = self.siglip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                text_features = nn.functional.normalize(text_features, dim=-1)
        image_features = image_features.reshape(batch_size, frame_count, -1)
        temporal, _ = self.temporal(image_features)
        fused = [temporal[:, -1]]
        if self.use_language:
            fused.append(text_features)
        if self.use_pose:
            fused.append(self.pose_encoder(pose))
        hidden = self.fusion(torch.cat(fused, dim=-1))
        return {name: head(hidden) for name, head in self.heads.items()}
