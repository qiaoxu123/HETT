"""Causal three-edge observation fusion; no goal labels or future poses."""
import torch
from torch import nn
from .candidate_relation_selector import candidate_anchor_geometry, candidate_pose_history_features


class RelationObservation(nn.Module):
    def __init__(self, feature_dim=256, language_dim=768, hidden=32):
        super().__init__()
        self.text = nn.Linear(language_dim, hidden)
        self.text_context = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
                                          nn.Conv1d(hidden, hidden, 3, padding=1))
        self.visual = nn.Linear(512, hidden)
        self.map = nn.Linear(feature_dim, hidden)
        self.edges = nn.Sequential(nn.Linear(23, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.score = nn.Linear(hidden, 1)
        self.feature = nn.Linear(hidden, feature_dim)
        self.arrival = nn.Sequential(nn.Linear(hidden + 9, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Exact inherited predictions before training the residual observations.
        nn.init.zeros_(self.score.weight); nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.feature.weight); nn.init.zeros_(self.feature.bias)
        nn.init.constant_(self.arrival[-1].bias, -4.)

    def forward(self, spatial, current_xy, heading_sc, language_tokens, language_mask,
                rgb_features, landmark_xy, landmark_extent, landmark_valid,
                landmark_text_mask, history_xy=None):
        b, c, h, w = spatial.shape
        axis_x = (torch.arange(w, device=spatial.device, dtype=spatial.dtype) + .5) / w
        axis_y = (torch.arange(h, device=spatial.device, dtype=spatial.dtype) + .5) / h
        yy, xx = torch.meshgrid(axis_y, axis_x, indexing='ij')
        cells = torch.stack((xx, yy), -1).reshape(1, h*w, 2).expand(b, -1, -1)
        candidate_uav = candidate_pose_history_features(cells, current_xy, heading_sc, history_xy)
        candidate_anchor = candidate_anchor_geometry(cells, landmark_xy, landmark_extent)
        # Same east/north convention; these encode UAV relative to each anchor.
        uav_anchor = candidate_anchor_geometry(current_xy[:, None], landmark_xy, landmark_extent)
        edge = torch.cat((candidate_anchor, uav_anchor.expand(-1, h*w, -1, -1),
                          candidate_uav[:, :, None].expand(-1, -1, landmark_xy.shape[1], -1)), -1)
        mask = language_mask.to(language_tokens.dtype)
        words = self.text(language_tokens) * mask[..., None]
        words = words + self.text_context(words.transpose(1, 2)).transpose(1, 2)
        global_text = (words * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        mentions = (landmark_text_mask & language_mask[:, None].bool()).to(words.dtype)
        names = torch.einsum('bml,bld->bmd', mentions, words) / mentions.sum(-1, keepdim=True).clamp_min(1)
        names = torch.where(mentions.any(-1)[..., None], names, global_text[:, None])
        visual = self.visual(rgb_features)
        map_features = self.map(spatial.flatten(2).transpose(1, 2))
        context = map_features + global_text[:, None] + visual[:, None]
        paired = self.norm(self.edges(edge) + names[:, None] + context[:, :, None])
        valid = landmark_valid.to(paired.dtype)[:, None, :, None]
        pooled = (paired * valid).sum(2) / valid.sum(2).clamp_min(1)
        # Landmark-free episodes retain visual/text/pose evidence without fake anchors.
        pooled = torch.where(landmark_valid.any(-1)[:, None, None], pooled,
                             self.norm(context))
        fused = self.norm(context + pooled)
        residual = self.score(fused).reshape(b, h, w)
        feature_residual = self.feature(fused).transpose(1, 2).reshape(b, c, h, w)
        return residual, feature_residual, fused, candidate_uav

    def stop(self, fused, candidate_uav, cell_ids, probabilities):
        rows = torch.arange(fused.shape[0], device=fused.device)
        confidence = probabilities.flatten(1).amax(-1, keepdim=True)
        entropy = -(probabilities.flatten(1).clamp_min(1e-8).log() * probabilities.flatten(1)).sum(-1, keepdim=True)
        return self.arrival(torch.cat((fused[rows, cell_ids], candidate_uav[rows, cell_ids],
                                       confidence, entropy), -1)).squeeze(-1)
