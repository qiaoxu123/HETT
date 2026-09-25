"""Small recurrent global-BEV fusion model with exact coordinate output."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalBEVFusion(nn.Module):
    """Fuse pose-projected RGB into a persistent world-aligned BEV state.

    Pose is deliberately absent from ``forward``: it is consumed by the fixed
    geometric projector before this model is called.  Target coordinates are
    supervision only and are likewise absent from the interface.
    """

    def __init__(self, grid_size=64, relation_dim=768, hidden=32):
        super().__init__()
        self.grid_size = int(grid_size)
        self.static = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.language = nn.Sequential(
            nn.Linear(relation_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.observation = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.gates = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1)
        self.proposal = nn.Conv2d(hidden * 2, hidden, 3, padding=1)
        self.belief_head = nn.Conv2d(hidden, 1, 1)
        self.offset_head = nn.Conv2d(hidden, 2, 1)

    def initial_state(self, initial_belief, landmark_bev, relation_context):
        static = torch.stack((initial_belief, landmark_bev), dim=1).float()
        context = self.language(relation_context.float())[:, :, None, None]
        return self.static(static) + context

    def fuse_step(self, state, rgb_bev, visibility):
        visible = visibility[:, None].float()
        observation = self.observation(torch.cat((rgb_bev.float(), visible), dim=1))
        joined = torch.cat((state, observation), dim=1)
        reset, update = torch.sigmoid(self.gates(joined)).chunk(2, dim=1)
        candidate = torch.tanh(self.proposal(torch.cat((reset * state, observation), 1)))
        proposed = (1.0 - update) * state + update * candidate
        # The persistent map is bitwise unchanged outside the current frustum.
        return torch.where(visible.bool(), proposed, state)

    def forward(self, initial_belief, landmark_bev, relation_context,
                rgb_bev, visibility, frame_valid=None):
        """Return per-frame cell logits and exact within-cell offsets.

        Args are target-free. ``rgb_bev`` is ``[B,T,3,H,W]`` after fixed pose
        projection and ``visibility`` is ``[B,T,H,W]``.
        """
        state = self.initial_state(initial_belief, landmark_bev, relation_context)
        logits, offsets = [], []
        for step in range(rgb_bev.shape[1]):
            updated = self.fuse_step(state, rgb_bev[:, step], visibility[:, step])
            if frame_valid is not None:
                valid = frame_valid[:, step, None, None, None].bool()
                state = torch.where(valid, updated, state)
            else:
                state = updated
            logits.append(
                torch.log(initial_belief.clamp_min(1e-8))
                + self.belief_head(state).squeeze(1)
            )
            # Offset is measured in cell widths and remains inside the cell.
            offsets.append(torch.tanh(self.offset_head(state)) * 0.5)
        return {
            "belief_logits": torch.stack(logits, 1),
            "offsets": torch.stack(offsets, 1),
            "final_state": state,
        }


def exact_location_loss(logits, offsets, target_xy, frame_valid,
                        offset_weight=2.0):
    """Cell classification plus continuous within-cell Smooth-L1 loss."""
    batch, steps, grid, _ = logits.shape
    cell_xy = torch.clamp((target_xy * grid).long(), 0, grid - 1)
    labels = cell_xy[:, 1] * grid + cell_xy[:, 0]
    repeated = labels[:, None].expand(-1, steps)
    selected = frame_valid.bool()
    classification = F.cross_entropy(
        logits.flatten(2)[selected], repeated[selected]
    )
    flat_offsets = offsets.permute(0, 1, 3, 4, 2).reshape(
        batch, steps, grid * grid, 2
    )
    gather = repeated[..., None, None].expand(-1, -1, 1, 2)
    predicted = flat_offsets.gather(2, gather).squeeze(2)
    exact_cell = target_xy * grid
    target_offset = exact_cell - torch.floor(exact_cell) - 0.5
    target_offset = target_offset[:, None].expand(-1, steps, -1)
    regression = F.smooth_l1_loss(predicted[selected], target_offset[selected])
    return classification + offset_weight * regression, {
        "classification": classification.detach(),
        "offset": regression.detach(),
    }


def decode_exact_coordinates(logits, offsets, frame_valid=None):
    """Decode the last valid frame to normalized continuous map coordinates."""
    batch, steps, grid, _ = logits.shape
    if frame_valid is None:
        indices = torch.full((batch,), steps - 1, device=logits.device, dtype=torch.long)
    else:
        indices = frame_valid.long().sum(1).clamp_min(1) - 1
    rows = torch.arange(batch, device=logits.device)
    final_logits = logits[rows, indices]
    cell = final_logits.flatten(1).argmax(-1)
    yy = torch.div(cell, grid, rounding_mode="floor")
    xx = cell % grid
    final_offsets = offsets[rows, indices]
    chosen = final_offsets[rows, :, yy, xx]
    return torch.stack((xx, yy), -1).float().add(0.5).add(chosen).div(grid)
