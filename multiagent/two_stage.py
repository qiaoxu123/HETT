"""Utilities for decoupled target-localisation and action training.

The target stage learns a stable global coordinate from deployment-available
inputs.  The action stage treats that coordinate as a detached condition and
only updates a small action-specific adapter plus the direction/progress heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


TRAINING_STAGES = ("joint", "target", "action", "fine", "arrival")
ACTION_MODULE_NAMES = (
    "target_conditioning",
    "multi_target_conditioning",
    "decoder_2_action_full",
    "decoder_2_progress_full",
    "decoder_2_stop_full",
)
FINE_MODULE_NAMES = (
    "fine_navigation_adapter",
    "decoder_2_local_control_full",
    "decoder_2_stop_full",
)
ARRIVAL_MODULE_NAMES = (
    "landmark_arrival_head",
)


@dataclass
class TargetLosses:
    coordinate: torch.Tensor
    distance: torch.Tensor
    bearing: torch.Tensor
    error_m: torch.Tensor


@dataclass
class QuadtreeBelief:
    """Sparse hierarchical target belief decoded from a quadtree beam."""

    coordinate: torch.Tensor
    leaf_centers: torch.Tensor
    leaf_probs: torch.Tensor
    hierarchy_loss: Optional[torch.Tensor]


class SparseQuadtreeBeliefHead(nn.Module):
    """Hierarchical sparse target localizer for normalized 2-D maps.

    Training uses teacher-forced 4-way decisions along the ground-truth path,
    so the classification cost is O(4 * depth). Inference keeps only the
    highest-scoring top-k nodes at every level, giving O(4 * top-k * depth)
    spatial hypotheses instead of scoring a dense HxW field.

    The final coordinate is the probability-weighted mean of the retained
    leaves. The leaf centers/probabilities are also returned so a later
    controller can consume multiple target hypotheses without early collapse.
    """

    _OFFSETS = (
        (-1.0, -1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (1.0, 1.0),
    )

    def __init__(
        self,
        d_model: int,
        depth: int = 5,
        topk: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("quadtree depth must be positive")
        if topk < 1:
            raise ValueError("quadtree topk must be positive")
        self.depth = depth
        self.topk = topk
        self.context_proj = nn.Linear(d_model, hidden_dim)
        self.box_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _children(
        centers: torch.Tensor,
        sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand square nodes into four children."""

        offsets = centers.new_tensor(SparseQuadtreeBeliefHead._OFFSETS)
        child_sizes = sizes.unsqueeze(-1).expand(*sizes.shape, 4) * 0.5
        child_centers = (
            centers.unsqueeze(-2)
            + offsets * (sizes.unsqueeze(-1).unsqueeze(-1) * 0.25)
        )
        return child_centers.clamp(0.0, 1.0), child_sizes

    def _score_children(
        self,
        context: torch.Tensor,
        child_centers: torch.Tensor,
        child_sizes: torch.Tensor,
    ) -> torch.Tensor:
        context_h = self.context_proj(context)
        while context_h.ndim < child_centers.ndim:
            context_h = context_h.unsqueeze(-2)
        geometry = torch.cat(
            (child_centers, child_sizes.unsqueeze(-1)), dim=-1
        )
        hidden = context_h + self.box_proj(geometry)
        return self.fusion(hidden).squeeze(-1)

    @staticmethod
    def _truth_child(
        truth: torch.Tensor,
        parent_centers: torch.Tensor,
    ) -> torch.Tensor:
        x_right = (truth[:, 0] >= parent_centers[:, 0]).long()
        y_bottom = (truth[:, 1] >= parent_centers[:, 1]).long()
        return x_right * 2 + y_bottom

    def hierarchy_loss(
        self,
        context: torch.Tensor,
        truth: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced hierarchical CE along the ground-truth path."""

        batch = context.shape[0]
        parent_centers = context.new_full((batch, 2), 0.5)
        parent_sizes = context.new_ones(batch)
        loss = context.new_zeros(())
        for _ in range(self.depth):
            child_centers, child_sizes = self._children(
                parent_centers, parent_sizes
            )
            logits = self._score_children(context, child_centers, child_sizes)
            labels = self._truth_child(truth, parent_centers)
            loss = loss + F.cross_entropy(logits, labels, reduction="sum")
            gather_index = labels.view(batch, 1, 1).expand(-1, 1, 2)
            parent_centers = child_centers.gather(
                1, gather_index
            ).squeeze(1)
            parent_sizes = child_sizes.gather(
                1, labels.view(batch, 1)
            ).squeeze(1)
        return loss

    def decode(
        self,
        context: torch.Tensor,
        topk: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Beam-decode a sparse set of quadtree leaves."""

        batch = context.shape[0]
        keep = self.topk if topk is None else topk
        centers = context.new_full((batch, 1, 2), 0.5)
        sizes = context.new_ones(batch, 1)
        log_scores = context.new_zeros(batch, 1)

        for _ in range(self.depth):
            child_centers, child_sizes = self._children(centers, sizes)
            bsz, beam, _, _ = child_centers.shape
            flat_centers = child_centers.reshape(bsz, beam * 4, 2)
            flat_sizes = child_sizes.reshape(bsz, beam * 4)

            logits = self._score_children(
                context.unsqueeze(1).expand(-1, beam, -1),
                child_centers,
                child_sizes,
            )
            child_log_probs = F.log_softmax(logits, dim=-1)
            flat_scores = (
                log_scores.unsqueeze(-1) + child_log_probs
            ).reshape(bsz, beam * 4)

            next_keep = min(keep, flat_scores.shape[1])
            log_scores, indices = torch.topk(
                flat_scores, k=next_keep, dim=-1
            )
            centers = flat_centers.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, 2)
            )
            sizes = flat_sizes.gather(1, indices)

        probs = torch.softmax(log_scores, dim=-1)
        coordinate = (centers * probs.unsqueeze(-1)).sum(dim=1)
        return coordinate, centers, probs

    def forward(
        self,
        context: torch.Tensor,
        truth: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
    ) -> QuadtreeBelief:
        coordinate, centers, probs = self.decode(context, topk=topk)
        loss = None if truth is None else self.hierarchy_loss(context, truth)
        return QuadtreeBelief(
            coordinate=coordinate,
            leaf_centers=centers,
            leaf_probs=probs,
            hierarchy_loss=loss,
        )


@dataclass
class CandidateBelief:
    """Sparse multi-hypothesis target belief over a fixed candidate set."""

    coordinate: torch.Tensor
    logits: torch.Tensor
    refined_coordinates: torch.Tensor
    topk_coordinates: torch.Tensor
    topk_probs: torch.Tensor
    topk_indices: torch.Tensor
    global_logits: torch.Tensor
    local_logits: torch.Tensor
    local_gate: torch.Tensor


class RelationalCandidateBeliefHead(nn.Module):
    """Distance-aware global/local sparse target grounding.

    The global branch only uses deployment-stable candidate geometry and full
    instruction tokens.  The local branch consumes the multimodal candidate
    tokens produced by ET (current RGB, pose, semantic map and history).
    A deterministic distance gate gives far-away candidates more global weight
    and nearby candidates more local evidence.  Top-K hypotheses are preserved
    without averaging them into a single belief point.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        hidden_dim: int = 256,
        topk: int = 4,
        max_offset: float = 0.0625,
        map_meters: float = 410.0,
        local_gate_center_m: float = 100.0,
        local_gate_temperature_m: float = 30.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if topk < 1:
            raise ValueError("candidate topk must be positive")
        if map_meters <= 0 or local_gate_temperature_m <= 0:
            raise ValueError("map scale and local gate temperature must be positive")
        self.topk = topk
        self.max_offset = max_offset
        self.map_meters = map_meters
        self.local_gate_center_m = local_gate_center_m
        self.local_gate_temperature_m = local_gate_temperature_m

        self.global_geometry_proj = nn.Sequential(
            nn.Linear(7, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.local_geometry_proj = nn.Sequential(
            nn.Linear(7, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.global_language_attention = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.global_norm = nn.LayerNorm(d_model)
        self.local_norm = nn.LayerNorm(d_model)
        self.global_scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.local_scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )

    @staticmethod
    def geometry(
        candidates: torch.Tensor,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        delta = candidates - current_positions.unsqueeze(1)
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        unit_delta = delta / distance.clamp_min(1e-6)
        return torch.cat((candidates, delta, distance, unit_delta), dim=-1)

    def distance_local_gate(
        self,
        candidates: torch.Tensor,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return local-evidence weight in [0, 1] for every candidate."""

        distance_m = torch.linalg.vector_norm(
            candidates - current_positions.unsqueeze(1), dim=-1
        ) * self.map_meters
        return torch.sigmoid(
            (self.local_gate_center_m - distance_m)
            / self.local_gate_temperature_m
        )

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        candidates: torch.Tensor,
        language_tokens: torch.Tensor,
        current_positions: torch.Tensor,
        language_mask: Optional[torch.Tensor] = None,
        global_candidate_tokens: Optional[torch.Tensor] = None,
    ) -> CandidateBelief:
        geometry = self.geometry(candidates, current_positions)
        if global_candidate_tokens is None:
            global_candidate_tokens = candidate_tokens

        global_tokens = (
            global_candidate_tokens + self.global_geometry_proj(geometry)
        )
        key_padding_mask = None
        if language_mask is not None:
            key_padding_mask = ~language_mask.bool()
        language_delta, _ = self.global_language_attention(
            query=global_tokens,
            key=language_tokens,
            value=language_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        global_grounded = self.global_norm(global_tokens + language_delta)

        local_grounded = self.local_norm(
            candidate_tokens + self.local_geometry_proj(geometry)
        )

        global_logits = self.global_scorer(global_grounded).squeeze(-1)
        local_logits = self.local_scorer(local_grounded).squeeze(-1)
        local_gate = self.distance_local_gate(candidates, current_positions)
        logits = (
            (1.0 - local_gate) * global_logits
            + local_gate * local_logits
        )

        fused_tokens = (
            (1.0 - local_gate.unsqueeze(-1)) * global_grounded
            + local_gate.unsqueeze(-1) * local_grounded
        )
        offsets = self.offset_head(fused_tokens) * self.max_offset
        refined = (candidates + offsets).clamp(0.0, 1.0)

        keep = min(self.topk, logits.shape[1])
        topk_logits, topk_indices = torch.topk(logits, k=keep, dim=-1)
        topk_coordinates = refined.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, 2)
        )
        topk_probs = torch.softmax(topk_logits, dim=-1)
        coordinate = topk_coordinates[:, 0]
        return CandidateBelief(
            coordinate=coordinate,
            logits=logits,
            refined_coordinates=refined,
            topk_coordinates=topk_coordinates,
            topk_probs=topk_probs,
            topk_indices=topk_indices,
            global_logits=global_logits,
            local_logits=local_logits,
            local_gate=local_gate,
        )


class MultiHypothesisConditioning(nn.Module):
    """Condition motion tokens on multiple target hypotheses via real attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.target_projection = nn.Sequential(
            nn.Linear(8, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.attention = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        motion_tokens: torch.Tensor,
        target_coordinates: torch.Tensor,
        target_probs: torch.Tensor,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        delta = target_coordinates - current_positions.unsqueeze(1)
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        unit_delta = delta / distance.clamp_min(1e-6)
        features = torch.cat(
            (
                target_coordinates,
                delta,
                distance,
                unit_delta,
                target_probs.unsqueeze(-1),
            ),
            dim=-1,
        )
        target_tokens = self.target_projection(features)
        attended, _ = self.attention(
            query=motion_tokens,
            key=target_tokens,
            value=target_tokens,
            need_weights=False,
        )
        motion_tokens = self.norm(motion_tokens + attended)
        return self.ffn_norm(motion_tokens + self.ffn(motion_tokens))


def candidate_supervision(
    logits: torch.Tensor,
    refined_coordinates: torch.Tensor,
    base_candidates: torch.Tensor,
    truth: torch.Tensor,
    map_meters: float,
    huber_delta_m: float,
    sample_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return candidate CE, GT-candidate offset loss and nearest candidate id."""

    distances = torch.linalg.vector_norm(
        base_candidates - truth.unsqueeze(1), dim=-1
    )
    labels = distances.argmin(dim=-1)
    classification_per_sample = F.cross_entropy(
        logits, labels, reduction="none"
    )
    selected = refined_coordinates.gather(
        1, labels.view(-1, 1, 1).expand(-1, 1, 2)
    ).squeeze(1)
    offset_per_sample = F.huber_loss(
        selected * map_meters,
        truth * map_meters,
        delta=huber_delta_m,
        reduction="none",
    ).sum(dim=-1)
    if sample_weights is None:
        sample_weights = torch.ones_like(classification_per_sample)
    classification = (classification_per_sample * sample_weights).sum()
    offset = (offset_per_sample * sample_weights).sum()
    return classification, offset, labels


def candidate_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Margin-rank the GT-associated candidate above the hardest negative.

    The hardest negative is the highest-scoring non-GT candidate in each
    sample. This directly optimizes the failure mode where a useful target is
    present in Top-K but is not ranked at Top-1.
    """

    if logits.ndim != 2:
        raise ValueError("candidate logits must have shape [B, N]")
    if logits.shape[1] < 2:
        return logits.new_zeros(())
    positive = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, labels.view(-1, 1), False)
    hardest_negative = logits.masked_fill(
        ~negative_mask, -torch.inf
    ).max(dim=-1).values
    per_sample = F.relu(margin - positive + hardest_negative)
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
    return per_sample.sum()


def candidate_recall(
    topk_coordinates: torch.Tensor,
    truth: torch.Tensor,
    map_meters: float,
    radius_m: float,
) -> torch.Tensor:
    """Per-sample Recall@K indicator for target candidates."""

    errors = torch.linalg.vector_norm(
        (topk_coordinates - truth.unsqueeze(1)) * map_meters,
        dim=-1,
    )
    return (errors.min(dim=-1).values <= radius_m).float()


class LandmarkArrivalHead(nn.Module):
    """Multi-evidence verifier for understanding arrival at the described landmark."""

    def __init__(self, d_model: int, depth_dim: int = 4, geom_dim: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.depth_proj = nn.Sequential(
            nn.Linear(depth_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
        )
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
        )
        fusion_dim = d_model * 2 + d_model // 4 + d_model // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.match_head = nn.Linear(d_model, 1)
        self.near_head = nn.Linear(d_model, 1)
        self.arrival_head = nn.Linear(d_model, 1)

    def forward(self, visual_token: torch.Tensor, language_token: torch.Tensor,
                depth_stats: torch.Tensor, topk_geometry: torch.Tensor):
        # Geometry is pooled conservatively across hypotheses; the model gets
        # min/mean proximity, directional spread and coarse confidence through
        # per-hypothesis features before pooling.
        geom = self.geom_proj(topk_geometry).mean(dim=1)
        depth = self.depth_proj(depth_stats)
        fused = self.fusion(torch.cat((visual_token, language_token, depth, geom), dim=-1))
        match_logit = self.match_head(fused)
        near_logit = self.near_head(fused)
        arrival_logit = self.arrival_head(fused)
        return match_logit, near_logit, arrival_logit

class FineNavigationAdapter(nn.Module):
    """Language-ground current motion/visual tokens for near-goal control."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.language_attention = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, motion_tokens: torch.Tensor, language_tokens: torch.Tensor,
                language_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None
        if language_mask is not None:
            key_padding_mask = ~language_mask.bool()
        delta, _ = self.language_attention(
            query=motion_tokens, key=language_tokens, value=language_tokens,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        motion_tokens = self.attn_norm(motion_tokens + delta)
        return self.ffn_norm(motion_tokens + self.ffn(motion_tokens))


def local_waypoint_target(current_positions: torch.Tensor,
                          target_positions: torch.Tensor,
                          map_meters: float, waypoint_meters: float,
                          success_radius_m: float = 0.0) -> torch.Tensor:
    """Return the next oracle waypoint toward the success region, not its center."""
    delta = target_positions - current_positions
    distance_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    distance_m = distance_norm * map_meters
    remaining_m = (distance_m - success_radius_m).clamp_min(0.0)
    step_m = torch.clamp(remaining_m, max=waypoint_meters)
    unit = delta / distance_norm.clamp_min(1e-6)
    return (current_positions + unit * (step_m / map_meters)).clamp(0.0, 1.0)

class TargetConditioning(nn.Module):
    """Inject a global target coordinate into motion tokens.

    In addition to absolute ``(x, y)``, the embedding receives relative
    displacement, distance and unit direction.  Every value is derived from
    the supplied coordinate and current pose, so no extra privileged input is
    introduced.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(7, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.attention = nn.MultiheadAttention(
            d_model, num_heads=1, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    @staticmethod
    def geometry(
        target_coordinates: torch.Tensor,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        delta = target_coordinates - current_positions
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        unit_delta = delta / distance.clamp_min(1e-6)
        return torch.cat(
            (target_coordinates, delta, distance, unit_delta), dim=-1
        )

    def forward(
        self,
        motion_tokens: torch.Tensor,
        target_coordinates: torch.Tensor,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        target_token = self.projection(
            self.geometry(target_coordinates, current_positions)
        ).unsqueeze(1)
        delta, _ = self.attention(
            query=motion_tokens,
            key=target_token,
            value=target_token,
            need_weights=False,
        )
        motion_tokens = self.norm(motion_tokens + delta)
        return self.ffn_norm(motion_tokens + self.ffn(motion_tokens))


def target_coordinate_losses(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    current_positions: torch.Tensor,
    map_meters: float,
    huber_delta_m: float,
) -> TargetLosses:
    """Return metre-scaled coordinate, range and bearing losses.

    Inputs use the normalized map coordinate system.  The main coordinate loss
    is computed in metres so it cannot be numerically drowned by normalized
    action losses as easily as the released coordinate MSE.
    """

    predicted_m = predicted * map_meters
    truth_m = truth * map_meters
    current_m = current_positions * map_meters

    coordinate = F.huber_loss(
        predicted_m,
        truth_m,
        delta=huber_delta_m,
        reduction="sum",
    )
    pred_delta = predicted_m - current_m
    truth_delta = truth_m - current_m
    pred_distance = torch.linalg.vector_norm(pred_delta, dim=-1)
    truth_distance = torch.linalg.vector_norm(truth_delta, dim=-1)
    distance = F.huber_loss(
        pred_distance,
        truth_distance,
        delta=huber_delta_m,
        reduction="sum",
    )
    pred_unit = pred_delta / pred_distance.unsqueeze(-1).clamp_min(1e-6)
    truth_unit = truth_delta / truth_distance.unsqueeze(-1).clamp_min(1e-6)
    bearing = (1.0 - (pred_unit * truth_unit).sum(dim=-1)).sum()
    error_m = torch.linalg.vector_norm(predicted_m - truth_m, dim=-1)
    return TargetLosses(coordinate, distance, bearing, error_m)


def coordinate_gt_probability(
    current_epoch: int,
    total_epochs: int,
    start_probability: float,
    end_probability: float,
) -> float:
    """Linear curriculum from oracle coordinates to predicted coordinates."""

    if total_epochs <= 1:
        fraction = 1.0
    else:
        fraction = min(max((current_epoch - 1) / (total_epochs - 1), 0.0), 1.0)
    return start_probability + fraction * (end_probability - start_probability)


def select_action_coordinates(
    predicted: torch.Tensor,
    truth: torch.Tensor | None,
    use_truth: torch.Tensor | None,
) -> torch.Tensor:
    """Select detached coordinates used by the action branch."""

    predicted = predicted.detach()
    if truth is None or use_truth is None:
        return predicted
    if use_truth.ndim == 1:
        use_truth = use_truth.unsqueeze(-1)
    return torch.where(use_truth.bool(), truth.detach(), predicted)


def configure_stage_parameters(
    language_model: nn.Module,
    vision_model: nn.Module,
    navigation_model: nn.Module,
    stage: str,
    freeze_target_backbones: bool = False,
) -> None:
    """Configure strict parameter ownership for a training stage."""

    if stage not in TRAINING_STAGES:
        raise ValueError(f"unsupported training stage: {stage}")

    for model in (language_model, vision_model, navigation_model):
        for parameter in model.parameters():
            parameter.requires_grad = True

    action_modules = [
        getattr(navigation_model, name)
        for name in ACTION_MODULE_NAMES
        if hasattr(navigation_model, name)
    ]
    if stage == "joint":
        # New two-stage-only modules are not on the released joint path.
        # Freezing them preserves the released optimizer parameter groups.
        for parameter in navigation_model.target_conditioning.parameters():
            parameter.requires_grad = False
        if hasattr(navigation_model, "quadtree_belief"):
            for parameter in navigation_model.quadtree_belief.parameters():
                parameter.requires_grad = False
        if hasattr(navigation_model, "candidate_belief"):
            for parameter in navigation_model.candidate_belief.parameters():
                parameter.requires_grad = False
        if hasattr(navigation_model, "multi_target_conditioning"):
            for parameter in navigation_model.multi_target_conditioning.parameters():
                parameter.requires_grad = False
        if hasattr(navigation_model, "decoder_2_stop_full"):
            for parameter in navigation_model.decoder_2_stop_full.parameters():
                parameter.requires_grad = False
    elif stage == "target":
        for module in action_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
        if freeze_target_backbones:
            for model in (language_model, vision_model):
                for parameter in model.parameters():
                    parameter.requires_grad = False
    elif stage == "action":
        for model in (language_model, vision_model, navigation_model):
            for parameter in model.parameters():
                parameter.requires_grad = False
        for module in action_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif stage == "fine":
        for model in (language_model, vision_model, navigation_model):
            for parameter in model.parameters():
                parameter.requires_grad = False
        for name in FINE_MODULE_NAMES:
            if hasattr(navigation_model, name):
                for parameter in getattr(navigation_model, name).parameters():
                    parameter.requires_grad = True
    elif stage == "arrival":
        for model in (language_model, vision_model, navigation_model):
            for parameter in model.parameters():
                parameter.requires_grad = False
        for name in ARRIVAL_MODULE_NAMES:
            if hasattr(navigation_model, name):
                for parameter in getattr(navigation_model, name).parameters():
                    parameter.requires_grad = True
