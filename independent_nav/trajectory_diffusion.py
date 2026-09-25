"""Lightweight Conditional 1-D U-Net for fixed-altitude trajectory diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, timestep):
        half = self.dimension // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequency = torch.exp(
            torch.arange(half, device=timestep.device) * -scale)
        value = timestep.float()[:, None] * frequency[None]
        return torch.cat((value.sin(), value.cos()), -1)


class ConditionEncoder(nn.Module):
    """Encode frozen belief/language/map features without target coordinates."""

    def __init__(self, output_dim=256):
        super().__init__()
        self.belief = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1))
        self.relation = nn.Sequential(
            nn.Linear(768, 64), nn.LayerNorm(64), nn.SiLU())
        self.cue = nn.Sequential(
            nn.Linear(12, 64), nn.LayerNorm(64), nn.SiLU())
        self.landmark = nn.Sequential(
            nn.Linear(2, 64), nn.LayerNorm(64), nn.SiLU())
        self.pose = nn.Sequential(
            nn.Linear(5, 64), nn.LayerNorm(64), nn.SiLU())
        self.output = nn.Sequential(
            nn.Linear(64 * 5, output_dim), nn.LayerNorm(output_dim), nn.SiLU())

    def forward(self, belief_map, relation, cue, landmark_centers,
                landmark_valid, pose_budget):
        belief = self.belief(belief_map[:, None].float()).flatten(1)
        valid = landmark_valid[..., None].float()
        count = valid.sum(1).clamp_min(1)
        relation = (self.relation(relation.float()) * valid).sum(1) / count
        cue = (self.cue(cue.float()) * valid).sum(1) / count
        landmark = (self.landmark(landmark_centers.float()) * valid).sum(1) / count
        pose = self.pose(pose_budget.float())
        return self.output(torch.cat((belief, relation, cue, landmark, pose), -1))


class EndpointConditionEncoder(nn.Module):
    """Encode scene context plus the explicit hypothesis endpoint.

    The endpoint is a belief hypothesis, never the metric-only target.  Making
    it explicit prevents the denoiser from inventing an unrelated path that
    must be bent toward the endpoint after sampling.
    """

    def __init__(self, output_dim=256):
        super().__init__()
        self.scene = ConditionEncoder(192)
        self.endpoint = nn.Sequential(
            nn.Linear(9, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU())
        self.output = nn.Sequential(
            nn.Linear(256, output_dim), nn.LayerNorm(output_dim), nn.SiLU())

    def forward(self, belief_map, relation, cue, landmark_centers,
                landmark_valid, pose_budget, endpoint_features):
        scene = self.scene(
            belief_map, relation, cue, landmark_centers,
            landmark_valid, pose_budget)
        endpoint = self.endpoint(endpoint_features.float())
        return self.output(torch.cat((scene, endpoint), -1))


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, channels * 2)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, value, condition):
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.condition(condition).chunk(2, -1)
        hidden = hidden * (1 + scale[..., None]) + shift[..., None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden


class ConditionalUnet1D(nn.Module):
    """Small U-Net over ``[T,4]`` continuous navigation actions."""

    def __init__(self, action_dim=4, condition_dim=256, time_dim=128,
                 base=64, horizon=20, trajectory_slots=8):
        super().__init__()
        self.horizon = horizon
        self.condition_encoder = ConditionEncoder(condition_dim)
        self.slot_embedding = nn.Embedding(trajectory_slots, condition_dim)
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim),
            nn.SiLU(), nn.Linear(time_dim, time_dim))
        joint = condition_dim + time_dim
        self.input = nn.Conv1d(action_dim, base, 3, padding=1)
        self.down0 = ConditionalResidualBlock1D(base, joint)
        self.downsample0 = nn.Conv1d(base, base * 2, 4, stride=2, padding=1)
        self.down1 = ConditionalResidualBlock1D(base * 2, joint)
        self.downsample1 = nn.Conv1d(base * 2, base * 4, 4, stride=2, padding=1)
        self.mid0 = ConditionalResidualBlock1D(base * 4, joint)
        self.mid1 = ConditionalResidualBlock1D(base * 4, joint)
        self.upsample1 = nn.ConvTranspose1d(
            base * 4, base * 2, 4, stride=2, padding=1)
        self.merge1 = nn.Conv1d(base * 4, base * 2, 1)
        self.up1 = ConditionalResidualBlock1D(base * 2, joint)
        self.upsample0 = nn.ConvTranspose1d(
            base * 2, base, 4, stride=2, padding=1)
        self.merge0 = nn.Conv1d(base * 2, base, 1)
        self.up0 = ConditionalResidualBlock1D(base, joint)
        self.output = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(), nn.Conv1d(base, action_dim, 3, padding=1))
        self.length_head = nn.Sequential(
            nn.Linear(condition_dim, 128), nn.SiLU(), nn.Linear(128, horizon + 1))

    def encode_condition(self, belief_map, relation, cue, landmark_centers,
                         landmark_valid, pose_budget):
        return self.condition_encoder(
            belief_map, relation, cue, landmark_centers,
            landmark_valid, pose_budget)

    def forward(self, noisy_actions, timestep, belief_map, relation, cue,
                landmark_centers, landmark_valid, pose_budget,
                trajectory_slot=None):
        global_condition = self.encode_condition(
            belief_map, relation, cue, landmark_centers,
            landmark_valid, pose_budget)
        if trajectory_slot is not None:
            global_condition = global_condition + self.slot_embedding(
                trajectory_slot.long())
        condition = torch.cat((global_condition, self.time(timestep)), -1)
        value = self.input(noisy_actions.float().transpose(1, 2))
        skip0 = self.down0(value, condition)
        skip1 = self.down1(self.downsample0(skip0), condition)
        value = self.mid1(self.mid0(self.downsample1(skip1), condition), condition)
        value = self.upsample1(value)
        value = self.up1(self.merge1(torch.cat((value, skip1), 1)), condition)
        value = self.upsample0(value)
        value = self.up0(self.merge0(torch.cat((value, skip0), 1)), condition)
        return {
            "noise": self.output(value).transpose(1, 2),
            "length_logits": self.length_head(global_condition),
        }


class EndpointResidualUnet1D(ConditionalUnet1D):
    """Diffuse lateral residuals around an endpoint-conditioned shortest path.

    The model predicts one scalar lateral offset per step.  Longitudinal
    progress comes from the shortest-path skeleton, so the learned component
    cannot freely create loops or reverse the route.  The length head predicts
    only zero, one, or two exploration steps beyond the minimum required.
    """

    def __init__(self, condition_dim=256, time_dim=128, base=32,
                 horizon=20, trajectory_slots=8):
        super().__init__(
            action_dim=1, condition_dim=condition_dim, time_dim=time_dim,
            base=base, horizon=horizon, trajectory_slots=trajectory_slots)
        self.condition_encoder = EndpointConditionEncoder(condition_dim)
        self.length_head = nn.Sequential(
            nn.Linear(condition_dim, 128), nn.SiLU(), nn.Linear(128, 3))

    def encode_condition(self, belief_map, relation, cue, landmark_centers,
                         landmark_valid, pose_budget, endpoint_features):
        return self.condition_encoder(
            belief_map, relation, cue, landmark_centers, landmark_valid,
            pose_budget, endpoint_features)

    def forward(self, noisy_residual, timestep, belief_map, relation, cue,
                landmark_centers, landmark_valid, pose_budget,
                endpoint_features, trajectory_slot=None):
        global_condition = self.encode_condition(
            belief_map, relation, cue, landmark_centers, landmark_valid,
            pose_budget, endpoint_features)
        if trajectory_slot is not None:
            global_condition = global_condition + self.slot_embedding(
                trajectory_slot.long())
        condition = torch.cat((global_condition, self.time(timestep)), -1)
        value = self.input(noisy_residual.float().transpose(1, 2))
        skip0 = self.down0(value, condition)
        skip1 = self.down1(self.downsample0(skip0), condition)
        value = self.mid1(self.mid0(self.downsample1(skip1), condition), condition)
        value = self.upsample1(value)
        value = self.up1(self.merge1(torch.cat((value, skip1), 1)), condition)
        value = self.upsample0(value)
        value = self.up0(self.merge0(torch.cat((value, skip0), 1)), condition)
        return {
            "noise": self.output(value).transpose(1, 2),
            "extra_step_logits": self.length_head(global_condition),
        }


def cosine_noise_schedule(steps, device=None, offset=0.008):
    """Cosine schedule whose final state is effectively pure noise.

    A fixed ``1e-4..0.02`` schedule is calibrated for roughly 1000 diffusion
    steps.  With only 100 steps it leaves too much signal at the terminal
    timestep, while sampling still starts from pure Gaussian noise.
    """
    time = torch.linspace(0, steps, steps + 1, device=device) / steps
    alpha_bar = torch.cos(
        (time + offset) / (1 + offset) * math.pi / 2).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999)
    alpha = 1.0 - beta
    return beta, alpha, torch.cumprod(alpha, 0)


def make_ddpm_scheduler(steps=100):
    """Official Diffusers scheduler used by Diffusion Policy-style training."""
    return DDPMScheduler(
        num_train_timesteps=steps, beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon", clip_sample=True)


def make_ddim_scheduler(steps=100):
    """Official deterministic inference scheduler with the same train noise."""
    return DDIMScheduler(
        num_train_timesteps=steps, beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon", clip_sample=True)


def endpoint_condition_features(starts, endpoints, minimum_steps,
                                remaining_steps, horizon=20):
    """Return explicit, target-free endpoint geometry for each hypothesis."""
    starts = torch.as_tensor(starts).float()
    endpoints = torch.as_tensor(endpoints).float()
    minimum_steps = torch.as_tensor(minimum_steps).float()
    remaining_steps = torch.as_tensor(remaining_steps).float()
    if starts.ndim == 2:
        starts = starts[:, None].expand(-1, endpoints.shape[1], -1)
    delta = endpoints - starts
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    direction = delta / distance.clamp_min(1e-8)
    if remaining_steps.ndim == 1:
        remaining_steps = remaining_steps[:, None].expand_as(minimum_steps)
    return torch.cat((
        endpoints, delta, direction, distance,
        minimum_steps[..., None] / horizon,
        remaining_steps[..., None] / horizon,
    ), -1)


def lateral_residual_targets(positions, movement_steps, starts, endpoints,
                             step_scale, horizon=20, maximum_residual=0.3):
    """Convert teacher paths into lateral residuals around a straight skeleton.

    Paths are resampled by arc length to ``minimum + at most two`` steps.
    Direct paths therefore have a zero target, while observation paths retain
    only their useful sideward deviation rather than redundant progress timing.
    """
    positions = torch.as_tensor(positions).float()
    movement_steps = torch.as_tensor(movement_steps).long()
    starts = torch.as_tensor(starts).float()
    endpoints = torch.as_tensor(endpoints).float()
    direct = torch.linalg.vector_norm(endpoints - starts[:, None], dim=-1)
    minimum = torch.ceil(direct / step_scale).long().clamp(1, horizon)
    extra = (movement_steps - minimum).clamp(0, 2)
    effective = (minimum + extra).clamp(1, horizon)
    residual = torch.zeros(*effective.shape, horizon, 1)
    for batch in range(positions.shape[0]):
        for slot in range(positions.shape[1]):
            source_count = max(1, int(movement_steps[batch, slot]))
            source = positions[batch, slot, :source_count + 1]
            segment = torch.linalg.vector_norm(source[1:] - source[:-1], dim=-1)
            cumulative = torch.cat((torch.zeros(1), segment.cumsum(0)))
            count = int(effective[batch, slot])
            if float(cumulative[-1]) <= 1e-8:
                sampled = source[:1].expand(count + 1, -1)
            else:
                targets = torch.linspace(0, float(cumulative[-1]), count + 1)
                sampled_values = []
                for target in targets:
                    upper = int(torch.searchsorted(cumulative, target, right=True))
                    upper = min(max(upper, 1), len(cumulative) - 1)
                    lower = upper - 1
                    span = float(cumulative[upper] - cumulative[lower])
                    weight = 0.0 if span <= 1e-8 else float(
                        (target - cumulative[lower]) / span)
                    sampled_values.append(
                        source[lower] * (1 - weight) + source[upper] * weight)
                sampled = torch.stack(sampled_values)
            delta = endpoints[batch, slot] - starts[batch]
            distance = torch.linalg.vector_norm(delta)
            direction = delta / distance.clamp_min(1e-8)
            perpendicular = torch.stack((-direction[1], direction[0]))
            fraction = torch.linspace(0, 1, count + 1)[:, None]
            skeleton = starts[batch] + fraction * delta
            lateral = ((sampled - skeleton) * perpendicular).sum(-1) / step_scale
            lateral[0] = 0; lateral[-1] = 0
            residual[batch, slot, :count, 0] = lateral[1:].clamp(
                -maximum_residual, maximum_residual)
    return residual, minimum, extra, effective


def residual_to_positions(residual, starts, endpoints, lengths, step_scale,
                          maximum_residual=0.3):
    """Apply sampled lateral offsets to a shortest-path skeleton."""
    residual = torch.as_tensor(residual).detach().cpu().float()
    starts = torch.as_tensor(starts).detach().cpu().float()
    endpoints = torch.as_tensor(endpoints).detach().cpu().float()
    lengths = torch.as_tensor(lengths).detach().cpu().long()
    batch, samples, horizon = residual.shape[:3]
    desired = torch.zeros(batch, samples, horizon + 1, 2)
    desired[:, :, 0] = starts[:, None]
    for batch_index in range(batch):
        for slot in range(samples):
            count = max(1, min(int(lengths[batch_index, slot]), horizon))
            delta = endpoints[batch_index, slot] - starts[batch_index]
            direction = delta / torch.linalg.vector_norm(delta).clamp_min(1e-8)
            perpendicular = torch.stack((-direction[1], direction[0]))
            for step in range(1, count + 1):
                fraction = step / count
                lateral = residual[batch_index, slot, step - 1, 0].clamp(
                    -maximum_residual, maximum_residual) * step_scale
                if step == count:
                    lateral = torch.zeros_like(lateral)
                desired[batch_index, slot, step] = (
                    starts[batch_index] + fraction * delta
                    + perpendicular * lateral)
            desired[batch_index, slot, count + 1:] = endpoints[batch_index, slot]
    actions = torch.zeros(batch, samples, horizon, 4)
    actions[..., :2] = (desired[..., 1:, :] - desired[..., :-1, :]) / step_scale
    return bounded_endpoint_projection(
        actions, lengths, starts, endpoints, step_scale)


def bounded_endpoint_projection(actions, lengths, starts, endpoints,
                                step_scale):
    """Project sampled paths onto target-free hypothesis endpoints.

    Each proposed step is retained whenever the endpoint remains reachable.
    Otherwise it is projected onto the boundary of the remaining reachable
    disk.  The final active step reaches the anchor exactly and every step is
    bounded by ``step_scale``.  ``endpoints`` come from belief hypotheses,
    never from target ground truth.
    """
    actions = actions.detach().cpu()
    lengths = lengths.detach().cpu().long()
    starts = starts.detach().cpu().float()
    endpoints = endpoints.detach().cpu().float()
    batch, samples, horizon = actions.shape[:3]
    direct = torch.linalg.vector_norm(
        endpoints - starts[:, None], dim=-1)
    required = torch.ceil(direct / step_scale).long().clamp(1, horizon)
    effective = torch.maximum(lengths.clamp(1, horizon), required)
    positions = torch.zeros(batch, samples, horizon + 1, 2)
    current = starts[:, None].expand(-1, samples, -1).clone()
    positions[:, :, 0] = current
    for step in range(horizon):
        active = step < effective
        proposal_delta = actions[:, :, step, :2]
        norm = torch.linalg.vector_norm(
            proposal_delta, dim=-1, keepdim=True).clamp_min(1.0)
        proposal = current + proposal_delta / norm * step_scale
        remaining = (effective - step - 1).clamp_min(0).float() * step_scale
        error = proposal - endpoints
        distance = torch.linalg.vector_norm(error, dim=-1)
        to_endpoint = endpoints - current
        direct_distance = torch.linalg.vector_norm(
            to_endpoint, dim=-1, keepdim=True)
        direct_step = current + to_endpoint / direct_distance.clamp_min(1e-8) * torch.minimum(
            direct_distance, torch.full_like(direct_distance, step_scale))
        proposal = torch.where(
            distance.gt(remaining)[..., None], direct_step, proposal)
        current = torch.where(active[..., None], proposal.clamp(0, 1), current)
        positions[:, :, step + 1] = current
    return positions, effective


@torch.inference_mode()
def ddim_sample(model, condition, samples=8, diffusion_steps=100,
                inference_steps=20):
    """Deterministically denoise multiple noise seeds under one condition."""
    device = next(model.parameters()).device
    batch = condition[0].shape[0]
    repeated = tuple(value.repeat_interleave(samples, 0).to(device)
                     for value in condition)
    trajectory_slot = torch.arange(samples, device=device).repeat(batch)
    value = torch.randn(batch * samples, model.horizon, 4, device=device)
    scheduler = make_ddim_scheduler(diffusion_steps)
    scheduler.set_timesteps(inference_steps, device=device)
    length_logits = None
    for timestep_value in scheduler.timesteps:
        timestep = timestep_value.expand(batch * samples)
        output = model(value, timestep, *repeated,
                       trajectory_slot=trajectory_slot)
        predicted_noise = output["noise"]
        length_logits = output["length_logits"]
        value = scheduler.step(
            predicted_noise, timestep_value, value).prev_sample
    length = length_logits.argmax(-1).reshape(batch, samples)
    return value.reshape(batch, samples, model.horizon, 4), length


@torch.inference_mode()
def ddim_sample_endpoint_residual(model, condition, endpoints, starts,
                                  minimum_steps, remaining_steps,
                                  samples=8, diffusion_steps=100,
                                  inference_steps=20,
                                  maximum_residual=0.3,
                                  step_scale=50.0 / 410.0):
    """Sample endpoint-conditioned lateral residual trajectories."""
    device = next(model.parameters()).device
    batch = condition[0].shape[0]
    if endpoints.shape[1] != samples:
        raise ValueError("one explicit endpoint is required per trajectory slot")
    features = endpoint_condition_features(
        starts, endpoints, minimum_steps, remaining_steps,
        horizon=model.horizon)
    repeated = tuple(value.repeat_interleave(samples, 0).to(device)
                     for value in condition)
    endpoint_features = features.reshape(batch * samples, -1).to(device)
    trajectory_slot = torch.arange(samples, device=device).repeat(batch)
    value = torch.randn(batch * samples, model.horizon, 1, device=device)
    scheduler = make_ddim_scheduler(diffusion_steps)
    scheduler.set_timesteps(inference_steps, device=device)
    extra_logits = None
    for timestep_value in scheduler.timesteps:
        timestep = timestep_value.expand(batch * samples)
        output = model(
            value, timestep, *repeated, endpoint_features,
            trajectory_slot=trajectory_slot)
        value = scheduler.step(
            output["noise"], timestep_value, value).prev_sample
        extra_logits = output["extra_step_logits"]
    extra = extra_logits.argmax(-1).reshape(batch, samples).cpu()
    remaining = torch.as_tensor(remaining_steps).long()
    if remaining.ndim == 1:
        remaining = remaining[:, None].expand_as(minimum_steps)
    lengths = torch.minimum(
        torch.as_tensor(minimum_steps).long() + extra,
        remaining.cpu()).clamp(1, model.horizon)
    positions, effective = residual_to_positions(
        value.reshape(batch, samples, model.horizon, 1), starts, endpoints,
        lengths, step_scale=step_scale,
        maximum_residual=maximum_residual)
    return positions, effective, extra
