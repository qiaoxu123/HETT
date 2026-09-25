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
