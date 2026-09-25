"""Target-free primitives for causal online semantic/RGB fusion samples."""

from __future__ import annotations

import numpy as np


FORBIDDEN_INFERENCE_KEYS = {
    "target", "target_coordinate", "target_coordinates", "target_object_id",
    "goal", "gt_trajectory",
}


def candidate_prior(candidates, grid_size=20, sigma_m=25.0, map_size_m=410.0):
    """Create an initial belief from ranked predicted candidates only."""
    candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 2)
    axis = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    cells = np.stack((xx, yy), axis=-1)
    belief = np.zeros((grid_size, grid_size), dtype=np.float32)
    sigma = sigma_m / map_size_m
    for rank, point in enumerate(candidates):
        distance2 = np.square(cells - point).sum(-1)
        belief += np.exp(-0.5 * distance2 / (sigma * sigma)) / (rank + 1.0)
    total = float(belief.sum())
    if total <= 0:
        raise ValueError("candidate prior has no probability mass")
    return belief / total


def previous_belief_reference(step):
    """Declare causal recurrent state without fabricating target-based history."""
    if step < 0:
        raise ValueError("step must be non-negative")
    return "initial_candidate_prior" if step == 0 else f"fusion_output_step_{step - 1}"


def contains_forbidden_inference_key(value):
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_INFERENCE_KEYS or contains_forbidden_inference_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_forbidden_inference_key(item) for item in value)
    return False


def validate_episode_sample(episode, max_actions=20, map_size_m=410.0):
    """Validate alignment, causality, budget and conservative visibility."""
    errors = []
    if contains_forbidden_inference_key(episode):
        errors.append("forbidden target/goal key in inference sample")
    prior = np.asarray(episode.get("initial_belief", []), dtype=np.float32)
    if prior.ndim != 2 or not np.isclose(prior.sum(), 1.0, atol=1e-5):
        errors.append("initial belief is not a normalized 2-D map")
    frames = episode.get("frames", [])
    previous_pose = None
    for step, frame in enumerate(frames):
        if frame.get("step") != step:
            errors.append(f"non-contiguous frame order at {step}")
        if frame.get("remaining_budget") != max_actions - step:
            errors.append(f"wrong remaining budget at {step}")
        if frame.get("previous_belief_ref") != previous_belief_reference(step):
            errors.append(f"non-causal previous belief reference at {step}")
        visible = set(frame.get("frustum_visible_indices", []))
        negative = set(frame.get("negative_evidence_allowed_indices", []))
        if not negative.issubset(visible):
            errors.append(f"negative evidence outside visibility at {step}")
        if negative and not frame.get("occlusion_verified", False):
            errors.append(f"negative evidence without occlusion verification at {step}")
        pose = frame.get("pose", {})
        current_pose = np.asarray([pose.get("x", np.nan), pose.get("y", np.nan)])
        if not np.isfinite(current_pose).all():
            errors.append(f"invalid pose at {step}")
        if previous_pose is not None:
            # One official outer action contains at most 10 x 5 m translations.
            if np.linalg.norm(current_pose - previous_pose) > 50.0 + 1e-4:
                errors.append(f"pose jump exceeds one official action at {step}")
        previous_pose = current_pose
    if len(frames) > max_actions:
        errors.append("episode exceeds action budget")
    return errors
