"""Target-free geometric teacher utilities for active trajectories."""

from __future__ import annotations

import torch


def geometric_trajectory_utility(positions, path_valid, endpoints,
                                 landmark_centers, landmark_valid,
                                 belief_modes, map_size_m=410.0,
                                 view_range_m=100.0):
    """Score joint visibility, candidate disambiguation and path cost.

    All inputs are inference-visible hypotheses.  No true target coordinate is
    accepted by this function.
    """
    valid = path_valid.float()
    path = positions.float()
    endpoint_distance = torch.linalg.vector_norm(
        path - endpoints[:, :, None], dim=-1) * map_size_m
    landmark_distance = torch.linalg.vector_norm(
        path[:, :, :, None] - landmark_centers[:, None, None], dim=-1
    ) * map_size_m
    landmark_distance = landmark_distance.masked_fill(
        ~landmark_valid[:, None, None], float("inf"))
    nearest_landmark = landmark_distance.amin(-1)
    joint = (torch.exp(-endpoint_distance / view_range_m)
             * torch.exp(-nearest_landmark / view_range_m) * valid).amax(-1)

    first, second = belief_modes[:, 0], belief_modes[:, 1]
    to_first = first[:, None, None] - path
    to_second = second[:, None, None] - path
    distance_first = torch.linalg.vector_norm(to_first, dim=-1).clamp_min(1e-6)
    distance_second = torch.linalg.vector_norm(to_second, dim=-1).clamp_min(1e-6)
    cosine = (to_first * to_second).sum(-1) / (distance_first * distance_second)
    angular_separation = ((1 - cosine.clamp(-1, 1)) / 2).sqrt()
    both_visible = torch.exp(
        -(distance_first + distance_second) * map_size_m / (2 * view_range_m))
    disambiguation = (angular_separation * both_visible * valid).amax(-1)

    step = torch.linalg.vector_norm(path[:, :, 1:] - path[:, :, :-1], dim=-1)
    path_length = step.sum(-1) * map_size_m
    cost = (path_length / (20 * 5.0)).clamp(0, 1)
    utility = 0.55 * joint + 0.35 * disambiguation - 0.10 * cost
    return {
        "utility": utility,
        "joint_visibility": joint,
        "disambiguation": disambiguation,
        "normalized_cost": cost,
    }
