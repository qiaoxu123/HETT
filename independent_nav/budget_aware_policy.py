"""Budget-aware path-local action scoring for active goal disambiguation.

The policy operates only on predicted belief modes and the current pose.  It
does not receive a reference target.  Each waypoint is both a navigation step
and a potential sensing step under the frozen 20-action protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from independent_nav.official_protocol import (
    BudgetedWaypoint, OfficialProtocol, generate_budgeted_waypoints,
    optimistic_translation_steps,
)


@dataclass(frozen=True)
class ActionUtility:
    name: str
    xy: tuple[float, float]
    utility: float
    goal_progress: float
    information_gain: float
    budget_penalty: float


def normalized_entropy(probabilities) -> float:
    """Entropy in [0, 1], including safe handling of exact zeroes."""
    value = np.asarray(probabilities, dtype=np.float64)
    value = np.clip(value / value.sum(), 1e-8, 1.0)
    return float(-(value * np.log(value)).sum() / np.log(len(value)))


def expected_goal_progress(current_xy, waypoint_xy, modes, probabilities,
                           protocol: OfficialProtocol) -> float:
    """Expected distance reduction, normalized by one action's reach."""
    current = np.asarray(current_xy, dtype=np.float64)
    waypoint = np.asarray(waypoint_xy, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    before = np.linalg.norm(modes - current[None], axis=1)
    after = np.linalg.norm(modes - waypoint[None], axis=1)
    progress_m = float(np.dot(probabilities, before - after) * protocol.map_size_m)
    return progress_m / protocol.maximum_translation_per_action_m


def joint_view_quality(waypoint_xy, modes, protocol: OfficialProtocol) -> float:
    """Target-free proxy for whether one view can compare two hypotheses.

    The official top-down crop covers a 100 x 100 m square at 50 m altitude.
    We use a conservative circular support, reward keeping both modes away from
    the crop edge, and reward spatial separation that can actually be resolved.
    """
    waypoint = np.asarray(waypoint_xy, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.float64)
    distances_m = np.linalg.norm(modes - waypoint[None], axis=1) * protocol.map_size_m
    edge_margin = np.clip(1.0 - distances_m.max() / protocol.altitude_m, 0.0, 1.0)
    separation_m = float(np.linalg.norm(modes[0] - modes[1]) * protocol.map_size_m)
    resolvability = np.clip(separation_m / 25.0, 0.0, 1.0)
    return float(edge_margin * resolvability)


def remaining_budget_penalty(waypoint_xy, modes, probabilities,
                             remaining_after_action: int,
                             protocol: OfficialProtocol) -> float:
    """Expected optimistic shortfall after taking an action."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    shortfalls = [max(0, optimistic_translation_steps(
        waypoint_xy, mode, protocol) - remaining_after_action) for mode in modes]
    return float(np.dot(probabilities, shortfalls))


def score_path_actions(current_xy, modes, probabilities, remaining_actions,
                       protocol=OfficialProtocol(), information_weight=0.5,
                       budget_weight=2.0, lateral_offset_m=10.0):
    """Score legal one-step waypoints by progress, information and budget."""
    modes = np.asarray(modes, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    committed = int(probabilities.argmax())
    alternate = 1 - committed
    generated = generate_budgeted_waypoints(
        current_xy, modes[committed], modes[alternate], remaining_actions,
        protocol, lateral_offset_m)
    entropy = normalized_entropy(probabilities)
    output = []
    for candidate in generated:
        progress = expected_goal_progress(
            current_xy, candidate.xy, modes, probabilities, protocol)
        information = joint_view_quality(candidate.xy, modes, protocol)
        penalty = remaining_budget_penalty(
            candidate.xy, modes, probabilities, remaining_actions - 1, protocol)
        utility = (progress + information_weight * entropy * information
                   - budget_weight * penalty)
        output.append(ActionUtility(
            candidate.name, candidate.xy, utility, progress, information,
            penalty))
    return sorted(output, key=lambda item: (-item.utility, item.name))


def choose_path_action(current_xy, modes, probabilities, remaining_actions,
                       protocol=OfficialProtocol(), **kwargs) -> ActionUtility:
    scored = score_path_actions(
        current_xy, modes, probabilities, remaining_actions, protocol, **kwargs)
    if scored:
        return scored[0]
    # The generator can be empty only at the final infeasible boundary.  A
    # direct one-action waypoint to the current posterior mode is the safest
    # deterministic fallback and still contains no target information.
    mode = tuple(np.asarray(modes)[int(np.argmax(probabilities))])
    return ActionUtility("fallback_commit", mode, -float("inf"), 0.0, 0.0, 0.0)
