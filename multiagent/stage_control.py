"""Small, dependency-free helpers for coarse/fine stage control."""

from dataclasses import dataclass


@dataclass
class RecoveryState:
    """Per-episode inference state; it contains no ground-truth information."""

    fine: bool = False
    far_count: int = 0
    stop_count: int = 0
    switches: int = 0
    recoveries: int = 0


def update_recovery_state(
    state,
    predicted_goal_distance,
    predicted_progress,
    enter_distance=5.0,
    recover_distance=20.0,
    recovery_patience=2,
    stop_threshold=0.95,
    stop_patience=2,
):
    """Choose ``coarse``, ``fine`` or ``stop`` from model predictions only.

    Two-threshold hysteresis avoids flipping stages on one noisy target.  A
    stop requires progress and target-distance predictions to agree; conflicting
    evidence resets the stop counter and can eventually trigger coarse recovery.
    """
    if enter_distance < 0 or recover_distance <= enter_distance:
        raise ValueError("recover_distance must be greater than enter_distance")
    if recovery_patience < 1 or stop_patience < 1:
        raise ValueError("patience values must be positive")

    if not state.fine:
        state.far_count = 0
        state.stop_count = 0
        if predicted_goal_distance <= enter_distance:
            state.fine = True
            state.switches += 1
            return "fine"
        return "coarse"

    if predicted_goal_distance >= recover_distance:
        state.far_count += 1
        state.stop_count = 0
        if state.far_count >= recovery_patience:
            state.fine = False
            state.far_count = 0
            state.recoveries += 1
            return "coarse"
        return "fine"

    state.far_count = 0
    if predicted_progress >= stop_threshold:
        state.stop_count += 1
        if state.stop_count >= stop_patience:
            return "stop"
    else:
        state.stop_count = 0
    return "fine"


def advance_teacher_stage1(stage1_steps, episode_index, move_iteration, trajectory_length):
    """Advance one episode and return its teacher trajectory index.

    ``stage1_steps`` is intentionally maintained per episode.  Returning ``-1``
    preserves the rollout's historical behavior of falling back to the final
    teacher pose when the requested index is beyond the available trajectory.
    """
    if move_iteration <= 0:
        raise ValueError("move_iteration must be positive")
    if trajectory_length <= 0:
        raise ValueError("teacher trajectory must not be empty")

    stage1_steps[episode_index] += 1
    trajectory_index = stage1_steps[episode_index] * move_iteration
    return trajectory_index if trajectory_index < trajectory_length else -1
