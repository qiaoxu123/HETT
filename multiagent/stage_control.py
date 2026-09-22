"""Small, dependency-free helpers for coarse/fine stage control."""


def target_progress(distance_m, scale_m=100.0):
    """Convert target distance to the progress supervision used by HETT."""
    if scale_m <= 0:
        raise ValueError("progress scale must be positive")
    return min(1.0, max(0.0, 1.0 - float(distance_m) / scale_m))


def advance_teacher_stage1(stage1_steps, episode_index, move_iteration, trajectory_length):
    if move_iteration <= 0:
        raise ValueError("move_iteration must be positive")
    if trajectory_length <= 0:
        raise ValueError("teacher trajectory must not be empty")
    stage1_steps[episode_index] += 1
    trajectory_index = stage1_steps[episode_index] * move_iteration
    return trajectory_index if trajectory_index < trajectory_length else -1
