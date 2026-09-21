"""Small, dependency-free helpers for coarse/fine stage control."""


def advance_teacher_stage1(stage1_steps, episode_index, move_iteration, trajectory_length):
    if move_iteration <= 0:
        raise ValueError("move_iteration must be positive")
    if trajectory_length <= 0:
        raise ValueError("teacher trajectory must not be empty")
    stage1_steps[episode_index] += 1
    trajectory_index = stage1_steps[episode_index] * move_iteration
    return trajectory_index if trajectory_index < trajectory_length else -1
