"""Minimal rollout book-keeping helpers (no simulator/model imports)."""


def should_record_pose(previous_pose, new_pose, *, ended):
    """Keep a newly executed terminal action in the scored trajectory.

    A stop without movement should not append a synthetic extra step.
    In-progress episodes retain legacy per-timestep recording semantics.
    """
    return not bool(ended) or new_pose != previous_pose
