import math

import pytest

from multiagent.scene_grounding.dataset import reject_forbidden_split, select_distance_steps, trajectory_pose


def test_samples_at_most_one_real_step_per_distance_bucket():
    trajectory = [[100, 0, 50, 1, 0, 0], [70, 0, 50, 1, 0, 0],
                  [30, 0, 50, 1, 0, 0], [10, 0, 50, 1, 0, 0]]
    selected = select_distance_steps(trajectory, (0, 0))
    indices = [value[0] for value in selected]
    assert indices == [3, 2, 1, 0]
    assert all(trajectory[index] in trajectory for index in indices)


def test_direction_vector_pose_conversion():
    pose = trajectory_pose([1, 2, 3, 0, 1, 0])
    assert pose[:3] == (1, 2, 3)
    assert pose.yaw == pytest.approx(math.pi / 2)


def test_forbidden_test_split():
    with pytest.raises(ValueError): reject_forbidden_split("test_unseen")
