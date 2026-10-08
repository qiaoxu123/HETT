import numpy as np
import pytest

from multiagent.visual_goal.leakage import assert_visual_only_inputs
from multiagent.visual_goal.retrieval import choose_distance_stratum
from multiagent.visual_goal.template_builder import build_split


def test_visual_encoder_only_receives_pixel_tensors():
    assert assert_visual_only_inputs({"pixel_values": np.zeros((1, 3, 4, 4))})
    with pytest.raises(ValueError):
        assert_visual_only_inputs({"pixel_values": np.zeros(1), "target_position": [0, 0]})
    with pytest.raises(ValueError):
        assert_visual_only_inputs({"pixel_values": np.zeros(1), "map_name": "private-map-label"})


def test_query_sampling_returns_only_real_trajectory_indices():
    poses = [[0, 0, 50, 0], [25, 0, 50, 0], [50, 0, 50, 0], [100, 0, 50, 0]]
    selected = choose_distance_stratum(poses, (0, 0))
    assert {i for i, _ in selected}.issubset(set(range(len(poses))))
    assert all(abs(d - np.linalg.norm(np.asarray(poses[i][:2]))) < 1e-6 for i, d in selected)


def test_test_unseen_cannot_be_built_or_opened():
    with pytest.raises(ValueError, match="test_unseen is intentionally disabled"):
        build_split("test_unseen", None, None, None)
