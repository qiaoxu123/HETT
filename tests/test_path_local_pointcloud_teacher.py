import inspect

from multiagent.space import Pose4D
from independent_nav.official_protocol import OfficialProtocol
from scripts.evaluate_path_local_pointcloud_teacher import (
    first_joint_path_observation, visible_in_official_crop,
)
from scripts.evaluate_path_local_teacher_closed_loop import (
    deployment_plans, rollout_teacher_plans,
)
from scripts.evaluate_path_local_rgb_evidence import reconstruct_observation_pose
from scripts.evaluate_path_local_rgb_unseen import geometry_records
from scripts.evaluate_online_rgb_closed_loop import inference_plans, rollout_online_rgb


def test_official_crop_visibility_rotates_with_yaw():
    pose = Pose4D(0, 0, 50, 0)
    assert visible_in_official_crop(pose, (49, 49))
    assert not visible_in_official_crop(pose, (51, 0))


def test_path_observation_selection_has_no_target_input_and_respects_budget():
    assert "target" not in inspect.signature(
        first_joint_path_observation).parameters
    protocol = OfficialProtocol()
    pose, step, visible = first_joint_path_observation(
        Pose4D(0, 0, 50, 0), (180, 0), (180, 30), protocol)
    assert visible
    assert 1 <= step < protocol.max_action_len
    assert pose.x > 0


def test_teacher_rollout_plan_strips_target_and_rollout_has_no_target_input():
    records = [{
        "prior": [0.6, 0.4], "evidence": [0.1, 0.9],
        "modes": [[0.2, 0.3], [0.7, 0.8]],
        "observation_action_step": 3, "target": [0.7, 0.8],
    }]
    plans = deployment_plans(records, beta=1.0)
    assert plans[0]["selected_rank"] == 1
    assert "target" not in plans[0]
    assert "target" not in inspect.signature(rollout_teacher_plans).parameters


def test_rgb_observation_pose_replay_has_no_target_input():
    assert "target" not in inspect.signature(reconstruct_observation_pose).parameters
    pose = reconstruct_observation_pose(
        Pose4D(0, 0, 50, 0), [(150, 0), (150, 20)], 2)
    assert pose.x > 0


def test_unseen_geometry_builder_has_no_target_argument():
    assert "target" not in inspect.signature(geometry_records).parameters


def test_strict_online_plan_and_rollout_exclude_target():
    record = {
        "index": 0, "modes": [[0.2, 0.3], [0.7, 0.8]],
        "prior": [0.6, 0.4], "evidence": [0.1, 0.9],
        "observation_action_step": 3, "target": [0.7, 0.8],
    }
    plan = inference_plans([record], [1])[0]
    assert "target" not in plan
    assert "target" not in inspect.signature(rollout_online_rgb).parameters
