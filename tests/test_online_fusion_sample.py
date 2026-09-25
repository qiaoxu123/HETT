import numpy as np

from independent_nav.online_fusion_sample import (
    candidate_prior, contains_forbidden_inference_key,
    previous_belief_reference, validate_episode_sample,
)


def valid_episode():
    return {
        "initial_belief": candidate_prior([[0.25, 0.25], [0.75, 0.75]]).tolist(),
        "frames": [
            {
                "step": 0, "remaining_budget": 20,
                "previous_belief_ref": "initial_candidate_prior",
                "frustum_visible_indices": [0, 1],
                "negative_evidence_allowed_indices": [],
                "occlusion_verified": False,
                "pose": {"x": 0.0, "y": 0.0},
            },
            {
                "step": 1, "remaining_budget": 19,
                "previous_belief_ref": "fusion_output_step_0",
                "frustum_visible_indices": [1, 2],
                "negative_evidence_allowed_indices": [1],
                "occlusion_verified": True,
                "pose": {"x": 10.0, "y": 0.0},
            },
        ],
    }


def test_candidate_prior_is_normalized_and_rank_sensitive():
    belief = candidate_prior([[0.25, 0.25], [0.75, 0.75]])
    assert belief.shape == (20, 20)
    assert np.isclose(belief.sum(), 1.0)
    assert belief[5, 5] > belief[15, 15]


def test_previous_belief_reference_is_causal():
    assert previous_belief_reference(0) == "initial_candidate_prior"
    assert previous_belief_reference(3) == "fusion_output_step_2"


def test_valid_online_episode_passes():
    assert validate_episode_sample(valid_episode()) == []


def test_unverified_negative_evidence_is_rejected():
    episode = valid_episode()
    episode["frames"][0]["negative_evidence_allowed_indices"] = [0]
    assert any("without occlusion" in error
               for error in validate_episode_sample(episode))


def test_target_coordinate_key_is_rejected():
    episode = valid_episode()
    episode["target"] = [0.1, 0.2]
    assert contains_forbidden_inference_key(episode)
    assert any("forbidden" in error for error in validate_episode_sample(episode))


def test_pose_jump_over_one_action_is_rejected():
    episode = valid_episode()
    episode["frames"][1]["pose"]["x"] = 51.0
    assert any("pose jump" in error for error in validate_episode_sample(episode))
