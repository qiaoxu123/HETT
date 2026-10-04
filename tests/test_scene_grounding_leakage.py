from multiagent.scene_grounding.dataset import leakage_audit
from pathlib import Path


def _row(split, episode, object_key, source="teacher_trajectory_pose"):
    return {"split": split, "episode_id": episode, "object_key": object_key,
            "sample_id": f"{episode}:0", "observation_source": source}


def test_no_episode_or_object_cross_split_leakage():
    audit = leakage_audit([_row("train_seen", "train:0", "map:1"),
                           _row("val_seen", "seen:0", "map:2"),
                           _row("val_unseen", "unseen:0", "map2:1")])
    assert not audit["forbidden_test_unseen"]
    assert not audit["candidate_centered_observations"]
    assert all(not any(values.values()) for values in audit["overlap"].values())


def test_candidate_centered_observation_is_detected():
    audit = leakage_audit([_row("train_seen", "train:0", "map:1", "candidate_center")])
    assert audit["candidate_centered_observations"]


def test_visual_inference_has_no_goal_future_or_map_id_input():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/extract_scene_evidence.py").read_text()
    assert "target_positions" not in source
    assert "future_rgb" not in source.casefold()
    assert "future_observation" not in source.casefold()
    assert 'processor(images=images' in source


def test_oracle_uses_local_scene_not_nearest_object_interface():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/evaluate_scene_oracle.py").read_text()
    assert "nearest" not in source.casefold()
    assert "expected_scene" in source
