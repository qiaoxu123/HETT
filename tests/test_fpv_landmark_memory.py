import numpy as np
from pathlib import Path
import importlib.util

from multiagent.fpv_landmark_memory.memory import AppearanceMemory, observation_quality
from multiagent.fpv_landmark_memory.projection import project_world_polygon

_builder_path = Path(__file__).resolve().parents[1] / "scripts/build_fpv_landmark_memory_dataset.py"
_spec = importlib.util.spec_from_file_location("fpv_builder", _builder_path)
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)
_eval_path = Path(__file__).resolve().parents[1] / "scripts/evaluate_fpv_landmark_memory.py"
_eval_spec = importlib.util.spec_from_file_location("fpv_eval", _eval_path)
_eval = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval)


def test_memory_is_bounded_and_keeps_best_quality():
    memory = AppearanceMemory(max_views_per_landmark=2)
    memory.update("a", np.array([1]), step=0, visible_ratio=.5, pixel_area=200,
                  distance_m=50, view_angle_deg=20)
    memory.update("a", np.array([2]), step=1, visible_ratio=.8, pixel_area=900,
                  distance_m=10, view_angle_deg=5)
    state = memory.update("a", np.array([3]), step=2, visible_ratio=.4, pixel_area=100,
                          distance_m=100, view_angle_deg=60)
    assert len(state.observations) == 2
    assert state.best.feature.tolist() == [2]
    assert memory.visible("a") and not memory.visible("missing")


def test_observation_quality_orders_useful_views():
    good = observation_quality(.8, 1500, 15, 10)
    poor = observation_quality(.1, 40, 180, 75)
    assert good > poor


def test_projection_returns_geometry_not_model_prediction():
    polygon = [(9, -2), (11, -2), (11, 2), (9, 2)]
    projected = project_world_polygon(polygon, (0, 0, 10, 0, 0), object_z=0)
    assert projected is not None
    assert projected["mask"].shape == (224, 224)
    assert projected["pixel_area"] > 0
    assert projected["projection"].endswith("no_occlusion")


def test_behind_camera_object_is_not_forced_into_view():
    polygon = [(-11, -2), (-9, -2), (-9, 2), (-11, 2)]
    assert project_world_polygon(polygon, (0, 0, 10, 0, 0), object_z=0) is None


def test_model_inputs_exclude_goal_and_gt_geometry():
    result = _builder.model_inputs({"instruction": "the red building", "image": "frame.jpg",
                                    "split": "val_seen", "sample_id": "x", "goal": [1, 2],
                                    "gt_mask": "mask.npy", "target_geometry_label": {"id": 4}})
    assert set(result) == {"instruction", "image", "split", "sample_id"}
    assert "goal" not in result and "gt_mask" not in result


def test_only_development_splits_are_allowed():
    assert _builder.SPLITS == ("train_seen", "val_seen", "val_unseen")
    assert "test_unseen" not in _builder.SPLITS


def test_history_contains_no_future_trajectory_step():
    history = _builder.history_steps(10)
    assert history[0] == (10, 0)
    assert all(step <= 10 and offset >= 0 for step, offset in history)


def test_no_visible_positive_is_not_counted_as_a_random_top1_hit():
    row = {"candidates": [{"landmark_id": 2, "world_xy": [0, 0]}],
           "referenced_landmark_ids": [2]}
    result = _eval.score_rank(row, {2: -float("inf")})
    assert result["top1"] == 0 and result["no_match"] == 1
    assert result["recall@20m"] == 0
