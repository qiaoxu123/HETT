import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from fpv_goal_approach.analyze_dataset import analyze
from fpv_goal_approach.build_dataset import pose5d, wrong_language
from fpv_goal_approach.dataset import read_jsonl, write_jsonl
from fpv_goal_approach.labels import (
    bearing_bin, clip_indices, distance_bin, geometric_arrival, phase,
    relative_bearing, wrap_radians,
)
from fpv_goal_approach.model import GoalApproachModel
from multiagent.space import Pose5D


def test_split_has_no_episode_leakage(tmp_path):
    metadata = tmp_path / "metadata"
    for split, episode in (("train_seen", "train-1"), ("val_seen", "seen-1"), ("val_unseen", "unseen-1")):
        write_jsonl(metadata / f"{split}.jsonl", [{
            "episode_id": episode, "target_match": 1, "visual_confirmed_arrival": 0,
            "distance_bin": 5, "bearing_bin": 2, "phase": 0, "negative_type": "none",
            "frames": ["x.png"], "render_source": "test",
        }])
    report = analyze(tmp_path)
    assert all(not overlap for overlap in report["leakage"].values())


def test_distance_bin_labels():
    assert [distance_bin(value) for value in (5, 15, 25, 35, 50, 80)] == [0, 1, 2, 3, 4, 5]


def test_bearing_bin_labels():
    assert [bearing_bin(math.radians(value)) for value in (-100, -40, 0, 40, 100)] == [0, 1, 2, 3, 4]


def test_wrong_language_negative_prefers_same_type_and_nearby():
    trajectory = SimpleNamespace(map_name="m", object_id=1)
    objects = {"m": {
        "1": {"position": [0, 0, 0], "object_type": "tower", "descriptions": ["goal"]},
        "2": {"position": [10, 0, 0], "object_type": "tower", "descriptions": ["hard negative"]},
        "3": {"position": [1, 0, 0], "object_type": "tree", "descriptions": ["easy negative"]},
    }}
    assert wrong_language(trajectory, objects) == ("hard negative", 2, 0)


def test_wrong_view_negative_changes_relative_bearing():
    original = relative_bearing(0, 0, 0, (10, 0))
    wrong = relative_bearing(0, 0, wrap_radians(math.radians(120)), (10, 0))
    assert bearing_bin(original) == 2
    assert bearing_bin(wrong) == 0


def test_clip_sampling():
    assert clip_indices(10) == [4, 7, 9, 10]
    assert clip_indices(2) == [0, 0, 1, 2]


def test_pose5d_preserves_pitch():
    pose = Pose5D(1, 2, 3, 0.4, -0.7)
    assert pose5d(pose) == [1.0, 2.0, 3.0, 0.4, -0.7]


def test_arrival_label():
    assert geometric_arrival(20.0) == 1
    assert geometric_arrival(20.01) == 0
    assert phase(19.0) == 3
    assert phase(25.0) == 2


def test_dataset_serialization(tmp_path):
    path = tmp_path / "sample.jsonl"
    rows = [{"episode_id": "e", "poses": [[1, 2, 3, 4, 5]]}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


class DummySiglip(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(projection_dim=8)
        self.anchor = nn.Parameter(torch.zeros(1))

    def get_image_features(self, pixel_values):
        return torch.ones(pixel_values.shape[0], 8, device=pixel_values.device) + self.anchor

    def get_text_features(self, input_ids, attention_mask):
        return torch.ones(input_ids.shape[0], 8, device=input_ids.device) + self.anchor


def test_model_forward_shape():
    model = GoalApproachModel(frames=4, hidden_dim=16, siglip_model=DummySiglip())
    output = model(
        torch.randn(2, 4, 3, 16, 16), torch.ones(2, 5, dtype=torch.long),
        torch.ones(2, 5, dtype=torch.long), torch.zeros(2, 5),
    )
    assert output["target_match"].shape == (2, 2)
    assert output["distance_bin"].shape == (2, 6)
    assert output["bearing_bin"].shape == (2, 5)

