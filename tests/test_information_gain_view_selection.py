import numpy as np

from scripts.evaluate_information_gain_view_selection import (
    choose_view,
    simulate_challenger,
    view_quality,
)


def test_view_quality_prefers_centered_candidates():
    candidates = np.asarray([[0.0, 0.0], [20.0, 0.0]])
    centered = view_quality(candidates, np.asarray([10.0, 0.0]), 50.0)
    offset = view_quality(candidates, np.asarray([-40.0, 0.0]), 50.0)
    assert centered.min() > offset.min()


def test_selector_chooses_joint_view_without_target_coordinates():
    candidates = np.asarray([[0.0, 0.0], [20.0, 0.0]])
    row = {
        "candidates": np.asarray([[0.0, 0.0], [0.05, 0.0]]),
        "candidates_world": candidates,
        "view_centers": np.asarray([[0.0, 0.0], [10.0, 0.0]]),
        "quality": np.asarray([[1.0, 0.1], [0.8, 0.8]]),
    }
    selected = choose_view(
        row, accumulated=np.zeros(2), seen_quality=np.zeros(2),
        current=np.asarray([0.0, 0.0]), selected=set(), beta=0.0,
        separation_weight=1.0, travel_weight=0.0)
    assert selected == 1


def test_challenger_uses_first_view_evidence_to_choose_second_candidate():
    row = {
        "candidates": np.zeros((3, 2)),
        "correct_evidence": np.asarray([
            [0.0, 0.0, 0.0],  # unused start option
            [0.1, 0.2, 0.9],  # first candidate-centered observation
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
    }
    _, selected = simulate_challenger(
        row, "correct_evidence", beta=0.1, views=2)
    assert selected == [1, 3]
