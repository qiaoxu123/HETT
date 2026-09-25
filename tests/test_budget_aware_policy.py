import numpy as np

from independent_nav.budget_aware_policy import (
    choose_path_action, normalized_entropy, score_path_actions,
)
from independent_nav.official_protocol import OfficialProtocol


def test_entropy_is_high_only_for_ambiguous_belief():
    assert normalized_entropy([0.5, 0.5]) > 0.99
    assert normalized_entropy([0.99, 0.01]) < 0.1


def test_uncertain_policy_values_midpoint_information():
    protocol = OfficialProtocol()
    scored = score_path_actions(
        (0.1, 0.5), [(0.5, 0.4), (0.5, 0.6)], [0.5, 0.5], 20,
        protocol, information_weight=2.0)
    by_name = {item.name: item for item in scored}
    assert by_name["toward_midpoint"].information_gain >= by_name["toward_top1"].information_gain
    assert scored[0].name == "toward_midpoint"


def test_confident_policy_commits_toward_top_mode():
    selected = choose_path_action(
        (0.1, 0.5), [(0.7, 0.4), (0.4, 0.6)], [0.99, 0.01], 20,
        information_weight=0.5)
    assert selected.name == "toward_top1"


def test_policy_inputs_do_not_need_reference_target():
    selected = choose_path_action(
        (0.2, 0.2), np.asarray([(0.8, 0.2), (0.8, 0.4)]),
        np.asarray([0.6, 0.4]), 8)
    assert np.isfinite(selected.xy).all()
