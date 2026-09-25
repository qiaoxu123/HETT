import torch

from scripts.audit_spatial_relation_probe import (
    canonical_prompt,
    cue_names,
    make_candidates,
)


def test_relation_cue_requires_direct_landmark_pattern():
    assert cue_names("The target is to the left of [MASK].") == ["left"]
    assert cue_names("The target has a door on its left near [MASK].") == []


def test_mirrored_candidates_cover_all_sign_combinations():
    geometry = make_candidates(torch.tensor([0.2, -0.1]))

    assert geometry.shape == (4, 5)
    assert torch.allclose(geometry[:, :2], torch.tensor([
        [0.2, -0.1], [-0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]]))


def test_canonical_prompt_matches_normalized_map_axes():
    assert canonical_prompt(torch.tensor([-0.2, -0.1])) == (
        "The target is left and above of [MASK].")
