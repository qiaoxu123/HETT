import numpy as np

from multiagent.scene_grounding.belief_update import additive_candidate_update, temporal_additive_update
from multiagent.scene_grounding.temporal_fusion import fuse_scores


def test_additive_update_preserves_distribution_and_strengthens_evidence():
    prior = np.asarray([0.5, 0.3, 0.2])
    posterior = additive_candidate_update(prior, [0, 1, -1])
    np.testing.assert_allclose(posterior.sum(), 1.0)
    assert posterior[1] > prior[1]


def test_temporal_update_uses_only_supplied_prefix():
    prior = np.asarray([0.5, 0.5])
    prefix = temporal_additive_update(prior, [[1, -1], [1, -1]], decay=0.8)
    with_future = temporal_additive_update(prior, [[1, -1], [1, -1], [-2, 2]], decay=0.8)
    assert prefix[0] > prefix[1]
    assert not np.allclose(prefix, with_future)


def test_three_frame_mean_is_causal():
    assert fuse_scores([1, 2, 9], window=2) == [1.0, 1.5, 5.5]
