import torch

from scripts.fuse_relation_type_scores import normalized, remap_features


def test_score_normalization_is_per_row():
    value = normalized(torch.tensor([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]]))
    assert torch.allclose(value.mean(-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(value[1], torch.zeros(3))


def test_semantic_feature_remapping_handles_reordered_rows():
    first = {"map_name": "a", "description": "first", "target": [0.1, 0.2]}
    second = {"map_name": "a", "description": "second", "target": [0.3, 0.4]}
    source = torch.tensor([[1.0], [2.0]])
    result = remap_features([first, second], source, [second, first])
    assert result.tolist() == [[2.0], [1.0]]
