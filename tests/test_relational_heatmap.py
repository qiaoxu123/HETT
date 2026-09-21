import torch

from multiagent.models.relational_heatmap import (GeometricCandidateSelector,
                                                  HETTBertCandidateSelector,
                                                  RelationalHeatmap)


def test_heatmap_shape_and_target_is_not_an_input():
    torch.manual_seed(0)
    model = RelationalHeatmap(vocab_size=32, max_landmarks=4,
                              angle_bases=9, distance_bases=6).eval()
    angles = torch.rand(2, 4, 9, 64, 64)
    distances = torch.rand(2, 4, 6, 64, 64)
    pair = torch.rand(2, 1, 64, 64)
    ref_tokens = torch.randint(1, 32, (2, 4, 12))
    global_tokens = torch.randint(1, 32, (2, 12))
    valid = torch.ones(2, 4)
    with torch.no_grad():
        logits, weights, gate = model(angles, distances, pair, ref_tokens, global_tokens, valid)
    assert logits.shape == (2, 1, 64, 64)
    assert weights.shape == (2, 4, 9, 6)
    assert gate.shape == (2,)
    torch.testing.assert_close(weights.sum((-1, -2)), torch.ones(2, 4))


def test_invalid_landmark_has_zero_mixture_weight():
    model = RelationalHeatmap(vocab_size=16, max_landmarks=4,
                              angle_bases=9, distance_bases=6).eval()
    valid = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        _, weights, _ = model(torch.rand(1, 4, 9, 64, 64), torch.rand(1, 4, 6, 64, 64),
                              torch.zeros(1, 1, 64, 64),
                              torch.randint(0, 16, (1, 4, 8)), torch.randint(1, 16, (1, 8)), valid)
    assert weights[0, 0].sum().item() > 0.99
    assert weights[0, 1:].abs().sum().item() == 0


def test_candidate_selector_masks_invalid_landmarks():
    model = GeometricCandidateSelector(vocab_size=16, max_landmarks=4,
                                       candidates_per_landmark=54).eval()
    valid = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        logits = model(torch.randint(1, 16, (1, 4, 8)), torch.randint(1, 16, (1, 8)),
                       valid, torch.tensor([0.0]))
    assert logits.shape == (1, 217)
    assert torch.all(logits[0, 54:] < -1000)


def test_hett_bert_selector_accepts_frozen_embeddings():
    model = HETTBertCandidateSelector(max_landmarks=4, candidates_per_landmark=54).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 4, 768), torch.randn(2, 768),
                       torch.ones(2, 4), torch.ones(2))
    assert logits.shape == (2, 217)
