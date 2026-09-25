import torch

from scripts.evaluate_per_landmark_rgb_signatures import InstanceSignatureRanker


def test_instance_signature_ranker_is_landmark_order_invariant():
    torch.manual_seed(0)
    model = InstanceSignatureRanker(visual_dim=4, hidden=32).eval()
    candidate = torch.randn(2, 3, 4)
    relation = torch.randn(2, 4, 768)
    cue = torch.randn(2, 4, 12)
    geometry = torch.randn(2, 3, 4, 10)
    visual = torch.randn(2, 3, 4, 4)
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    order = torch.tensor([2, 0, 3, 1])
    with torch.inference_mode():
        expected = model(candidate, relation, cue, geometry, visual, valid)
        actual = model(candidate, relation[:, order], cue[:, order],
                       geometry[:, :, order], visual[:, :, order], valid[:, order])
    assert torch.allclose(expected, actual, atol=1e-5)
