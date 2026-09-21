import torch

from multiagent.models.token_grounding_heatmap import TokenGroundingHeatmap


BERT = "/home/tenant2/dataext/home-cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"


def test_shapes_frozen_language_and_invalid_landmark():
    model = TokenGroundingHeatmap(BERT, 4, 9, 6, hidden_dim=32)
    model.train()
    assert not model.bert.training
    assert not any(parameter.requires_grad for parameter in model.bert.parameters())
    batch = 2
    output = model(
        torch.rand(batch, 4, 9, 8, 8), torch.rand(batch, 4, 6, 8, 8),
        torch.rand(batch, 1, 8, 8), torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.]]),
        torch.tensor([[101, 2187, 1997, 2311, 102], [101, 2157, 1997, 2311, 102]]),
        torch.ones(batch, 5, dtype=torch.long),
        torch.tensor([[[101, 2311, 102]] * 4] * batch),
        torch.tensor([[[0, 1, 0]] * 4] * batch),
    )
    logits, weights, gate, attention = output
    assert logits.shape == (batch, 1, 8, 8)
    assert weights.shape == (batch, 4, 9, 6)
    assert gate.shape == (batch,)
    assert attention.shape == (batch, 4, 5)
    assert weights[0, 2:].abs().sum().item() == 0
