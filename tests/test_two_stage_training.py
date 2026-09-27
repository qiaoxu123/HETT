from __future__ import annotations

import torch
from torch import nn

from multiagent.two_stage import (
    MultiHypothesisConditioning,
    RelationalCandidateBeliefHead,
    SparseQuadtreeBeliefHead,
    TargetConditioning,
    candidate_recall,
    candidate_supervision,
    configure_stage_parameters,
    coordinate_gt_probability,
    select_action_coordinates,
    target_coordinate_losses,
)


class TinyNavigationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.target_head = nn.Linear(4, 2)
        self.target_conditioning = TargetConditioning(4, dropout=0.0)
        self.decoder_2_action_full = nn.Linear(4, 2)
        self.decoder_2_progress_full = nn.Linear(4, 1)


def test_target_losses_are_zero_at_truth_and_use_metre_scale():
    truth = torch.tensor([[0.25, 0.75]])
    current = torch.tensor([[0.10, 0.20]])
    exact = target_coordinate_losses(truth, truth, current, 410.0, 10.0)
    shifted = target_coordinate_losses(
        truth + torch.tensor([[10.0 / 410.0, 0.0]]),
        truth,
        current,
        410.0,
        10.0,
    )

    assert exact.coordinate.item() == 0.0
    assert exact.distance.item() == 0.0
    assert exact.bearing.item() < 1e-6
    assert torch.allclose(shifted.error_m, torch.tensor([10.0]), atol=1e-4)
    assert shifted.coordinate.item() > exact.coordinate.item()


def test_action_coordinate_curriculum_reaches_configured_endpoints():
    assert coordinate_gt_probability(1, 10, 1.0, 0.1) == 1.0
    assert abs(coordinate_gt_probability(10, 10, 1.0, 0.1) - 0.1) < 1e-8


def test_selected_prediction_is_detached_and_mask_selects_truth():
    predicted = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
    truth = torch.tensor([[0.8, 0.9], [0.6, 0.7]], requires_grad=True)
    selected = select_action_coordinates(
        predicted, truth, torch.tensor([True, False])
    )

    assert torch.allclose(selected[0], truth.detach()[0])
    assert torch.allclose(selected[1], predicted.detach()[1])
    assert not selected.requires_grad


def test_target_conditioning_changes_motion_without_changing_shape():
    torch.manual_seed(0)
    module = TargetConditioning(d_model=8, dropout=0.0).eval()
    motion = torch.randn(2, 2, 8)
    current = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
    target_a = torch.tensor([[0.2, 0.2], [0.3, 0.3]])
    target_b = torch.tensor([[0.8, 0.8], [0.9, 0.9]])

    output_a = module(motion, target_a, current)
    output_b = module(motion, target_b, current)

    assert output_a.shape == motion.shape
    assert not torch.allclose(output_a, output_b)


def test_stage_parameter_ownership_is_strict():
    language = nn.Linear(4, 4)
    vision = nn.Linear(4, 4)
    navigation = TinyNavigationModel()

    configure_stage_parameters(language, vision, navigation, 'joint')
    assert all(parameter.requires_grad for parameter in navigation.backbone.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in navigation.target_conditioning.parameters()
    )

    configure_stage_parameters(language, vision, navigation, 'target')
    assert all(parameter.requires_grad for parameter in language.parameters())
    assert all(parameter.requires_grad for parameter in navigation.target_head.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in navigation.decoder_2_action_full.parameters()
    )

    configure_stage_parameters(language, vision, navigation, 'action')
    assert not any(parameter.requires_grad for parameter in language.parameters())
    assert not any(parameter.requires_grad for parameter in vision.parameters())
    assert not any(parameter.requires_grad for parameter in navigation.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in navigation.target_head.parameters())
    assert all(
        parameter.requires_grad
        for parameter in navigation.target_conditioning.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in navigation.decoder_2_action_full.parameters()
    )


def test_action_optimizer_cannot_mutate_frozen_predictor():
    language = nn.Linear(4, 4)
    vision = nn.Linear(4, 4)
    navigation = TinyNavigationModel()
    configure_stage_parameters(language, vision, navigation, 'action')
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in navigation.named_parameters()
        if not parameter.requires_grad
    }
    trainable_before = navigation.decoder_2_action_full.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in navigation.parameters() if parameter.requires_grad],
        lr=1e-2,
    )

    motion = navigation.backbone(torch.randn(2, 4)).detach().unsqueeze(1).repeat(1, 2, 1)
    conditioned = navigation.target_conditioning(
        motion,
        torch.tensor([[0.8, 0.7], [0.6, 0.5]]),
        torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
    )
    loss = navigation.decoder_2_action_full(conditioned[:, 0]).square().mean()
    loss = loss + navigation.decoder_2_progress_full(conditioned[:, 1]).square().mean()
    loss.backward()
    optimizer.step()

    for name, parameter in navigation.named_parameters():
        if name in frozen_before:
            assert torch.equal(parameter.detach(), frozen_before[name])
    assert not torch.equal(
        navigation.decoder_2_action_full.weight.detach(), trainable_before
    )


def test_sparse_quadtree_belief_shapes_and_bounds():
    torch.manual_seed(0)
    head = SparseQuadtreeBeliefHead(
        d_model=8,
        depth=4,
        topk=3,
        hidden_dim=16,
    )
    context = torch.randn(2, 8)
    truth = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    output = head(context, truth=truth)

    assert output.coordinate.shape == (2, 2)
    assert output.leaf_centers.shape == (2, 3, 2)
    assert output.leaf_probs.shape == (2, 3)
    assert output.hierarchy_loss is not None
    assert torch.isfinite(output.hierarchy_loss)
    assert torch.all(output.coordinate >= 0)
    assert torch.all(output.coordinate <= 1)
    assert torch.allclose(
        output.leaf_probs.sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )


def test_sparse_quadtree_teacher_path_reaches_expected_leaf_center():
    head = SparseQuadtreeBeliefHead(
        d_model=4,
        depth=3,
        topk=1,
        hidden_dim=8,
    )
    truth = torch.tensor([[0.90, 0.10]])
    center = torch.full((1, 2), 0.5)
    size = torch.ones(1)
    for _ in range(head.depth):
        children, child_sizes = head._children(center, size)
        label = head._truth_child(truth, center)
        center = children.gather(
            1, label.view(1, 1, 1).expand(-1, -1, 2)
        ).squeeze(1)
        size = child_sizes.gather(1, label.view(1, 1)).squeeze(1)

    # At depth 3 each leaf spans 1/8 of the map and its center must contain truth.
    assert torch.all(torch.abs(center - truth) <= size.unsqueeze(-1) * 0.5)


def test_relational_candidate_belief_returns_topk_without_averaging():
    torch.manual_seed(0)
    head = RelationalCandidateBeliefHead(
        d_model=8,
        num_heads=2,
        hidden_dim=16,
        topk=3,
        max_offset=0.05,
        dropout=0.0,
    ).eval()
    tokens = torch.randn(2, 6, 8)
    candidates = torch.rand(2, 6, 2)
    language = torch.randn(2, 5, 8)
    current = torch.rand(2, 2)
    mask = torch.ones(2, 5, dtype=torch.long)
    output = head(tokens, candidates, language, current, mask)

    assert output.coordinate.shape == (2, 2)
    assert output.logits.shape == (2, 6)
    assert output.refined_coordinates.shape == (2, 6, 2)
    assert output.topk_coordinates.shape == (2, 3, 2)
    assert output.topk_probs.shape == (2, 3)
    assert torch.allclose(output.coordinate, output.topk_coordinates[:, 0])
    assert torch.allclose(output.topk_probs.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(output.refined_coordinates >= 0)
    assert torch.all(output.refined_coordinates <= 1)


def test_candidate_supervision_and_recall_are_finite():
    logits = torch.tensor([[0.0, 2.0, -1.0]])
    base = torch.tensor([[[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]])
    refined = base.clone()
    truth = torch.tensor([[0.52, 0.48]])
    cls, offset, labels = candidate_supervision(
        logits, refined, base, truth, 410.0, 10.0
    )
    assert labels.item() == 1
    assert torch.isfinite(cls)
    assert torch.isfinite(offset)
    recall = candidate_recall(
        refined[:, :2], truth, 410.0, radius_m=20.0
    )
    assert recall.item() == 1.0


def test_multi_hypothesis_conditioning_uses_multiple_targets():
    torch.manual_seed(0)
    module = MultiHypothesisConditioning(
        d_model=8, num_heads=2, dropout=0.0
    ).eval()
    motion = torch.randn(1, 2, 8)
    current = torch.tensor([[0.2, 0.2]])
    targets_a = torch.tensor([[[0.3, 0.3], [0.8, 0.8]]])
    targets_b = torch.tensor([[[0.3, 0.3], [0.1, 0.9]]])
    probs = torch.tensor([[0.6, 0.4]])
    out_a = module(motion, targets_a, probs, current)
    out_b = module(motion, targets_b, probs, current)
    assert out_a.shape == motion.shape
    assert not torch.allclose(out_a, out_b)
