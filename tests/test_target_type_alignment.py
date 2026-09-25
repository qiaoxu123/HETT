import torch

from scripts.audit_target_type_alignment import (
    classification_metrics,
    first_type_match,
)
from scripts.audit_multiscale_type_matching import type_agreement


def test_classification_metrics_report_balanced_perfect_result():
    labels = torch.tensor([0, 0, 1, 2, 3])

    result = classification_metrics(labels, labels)

    assert result["accuracy"] == 100.0
    assert result["balanced_accuracy"] == 100.0
    assert result["macro_f1"] == 100.0


def test_first_type_match_preserves_geometry_order_and_falls_back():
    text = torch.tensor([2, 3, 1])
    images = torch.tensor([[0, 2, 2], [3, 3, 0], [0, 2, 3]])

    rank, has_match = first_type_match(text, images)

    assert rank.tolist() == [1, 0, 0]
    assert has_match.tolist() == [True, True, False]


def test_type_agreement_separates_agreement_from_joint_correctness():
    text = torch.tensor([0, 1, 2, 3])
    image = torch.tensor([0, 1, 3, 2])
    truth = torch.tensor([0, 0, 2, 3])

    result = type_agreement(text, image, truth)

    assert result["prediction_agreement"] == 50.0
    assert result["both_correct"] == 25.0
    assert result["agreement_and_correct"] == 25.0
