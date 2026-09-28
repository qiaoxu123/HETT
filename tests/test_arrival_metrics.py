import math

from multiagent.arrival_metrics import (
    binary_auprc,
    binary_auroc,
    expected_calibration_error,
    summarize_arrival,
)


def test_perfect_arrival_ranking_has_unit_auc():
    probs = [0.05, 0.10, 0.90, 0.95]
    labels = [0, 0, 1, 1]
    assert math.isclose(binary_auroc(probs, labels), 1.0)
    assert math.isclose(binary_auprc(probs, labels), 1.0)


def test_arrival_summary_reports_threshold_metrics():
    probs = [0.1, 0.4, 0.6, 0.9]
    labels = [0, 0, 1, 1]
    summary = summarize_arrival(probs, labels, threshold=0.5)
    assert summary["accuracy"] == 1.0
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["f1"] == 1.0
    assert summary["fpr"] == 0.0
    assert 0.0 <= summary["ece"] <= 1.0


def test_calibration_error_is_small_for_confident_correct_predictions():
    probs = [0.01, 0.02, 0.98, 0.99]
    labels = [0, 0, 1, 1]
    assert expected_calibration_error(probs, labels, bins=5) < 0.05
