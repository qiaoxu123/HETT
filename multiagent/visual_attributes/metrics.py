"""Dependency-free classification, calibration, and consistency metrics."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(probabilities, targets, bins=10):
    probabilities = np.asarray(probabilities); targets = np.asarray(targets)
    confidence = probabilities.max(1); prediction = probabilities.argmax(1); result = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        selected = (confidence >= lower) & (confidence < lower + 1 / bins)
        if selected.any():
            result += selected.mean() * abs((prediction[selected] == targets[selected]).mean() - confidence[selected].mean())
    return float(result)


def macro_f1(prediction, target, classes):
    values = []
    for label in range(classes):
        tp = np.sum((prediction == label) & (target == label)); fp = np.sum((prediction == label) & (target != label)); fn = np.sum((prediction != label) & (target == label))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def _binary_auroc(scores, positives):
    """AUROC from positive/negative pair comparisons, with half credit for ties."""
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    positive_scores = scores[positives]
    negative_scores = scores[~positives]
    if not len(positive_scores) or not len(negative_scores):
        return None
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def macro_auroc(probabilities, targets):
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    values = [
        _binary_auroc(probabilities[:, label], targets == label)
        for label in range(probabilities.shape[1])
    ]
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def classification_metrics(probabilities, targets):
    probabilities = np.asarray(probabilities); targets = np.asarray(targets); prediction = probabilities.argmax(1)
    classes = probabilities.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)
    for truth, pred in zip(targets, prediction): confusion[truth, pred] += 1
    precisions = []; recalls = []
    for label in range(classes):
        tp = np.sum((prediction == label) & (targets == label))
        fp = np.sum((prediction == label) & (targets != label))
        fn = np.sum((prediction != label) & (targets == label))
        precisions.append(tp / max(tp + fp, 1)); recalls.append(tp / max(tp + fn, 1))
    return {"samples": len(targets), "accuracy": float((prediction == targets).mean()),
            "macro_precision": float(np.mean(precisions)), "macro_recall": float(np.mean(recalls)),
            "macro_f1": macro_f1(prediction, targets, classes),
            "macro_auroc": macro_auroc(probabilities, targets),
            "ece": expected_calibration_error(probabilities, targets), "calibration_score": 1 - expected_calibration_error(probabilities, targets),
            "confusion_matrix": confusion.tolist()}


def grouped_consistency(probabilities, object_keys, condition_values):
    groups = {}
    for probability, key, condition in zip(probabilities, object_keys, condition_values):
        groups.setdefault(key, {}).setdefault(condition, []).append(probability)
    agreements = []; divergences = []
    for values in groups.values():
        arrays = [np.mean(items, axis=0) for items in values.values()]
        predictions = [value.argmax() for value in arrays]
        if len(predictions) > 1:
            agreements.append(sum(item == predictions[0] for item in predictions[1:]) / (len(predictions) - 1))
            mean = np.mean(arrays, 0)
            divergences.extend([0.5 * np.sum(a * np.log((a + 1e-8) / (mean + 1e-8))) for a in arrays])
    return {"agreement": float(np.mean(agreements)) if agreements else 0.0,
            "js_divergence": float(np.mean(divergences)) if divergences else 0.0}


def observability_score(val_unseen_f1, cross_view_consistency, cross_height_consistency, calibration_score):
    """Benchmark score from four independently reported [0, 1] components."""
    values = (val_unseen_f1, cross_view_consistency, cross_height_consistency, calibration_score)
    if any(value is None or not 0 <= value <= 1 for value in values):
        raise ValueError("observability components must all be available in [0, 1]")
    return float(0.4 * val_unseen_f1 + 0.2 * cross_view_consistency +
                 0.2 * cross_height_consistency + 0.2 * calibration_score)
