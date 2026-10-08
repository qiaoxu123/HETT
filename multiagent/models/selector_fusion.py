"""Bounded SBF semantic correction to HETT candidate logit ranking."""
import math
import torch


def bounded_selector_fusion(field_logits, selector_logits, weight=0.25):
    """Keep strong heatmap preferences; allow semantic reordering near ties.

    The semantic correction is in [-weight, weight]. A field standardized
    top1 margin greater than 2*weight therefore cannot be overturned.
    Inputs contain model predictions only. No calibration on validation labels.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('weight must be finite and nonnegative')
    if field_logits.shape != selector_logits.shape or field_logits.ndim != 2:
        raise ValueError('scores must have matching [B,K] shape')
    if field_logits.shape[1] == 0:
        raise ValueError('at least one candidate is required')
    if not torch.isfinite(field_logits).all() or not torch.isfinite(selector_logits).all():
        raise ValueError('scores must be finite')
    def standardize(x):
        x=x.float()
        return (x-x.mean(-1,keepdim=True))/x.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)
    return standardize(field_logits)+weight*torch.tanh(standardize(selector_logits))
