"""Memory-friendly gradient accumulation for independent teacher/student rollouts.

Backpropagate each rollout's loss immediately, before constructing the next
rollout graph. Both gradients are accumulated on the same parameters; optimizer
step and gradient clipping remain outside this helper and happen only once.
"""
from __future__ import annotations

from typing import Callable

import torch


def backward_rollout_pair(
    teacher_forward: Callable[[], torch.Tensor],
    student_forward: Callable[[], torch.Tensor],
    *,
    backward: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Return detached total loss after accumulating both gradients.

    Forward callbacks must construct independent graphs and MUST NOT update
    weights. This cannot be used when the student forward depends on the
    *live* teacher autograd graph.
    """
    apply_backward = backward if backward is not None else lambda loss: loss.backward()
    teacher_loss = teacher_forward()
    if not torch.is_tensor(teacher_loss) or not teacher_loss.requires_grad:
        raise ValueError("teacher rollout must return a differentiable scalar loss")
    detached_total = teacher_loss.detach()
    apply_backward(teacher_loss)
    del teacher_loss

    student_loss = student_forward()
    if not torch.is_tensor(student_loss) or not student_loss.requires_grad:
        raise ValueError("student rollout must return a differentiable scalar loss")
    detached_total = detached_total + student_loss.detach()
    apply_backward(student_loss)
    return detached_total
