"""Gradient and optimizer equivalence for memory-saving two-rollout backward."""
import copy
import unittest

import torch
from torch import nn

from multiagent.rollout_backward import backward_rollout_pair


class SequentialBackwardRegression(unittest.TestCase):
    def test_sum_gradient_matches_joint_backward(self):
        torch.manual_seed(71)
        inputs = (torch.randn(7, 5), torch.randn(8, 5))
        targets = (torch.randn(7, 3), torch.randn(8, 3))
        original = nn.Sequential(nn.Linear(5, 12), nn.GELU(), nn.Linear(12, 3))
        parallel = copy.deepcopy(original)
        sequential = copy.deepcopy(original)

        def rollout(model, idx):
            return (model(inputs[idx]) - targets[idx]).square().mean() * (0.3 if idx == 0 else 0.7)

        joint_loss = rollout(parallel, 0) + rollout(parallel, 1)
        joint_loss.backward()
        pair_loss = backward_rollout_pair(
            lambda: rollout(sequential, 0),
            lambda: rollout(sequential, 1))
        self.assertFalse(pair_loss.requires_grad)
        self.assertTrue(torch.allclose(pair_loss, joint_loss.detach(), rtol=1e-6, atol=1e-7))
        for p, q in zip(parallel.parameters(), sequential.parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, rtol=1e-6, atol=1e-7))
        optimizer_joint = torch.optim.AdamW(parallel.parameters(), lr=1e-3)
        optimizer_pair = torch.optim.AdamW(sequential.parameters(), lr=1e-3)
        nn.utils.clip_grad_norm_(parallel.parameters(), 1.0)
        nn.utils.clip_grad_norm_(sequential.parameters(), 1.0)
        optimizer_joint.step()
        optimizer_pair.step()
        for p, q in zip(parallel.parameters(), sequential.parameters()):
            self.assertTrue(torch.allclose(p, q, rtol=1e-6, atol=1e-7))

    def test_custom_backward_called_for_both_losses(self):
        model = nn.Linear(2, 1)
        calls = []
        t = torch.tensor([[1., 2.]])
        result = backward_rollout_pair(
            lambda: model(t).square().sum(),
            lambda: model(t * 2).abs().sum(),
            backward=lambda loss: (calls.append(loss.detach().item()), loss.backward()))
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.isfinite(result))
        self.assertTrue(torch.isfinite(model.weight.grad).all())

    def test_refuse_nondifferentiable_rollouts(self):
        with self.assertRaisesRegex(ValueError, "teacher rollout"):
            backward_rollout_pair(lambda: torch.tensor(1.0), lambda: torch.tensor(2.0))
        with self.assertRaisesRegex(ValueError, "student rollout"):
            backward_rollout_pair(
                lambda: torch.tensor(1.0, requires_grad=True),
                lambda: torch.tensor(2.0))


if __name__ == "__main__":
    unittest.main()
