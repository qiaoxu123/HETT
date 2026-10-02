import unittest

import torch
import torch.nn.functional as F

from multiagent.agent import masked_navigation_losses
from multiagent.models.encodings import MapPosEncoding
from multiagent.models.ET_haa import aggregate_history_grid


def reference_navigation_losses(pred_direction, pred_progress, pred_goals,
                                gt_direction, gt_progress, gt_goal, active):
    direction_loss = pred_direction.new_zeros(())
    progress_loss = pred_direction.new_zeros(())
    goal_loss = pred_direction.new_zeros(())
    for i in range(pred_direction.shape[0]):
        if active[i]:
            true_direction = torch.stack((
                torch.sin(gt_direction[i]), torch.cos(gt_direction[i])
            ))
            direction_loss = direction_loss + F.mse_loss(
                pred_direction[i].reshape(-1), true_direction, reduction='sum'
            )
            progress_loss = progress_loss + F.mse_loss(
                pred_progress[i].reshape(-1), gt_progress[i].reshape(-1),
                reduction='sum'
            )
            goal_loss = goal_loss + F.mse_loss(
                pred_goals[i].reshape(-1), gt_goal[i].reshape(-1)
            )
    return direction_loss, progress_loss, goal_loss


def reference_history_grid(grid_fts, grid_indices, text_fts, grid_proj,
                           cell_count):
    batch_size, _, feature_dim = grid_fts.shape
    result = grid_fts.new_zeros((batch_size, cell_count, feature_dim))
    for b in range(batch_size):
        history = grid_fts[b].to(torch.float32)
        if history.shape[0] == 0:
            continue
        scores = (history @ text_fts[b]).max(dim=-1).values
        projected = grid_proj(history)
        for cell in range(cell_count):
            mask = grid_indices[b] == cell
            cell_features = projected[mask]
            if cell_features.shape[0]:
                result[b, cell] = (
                    cell_features * torch.softmax(scores[mask], dim=-1).unsqueeze(-1)
                ).sum(dim=-2)
    return result


class PerformanceEquivalenceTest(unittest.TestCase):
    def test_vectorized_map_position_encoding_matches_reference(self):
        torch.manual_seed(2)
        batch_size, width, language_len = 4, 16, 9
        module = MapPosEncoding(width, max_len=100)
        values = (
            torch.randn(batch_size, language_len, width),
            torch.randn(batch_size, 1, width),
            torch.randn(batch_size, 1, width),
            torch.randn(batch_size, 1, width),
            torch.randn(batch_size, 25, width),
        )
        expected = [value.clone() for value in values]
        enc = module.pe[:, :language_len + 1 + 25] / (width ** 0.5)
        expected[0] = expected[0] + enc[:, :language_len]
        for batch_idx in range(batch_size):
            expected[1][batch_idx] += enc[0, language_len:language_len + 1]
            expected[2][batch_idx] += enc[0, language_len:language_len + 1]
            expected[3][batch_idx] += enc[0, language_len:language_len + 1]
            expected[4][batch_idx] += enc[0, language_len + 1:language_len + 26]

        actual = module(*(value.clone() for value in values), language_len)
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_masked_navigation_losses_match_reference_and_gradients(self):
        torch.manual_seed(0)
        batch_size = 8
        tensors = [
            torch.randn(batch_size, 2, dtype=torch.float64, requires_grad=True),
            torch.randn(batch_size, 1, dtype=torch.float64, requires_grad=True),
            torch.randn(batch_size, 2, dtype=torch.float64, requires_grad=True),
        ]
        copies = [tensor.detach().clone().requires_grad_() for tensor in tensors]
        gt_direction = torch.randn(batch_size, dtype=torch.float64)
        gt_progress = torch.randn(batch_size, dtype=torch.float64)
        gt_goal = torch.randn(batch_size, 2, dtype=torch.float64)
        active = torch.tensor([True, False, True, True, False, True, False, True])

        expected = reference_navigation_losses(
            *tensors, gt_direction, gt_progress, gt_goal, active
        )
        actual = masked_navigation_losses(
            *copies, gt_direction, gt_progress, gt_goal, active
        )
        for expected_loss, actual_loss in zip(expected, actual):
            torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-12, atol=1e-12)

        sum(expected).backward()
        sum(actual).backward()
        for reference_tensor, actual_tensor in zip(tensors, copies):
            torch.testing.assert_close(
                actual_tensor.grad, reference_tensor.grad, rtol=1e-12, atol=1e-12
            )

    def test_vectorized_history_grid_matches_reference_and_gradients(self):
        torch.manual_seed(1)
        batch_size, history_len, feature_dim = 3, 20, 16
        cell_count, language_len = 25, 11
        grid = torch.randn(
            batch_size, history_len, feature_dim, dtype=torch.float32,
            requires_grad=True,
        )
        grid_copy = grid.detach().clone().requires_grad_()
        indices = torch.randint(0, cell_count, (batch_size, history_len))
        text = torch.randn(batch_size, feature_dim, language_len)
        projection = torch.nn.Linear(feature_dim, feature_dim)
        projection_copy = torch.nn.Linear(feature_dim, feature_dim)
        projection_copy.load_state_dict(projection.state_dict())

        expected = reference_history_grid(
            grid, indices, text, projection, cell_count
        )
        actual = aggregate_history_grid(
            grid_copy, indices, text, projection_copy, cell_count
        )
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(grid_copy.grad, grid.grad, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(
            projection_copy.weight.grad, projection.weight.grad,
            rtol=2e-5, atol=2e-5,
        )

    def test_vectorized_history_grid_handles_empty_history(self):
        projection = torch.nn.Linear(8, 8)
        result = aggregate_history_grid(
            torch.empty(2, 0, 8),
            torch.empty(2, 0),
            torch.empty(2, 8, 4),
            projection,
            25,
        )
        self.assertEqual(result.shape, (2, 25, 8))
        self.assertEqual(torch.count_nonzero(result).item(), 0)


if __name__ == '__main__':
    unittest.main()
