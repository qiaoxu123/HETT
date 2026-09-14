import unittest
from types import SimpleNamespace

import torch

from multiagent.models.ET_haa import CoarseToFineGoalDecoder, ET


class CoarseToFineGoalDecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.decoder = CoarseToFineGoalDecoder(
            d_model=16, grid_size=2, topk=1, temperature=1.0
        )
        self.tokens = torch.randn(2, 4, 16, requires_grad=True)
        self.candidates = torch.tensor([
            [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]],
            [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]],
        ])

    def test_top_one_goal_stays_inside_selected_cell(self):
        logits = torch.tensor([
            [[9.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [9.0]],
        ], requires_grad=True)
        goals, offsets = self.decoder(self.tokens, self.candidates, logits)
        self.assertTrue(torch.all(goals[0] >= 0.0) and torch.all(goals[0] <= 0.5))
        self.assertTrue(torch.all(goals[1] >= 0.5) and torch.all(goals[1] <= 1.0))
        self.assertEqual(tuple(offsets.shape), (2, 4, 2))

    def test_changing_coarse_cell_changes_goal(self):
        first = torch.tensor([[[9.0], [0.0], [0.0], [0.0]]] * 2)
        last = torch.tensor([[[0.0], [0.0], [0.0], [9.0]]] * 2)
        goal_first, _ = self.decoder(self.tokens, self.candidates, first)
        goal_last, _ = self.decoder(self.tokens, self.candidates, last)
        self.assertGreater((goal_first - goal_last).abs().max().item(), 0.49)

    def test_goal_loss_reaches_selected_offset_head(self):
        logits = torch.tensor([
            [[9.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [9.0]],
        ], requires_grad=True)
        goals, _ = self.decoder(self.tokens, self.candidates, logits)
        goals.square().sum().backward()
        gradients = [parameter.grad for parameter in self.decoder.offset_head.parameters()]
        self.assertTrue(all(
            gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
            for gradient in gradients
        ))
        self.assertTrue(torch.isfinite(self.tokens.grad).all() and self.tokens.grad.norm() > 0)


class CoarseToFineETIntegrationTest(unittest.TestCase):
    def test_full_model_routes_existing_losses_into_coarse_and_fine_heads(self):
        args = SimpleNamespace(
            grid_size=5, demb=768, encoder_heads=12, encoder_layers=2,
            dropout_transformer_encoder=0.1, num_input_actions=1,
            dropout_emb=0.0, disable_task_interaction=True,
            coarse_to_fine_target=True, target_topk=3, target_temperature=1.0,
        )
        model = ET(args)
        batch = 2
        inputs = {
            'lang': torch.randn(batch, 6, 768, requires_grad=True),
            'maps': torch.randn(batch, 3, 240, 240),
            'candidates': torch.tensor([
                [[i / 5, j / 5] for i in range(5) for j in range(5)]
            ] * batch),
            'directions': torch.rand(batch, 1, 4),
            'frames': torch.randn(batch, 1, 512, 49),
            'grid_fts': torch.zeros(batch, 0, 768),
            'grid_index': torch.zeros(batch, 0),
            'lang_cls': torch.randn(batch, 49),
            'lang_mask': torch.ones(batch, 6, dtype=torch.bool),
        }
        direction, progress, goals, logits, history = model(**inputs)
        self.assertEqual(tuple(goals.shape), (batch, 2))
        self.assertTrue(torch.all(goals >= 0) and torch.all(goals <= 1))
        target = torch.tensor([0, 24])
        loss = torch.nn.functional.cross_entropy(logits.squeeze(-1), target)
        loss = loss + goals.square().mean() + direction.square().mean() + progress.square().mean()
        loss.backward()
        coarse_gradients = [parameter.grad for parameter in model.decoder_2_logits_full.parameters()]
        self.assertTrue(all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in coarse_gradients
        ))
        # A shared scalar-logit bias is shift-invariant under cross entropy, so
        # require a real gradient in the head without incorrectly requiring
        # every individual parameter to be non-zero.
        self.assertGreater(sum(gradient.norm() for gradient in coarse_gradients), 0)
        fine_gradients = [
            parameter.grad for parameter in model.coarse_to_fine_goal.offset_head.parameters()
        ]
        self.assertTrue(all(
            gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
            for gradient in fine_gradients
        ))
        self.assertEqual(tuple(history.shape), (batch, 1, 768))


if __name__ == '__main__':
    unittest.main()
