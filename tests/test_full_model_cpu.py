import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from multiagent.models.ET_haa import ET


def args(hypothesis):
    return SimpleNamespace(
        grid_size=5, demb=768, encoder_heads=12, encoder_layers=2,
        dropout_transformer_encoder=0.1, num_input_actions=1,
        dropout_emb=0.0, disable_task_interaction=True,
        enable_multi_hypothesis=hypothesis,
    )


def inputs(batch=2):
    return {
        'lang': torch.randn(batch, 6, 768, requires_grad=True),
        'maps': torch.randn(batch, 3, 240, 240),
        'candidates': torch.rand(batch, 25, 2),
        'directions': torch.rand(batch, 1, 4),
        'frames': torch.randn(batch, 1, 512, 49, requires_grad=True),
        'grid_fts': torch.zeros(batch, 0, 768),
        'grid_index': torch.zeros(batch, 0),
        'lang_cls': torch.randn(batch, 49),
        'lang_mask': torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool),
    }


class FullHypothesisETForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_hypothesis_path_full_forward_and_backward(self):
        model = ET(args(True))
        output = model(**inputs())
        self.assertEqual([tuple(value.shape) if value is not None else None for value in output],
                         [(2, 2), (2, 1), (2, 2), (2, 25, 1), (2, 1, 768), (2, 25, 2)])
        sum(value.square().mean() for value in output if value is not None).backward()
        gradients = [parameter.grad for parameter in model.hypothesis_offset_head.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                            for gradient in gradients))

    def test_disabled_path_keeps_history_and_no_offsets(self):
        model = ET(args(False))
        output = model(**inputs())
        self.assertEqual(tuple(output[4].shape), (2, 1, 768))
        self.assertIsNone(output[5])

    def test_selected_cell_offset_supervision_reaches_head_and_language(self):
        model = ET(args(True))
        sample = inputs()
        offsets = model(**sample)[5]
        cells = torch.tensor([0, 24])
        predictions = offsets[torch.arange(2), cells]
        targets = torch.tensor([[0.03, 0.12], [0.18, 0.06]])
        F.mse_loss(predictions, targets).backward()
        for name, parameter in model.hypothesis_offset_head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.norm().item(), 0, name)
        self.assertIsNotNone(sample['lang'].grad)
        self.assertTrue(torch.isfinite(sample['lang'].grad).all())
        self.assertGreater(sample['lang'].grad.norm().item(), 0)


if __name__ == '__main__':
    unittest.main()
