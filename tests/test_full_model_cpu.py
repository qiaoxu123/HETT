import unittest
from types import SimpleNamespace

import torch

from multiagent.models.ET_haa import ET


def args(region):
    return SimpleNamespace(
        grid_size=5, demb=768, encoder_heads=12, encoder_layers=2,
        dropout_transformer_encoder=0.1, num_input_actions=1,
        dropout_emb=0.0, disable_task_interaction=True,
        enable_region_grounding=region,
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


class FullETForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_grounding_path_full_forward_and_backward(self):
        model = ET(args(True))
        output = model(**inputs())
        self.assertEqual([tuple(value.shape) if value is not None else None for value in output],
                         [(2, 2), (2, 1), (2, 2), (2, 25, 1), (2, 1, 768), (2, 50)])
        sum(value.square().mean() for value in output if value is not None).backward()
        gradients = [parameter.grad for parameter in model.region_grounding.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                            for gradient in gradients))

    def test_disabled_path_keeps_single_history_token(self):
        model = ET(args(False))
        output = model(**inputs())
        self.assertEqual(tuple(output[4].shape), (2, 1, 768))
        self.assertIsNone(output[5])


if __name__ == '__main__':
    unittest.main()
