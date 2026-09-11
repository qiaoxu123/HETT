import unittest
from types import SimpleNamespace

import torch

from multiagent.models.ET_haa import ET


def args(disabled=False):
    return SimpleNamespace(
        grid_size=5, demb=768, encoder_heads=12, encoder_layers=2,
        dropout_transformer_encoder=0.1, num_input_actions=1,
        dropout_emb=0.0, disable_task_interaction=disabled,
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


class FullBidirectionalETTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_cross_attention_directions_receive_gradients(self):
        model = ET(args(False))
        output = model(**inputs())
        self.assertEqual([tuple(value.shape) for value in output],
                         [(2, 2), (2, 1), (2, 2), (2, 25, 1), (2, 1, 768)])
        sum(value.square().mean() for value in output).backward()
        for name in ('target_from_motion', 'motion_from_target'):
            module = getattr(model.task_interaction, name)
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                and gradient.norm() > 0 for gradient in gradients), name)

    def test_enabled_and_disabled_heads_are_not_identical(self):
        model = ET(args(False)).eval()
        sample = inputs()
        with torch.no_grad():
            enabled = model(**sample)
            model.args.disable_task_interaction = True
            disabled = model(**sample)
        deltas = [(left-right).abs().max().item()
                  for left, right in zip(enabled[:4], disabled[:4])]
        self.assertTrue(all(delta > 0 for delta in deltas), deltas)


if __name__ == '__main__':
    unittest.main()
