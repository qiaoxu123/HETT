import unittest
from types import SimpleNamespace

import torch

from multiagent.models.ET_haa import ET, FirstPersonSpatialFusion


class FirstPersonSpatialFusionTest(unittest.TestCase):
    def test_zero_gate_matches_full_heatmap_pathway(self):
        torch.manual_seed(3)
        args = SimpleNamespace(
            demb=768, encoder_heads=12, encoder_layers=2,
            dropout_transformer_encoder=0.1, num_input_actions=1,
            dropout_emb=0.0, grid_size=7,
            disable_first_person_spatial=False,
        )
        model = ET(args).eval()
        inputs = {
            'lang': torch.randn(1, 3, 768),
            'lang_mask': torch.ones(1, 3, dtype=torch.bool),
            'lang_cls': torch.randn(1, 49),
            'frames': torch.randn(1, 1, 512, 49),
            'directions': torch.tensor([[[0.0, 1.0, 0.3, 0.4]]]),
            'maps': torch.randn(1, 4, 240, 240),
            'candidates': torch.rand(1, 49, 2),
            'grid_fts': torch.empty(1, 0, 768),
            'grid_index': torch.empty(1, 0),
            'landmark_anchors': torch.zeros(1, 8, 2),
            'landmark_anchor_mask': torch.zeros(1, 8, dtype=torch.bool),
        }
        with torch.inference_mode():
            enabled = model(**inputs)
            args.disable_first_person_spatial = True
            disabled = model(**inputs)
        for actual, expected in zip(enabled, disabled):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_gate_preserves_baseline_and_nonzero_gate_uses_rgb(self):
        torch.manual_seed(0)
        fusion = FirstPersonSpatialFusion(
            candidate_dim=16, image_channels=8, attention_dim=4,
            image_grid_size=2,
        )
        candidates = torch.randn(2, 3, 16)
        candidate_xy = torch.rand(2, 3, 2)
        current_state = torch.tensor([
            [0.0, 1.0, 0.3, 0.4],
            [1.0, 0.0, 0.6, 0.2],
        ])
        rgb = torch.randn(2, 8, 4)

        torch.testing.assert_close(
            fusion(candidates, candidate_xy, current_state, rgb),
            candidates,
            rtol=0,
            atol=0,
        )

        with torch.no_grad():
            fusion.gate.fill_(1.0)
        output_a = fusion(candidates, candidate_xy, current_state, rgb)
        output_b = fusion(candidates, candidate_xy, current_state, rgb + 1.0)
        self.assertFalse(torch.allclose(output_a, output_b))
        output_a.square().sum().backward()
        self.assertTrue(torch.isfinite(fusion.gate.grad))
        self.assertGreater(fusion.gate.grad.abs().item(), 0)


if __name__ == '__main__':
    unittest.main()
