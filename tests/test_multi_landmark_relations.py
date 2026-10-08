"""Regression tests for named landmark alignment and relation-aware heatmap."""
import unittest

import torch

from multiagent.models.multi_landmark import (
    build_landmark_batch, candidate_landmark_geometry, MultiLandmarkRelationHead,
)
from multiagent.models.spatial_belief import CompactSpatialBelief


class TinyTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, phrase, add_special_tokens=False):
        result = []
        for token in phrase.lower().split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 5
            result.append(self.vocab[token])
        return ([1] + result + [2]) if add_special_tokens else result


class MultiLandmarkTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_name_spans_preserve_own_geometry_and_padding(self):
        tok = TinyTokenizer()
        instruction = tok.encode('east of red church north of blue tower')
        names = [
            {"name": "blue tower", "center_xy": (0.1, 0.7), "extent_xy": (.2, .1)},
            {"name": "red church", "center_xy": (.8, .2), "extent_xy": (.1, .2)},
        ]
        one = build_landmark_batch(
            [{"reference_landmarks": names}, {"reference_landmarks": []}],
            tok, torch.tensor([instruction, instruction]), max_landmarks=8,
        )
        self.assertEqual(tuple(one['landmark_xy'].shape), (2, 2, 2))
        # Sorted by mention order: red church precedes blue tower.
        self.assertTrue(torch.allclose(one['landmark_xy'][0, 0], torch.tensor([.8, .2])))
        self.assertTrue(torch.allclose(one['landmark_xy'][0, 1], torch.tensor([.1, .7])))
        self.assertEqual(int(one['landmark_text_mask'][0, 0].sum()), 2)
        self.assertEqual(int(one['landmark_text_mask'][0, 1].sum()), 2)
        self.assertFalse(one['landmark_valid'][1].any())

    def test_missing_name_does_not_fabricate_span(self):
        tok = TinyTokenizer()
        instructions = torch.tensor([tok.encode('go to bridge')])
        anchors = [{"reference_landmarks": [
            {"name": "church", "center_xy": (.3, .4), "extent_xy": (.1, .1)}
        ]}]
        inp = build_landmark_batch(anchors, tok, instructions)
        self.assertTrue(inp['landmark_valid'].any())
        self.assertFalse(inp['landmark_text_mask'].any())

    def test_cap_is_reported(self):
        tok = TinyTokenizer()
        names = [
            {"name": f"n{i}", "center_xy": (.1 * i, .3), "extent_xy": (.1, .1)}
            for i in range(6)
        ]
        ids = tok.encode(' '.join(x["name"] for x in names))
        out = build_landmark_batch([{"reference_landmarks": names}],
                                   tok, torch.tensor([ids]), max_landmarks=3)
        self.assertEqual(out['landmark_truncated'], 3)
        self.assertEqual(out['landmark_valid'].sum().item(), 3)

    def test_candidate_geometry_uses_north_positive_world_coordinates(self):
        xy = torch.tensor([[[.5, .5]]])
        sizes = torch.tensor([[[.1, .1]]])
        f = candidate_landmark_geometry(xy, sizes, field_size=3)
        self.assertEqual(tuple(f.shape), (1, 9, 1, 8))
        # Row 0 center cell lies north of landmark.
        self.assertAlmostEqual(float(f[0, 1, 0, 3]), 1., places=5)
        # Row 1 col 2 lies east of landmark.
        self.assertAlmostEqual(float(f[0, 5, 0, 4]), 1., places=5)

    def _model_inputs(self):
        head = MultiLandmarkRelationHead(
            map_dim=24, language_dim=16, hidden_dim=24,
            attention_heads=4, dropout=0.0
        ).eval()
        feature = torch.randn(2, 24, 5, 5)
        lang = torch.randn(2, 6, 16)
        xy = torch.tensor([
            [[.2, .2], [.8, .7]], [[.5, .4], [.0, .0]],
        ])
        ext = torch.ones_like(xy) * .05
        valid = torch.tensor([[True, True], [True, False]])
        spans = torch.zeros(2, 2, 6, dtype=torch.bool)
        spans[0, 0, 1] = True
        spans[0, 1, 4] = True
        spans[1, 0, 2] = True
        return head, (feature, lang, xy, ext, valid, spans)

    def test_joint_head_is_permutation_invariant_for_name_geometry_pairs(self):
        head, args = self._model_inputs()
        a = head(*args)
        feat, lang, xy, ext, valid, spans = args
        p = torch.tensor([1, 0])
        b = head(feat, lang, xy[:, p], ext[:, p], valid[:, p], spans[:, p])
        self.assertEqual(tuple(a.shape), (2, 5, 5))
        self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-5))

    def test_name_geometry_mismatch_changes_output(self):
        head, args = self._model_inputs()
        baseline = head(*args)
        feat, lang, xy, ext, valid, spans = args
        swapped_names = spans.clone()
        swapped_names[0] = swapped_names[0, [1, 0]]
        changed = head(feat, lang, xy, ext, valid, swapped_names)
        self.assertGreater(float((baseline[0] - changed[0]).abs().max()), 1e-7)

    def test_missing_anchor_rows_are_finite_and_zero(self):
        head, args = self._model_inputs()
        feat, lang, xy, ext, valid, spans = args
        valid = valid.clone()
        valid[1] = False
        values = head(feat, lang, xy, ext, valid, spans)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.equal(values[1], torch.zeros_like(values[1])))
        values[0].sum().backward()
        self.assertTrue(torch.isfinite(head.geometry_proj[0].weight.grad).all())

    def test_heatmap_outputs_and_relation_gate_gradients(self):
        model = CompactSpatialBelief(
            input_channels=4, field_size=5, hidden_dim=32,
            language_dim=16, attention_heads=4, dropout=0.0
        )
        lang = torch.randn(2, 6, 16)
        maps = torch.randn(2, 4, 64, 64)
        xy = torch.tensor([[[.2, .2], [.8, .7]], [[.5, .5], [.0, .0]]])
        extent = torch.ones_like(xy) * .1
        valid = torch.tensor([[True, True], [True, False]])
        spans = torch.zeros(2, 2, 6, dtype=torch.bool)
        spans[:, 0, 2] = True
        spans[0, 1, 4] = True
        output = model(maps, lang, landmark_xy=xy, landmark_extent=extent,
                       landmark_valid=valid, landmark_text_mask=spans)
        self.assertEqual(tuple(output.logits.shape), (2, 5, 5))
        self.assertTrue(torch.allclose(output.probabilities.sum((1, 2)),
                                       torch.ones(2), atol=1e-5))
        output.logits.square().mean().backward()
        self.assertTrue(torch.isfinite(model.multi_landmark_gate.grad))


if __name__ == "__main__":
    unittest.main()
