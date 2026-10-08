import unittest
import torch

from multiagent.models.relative_geometry import dense_relative_geometry
from multiagent.models.spatial_belief import CompactSpatialBelief, greedy_nms_topk


class RelativeGeometryTest(unittest.TestCase):
    def test_cardinal_angles_and_world_y_flip(self):
        # Position at map center, yaw east (sin=0, cos=1).
        f = dense_relative_geometry(
            torch.tensor([[0.5, 0.5]]), torch.tensor([[0., 1.]]),
            torch.tensor([[0.5, 0.5]]), torch.tensor([True]),
            field_size=3, map_meters=410.,
        )
        self.assertEqual(tuple(f.shape), (1, 13, 3, 3))
        self.assertAlmostEqual(float(f[0, 4, 1, 2]), 1.0, places=4)
        self.assertAlmostEqual(float(f[0, 3, 0, 1]), 1.0, places=4)
        # Same north candidate is to the left when UAV faces east.
        self.assertAlmostEqual(float(f[0, 5, 0, 1]), 1.0, places=4)

    def test_heading_changes_egocentric_not_absolute_bearing(self):
        p = torch.tensor([[0.5, 0.5]])
        anchor = torch.tensor([[0.5, 0.5]])
        east = dense_relative_geometry(p, torch.tensor([[0., 1.]]), anchor,
                                       torch.tensor([True]), field_size=3, map_meters=410.)
        north = dense_relative_geometry(p, torch.tensor([[1., 0.]]), anchor,
                                        torch.tensor([True]), field_size=3, map_meters=410.)
        self.assertTrue(torch.allclose(east[:, 3:5], north[:, 3:5]))
        self.assertFalse(torch.allclose(east[:, 5:7], north[:, 5:7]))

    def test_invalid_reference_zeroes_reference_channels(self):
        f = dense_relative_geometry(
            torch.tensor([[0.4, 0.5]]), torch.tensor([[0., 1.]]),
            torch.tensor([[0., 0.]]), torch.tensor([False]),
            field_size=3, map_meters=410.)
        self.assertTrue(torch.all(f[:, 7:] == 0))

    def test_dense_belief_geometry_gradient_and_probabilities(self):
        model = CompactSpatialBelief(input_channels=4, field_size=7,
                                     hidden_dim=32, language_dim=32,
                                     attention_heads=4, dropout=0)
        geometry = dense_relative_geometry(
            torch.tensor([[0.5, 0.5]]), torch.tensor([[0., 1.]]),
            torch.tensor([[0.4, 0.4]]), torch.tensor([True]),
            field_size=7, map_meters=410.)
        out = model(torch.zeros(1, 4, 80, 80), torch.randn(1, 4, 32),
                    geometry=geometry)
        self.assertEqual(tuple(out.logits.shape), (1, 7, 7))
        self.assertAlmostEqual(float(out.probabilities.sum()), 1., places=5)
        out.logits.mean().backward()
        self.assertIsNotNone(model.geometry_gate.grad)

    def test_twenty_nms_candidates(self):
        field = torch.rand(1, 28, 28)
        ids = greedy_nms_topk(field, top_k=20, kernel_size=3)
        self.assertEqual(tuple(ids.shape), (1, 20))
        self.assertEqual(len(set(ids.flatten().tolist())), 20)


if __name__ == '__main__':
    unittest.main()
