"""Fast checks for oracle reference isolation and the one-shot predictor."""

import unittest

import torch

from multiagent.grounded_goal_predictor import LandmarkRelativeGoalPredictor
from multiagent.grounded_instruction_memory import OracleReferenceResolver, relation_labels
from multiagent.landmark_multimodal_map import LandmarkMultimodalMap, LandmarkRecord


class StaticLocalizationTest(unittest.TestCase):
    def test_reference_lookup_excludes_target_even_on_name_collision(self):
        records = [
            LandmarkRecord("m", 1, "Old Hall", (0.0, 0.0), (0.0, 0.0)),
            LandmarkRecord("m", 2, "Old Hall", (10.0, 0.0), (0.1, 0.0)),
        ]
        resolver = OracleReferenceResolver(LandmarkMultimodalMap(records))
        memory = resolver.compile("episode", "m", 1, "right of Old Hall", {
            "target": "Old Hall", "landmarks": ["Old Hall"], "surroundings": [],
        })
        self.assertEqual(memory.references[0].record.landmark_id, 2)
        self.assertNotIn("target_xy", memory.__dataclass_fields__)

    def test_relation_labels(self):
        labels = relation_labels("the second house between the hall and church")
        self.assertEqual(labels[4], 1)
        self.assertEqual(labels[9], 1)

    def test_predictor_with_and_without_references(self):
        torch.manual_seed(0)
        model = LandmarkRelativeGoalPredictor(text_dim=8, visual_dim=6, hidden_dim=16)
        inputs = [
            torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8),
            torch.randn(2, 3, 8), torch.randn(2, 3, 6),
            torch.randn(2, 3, 11), torch.randn(2, 3, 2),
            torch.tensor([[True, True, False], [False, False, False]]),
        ]
        prediction, attention, relation_logits = model(*inputs)
        self.assertEqual(tuple(prediction.shape), (2, 2))
        self.assertEqual(tuple(attention.shape), (2, 4))
        self.assertEqual(tuple(relation_logits.shape), (2, 10))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(float(attention[1, 0].detach()), 1.0)
        self.assertEqual(float(attention[0, 0].detach()), 0.0)
        prediction.sum().backward()
        self.assertIsNotNone(model.offset[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
