import unittest

import numpy as np
import torch

from multiagent.models.static_belief import StaticBeliefModel
from multiagent.navigation_state import CandidateStatus, NavigationMemory, nms_topk_from_belief
from multiagent.static_belief_metrics import candidate_coverage, gaussian_field_target


class StaticBeliefMemoryTest(unittest.TestCase):
    def test_static_belief_is_normalized(self):
        model = StaticBeliefModel(input_channels=4, language_dim=32, hidden_dim=32, attention_heads=4)
        static_map = torch.zeros(2, 4, 224, 224)
        language = torch.randn(2, 6, 32)
        mask = torch.ones(2, 6, dtype=torch.bool)
        output = model(static_map, language, mask)
        self.assertEqual(output.probabilities.ndim, 3)
        self.assertTrue(torch.allclose(output.probabilities.sum(dim=(1, 2)), torch.ones(2), atol=1e-5))

    def test_nms_returns_separated_topk(self):
        belief = torch.zeros(7, 7)
        belief[1, 1] = 0.5
        belief[1, 2] = 0.49
        belief[5, 5] = 0.4
        candidates = nms_topk_from_belief(belief, top_k=2, kernel_size=3)
        self.assertEqual((candidates[0].row, candidates[0].col), (1, 1))
        self.assertEqual((candidates[1].row, candidates[1].col), (5, 5))

    def test_navigation_memory_keeps_structured_history(self):
        memory = NavigationMemory()
        memory.update_pose((1.0, 2.0))
        memory.mark_landmark_seen("church", 0.8)
        memory.set_candidate_status(2, CandidateStatus.REJECTED)
        memory.update_explored(np.array([[1, 0], [0, 0]], dtype=bool))
        memory.update_explored(np.array([[0, 1], [0, 0]], dtype=bool))
        self.assertEqual(memory.trajectory_xy, [(1.0, 2.0)])
        self.assertAlmostEqual(memory.seen_landmarks["church"], 0.8)
        self.assertEqual(memory.candidate_status[2], CandidateStatus.REJECTED)
        self.assertEqual(int(memory.explored_mask.sum()), 2)

    def test_gaussian_target_and_coverage(self):
        target = gaussian_field_target(7, 7, (3.0, 3.0), sigma_cells=1.0)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=5)
        metrics = candidate_coverage(target, (3.0, 3.0), ks=(1,), radii_cells=(1.0,))
        self.assertEqual(metrics["recall@1/1cells"], 1.0)


if __name__ == "__main__":
    unittest.main()
