import unittest

import torch

from multiagent.hypothesis_control import TemporalHypothesisTracker


class TemporalHypothesisTrackerTest(unittest.TestCase):
    def setUp(self):
        self.positions = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])

    def test_keeps_ranked_alternatives(self):
        tracker = TemporalHypothesisTracker(4, top_k=3)
        position, info = tracker.update(torch.tensor([1., 4., 3., 2.]), self.positions)
        torch.testing.assert_close(position, self.positions[1])
        self.assertEqual(info['indices'], [1, 2, 3])

    def test_uniform_observation_does_not_create_false_confidence(self):
        tracker = TemporalHypothesisTracker(4)
        _, first = tracker.update(torch.zeros(4), self.positions)
        _, repeated = tracker.update(torch.zeros(4), self.positions)
        self.assertAlmostEqual(first['confidence'], 0.0, places=6)
        self.assertAlmostEqual(repeated['confidence'], 0.0, places=6)
        self.assertAlmostEqual(repeated['probabilities'][0], 0.25, places=6)

    def test_repeated_contradictory_evidence_can_change_map(self):
        tracker = TemporalHypothesisTracker(4, decay=0.5)
        first, _ = tracker.update(torch.tensor([8., 0., 0., 0.]), self.positions)
        tracker.update(torch.tensor([0., 9., 0., 0.]), self.positions)
        second, _ = tracker.update(torch.tensor([0., 9., 0., 0.]), self.positions)
        torch.testing.assert_close(first, self.positions[0])
        torch.testing.assert_close(second, self.positions[1])

    def test_nonfinite_or_wrong_shape_is_rejected(self):
        tracker = TemporalHypothesisTracker(4)
        with self.assertRaises(ValueError):
            tracker.update(torch.zeros(3), self.positions)
        with self.assertRaises(ValueError):
            tracker.update(torch.tensor([0., 0., float('nan'), 0.]), self.positions)


if __name__ == '__main__':
    unittest.main()
