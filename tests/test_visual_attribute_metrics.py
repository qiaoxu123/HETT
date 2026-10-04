import unittest

import numpy as np

from multiagent.visual_attributes.metrics import classification_metrics, grouped_consistency, macro_auroc, observability_score


class VisualAttributeMetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        probabilities = np.asarray([[0.99, 0.01], [0.02, 0.98]])
        metrics = classification_metrics(probabilities, np.asarray([0, 1]))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["macro_precision"], 1.0)
        self.assertEqual(metrics["macro_recall"], 1.0)
        self.assertEqual(metrics["macro_auroc"], 1.0)

    def test_auroc_ties_receive_half_credit(self):
        probabilities = np.asarray([[0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(macro_auroc(probabilities, np.asarray([0, 1])), 0.5)

    def test_cross_height_consistency(self):
        probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        metrics = grouped_consistency(probabilities, ["a", "a", "b", "b"], [20, 80, 20, 80])
        self.assertEqual(metrics["agreement"], 1.0)

    def test_observability_requires_all_components(self):
        self.assertAlmostEqual(observability_score(0.5, 0.8, 0.7, 0.9), 0.68)
        with self.assertRaises(ValueError):
            observability_score(0.5, None, 0.7, 0.9)


if __name__ == "__main__": unittest.main()
