import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CoarseToFineProtocolTest(unittest.TestCase):
    def test_protocol_does_not_enable_test_split(self):
        source = (ROOT / 'scripts/run_coarse_to_fine_smoke.py').read_text()
        self.assertNotIn('--include_test_unseen', source)

    def test_screening_is_finetuning_from_corrected_checkpoint(self):
        source = (ROOT / 'scripts/run_coarse_to_fine_smoke.py').read_text()
        self.assertIn("'--epochs', '1'", source)
        self.assertIn("'--max-episodes', '512'", source)
        self.assertIn("'--variant-arg=--checkpoint'", source)
        self.assertIn("'--variant-arg=--coarse_to_fine_target'", source)

    def test_paused_predecessor_requires_explicit_opt_in(self):
        source = (ROOT / 'scripts/run_coarse_to_fine_smoke.py').read_text()
        self.assertIn("parser.add_argument('--allow-paused-predecessor', action='store_true'", source)
        self.assertIn("and not args.allow_paused_predecessor", source)

    def test_full_run_gate_is_predeclared(self):
        source = (ROOT / 'scripts/run_coarse_to_fine_smoke.py').read_text()
        self.assertIn("metrics['first_cell_change_percent'] >= 25.0", source)
        self.assertIn("metrics['mean_first_prediction_correct_advantage_m'] > 0.0", source)
        self.assertIn("metrics['correct_first_prediction_better_percent'] >= 55.0", source)


if __name__ == '__main__':
    unittest.main()
