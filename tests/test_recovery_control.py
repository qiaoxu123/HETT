import unittest

from multiagent.stage_control import RecoveryState, update_recovery_state


class RecoveryControlTest(unittest.TestCase):
    def test_enters_fine_at_predicted_target(self):
        state = RecoveryState()
        self.assertEqual(update_recovery_state(state, 4.9, 0.2), "fine")
        self.assertTrue(state.fine)
        self.assertEqual(state.switches, 1)

    def test_single_far_prediction_does_not_flip_stage(self):
        state = RecoveryState(fine=True)
        self.assertEqual(update_recovery_state(state, 21.0, 0.2), "fine")
        self.assertTrue(state.fine)
        self.assertEqual(state.far_count, 1)

    def test_repeated_far_predictions_recover_to_coarse(self):
        state = RecoveryState(fine=True)
        update_recovery_state(state, 21.0, 0.2)
        self.assertEqual(update_recovery_state(state, 24.0, 0.2), "coarse")
        self.assertFalse(state.fine)
        self.assertEqual(state.recoveries, 1)

    def test_hysteresis_prevents_boundary_oscillation(self):
        state = RecoveryState(fine=True)
        for distance in (6.0, 18.0, 7.0, 19.9):
            self.assertEqual(update_recovery_state(state, distance, 0.2), "fine")
        self.assertEqual(state.switches, 0)
        self.assertEqual(state.recoveries, 0)

    def test_stop_needs_two_consistent_predictions(self):
        state = RecoveryState(fine=True)
        self.assertEqual(update_recovery_state(state, 8.0, 0.96), "fine")
        self.assertEqual(update_recovery_state(state, 7.0, 0.97), "stop")

    def test_far_target_blocks_false_stop(self):
        state = RecoveryState(fine=True)
        update_recovery_state(state, 8.0, 0.99)
        self.assertEqual(update_recovery_state(state, 25.0, 0.99), "fine")
        self.assertEqual(state.stop_count, 0)

    def test_can_enter_fine_again_after_recovery(self):
        state = RecoveryState(fine=True)
        update_recovery_state(state, 21.0, 0.2)
        update_recovery_state(state, 22.0, 0.2)
        self.assertEqual(update_recovery_state(state, 4.0, 0.3), "fine")
        self.assertTrue(state.fine)
        self.assertEqual(state.switches, 1)

    def test_rejects_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            update_recovery_state(RecoveryState(), 1.0, 0.1,
                                  enter_distance=20.0, recover_distance=5.0)


if __name__ == "__main__":
    unittest.main()
