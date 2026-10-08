import unittest
import numpy as np
from multiagent.space import Pose4D
from multiagent.heatmap_execution import bounded_heatmap_step

class ExecutionTest(unittest.TestCase):
    def test_far_target_is_bounded(self):
        p=bounded_heatmap_step(Pose4D(0,0,50,1),[60,80])
        np.testing.assert_allclose(p[:3],[30,40,50])
    def test_near_target_no_overshoot(self):
        p=bounded_heatmap_step(Pose4D(0,0,50,1),[3,4])
        np.testing.assert_allclose(p[:3],[3,4,50])
    def test_stationary_keeps_yaw_and_altitude(self):
        p=Pose4D(10,20,30,1.23)
        self.assertEqual(bounded_heatmap_step(p,p.xy),p)
    def test_negative_coordinates(self):
        p=bounded_heatmap_step(Pose4D(0,0,50,1),[-3,-4])
        np.testing.assert_allclose(p[:2],[-3,-4])
    def test_invalid_inputs(self):
        for step in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):bounded_heatmap_step(Pose4D(0,0,50,0),[1,1],step)
        with self.assertRaises(ValueError):bounded_heatmap_step(Pose4D(0,0,50,0),[float('nan'),0])
    def test_repeated_target_converges(self):
        p=Pose4D(0,0,50,0)
        for _ in range(10):p=bounded_heatmap_step(p,[123,234])
        np.testing.assert_allclose(p[:2],[123,234])

class ExecutionConfigTest(unittest.TestCase):
    def test_default_preserves_two_stage(self):
        import sys
        from unittest.mock import patch
        from multiagent.parser import parse_args
        with patch.object(sys,'argv',['test']):
            self.assertEqual(parse_args().heatmap_execution,'two_stage')
    def test_eval_enables_waypoint(self):
        import sys
        from unittest.mock import patch
        from multiagent.parser import parse_args
        with patch.object(sys,'argv',['test','--mode','eval','--heatmap_execution','waypoint']):
            self.assertEqual(parse_args().heatmap_execution,'waypoint')
    def test_training_rejects_inference_only_mode(self):
        import sys
        from unittest.mock import patch
        from multiagent.parser import parse_args
        with patch.object(sys,'argv',['test','--heatmap_execution','waypoint']):
            with self.assertRaises(ValueError):parse_args()

if __name__=='__main__':unittest.main()
