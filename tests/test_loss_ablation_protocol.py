import sys
import unittest

from multiagent.parser import parse_args


def parsed(*extra):
    saved = sys.argv
    try:
        sys.argv = ['loss-ablation-test', '--mode', 'train', *extra]
        return parse_args()
    finally:
        sys.argv = saved


class LossAblationProtocolTest(unittest.TestCase):
    def test_released_weights(self):
        args = parsed()
        self.assertEqual((args.goal_loss_weight, args.direction_loss_weight,
                          args.progress_loss_weight, args.target_loss_weight),
                         (2.0, 1.5, 0.1, 0.1))

    def test_paper_only_weights(self):
        args = parsed('--target_loss_weight', '0')
        self.assertEqual((args.progress_loss_weight, args.target_loss_weight), (0.1, 0.0))

    def test_no_progress_weights(self):
        args = parsed('--progress_loss_weight', '0')
        self.assertEqual((args.progress_loss_weight, args.target_loss_weight), (0.0, 0.1))

    def test_neither_auxiliary_weight(self):
        args = parsed('--progress_loss_weight', '0', '--target_loss_weight', '0')
        self.assertEqual((args.progress_loss_weight, args.target_loss_weight), (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
