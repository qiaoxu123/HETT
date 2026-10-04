import unittest

import numpy as np

from multiagent.agent import advance_teacher_stage1_index


class TeacherStage1IndexTest(unittest.TestCase):
    def test_first_step_is_independent_of_batch_position(self):
        steps = np.zeros(16, dtype=np.int32)
        indices = [
            advance_teacher_stage1_index(steps, i, 10, 100)
            for i in range(16)
        ]
        self.assertEqual(indices, [10] * 16)
        np.testing.assert_array_equal(steps, np.ones(16, dtype=np.int32))

    def test_only_selected_episode_advances_and_clamps_at_end(self):
        steps = np.zeros(2, dtype=np.int32)
        self.assertEqual(advance_teacher_stage1_index(steps, 1, 10, 25), 10)
        self.assertEqual(advance_teacher_stage1_index(steps, 1, 10, 25), 20)
        self.assertEqual(advance_teacher_stage1_index(steps, 1, 10, 25), -1)
        self.assertEqual(advance_teacher_stage1_index(steps, 0, 10, 25), 10)
        np.testing.assert_array_equal(steps, [1, 3])


if __name__ == '__main__':
    unittest.main()
