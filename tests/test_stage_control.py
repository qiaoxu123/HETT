import unittest

from multiagent.stage_control import advance_teacher_stage1


class TeacherStage1CounterTest(unittest.TestCase):
    def test_batch_episodes_advance_independently(self):
        steps = [0, 0]

        first_round = [
            advance_teacher_stage1(steps, episode, 10, 100)
            for episode in range(2)
        ]
        second_round = [
            advance_teacher_stage1(steps, episode, 10, 100)
            for episode in range(2)
        ]

        self.assertEqual(first_round, [10, 10])
        self.assertEqual(second_round, [20, 20])
        self.assertEqual(steps, [2, 2])

    def test_finished_episode_does_not_advance_other_counter(self):
        steps = [1, 1]

        second_episode_index = advance_teacher_stage1(steps, 1, 10, 100)

        self.assertEqual(second_episode_index, 20)
        self.assertEqual(steps, [1, 2])

    def test_out_of_range_falls_back_to_final_pose(self):
        steps = [2]

        index = advance_teacher_stage1(steps, 0, 10, 30)

        self.assertEqual(index, -1)
        self.assertEqual(steps, [3])

    def test_invalid_trajectory_is_rejected(self):
        with self.assertRaises(ValueError):
            advance_teacher_stage1([0], 0, 10, 0)


if __name__ == "__main__":
    unittest.main()
