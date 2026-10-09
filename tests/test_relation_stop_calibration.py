import unittest
from scripts.calibrate_relation_stop import replay

class RelationStopReplayTest(unittest.TestCase):
    def episode(self):
        return dict(episode_id=['map',1,0],goal_xy=[100.,0.],path_xy=[[0.,0.],[50.,0.],[100.,0.]],
                    navigation_steps=[dict(path_index=0,stop_probability=.1),dict(path_index=1,stop_probability=.6)])
    def test_disabled_stop_keeps_official_straight_path_metrics(self):
        summary,eps=replay([self.episode()],1.01)
        self.assertEqual(summary['sr'],100.);self.assertEqual(summary['spl'],100.)
        self.assertFalse(eps[0]['stop']);self.assertEqual(eps[0]['ne'],0.)
    def test_stop_before_next_action_records_false_positive(self):
        summary,eps=replay([self.episode()],.5)
        self.assertEqual(eps[0]['path_xy'],[[0.,0.],[50.,0.]])
        self.assertEqual(summary['stop_FP'],1);self.assertEqual(summary['sr'],0.)
    def test_gt_does_not_choose_stop_time(self):
        a=self.episode();b=self.episode();b['goal_xy']=[-500.,0.]
        self.assertEqual(replay([a],.5)[1][0]['path_xy'],replay([b],.5)[1][0]['path_xy'])
    def test_t0_stop_adds_no_motion(self):
        e=self.episode();e['navigation_steps'][0]['stop_probability']=.9
        self.assertEqual(replay([e],.5)[1][0]['path_xy'],[[0.,0.]])

if __name__=='__main__':unittest.main()
