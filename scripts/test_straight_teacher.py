import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.teacher.straight_probe import straight_teacher,balanced_subset
from multiagent.space import Pose4D,Point3D
class Tests(unittest.TestCase):
    def test_path(self):
        start=Pose4D(0,0,50,1);goal=Point3D(30,40,0)
        p=straight_teacher(start,goal)
        self.assertEqual(p[0],start);self.assertEqual(p[-1][:3],(30,40,50))
        self.assertTrue(all(abs(x.y*.6-x.x*.8)<1e-6 for x in p))
        self.assertTrue(all(np.linalg.norm(np.array(b[:2])-a[:2])<=5.00001 for a,b in zip(p,p[1:])))
    def test_zero_and_short(self):
        p=Pose4D(0,0,50,1)
        self.assertEqual(straight_teacher(p,Point3D(0,0,0)),[p])
        self.assertEqual(len(straight_teacher(p,Point3D(1,1,0))),2)
    def test_balanced(self):
        ts=[SimpleNamespace(map_name=m,id=i) for m in ['a','b'] for i in range(20)]
        a=balanced_subset(ts,10,0);b=balanced_subset(ts,10,0)
        self.assertEqual([t.id for t in a],[t.id for t in b]);self.assertEqual(sum(t.map_name=='a' for t in a),5)
    def test_training_only_and_reference_preserved(self):
        from multiagent.env import CityNavBatch
        args=SimpleNamespace(altitude=50,max_episodes=4,balanced_screen=True,teacher_path_mode='straight')
        with patch('multiagent.env.cropclient.load_image_cache'):
            train=CityNavBatch('train_seen',args);val=CityNavBatch('val_unseen',args)
            self.assertEqual(len(train.teacher_paths),len(train.data));self.assertEqual(val.teacher_paths,{})
            for ep in train.data:
                p=train.teacher_paths[ep.id]
                self.assertIsNot(p,ep.trajectory);self.assertEqual(p[0],ep.start_pose)
                self.assertLess(np.linalg.norm(np.array(p[-1][:2])-ep.target_position[:2]),1e-6)
if __name__=='__main__':unittest.main()
