import unittest
import torch
from multiagent.models.candidate_reranker import CandidateReranker

def fixture():
 torch.manual_seed(0)
 return {'map':torch.randn(2,20,256),'language':torch.randn(2,768),'agent':torch.randn(2,20,7),'landmark_geometry':torch.randn(2,20,16,8),'landmark_names':torch.randn(2,16,96),'landmark_valid':torch.ones(2,16,dtype=torch.bool),'field':torch.randn(2,20,2)}
class RerankerTest(unittest.TestCase):
 def test_zero_initialized_exact_scores(self):
  f=fixture();m=CandidateReranker();torch.testing.assert_close(m(f),f['field'][...,0],rtol=0,atol=0)
 def test_gt_labels_are_not_features(self):
  f=fixture();m=CandidateReranker();a=m(f);f['target_xy']=torch.randn(2,2)*100000;f['distances']=torch.randn(2,20);torch.testing.assert_close(a,m(f))
 def test_no_landmarks_finite(self):
  f=fixture();f['landmark_valid'][:]=False;m=CandidateReranker();self.assertTrue(torch.isfinite(m(f)).all())
 def test_gradients_and_learning(self):
  f=fixture();m=CandidateReranker();scores=m(f);loss=torch.nn.functional.cross_entropy(scores,torch.tensor([2,3]));loss.backward();self.assertGreater(m.head[-1].weight.grad.abs().sum().item(),0)
 def test_candidate_permutation(self):
  f=fixture();m=CandidateReranker();torch.nn.init.normal_(m.head[-1].weight,std=.01);order=torch.randperm(20);g={k:v[:,order] if k in ['map','agent','landmark_geometry','field'] else v for k,v in f.items()};torch.testing.assert_close(m(g),m(f)[:,order])
 def test_landmark_permutation(self):
  f=fixture();m=CandidateReranker();torch.nn.init.normal_(m.head[-1].weight,std=.01);order=torch.randperm(16);g=dict(f);g['landmark_geometry']=f['landmark_geometry'][:,:,order];g['landmark_names']=f['landmark_names'][:,order];g['landmark_valid']=f['landmark_valid'][:,order];torch.testing.assert_close(m(g),m(f),atol=1e-6,rtol=1e-5)
if __name__=='__main__':unittest.main()
