import unittest
import torch
from multiagent.models.selector_fusion import bounded_selector_fusion

class FusionTests(unittest.TestCase):
    def test_zero_preserves_field(self):
        a=torch.randn(10,20);b=torch.randn_like(a)
        self.assertTrue(torch.equal(a.argmax(-1),bounded_selector_fusion(a,b,0).argmax(-1)))
    def test_strong_margin_protected(self):
        a=torch.tensor([[10.,0.,0.]]);b=torch.tensor([[-100.,100.,100.]])
        self.assertEqual(bounded_selector_fusion(a,b).argmax(-1).item(),0)
    def test_near_tie_can_change(self):
        a=torch.tensor([[1.,.99,-3.]]);b=torch.tensor([[-10.,10.,0.]])
        self.assertEqual(bounded_selector_fusion(a,b).argmax(-1).item(),1)
    def test_flat_and_single_finite(self):
        for k in (1,20):
            self.assertTrue(torch.isfinite(bounded_selector_fusion(torch.ones(2,k),torch.ones(2,k))).all())
    def test_permutation(self):
        a=torch.randn(2,20);b=torch.randn_like(a);p=torch.randperm(20)
        self.assertTrue(torch.allclose(bounded_selector_fusion(a[:,p],b[:,p]),bounded_selector_fusion(a,b)[:,p],atol=1e-6))
    def test_bad_input(self):
        a=torch.ones(2,3)
        for w in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):bounded_selector_fusion(a,a,w)
        with self.assertRaises(ValueError):bounded_selector_fusion(a,a*float('nan'))
if __name__=='__main__':unittest.main()
