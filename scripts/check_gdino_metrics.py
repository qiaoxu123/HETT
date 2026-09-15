import unittest
from test_gdino_frames import iou,noun_prompt
class Checks(unittest.TestCase):
    def test_exact(self): self.assertAlmostEqual(iou([1,2,11,12],[1,2,11,12]),1)
    def test_disjoint(self): self.assertEqual(iou([0,0,1,1],[2,2,3,3]),0)
    def test_overlap(self): self.assertAlmostEqual(iou([0,0,2,2],[1,0,3,2]),1/3)
    def test_prompt_uses_text_only(self): self.assertEqual(noun_prompt('a singular silver car.'),'car.')
if __name__=='__main__': unittest.main()
