import unittest
import cv2
import numpy as np
from probe_contour_evidence import boundary_points, scores, edge_distance, valid_pixels


class ProbeTests(unittest.TestCase):
    def test_synthetic_aligned_boundary_wins(self):
        contour = np.float32([[60,60],[150,60],[150,150],[60,150]])
        points, _ = boundary_points(contour)
        image = np.full((224,224,3), 20, np.uint8)
        cv2.fillPoly(image, [contour.astype(int)], (255,255,255))
        _, distance = edge_distance(image)
        values = scores(distance, points)
        self.assertGreater(values[0], .95)
        self.assertGreater(values[0], max(values[1:]))

    def test_crop_border_is_not_polygon_boundary(self):
        points, fraction = boundary_points(np.float32([[-50,-50],[280,-50],[280,280],[-50,280]]))
        self.assertEqual(len(points), 0)
        self.assertEqual(fraction, 0)

    def test_blank_image_has_no_evidence(self):
        _, distance = edge_distance(np.zeros((224,224,3),np.uint8))
        values = scores(distance,np.float32([[100,100],[110,110]]))
        self.assertEqual(values,[0.]*9)

    def test_no_data_border_removed_not_small_dark_details(self):
        rgb=np.full((224,224,3),100,np.uint8)
        rgb[:,:40]=0
        rgb[100,100]=0
        valid=valid_pixels(rgb)
        self.assertEqual(valid[100,30],0)
        self.assertEqual(valid[100,100],1)

if __name__ == '__main__': unittest.main()
