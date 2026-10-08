import numpy as np
from affine import Affine
from shapely.geometry import box
from multiagent.visual_goal.overlap import pose_corners,crop_transform,world_to_image,valid_mask,coverage,bin_mask


def test_xy_colrow_yaw_origin_and_scale():
    for affine in (Affine(.1,0,100,0,-.1,200),Affine(.2,0,-50,0,-.3,500)):
        center=affine*(500.5,500.5)
        H,full,valid=crop_transform(affine,1000,1000,center,10,0)
        pixels=world_to_image([center,np.asarray(center)+[2,0],np.asarray(center)+[0,2]],affine,H)
        assert np.linalg.norm(pixels[0]-111.5)<3
        assert pixels[1,1]<pixels[0,1] # +worldX is up
        assert pixels[2,0]<pixels[0,0] # +worldY is left
        assert abs(full.area-400)<20


def test_padding_not_valid_area_and_rotation():
    H,full,valid=crop_transform(Affine(1,0,0,0,-1,100),100,100,[0,50],10,np.pi/4)
    assert .4<valid.area/full.area<.6
    assert .4<valid_mask(H,100,100).mean()<.6


def test_overlap_ratios_and_no_overlap():
    v=coverage(box(0,0,2,2),box(1,0,3,2))
    assert v['intersection_m2']==2 and v['iou']==1/3
    assert v['query_coverage']==v['template_coverage']==.5
    assert coverage(box(0,0,1,1),box(2,2,3,3))['iou']==0


def test_bins_include_exact_boundaries():
    assert bin_mask([0,.1,.1001],0,.1).tolist()==[False,True,False]
    assert bin_mask([0,.001],0,0).tolist()==[True,False]
