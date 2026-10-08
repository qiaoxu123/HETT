import numpy as np
from multiagent.visual_goal.geometry_matching import symmetric_chamfer,observed_edges

def test_geometry_identity_and_empty():
    e=np.zeros((32,32),np.float32);e[5:20,8]=1;e[20,8:25]=1
    assert symmetric_chamfer(e,e)>.99
    assert symmetric_chamfer(e,np.zeros_like(e))==0

def test_observed_edges_exclude_padding():
    rgb=np.zeros((224,224,3),np.uint8);rgb[:,112:]=255
    valid=np.zeros((224,224),bool);valid[:,120:]=True
    assert not observed_edges(rgb,valid).any()
