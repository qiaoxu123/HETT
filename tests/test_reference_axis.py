import numpy as np
from multiagent.spatial_program.axis import resolve_axis
from multiagent.spatial_program.parser import parse_spatial_program

def test_start_to_anchor_axis_not_global_x():
    p=parse_spatial_program("Fly past the church and stop at the building")
    axis,_=resolve_axis(p,[{'centroid':[0,10],'contour':[[-2,9],[2,9],[2,11],[-2,11]]}],[0,0],0,[])
    assert axis['type']=='start_to_anchor'
    assert np.allclose(axis['vector'],[0,1])
