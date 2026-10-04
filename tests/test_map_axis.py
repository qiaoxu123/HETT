import numpy as np
from multiagent.spatial_program.map_axis import principal_axis

def test_principal_axis_follows_long_dimension():
    axis,confidence=principal_axis([[-10,-1],[-10,1],[10,-1],[10,1]])
    assert abs(axis[0])>.99 and confidence>.9
