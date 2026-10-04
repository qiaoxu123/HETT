import numpy as np
from multiagent.spatial_program.executor import ProgramExecutor
from multiagent.spatial_program.parser import parse_spatial_program

def test_sequential_soft_program_prefers_right_and_past():
    p=parse_spatial_program("Fly past the church and stop at the building on the right")
    points=[[0,20],[10,20],[-10,-20]];anchor={'centroid':[0,10],'contour':[[-1,9],[1,9],[1,11],[-1,11]]}
    score,debug=ProgramExecutor().execute(p,points,[1/3]*3,[anchor],[0,0],0,[])
    assert np.isclose(score.sum(),1) and len(debug['steps'])==2

def test_hard_and_soft_execution_exist():
    p=parse_spatial_program("Stop at the building near the church")
    a={'centroid':[0,0],'contour':[[-1,-1],[1,-1],[1,1],[-1,1]]}
    for soft in (True,False):
        score,_=ProgramExecutor(soft=soft).execute(p,[[1,0],[100,0]],[.5,.5],[a],[0,-10],0,[])
        assert np.isclose(score.sum(),1)
