import inspect
import numpy as np
import pytest
from multiagent.visual_goal import local_matching, geometry_matching,score_fusion
from multiagent.visual_goal.diagnosis_data import load_split
from multiagent.visual_goal.failure_analysis import query_outcomes,clustered_mean_ci

def test_matcher_interfaces_cannot_take_target_metadata():
    for f in (local_matching.pair_scores,geometry_matching.symmetric_chamfer,geometry_matching.observed_edges,score_fusion.fuse):
        assert not set(inspect.signature(f).parameters)&{'target_id','target_xy','map_name','scene_key','overlap','yaw'}

def test_no_unseen_alpha_selection_or_test_access():
    with pytest.raises(ValueError): score_fusion.select_alpha({},'val_unseen')
    with pytest.raises(ValueError): load_split('test_unseen')

def test_candidate_permutation_and_tie_fairness():
    q=[dict(scene_key='a',map_name='m')];c=[dict(scene_key=x,map_name='m') for x in 'abc']
    s=np.array([[.5,.5,.1]]);a,_,_=query_outcomes(s,q,c);b,_,_=query_outcomes(s[:,::-1],q,c[::-1])
    assert a['R@1'][0]==b['R@1'][0]==.5
    assert a['hard_accuracy'][0]==0

def test_bootstrap_resamples_whole_episode():
    a=clustered_mean_ci([0,0,1,1],['a','a','b','b'])
    assert a==[0.,1.]

def test_fusion_endpoints_monotone():
    a=np.array([[.1,.4,.2]]);b=a[:,::-1]
    assert np.argmax(score_fusion.fuse(a,b,1))==np.argmax(a)
