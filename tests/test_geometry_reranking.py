import numpy as np
from multiagent.geometry_reasoning.parser import parse_geometry
from multiagent.geometry_reasoning.scorer import GeometryReasoner
def test_right_rule_reranks_right_candidate():
 p=parse_geometry('Go right of the church.',('church',));anchors=[{'query':'church','matches':[{'centroid':[10,0],'orientation':[1,0]}]}];scores=GeometryReasoner().score(p,anchors,[(10,-20),(10,20)],(0,0),0);assert scores[0]['total']>scores[1]['total']
def test_soft_conflicts_remain_finite():
 p=parse_geometry('second building right of and behind church',('church',));a=[{'query':'church','matches':[{'centroid':[10,0],'orientation':[1,0]}]}];v=GeometryReasoner().score(p,a,[(1,1),(2,2)],(0,0),0);assert np.isfinite([x['total'] for x in v]).all()
