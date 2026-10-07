import numpy as np
from sensaturban_fpv.spatial_graph.semantic_canonicalizer import (parse_clauses,hypotheses,between_score,
    final_approach,route_frame,anchor_frame,marginalize,weighted)
from sensaturban_fpv.spatial_graph.semantic_deepseek import request,validate

def test_multi_clause_and_context():
    p=parse_clauses('white building behind the church on Aldridge Road')
    assert len(p['clauses'])==2
    assert hypotheses('on','road')!=hypotheses('on','building')

def test_no_leakage():
    blob=str(request('white building behind church'))
    assert all(x not in blob for x in ('target_id','target_position','candidate_index','object_ids'))

def test_frames():
    t=np.array([[0,0,0,1,0,0],[1,0,0,1,0,0],[2,1,0,1,0,0]])
    assert np.allclose(route_frame(t),np.array([2,1])/np.sqrt(5))
    assert np.allclose(final_approach(t,2),np.array([1,1])/np.sqrt(2))
    assert abs(anchor_frame([[0,0],[4,0],[4,1],[0,1]])[0])>.9

def test_between_and_marginal():
    assert between_score([5,0],[0,0],[10,0])>between_score([15,0],[0,0],[10,0])
    assert np.isclose(marginalize(np.log([.5,.5]),[1,1]),1)
    assert weighted(2,.25)==.5

def test_directional_counterfactual():
    from sensaturban_fpv.relation_v2 import GEOM_COLUMNS
    i={x:j for j,x in enumerate(GEOM_COLUMNS)}
    g=np.zeros((2,len(GEOM_COLUMNS)));g[:,i['agent_lateral']]=[-1,1]
    assert np.argmax(-g[:,i['agent_lateral']])!=np.argmax(g[:,i['agent_lateral']])

def test_schema_rejects_bad_family():
    assert not validate({'target_phrase':'house','clauses':[{'raw_phrase':'near','anchor_phrase':'road','semantic_family':'BAD','arity':1,'confidence':.5}]})
