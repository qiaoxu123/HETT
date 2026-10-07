import numpy as np
from types import SimpleNamespace
from sensaturban_fpv.spatial_graph.gate_f_geometry import (trajectory_frames,anchor_axes,
    road_projection,between_features,frame_axes,GEOMETRY_COLUMNS,GEOMETRY_INDEX)
from sensaturban_fpv.spatial_graph.gate_f_binding import explicit_between_second,annotation_clauses
from sensaturban_fpv.spatial_graph.gate_f_programs import programs,score_candidates
from sensaturban_fpv.spatial_graph.node_types import Node,NodeKind
from sensaturban_fpv.spatial_graph.graph_builder import BlockGraph
from sensaturban_fpv.spatial_graph.edge_features import EdgeContext


def node(i,xy,kind=NodeKind.BUILDING,name=''):
    x,y=xy
    return Node(i,kind,kind.value,name,np.array([x,y,0.]),np.array([2.,2.,2.]),
                np.array([[x-1,y-1],[x+1,y-1],[x+1,y+1],[x-1,y+1]]),np.array([1.,0.]))


def trajectory(points):
    return np.array([[x,y,0,1,0,0] for x,y in points],float)


def test_trajectory_windows_and_zero_motion():
    f=trajectory_frames(trajectory([(0,0),(1,0),(2,0),(3,0),(3,1)]))
    assert np.allclose(f['final_heading_3'],[1/np.sqrt(2),1/np.sqrt(2)])
    assert np.allclose(f['final_heading_5'],[3/np.sqrt(10),1/np.sqrt(10)])
    assert np.allclose(f['final_heading_10'],f['route_heading'])
    assert np.allclose(f['final_heading_20'],f['route_heading'])
    assert np.dot(f['route_pca_heading'],f['route_heading'])>0
    z=trajectory_frames(trajectory([(0,0),(0,0),(0,0)]))
    assert z['final_heading_3'] is None and z['route_heading'] is None


def test_anchor_axes_and_no_target_frame_leak():
    major,minor=anchor_axes([[0,0],[6,0],[6,1],[0,1]])
    assert abs(major[0])>.99 and abs(major@minor)<1e-8
    a=node(1,(0,0))
    row={'start_heading':[1,0],'route_heading':[0,1],'target_position':[1000,1000]}
    x=frame_axes(row,a);row['target_position']=[-1000,-1000]
    y=frame_axes(row,a)
    assert all((x[k] is None and y[k] is None) or np.allclose(x[k],y[k]) for k in x)


def test_local_road_tangent_normal_and_side():
    members=[node(1,(0,0),NodeKind.ROAD_SEGMENT,'A Road'),
             node(2,(8,0),NodeKind.ROAD_SEGMENT,'A Road'),
             node(3,(8,8),NodeKind.ROAD_SEGMENT,'A Road')]
    region=node(4,(5,3),NodeKind.ROAD_REGION,'A Road')
    near_horizontal=road_projection([4,2],region,members)
    near_vertical=road_projection([10,7],region,members)
    assert abs(near_horizontal.tangent@near_vertical.tangent)<.9
    assert abs(near_horizontal.tangent@near_horizontal.normal)<1e-8
    assert road_projection([4,-2],region,members).side * near_horizontal.side < 0


def test_ternary_second_anchor_and_shuffle():
    a=[0,0];b=[10,0];t=[5,0]
    f=between_features(t,a,b);wrong=between_features(t,a,[0,10])
    assert f['between_inside_segment']==1 and f['between_perpendicular']==0
    assert wrong['between_perpendicular']>f['between_perpendicular']
    assert explicit_between_second('between Leslie and Willmore','Leslie Road',['Leslie Road','Willmore Road'])=='Willmore Road'
    assert explicit_between_second('near Leslie Road','Leslie Road',['Leslie Road','Willmore Road']) is None


def test_no_gt_nearest_anchor_and_candidate_order():
    a=node(1,(0,0),name='Church');b=node(2,(10,0),name='Church')
    c=node(3,(1,0));d=node(4,(9,0))
    graph=BlockGraph('x',[a,b,c,d],None,EdgeContext(),[],[])
    row={'start_heading':[1,0]}
    one=score_candidates('CENTER_DISTANCE',row,[c,d],[a,b],[],graph,{})
    two=score_candidates('CENTER_DISTANCE',row,[d,c],[a,b],[],graph,{})
    assert np.allclose(one,two[::-1])
    assert np.isclose(one[0],one[1]) # both name hypotheses retained


def test_road_context_and_schema():
    assert 'ROAD_ASSOCIATION' in programs('on','road_region')
    assert programs('on','building')==[]
    assert programs('across from','road_region')==[]
    assert set(programs('across from','road_region',True))=={'OPPOSITE_SIDE_OF_ROAD','LINE_CROSSES_ROAD'}
    assert 'ALONG_ROAD_TANGENT' in programs('along','road_region')
    assert len(GEOMETRY_COLUMNS)==len(GEOMETRY_INDEX)


def test_short_preposition_rejection():
    from sensaturban_fpv.spatial_graph.semantic_canonicalizer import parse_clauses
    assert all(c['raw_phrase']!='in' for c in parse_clauses('the white house in the parking lot')['clauses'])
    assert all(c['raw_phrase']!='on' for c in parse_clauses('the shop on a sunny day')['clauses'])
    assert any(c['raw_phrase']=='on' for c in parse_clauses('the building on Wellington Road')['clauses'])


def test_split_safe_calibration_probability_and_reliability():
    from sensaturban_fpv.spatial_graph.gate_f_calibration import fit_calibration
    train={'P':{'auc':.72,'n':40},'Q':{'auc':.51,'n':40}}
    val={'P':{'auc':.68,'n':25},'Q':{'auc':.49,'n':25}}
    best,prob,rho=fit_calibration(train,val)
    assert best=='P' and abs(sum(prob.values())-1)<1e-12 and 0<rho<1
    assert 'UNKNOWN' in prob
    # The API only accepts train and val_seen: unseen results cannot select a program.
    assert fit_calibration(train,val)==(best,prob,rho)


def test_parser_between_uses_text_only():
    from sensaturban_fpv.spatial_graph.gate_f_binding import parser_clauses
    sample={'instruction':'The building between Leslie and Willmore.',
            'annotation_phrases':['UNRELATED ANSWER ONLY'],
            'annotation_relations':[{'phrase':'between'}]}
    clauses=[c for c in parser_clauses(sample) if c['phrase']=='between']
    assert clauses and clauses[0]['anchor_phrase']=='Leslie'
    assert clauses[0]['second_anchor_phrase']=='Willmore'


def test_across_road_crossing_and_side():
    from sensaturban_fpv.spatial_graph.gate_f_geometry import road_crosses
    road=node(1,(0,0),NodeKind.ROAD_SEGMENT,'Road')
    assert road_crosses([-3,0],[3,0],[road])
    assert not road_crosses([-3,3],[3,3],[road])


def test_ternary_program_uses_second_anchor():
    a=node(1,(0,0),name='A');b=node(2,(10,0),name='B');wrong=node(3,(0,10),name='C')
    target=node(4,(5,0));distractor=node(5,(0,5))
    graph=BlockGraph('x',[a,b,wrong,target,distractor],None,EdgeContext(),[],[])
    row={}
    right=score_candidates('BETWEEN_TERNARY',row,[target,distractor],[a],[b],graph,{})
    shuffled=score_candidates('BETWEEN_TERNARY',row,[target,distractor],[a],[wrong],graph,{})
    assert right[0]>right[1] and shuffled[0]<shuffled[1]


def test_no_target_fields_in_deepseek_request():
    from sensaturban_fpv.spatial_graph.semantic_deepseek import request
    body=str(request('The building behind the church'))
    for key in ('target_id','target_position','candidate_index','object_ids','gt_rank'):
        assert key not in body


def test_old_hard_ontology_program_matches_existing_geometry():
    from sensaturban_fpv.spatial_graph.relation_geometry import satisfaction,RelationContext
    a=node(1,(0,0));c=node(2,(4,0))
    graph=BlockGraph('x',[a,c],None,EdgeContext(),[],[])
    got=score_candidates('OLD_NEAR',{},[c],[a],[],graph,{})
    assert np.isclose(got[0],satisfaction('near',a,c,RelationContext()))


def test_across_program_requires_independent_second_anchor():
    road=node(1,(0,0),NodeKind.ROAD_REGION,'Main Road')
    segment=node(2,(0,0),NodeKind.ROAD_SEGMENT,'Main Road')
    landmark=node(3,(-3,0),name='Church');candidate=node(4,(3,0))
    graph=BlockGraph('x',[road,segment,landmark,candidate],None,EdgeContext(road_regions=[road],road_members={0:[segment]}),[road],[segment])
    assert score_candidates('LINE_CROSSES_ROAD',{},[candidate],[road],[],graph,{}) is None
    got=score_candidates('LINE_CROSSES_ROAD',{},[candidate],[road],[landmark],graph,{})
    assert got is not None and got[0]==1


def test_probability_marginalization_keeps_unknown_mass():
    from scripts.eval_gate_f import combine
    scores={'P':np.array([1.,0.])}
    a=combine(scores,{'P':1.,'UNKNOWN':0.})
    b=combine(scores,{'P':.5,'UNKNOWN':.5})
    assert abs(a[0]-a[1])>abs(b[0]-b[1])
    assert np.allclose(combine(scores,{'P':.5,'UNKNOWN':.5},0),0)


def test_front_back_counterfactual_scores_reverse():
    a=node(1,(0,0));front=node(2,(4,0));back=node(3,(-4,0))
    graph=BlockGraph('x',[a,front,back],None,EdgeContext(),[],[])
    row={'start_heading':[1,0]}
    f=score_candidates('FRONT_START',row,[front,back],[a],[],graph,{})
    b=score_candidates('BACK_START',row,[front,back],[a],[],graph,{})
    assert np.argmax(f)!=np.argmax(b)
    assert np.allclose(f,-b)


def test_deepseek_clause_span_normalization_and_unknown():
    from sensaturban_fpv.spatial_graph.semantic_canonicalizer import normalize_relation_phrase
    assert normalize_relation_phrase('behind the church','VIEW_DEPENDENT_DIRECTION')=='behind'
    assert normalize_relation_phrase('on Aldridge Road','ROAD_ASSOCIATION')=='on'
    assert normalize_relation_phrase('in the middle of A and B','BETWEEN')=='between'
    assert normalize_relation_phrase('somewhere nearby-ish','UNKNOWN')=='UNKNOWN'


def test_relational_program_rejects_self_binding():
    a=node(1,(0,0),name='Church');other=node(2,(4,0))
    graph=BlockGraph('x',[a,other],None,EdgeContext(),[],[])
    scores=score_candidates('CENTER_DISTANCE',{},[a,other],[a],[],graph,{})
    assert scores[0]<scores[1]
