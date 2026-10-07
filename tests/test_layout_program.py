import numpy as np
import pytest
from sensaturban_fpv.spatial_graph.layout_program import (parse_layout,footprint_axes,oriented_frame,
    parking_layout,score_order,road_s,road_sequence_score,soft_rank)

def test_parse_explicit_vs_generic():
    assert not parse_layout('building left of church')['layout_programs']
    p=parse_layout('2nd car from the left in the 20th row')
    assert [x['type'] for x in p['layout_programs']]==['NTH_FROM_LEFT','ROW_INDEX']
    assert [x['index'] for x in p['layout_programs']]==[2,20]
    assert parse_layout('3rd car from the right')['layout_programs'][0]['index']==3
    assert parse_layout('first column on the left')['layout_programs'][0]['type']=='COLUMN_INDEX'
    assert parse_layout('top row')['layout_programs'][0]['type']=='ROW_TOP'
    assert parse_layout('bottom row')['layout_programs'][0]['type']=='ROW_BOTTOM'

def test_frame_sign_square_and_no_answer():
    rect=footprint_axes([[0,0],[8,0],[8,2],[0,2]])
    assert rect['confidence']>.7
    assert footprint_axes([[0,0],[2,0],[2,2],[0,2]]) is None
    right=oriented_frame(rect['major'],'right');left=oriented_frame(rect['major'],'left')
    assert np.allclose(right,-left)
    assert parse_layout('with the long side of church to the right')['frame']['cues'][0]['axis_source']=='building_long_axis'
    assert parse_layout('with the short side of church to the left')['frame']['cues'][0]['axis_source']=='building_short_axis'

def test_parking_grid_soft_rank_and_counterfactual():
    points=np.array([[0,0],[4,0],[8,0],[0,8],[4,8],[8,8]],float)
    lay=parking_layout(points,np.eye(2))
    assert len(set(lay['rows']))==2 and len(set(lay['columns']))==3
    a=score_order({'type':'NTH_FROM_LEFT','index':2},lay['local'],lay['rows'])
    b=score_order({'type':'NTH_FROM_RIGHT','index':2},lay['local'],lay['rows'])
    assert np.allclose(a,b)  # middle of three is invariant; test edge separately
    left=score_order({'type':'NTH_FROM_LEFT','index':1},lay['local'],lay['rows'])
    right=score_order({'type':'NTH_FROM_RIGHT','index':1},lay['local'],lay['rows'])
    assert not np.allclose(left,right)
    top=score_order({'type':'ROW_TOP','index':1},lay['local'],lay['rows'])
    bottom=score_order({'type':'ROW_BOTTOM','index':1},lay['local'],lay['rows'])
    assert not np.allclose(top,bottom)
    assert soft_rank(2,2)>soft_rank(1,2)
    shuffled=score_order({'type':'NTH_FROM_LEFT','index':3},lay['local'],lay['rows'])
    assert not np.allclose(a,shuffled)
    assert np.allclose(a,score_order({'type':'NTH_FROM_LEFT','index':2},lay['local'][::-1],lay['rows'][::-1])[::-1])

def test_road_curvilinear_and_sequence():
    line=[[0,0],[10,0],[10,10]];p=[[1,1],[9,1],[10,9]]
    s=road_s(p,line)
    assert np.allclose(s,[.05,.45,.95])
    first=road_sequence_score({'type':'ROAD_SEQUENCE_ORDER','index':1},s)
    last=road_sequence_score({'type':'ROAD_SEQUENCE_ORDER','index':-1},s)
    assert np.argmax(first)==0 and np.argmax(last)==2
    third=road_sequence_score({'type':'NTH_FROM_CORNER','index':3},s)
    assert np.argmax(third)==2
    half=road_sequence_score({'type':'HALFWAY_ALONG','index':.5},s)
    assert np.argmax(half)==1
    assert parse_layout('3rd building from the corner')['layout_programs'][0]['type']=='NTH_FROM_CORNER'
    assert parse_layout('halfway down Alpha Road')['layout_programs'][0]['type']=='HALFWAY_ALONG'

def test_no_gt_parser_input_and_tie_order_invariance():
    import inspect
    from scripts.eval_layout_gate_g import metric
    from sensaturban_fpv.spatial_graph.layout_deepseek import request
    body=request('2nd car from the left')
    blob=str(body).lower()
    assert all(token not in blob for token in ('target_id','target_position','candidate_ids','correct_answer'))
    assert tuple(inspect.signature(parse_layout).parameters)==('instruction',)
    assert tuple(inspect.signature(parking_layout).parameters)==('centers','frame')
    x=metric([1,1,0],0);y=metric([1,0,1],2)
    assert x==y and x['top1']==.5

def test_reference_frame_counterfactual_and_column_order():
    points=np.array([[0,0],[4,0],[0,8],[4,8]],float)
    right=oriented_frame([1,0],'right');left=oriented_frame([1,0],'left')
    r=parking_layout(points,right);l=parking_layout(points,left)
    program={'type':'NTH_FROM_LEFT','index':1}
    assert not np.allclose(score_order(program,r['local'],r['rows']),score_order(program,l['local'],l['rows']))
    cols=score_order({'type':'COLUMN_LEFT','index':1},r['local'],r['rows'],r['columns'])
    assert cols[0]>cols[1]

@pytest.mark.parametrize('text,operator,index',[
    ('2nd from the left','NTH_FROM_LEFT',2),('5th from the right','NTH_FROM_RIGHT',5),
    ('third car from the top','NTH_FROM_TOP',3),('fourth car from the bottom','NTH_FROM_BOTTOM',4),
    ('20th row','ROW_INDEX',20),('second column','COLUMN_INDEX',2),
    ('top row','ROW_TOP',1),('bottom row','ROW_BOTTOM',1),
    ('leftmost column','COLUMN_LEFT',1),('rightmost column','COLUMN_RIGHT',1),
    ('3rd house from the corner','NTH_FROM_CORNER',3),('halfway down Alpha Road','HALFWAY_ALONG',.5),
    ('first building along Alpha Road','ROAD_SEQUENCE_ORDER',1),('last building along Alpha Road','ROAD_SEQUENCE_ORDER',-1),
])
def test_layout_phrase_family(text,operator,index):
    program=parse_layout(text)['layout_programs'][0]
    assert (program['type'],program['index'])==(operator,index)

def test_scope_selection_answer_blind():
    import inspect
    from scripts.eval_layout_gate_g import select_scope,cue_frame
    assert tuple(inspect.signature(select_scope).parameters)==('text','objects')
    assert tuple(inspect.signature(cue_frame).parameters)==('cues','objects','scope_center')

def test_val_unseen_not_in_calibration():
    from sensaturban_fpv.spatial_graph.layout_program import soft_rank
    assert tuple(__import__('inspect').signature(soft_rank).parameters)==('predicted','wanted','tau')
    assert soft_rank(2,2)>soft_rank(3,2)

def test_named_road_frame_uses_scope_not_target():
    from types import SimpleNamespace
    from scripts.eval_layout_gate_g import cue_frame
    road=lambda i,xy:SimpleNamespace(id=i,name='Wellington Road',object_type='TrafficRoad',position=xy,contour=[])
    cue={'axis_source':'view_cue','reference_phrase':'Wellington Road','direction':'right'}
    frame,selected,source=cue_frame([cue],[road(1,[5,0]),road(2,[50,0])],np.array([0.,0.]))
    assert selected==1 and source=='resolved_by_reference_position'
    assert frame[0,0]>.99

def test_row_index_start_cue_changes_target():
    p=np.array([[0.,0.],[2.,0.],[0.,10.],[2.,10.]])
    layout=parking_layout(p,np.eye(2))
    bottom=score_order({'type':'ROW_INDEX','index':1,'start':'bottom'},layout['local'],layout['rows'])
    top=score_order({'type':'ROW_INDEX','index':1,'start':'top'},layout['local'],layout['rows'])
    assert not np.allclose(bottom,top)

def test_selective_constraints_and_split_safe_weight():
    import inspect
    from sensaturban_fpv.spatial_graph.selective_layout_graph import Constraint,admissible,combine,calibrate_lambda
    assert admissible(Constraint('NEXT_TO','church'))
    assert not admissible(Constraint('BEHIND','church'))
    assert admissible(Constraint('BEHIND','church',True))
    assert np.allclose(combine([1,0],[0,1],.75),[.75,.25])
    assert tuple(inspect.signature(calibrate_lambda).parameters)==('train_seen','val_seen')
    rows=[([1,0],[0,1],0)]
    assert calibrate_lambda(rows,rows)>.5
