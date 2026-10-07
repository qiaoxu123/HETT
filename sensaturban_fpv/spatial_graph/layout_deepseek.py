"""Language-only DeepSeek layout role extraction."""
import random
from .deepseek_parser import call_model,check_no_leakage

OPERATORS=('ROW_INDEX','COLUMN_INDEX','NTH_FROM_LEFT','NTH_FROM_RIGHT','NTH_FROM_TOP','NTH_FROM_BOTTOM',
           'LEFTMOST','RIGHTMOST','TOPMOST','BOTTOMMOST','FIRST','LAST','NEAREST_TO_END','NEAREST_TO_CORNER',
           'NTH_FROM_CORNER','HALFWAY_ALONG','ALONG_AXIS_ORDER','ROAD_SEQUENCE_ORDER','PARKING_ROW_ORDER',
           'OBJECT_LINE_ORDER','GRID_POSITION')
AXES=('building_long_axis','building_short_axis','road_axis','view_cue','unknown')
SYSTEM='Extract explicit frame cues and layout/ordinal programs from an instruction. Return JSON only. Do not choose any map object, candidate, geometry score, or correct answer. If no explicit frame cue is present, frame.is_explicit=false. Preserve all clauses; do not reinterpret generic left-of as an ordinal layout.'

def request(instruction,shuffled=False):
    operators=list(OPERATORS)
    if shuffled:random.Random(20261007).shuffle(operators)
    schema={'target_phrase':'text','anchors':[{'phrase':'text','type':'building|road|landmark|object|unknown'}],
            'frame':{'is_explicit':False,'reference_phrase':'text','reference_type':'text','axis_source':'building_long_axis|building_short_axis|road_axis|view_cue|unknown','orientation_constraints':[{'entity':'text','direction':'left|right|top|bottom'}]},
            'layout_programs':[{'type':'operator','index':1,'scope':'row|column|object_line|parking_grid|road_sequence|unknown'}]}
    body={'model':'deepseek-chat','temperature':0,'response_format':{'type':'json_object'},'messages':[
        {'role':'system','content':SYSTEM},
        {'role':'user','content':'Instruction:\n'+instruction+'\n\nAllowed operators: '+', '.join(operators)+'\nAllowed axis sources: '+', '.join(AXES)+'\nSchema example (replace values): '+str(schema)}]}
    check_no_leakage(body);return body

def valid(p):
    if not isinstance(p,dict) or not isinstance(p.get('target_phrase'),str) or not isinstance(p.get('anchors'),list) or not isinstance(p.get('frame'),dict) or not isinstance(p.get('layout_programs'),list):return False
    f=p['frame']
    if not isinstance(f.get('is_explicit'),bool) or not isinstance(f.get('reference_phrase'),str) or not isinstance(f.get('reference_type'),str) or f.get('axis_source') not in AXES or not isinstance(f.get('orientation_constraints'),list):return False
    if not all(isinstance(a,dict) and isinstance(a.get('phrase'),str) and a.get('type') in ('building','road','landmark','object','unknown') for a in p['anchors']):return False
    if not all(isinstance(c,dict) and isinstance(c.get('entity'),str) and c.get('direction') in ('left','right','top','bottom') for c in f['orientation_constraints']):return False
    return all(isinstance(q,dict) and q.get('type') in OPERATORS and (isinstance(q.get('index'),int) or q.get('index') is None) and q.get('scope') in ('row','column','object_line','parking_grid','road_sequence','unknown') for q in p['layout_programs'])

def call(instruction,key,cache,shuffled=False):
    p,_=call_model(request(instruction,shuffled),key,cache=cache)
    return p if valid(p) else None
