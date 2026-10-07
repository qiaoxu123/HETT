"""DeepSeek receives only instruction text and semantic family definitions."""
from __future__ import annotations
import json
from .deepseek_parser import call_model, check_no_leakage
FAMILIES=('PROXIMITY','DISTANCE','GLOBAL_DIRECTION','VIEW_DEPENDENT_DIRECTION',
          'ROAD_ASSOCIATION','ROAD_SIDE','ROAD_ACROSS','ROAD_ALONG','BETWEEN',
          'ALIGNMENT','CONTAINMENT','TOPOLOGICAL_ASSOCIATION','UNKNOWN')
SYSTEM=('Extract spatial language roles. Return JSON only: '
        '{"target_phrase":"...","clauses":[{"raw_phrase":"...",'
        '"anchor_phrase":"...","semantic_family":"...","arity":1,'
        '"view_dependency":"none|possible|required","confidence":0.5}]}. '
        'Preserve multiple clauses. BETWEEN has arity 2. UNKNOWN is allowed. '
        'Do not choose a map entity or geometry program.')
def request(instruction,shuffle=False):
    import random
    families=list(FAMILIES)
    if shuffle:random.Random(1729).shuffle(families)
    body={'model':'deepseek-chat','temperature':0,'response_format':{'type':'json_object'},
          'messages':[{'role':'system','content':SYSTEM},
                      {'role':'user','content':'Instruction:\n'+instruction+'\n\nSemantic families:\n'+', '.join(families)}]}
    check_no_leakage(body);return body
def validate(parsed):
    if not isinstance(parsed,dict) or not isinstance(parsed.get('target_phrase'),str) or not isinstance(parsed.get('clauses'),list):return False
    return all(isinstance(c,dict) and isinstance(c.get('raw_phrase'),str) and
               isinstance(c.get('anchor_phrase'),str) and c.get('semantic_family') in FAMILIES and
               c.get('arity') in (1,2) and isinstance(c.get('confidence'),(int,float)) and
               0<=c['confidence']<=1 for c in parsed['clauses'])
def call(instruction,key,cache=None,shuffle=False):
    parsed,_=call_model(request(instruction,shuffle),key,cache=cache)
    return parsed if validate(parsed) else None
