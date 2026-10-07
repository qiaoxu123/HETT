"""Text-only spatial clauses and executable, answer-blind program hypotheses."""
from __future__ import annotations

import re
import numpy as np

PHRASES = {
    'in front of':'VIEW_DEPENDENT_DIRECTION', 'front of':'VIEW_DEPENDENT_DIRECTION',
    'behind':'VIEW_DEPENDENT_DIRECTION', 'left of':'VIEW_DEPENDENT_DIRECTION',
    'right of':'VIEW_DEPENDENT_DIRECTION', 'to the left of':'VIEW_DEPENDENT_DIRECTION',
    'to the right of':'VIEW_DEPENDENT_DIRECTION', 'next to':'PROXIMITY',
    'beside':'PROXIMITY', 'near':'PROXIMITY', 'close to':'PROXIMITY',
    'between':'BETWEEN', 'on':'ROAD_ASSOCIATION', 'in':'CONTAINMENT',
    'off':'ROAD_ASSOCIATION', 'inside':'CONTAINMENT', 'on top of':'CONTAINMENT', 'along':'ROAD_ALONG', 'across from':'ROAD_ACROSS', 'across':'ROAD_ACROSS',
    'bordered by':'TOPOLOGICAL_ASSOCIATION', 'surrounded by':'CONTAINMENT',
    'connected to':'TOPOLOGICAL_ASSOCIATION', 'far from':'DISTANCE',
    'north of':'GLOBAL_DIRECTION', 'south of':'GLOBAL_DIRECTION',
    'east of':'GLOBAL_DIRECTION', 'west of':'GLOBAL_DIRECTION',
    'parallel to':'ALIGNMENT', 'perpendicular to':'ALIGNMENT',
}
ORDER = sorted(PHRASES, key=len, reverse=True)
REGEX = re.compile(r'\b(?:' + '|'.join(re.escape(p) for p in ORDER) + r')\b', re.I)
ROAD = re.compile(r'\b(road|street|avenue|lane|highway|roundabout|intersection|junction|bridge)\b', re.I)

def anchor_type(text):
    return 'road' if ROAD.search(text or '') else 'building' if re.search(r'\b(building|house|church|library|tower|shop|school|hotel|warehouse)\b', text or '', re.I) else 'unknown'

def parse_clauses(instruction):
    """Conservative fallback parser; preserves each clause and UNKNOWN."""
    raw_matches = list(REGEX.finditer(instruction))
    matches = []
    for i, match in enumerate(raw_matches):
        phrase = match.group().lower()
        following = instruction[match.end():raw_matches[i+1].start() if i+1 < len(raw_matches) else len(instruction)]
        # Bare 'in' frequently introduces a place/attribute, with no asserted
        # target-anchor geometry. Bare 'on' needs a typed surface or road.
        if phrase == 'in': continue
        if phrase in ('on', 'off') and not (ROAD.search(following) or re.search(r'\b(building|roof|wall|bridge)\b', following, re.I)):
            continue
        matches.append(match)
    target = instruction[:matches[0].start()].strip(' ,.;') if matches else instruction.strip()
    clauses = []
    for i, m in enumerate(matches):
        following = instruction[m.end():matches[i+1].start() if i+1 < len(matches) else len(instruction)].strip(' ,.;')
        if not following: continue
        phrase = m.group().lower()
        anchor = re.split(r'[.,;]', following, maxsplit=1)[0].strip()
        clauses.append({'raw_phrase':phrase, 'anchor_phrase':anchor,
                        'semantic_family':PHRASES[phrase], 'arity':2 if phrase == 'between' else 1,
                        'view_dependency':'possible' if PHRASES[phrase] == 'VIEW_DEPENDENT_DIRECTION' else 'none',
                        'confidence':0.5})
    return {'target_phrase':target, 'clauses':clauses}

def hypotheses(phrase, kind='unknown'):
    p = phrase.lower()
    if p in ('near','next to','beside','close to'):
        return ['CENTER_DISTANCE','FOOTPRINT_DISTANCE','ROAD_AWARE_DISTANCE','REGION_DISTANCE']
    if p == 'between': return ['BETWEEN_TERNARY']
    if p in ('behind','in front of','front of','left of','right of','to the left of','to the right of'):
        stem = 'BEHIND' if p == 'behind' else 'FRONT' if 'front' in p else 'LEFT' if 'left' in p else 'RIGHT'
        return [stem+'_'+f for f in ('GLOBAL','START_AGENT','CURRENT_AGENT','FINAL_APPROACH','ROUTE','ROAD','ANCHOR')]
    if p == 'on' and kind == 'road': return ['NEAR_ROAD_REGION','ROAD_ASSOCIATION','ON_ROAD_CORRIDOR']
    if p == 'along' and kind == 'road': return ['ROAD_TANGENT_ALIGNMENT','ROAD_PROJECTION']
    if p == 'across' and kind == 'road': return ['OPPOSITE_SIDE_OF_ROAD','LINE_CROSSES_ROAD']
    return ['UNKNOWN']

def unit(v):
    v = np.asarray(v, dtype=float)[:2]; n = np.linalg.norm(v)
    return v/n if n > 1e-9 else np.array([1.,0.])

def final_approach(trajectory, n=5):
    a=np.asarray(trajectory, dtype=float)
    if len(a)<2: return None
    return unit(a[-1,:2]-a[max(0,len(a)-n),:2])

def route_frame(trajectory):
    a=np.asarray(trajectory,dtype=float)
    return unit(a[-1,:2]-a[0,:2]) if len(a)>1 else None

def anchor_frame(footprint):
    p=np.asarray(footprint,dtype=float)[:,:2]
    if len(p)<2: return None
    _,_,vt=np.linalg.svd(p-p.mean(axis=0),full_matrices=False)
    axis=unit(vt[0]); return axis if axis[0]>0 or (axis[0]==0 and axis[1]>0) else -axis

def between_score(target, a, b):
    a=np.asarray(a)[:2];b=np.asarray(b)[:2];t=np.asarray(target)[:2]
    d=b-a; span=np.linalg.norm(d)
    if span<1e-6:return -10.
    u=np.dot(t-a,d)/span**2; perp=np.linalg.norm(t-(a+u*d))
    return -abs(u-.5)-perp/span-2*max(0,-u,u-1)

def marginalize(log_probs, scores):
    x=np.asarray(log_probs)+np.asarray(scores); m=np.max(x)
    return float(m+np.log(np.exp(x-m).sum()))

def weighted(score,rho): return float(score*rho)


def normalize_relation_phrase(raw_phrase, semantic_family=None):
    """Map an extracted clause span to a surface relation without map/answer data."""
    raw=(raw_phrase or '').lower()
    if semantic_family=='BETWEEN' and re.search(r'\b(in the middle of|midway between)\b',raw):
        return 'between'
    match=REGEX.search(raw)
    if match:
        phrase=match.group().lower()
        # Do not force an ambiguous short preposition into an unrelated family.
        if phrase in ('in','on','off') and semantic_family == 'UNKNOWN':return 'UNKNOWN'
        return phrase
    if semantic_family=='BETWEEN' and re.search(r'\b(in the middle of|midway between)\b',raw):
        return 'between'
    return 'UNKNOWN'
