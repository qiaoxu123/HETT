"""Executable Gate F programs; no answer fields enter scoring."""
from __future__ import annotations
import numpy as np
from .gate_f_geometry import (FRAME_NAMES, frame_axes, perpendicular, road_projection,
                              road_crosses, between_features)
from .edge_features import footprint_distance
from .node_types import NodeKind

DIRECTION = {'behind':('BACK',-1),'in front of':('FRONT',1),'front of':('FRONT',1),
             'left of':('LEFT',1),'to the left':('LEFT',1),'to the left of':('LEFT',1),
             'right of':('RIGHT',-1),'to the right':('RIGHT',-1),'to the right of':('RIGHT',-1)}
PROXIMITY={'near','next to','beside','close to','adjacent to'}
ROAD_WORDS={'on','off','along','across','across from'}

def family(phrase, anchor_kind):
    p=phrase.lower()
    if p in DIRECTION:return 'DIRECTION'
    if p=='between':return 'BETWEEN'
    if p in PROXIMITY:return 'PROXIMITY'
    if p in ROAD_WORDS and anchor_kind=='road_region':return 'ROAD'
    return 'UNKNOWN'

def programs(phrase, anchor_kind, has_second=False):
    f=family(phrase,anchor_kind)
    if f=='DIRECTION':return [f'{DIRECTION[phrase.lower()][0]}_{name}' for name in FRAME_NAMES]
    if f=='BETWEEN':return ['DISTANCE_PRIOR','PAIRWISE_OLD','BETWEEN_TERNARY'] if has_second else []
    if f=='PROXIMITY':return ['CENTER_DISTANCE','FOOTPRINT_DISTANCE']
    if f=='ROAD':
        p=phrase.lower()
        if p=='on':return ['NEAR_ROAD_REGION','IN_ROAD_BUFFER','ROAD_ASSOCIATION','ALONG_ROAD']
        if p=='off':return ['NEAR_ROAD_REGION','ROAD_ASSOCIATION','OFF_ROAD_BUFFER']
        if p=='along':return ['ALONG_ROAD_TANGENT','NEAR_ROAD_REGION']
        if p in ('across','across from'):
            return ['OPPOSITE_SIDE_OF_ROAD','LINE_CROSSES_ROAD'] if has_second else []
    return []

def anchor_point(node,candidate,graph,road_cache):
    if node.kind is not NodeKind.ROAD_REGION:return node.center[:2]
    return cached_road(candidate,node,graph,road_cache).nearest_point

def cached_road(candidate,region,graph,cache):
    key=(graph.map_name,candidate.node_id,region.node_id)
    if key not in cache:
        idx=next(i for i,r in enumerate(graph.regions) if r.node_id==region.node_id)
        cache[key]=road_projection(candidate.center,region,graph.context.road_members[idx])
    return cache[key]

def _one(program,row,candidate,anchor,second,graph,road_cache):
    if candidate.node_id==anchor.node_id or (second is not None and candidate.node_id==second.node_id):
        return -1000.0
    xy=np.asarray(candidate.center[:2]);a=np.asarray(anchor.center[:2]);delta=xy-a
    if program=='OLD_NEAR':
        from .relation_geometry import satisfaction, RelationContext
        return satisfaction('near',anchor,candidate,RelationContext())
    if program=='OLD_BETWEEN':
        if second is None:return None
        from .relation_geometry import satisfaction, RelationContext
        return satisfaction('between',anchor,candidate,RelationContext(),second=second)
    if program=='CENTER_DISTANCE':return -float(np.linalg.norm(delta))
    if program=='FOOTPRINT_DISTANCE':return -footprint_distance(anchor,candidate)
    if program=='DISTANCE_PRIOR':return -float(np.linalg.norm(delta))
    if program=='PAIRWISE_OLD':
        if second is None:return None
        return -max(float(np.linalg.norm(xy-a)),float(np.linalg.norm(xy-np.asarray(second.center[:2]))))
    if program=='BETWEEN_TERNARY':
        if second is None or second.node_id==anchor.node_id:return None
        aa=anchor_point(anchor,candidate,graph,road_cache);bb=anchor_point(second,candidate,graph,road_cache)
        feat=between_features(xy,aa,bb)
        if feat is None:return None
        span=feat['between_anchor_separation']
        return (-feat['between_perpendicular']/max(span,5.)
                -max(0.,-feat['between_t'],feat['between_t']-1.)
                -0.25*feat['between_balance'])
    if program.startswith(('BACK_','FRONT_','LEFT_','RIGHT_')):
        direction,frame=program.split('_',1)
        road=None
        if frame=='ROAD':
            if anchor.kind is not NodeKind.ROAD_REGION:return None
            road=cached_road(candidate,anchor,graph,road_cache)
            delta=xy-road.nearest_point
        axes=frame_axes(row,anchor if anchor.kind is not NodeKind.ROAD_REGION else None,road)
        axis=axes.get(frame)
        if axis is None:return None
        if direction in ('LEFT','RIGHT'):axis=perpendicular(axis)
        sign=-1 if direction in ('BACK','RIGHT') else 1
        return float(sign*delta@axis)
    if anchor.kind is not NodeKind.ROAD_REGION:return None
    proj=cached_road(candidate,anchor,graph,road_cache)
    if program=='NEAR_ROAD_REGION':return -proj.distance
    if program=='IN_ROAD_BUFFER':return 2.0*float(proj.corridor)-min(proj.distance,20.)/20.
    if program=='ROAD_ASSOCIATION':return -np.log1p(proj.distance)-min(proj.end_distance,50.)/100.
    if program=='OFF_ROAD_BUFFER':return 2.0*float(not proj.corridor)-min(proj.distance,40.)/40.
    if program in ('ALONG_ROAD','ALONG_ROAD_TANGENT'):
        # A named road gives a corridor and a local projection; no arbitrary global direction.
        return -proj.distance/10.+min(proj.end_distance,50.)/50.
    if program=='OPPOSITE_SIDE_OF_ROAD':
        if second is None:return None
        other=cached_road(second,anchor,graph,road_cache)
        if abs(other.side)<1. or abs(proj.side)<1.:return 0.
        return -np.sign(other.side)*proj.side
    if program=='LINE_CROSSES_ROAD':
        if second is None:return None
        idx=next(i for i,r in enumerate(graph.regions) if r.node_id==anchor.node_id)
        return float(road_crosses(second.center,candidate.center,graph.context.road_members[idx]))
    return None

def score_candidates(program,row,candidates,anchors,seconds,graph,road_cache):
    """Marginalise all legal named hypotheses without looking at GT."""
    if not anchors:return None
    output=[]
    for candidate in candidates:
        values=[]
        for anchor in anchors:
            partners=seconds if program in ('PAIRWISE_OLD','BETWEEN_TERNARY','OLD_BETWEEN','OPPOSITE_SIDE_OF_ROAD','LINE_CROSSES_ROAD') else [None]
            for second in partners:
                value=_one(program,row,candidate,anchor,second,graph,road_cache)
                if value is not None and np.isfinite(value):values.append(value)
        output.append(max(values) if values else np.nan)
    arr=np.asarray(output,dtype=float)
    return arr if np.isfinite(arr).all() else None
