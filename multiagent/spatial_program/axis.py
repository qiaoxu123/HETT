"""Resolve an explicit, auditable coordinate frame for each relation."""
from __future__ import annotations
import math
import numpy as np
from .map_axis import local_road_axis,principal_axis

def unit(x):
    x=np.asarray(x,float);n=np.linalg.norm(x);return x/n if n>1e-6 else np.asarray([1.,0.])

def resolve_axis(program,anchor_records,start_xy,start_yaw,map_objects=()):
    by_entity={}
    refs=[e for e in program.entities if e.role=="reference"]
    for entity,record in zip(refs,anchor_records):by_entity[entity.id]=record
    primary=next((by_entity.get(a.anchor) for a in program.axes if a.anchor),None)
    if primary:
        center=np.asarray(primary["centroid"],float);travel=unit(center-np.asarray(start_xy,float))
        candidates=[{"type":"start_to_anchor","vector":travel.tolist(),"origin":center.tolist(),"confidence":.95}]
        contour=primary.get("contour",[]);ba,bc=principal_axis(contour)
        candidates.append({"type":"building_axis","vector":ba.tolist(),"origin":center.tolist(),"confidence":.45*bc})
        ra,rc=local_road_axis(map_objects,center)
        candidates.append({"type":"road_axis","vector":ra.tolist(),"origin":center.tolist(),"confidence":.7*rc})
    else:candidates=[]
    if len(anchor_records)>=2:
        a,b=(np.asarray(x["centroid"],float) for x in anchor_records[:2]);candidates.append({"type":"anchor_pair_axis","vector":unit(b-a).tolist(),"origin":a.tolist(),"confidence":.85})
    candidates.append({"type":"start_heading","vector":[math.cos(start_yaw),math.sin(start_yaw)],"origin":list(map(float,start_xy)),"confidence":.65})
    # Travel axis is preferred for sequence/side/ordinal; road axis only wins when highly linear.
    best=max(candidates,key=lambda x:x["confidence"])
    return best,sorted(candidates,key=lambda x:-x["confidence"])
