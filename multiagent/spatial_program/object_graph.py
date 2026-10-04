"""Lightweight cached spatial database; no learned graph model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ObjectNode:
    id:int;name:str;object_type:str;centroid:tuple;contour:tuple;area:float

class ObjectGraph:
    def __init__(self,map_name,objects):
        self.map_name=map_name;self.nodes=[]
        for o in objects:
            self.nodes.append(ObjectNode(int(o.id),o.name,o.object_type,(float(o.position.x),float(o.position.y)),tuple((float(p.x),float(p.y)) for p in o.contour),float(o.area)))
    def nearby(self,xy,radius=30.,compatible=None):
        xy=np.asarray(xy,float);found=[]
        for n in self.nodes:
            if compatible and compatible not in ("region","target") and compatible not in (n.object_type+" "+n.name).casefold():continue
            d=float(np.linalg.norm(np.asarray(n.centroid)-xy))
            if d<=radius:found.append((n,d))
        return sorted(found,key=lambda x:x[1])

_CACHE={}
def get_object_graph(map_name,objects):
    if map_name not in _CACHE:_CACHE[map_name]=ObjectGraph(map_name,objects)
    return _CACHE[map_name]
