"""Sequential hard/soft execution over B0 candidates."""
from __future__ import annotations
import numpy as np
from .axis import resolve_axis
from .ordinal import ordinal_scores
from .scorer import between_score,relation_score

class ProgramExecutor:
    def __init__(self,soft=True,epsilon=.05):self.soft=soft;self.epsilon=epsilon
    def execute(self,program,points,b0,anchors,start_xy,start_yaw,map_objects=(),enabled=("axis","ordinal")):
        points=np.asarray(points,float);scores=np.asarray(b0,float).copy();details=[]
        axis,alternatives=resolve_axis(program,anchors,start_xy,start_yaw,map_objects) if "axis" in enabled else ({"type":"start_heading","vector":[np.cos(start_yaw),np.sin(start_yaw)],"origin":list(start_xy),"confidence":.4},[])
        entity_records={}
        refs=[e for e in program.entities if e.role=="reference"]
        for e,r in zip(refs,anchors):entity_records[e.id]=r
        for clause in program.clauses:
            if not clause.supported:continue
            anchor=entity_records.get(clause.object)
            if clause.relation=="ordinal" and "ordinal" in enabled:
                rel,diag=ordinal_scores(points,scores,axis["vector"],anchor["centroid"] if anchor else axis["origin"],clause.value or 1);extra=diag
            elif clause.relation=="between" and len(anchors)>=2:
                rel=between_score(points,anchors[0]["centroid"],anchors[1]["centroid"]);extra={}
            elif anchor:
                rel=relation_score(clause.relation,points,anchor["centroid"],axis["vector"]);extra={}
            else:continue
            factor=np.maximum(rel,self.epsilon) if self.soft else (rel>=.5).astype(float)
            scores*=factor;details.append({"clause":clause.id,"relation":clause.relation,"scores":rel.tolist(),**extra})
        total=scores.sum();scores=scores/total if total>0 else np.asarray(b0)/max(np.sum(b0),1e-9)
        return scores,{"axis":axis,"alternative_axes":alternatives,"steps":details}
