"""Explainable geometry-only candidate scorer."""
from __future__ import annotations
import numpy as np
from .relations import between, directional, far, near, reference_frame, sequence
from .ordering import candidate_order, ordinal_scores

class GeometryReasoner:
    def __init__(self, sigma_near=30.0, sigma_direction=20.0): self.sigma_near=sigma_near; self.sigma_direction=sigma_direction
    def score(self, program, anchors, candidates_xy, pose_xy, yaw, relation_filter=None, include_anchor_distance=False):
        resolved=[match for group in anchors for match in group["matches"]]
        if not resolved: return [{"total":0.0,"components":{},"valid":False,"reference_frame":"none"} for _ in candidates_xy]
        anchor=np.asarray(resolved[0]["centroid"]); heading,frame=reference_frame(anchor,pose_xy,yaw,resolved[0]["orientation"])
        ranks, projections=candidate_order(candidates_xy,anchor,heading)
        components=[{} for _ in candidates_xy]
        if include_anchor_distance:
            for i,xy in enumerate(candidates_xy): components[i]["anchor_distance"]=near(xy,anchor,self.sigma_near)
        for constraint in program.constraints:
            kind=constraint["type"]
            if not constraint.get("supported",False) or (relation_filter and kind not in relation_filter): continue
            if kind in ("left_of","right_of","front_of","behind"):
                values=[directional(xy,anchor,heading,kind,self.sigma_direction) for xy in candidates_xy]
            elif kind in ("near","nearest"): values=[near(xy,anchor,self.sigma_near) for xy in candidates_xy]
            elif kind=="farthest": values=[far(xy,anchor) for xy in candidates_xy]
            elif kind in ("before","after","past"): values=[sequence(xy,anchor,heading,kind,self.sigma_direction) for xy in candidates_xy]
            elif kind=="ordinal": values=ordinal_scores(ranks,constraint["value"])
            elif kind=="between" and len(resolved)>=2: values=[between(xy,resolved[0]["centroid"],resolved[1]["centroid"]) for xy in candidates_xy]
            else: continue
            for item,value in zip(components,values): item[f"{kind}"]=value
        return [{"total":float(np.mean(list(item.values()))) if item else 0.0,"components":item,"valid":bool(item),
                 "reference_frame":frame,"candidate_order":ranks[i],"ordering_projection":projections[i]} for i,item in enumerate(components)]
