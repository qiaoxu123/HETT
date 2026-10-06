"""Dataset construction and leakage checks for real teacher-pose RGB.

GT reference masks are emitted as labels.  They are never returned by
``inference_inputs`` and must not be passed into a grounding model.
"""
from __future__ import annotations
import re
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
from multiagent.mapdata import GROUND_LEVEL
from multiagent.scene_grounding.dataset import trajectory_pose, world_to_image

ALLOWED_SPLITS=("train_seen","val_seen","val_unseen")
def normalize_name(value):return re.sub(r"[^a-z0-9]","",value.casefold())

def resolve_named_landmarks(names,objects):
    result=[]
    for name in names:
        key=normalize_name(name);matches=[o for o in objects if o.name and normalize_name(o.name)==key]
        if not matches:matches=[o for o in objects if o.name and (key in normalize_name(o.name) or normalize_name(o.name) in key)]
        if matches:result.append(max(matches,key=lambda o:o.area))
    seen=set();return [o for o in result if not (int(o.id) in seen or seen.add(int(o.id)))]

def projected_object(obj,pose,map_name,image_size):
    ground=GROUND_LEVEL[map_name];fov=Polygon(__import__('multiagent.space',fromlist=['view_area_corners']).view_area_corners(pose,ground));polygon=obj.contour_polygon
    if not polygon.is_valid:polygon=polygon.buffer(0)
    clipped=polygon.intersection(fov)
    if clipped.is_empty:return None
    geometry=max(getattr(clipped,"geoms",[clipped]),key=lambda x:x.area)
    if not hasattr(geometry,"exterior"):return None
    uv=world_to_image(np.asarray(geometry.exterior.coords),pose,ground,image_size);mask=np.zeros((image_size,image_size),np.uint8);cv2.fillPoly(mask,[np.rint(uv).astype(np.int32)],1)
    if not mask.any():return None
    y,x=np.where(mask);return {"mask":mask,"bbox_xyxy":[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],"visible_pixels":int(mask.sum()),"visible_fraction":float(clipped.area/max(polygon.area,1e-6)),"center_uv":[float(x.mean()),float(y.mean())]}

def build_frame_labels(row,episode,objects,image_size=256):
    oid,ann=int(episode['object_ids'][0]),int(episode['ann_ids'][0]);target=objects[oid]
    if ann>=len(target.processed_descriptions):return None
    processed=target.processed_descriptions[ann];refs=resolve_named_landmarks(processed.landmarks,objects.values());pose=trajectory_pose(episode['trajectory'][row['step']])
    positives=[];candidates=[]
    for obj in objects.values():
        if not obj.name:continue
        projected=projected_object(obj,pose,row['map_name'],image_size)
        if projected is None:continue
        record={k:v for k,v in projected.items() if k!='mask'}|{"landmark_id":int(obj.id),"name":obj.name,"category":obj.object_type,"world_xy":[float(obj.position.x),float(obj.position.y)]}
        candidates.append((record,projected['mask']))
        if any(int(obj.id)==int(x.id) for x in refs):positives.append((record,projected['mask']))
    if not positives:return None
    union=np.maximum.reduce([m for _,m in positives]);coarse=cv2.resize(union.astype(np.float32),(28,28),interpolation=cv2.INTER_AREA)
    # Candidate masks are labels/pooling geometry, never model channels.
    return {"instruction":target.descriptions[ann],"referenced_landmark_ids":[int(x.id) for x in refs],"primary_referenced_landmark_id":int(refs[0].id) if refs else None,"context_referenced_landmark_ids":[int(x.id) for x in refs[1:]],"referenced_landmark_names":[x.name for x in refs],"referenced_phrase":processed.landmarks[0] if processed.landmarks else "","target_phrase":processed.target,"gt_mask":union,"coarse_target":coarse,"positives":[r for r,_ in positives],"candidates":[r for r,_ in candidates],"candidate_masks":{str(r['landmark_id']):m for r,m in candidates}}

def inference_inputs(record):
    allowed={"sample_id","instruction","referenced_phrase","image","candidate_landmarks"}
    return {k:record[k] for k in allowed if k in record}

def leakage_audit(records):
    by={s:{"episodes":set(),"objects":set(),"samples":set()} for s in ALLOWED_SPLITS}
    forbidden=False
    for r in records:
        if r['split'] not in ALLOWED_SPLITS:forbidden=True;continue
        by[r['split']]['episodes'].add(r['episode_id']);by[r['split']]['objects'].update(f"{r['map_name']}:{x}" for x in r['referenced_landmark_ids']);by[r['split']]['samples'].add(r['sample_id'])
    overlap={}
    for i,a in enumerate(ALLOWED_SPLITS):
        for b in ALLOWED_SPLITS[i+1:]:overlap[f"{a}/{b}"]={k:len(by[a][k]&by[b][k]) for k in by[a]}
    return {"overlap":overlap,"forbidden_split":forbidden,"gt_mask_in_inference":any('gt_mask' in inference_inputs(r) for r in records),"gt_goal_in_inference":any('goal' in inference_inputs(r) for r in records)}
