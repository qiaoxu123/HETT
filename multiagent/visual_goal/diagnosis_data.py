"""Immutable prior-run dataset adapter and content-addressed experiment inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from multiagent.visual_goal.template_builder import opaque_key

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT/'artifacts/visual_goal_abstraction/dataset_v2'
OUT = ROOT/'artifacts/visual_overlap_geometry_diagnosis'
ALLOWED = ('train_seen','val_seen','val_unseen')


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def load_split(split):
    if split not in ALLOWED:
        raise ValueError('split prohibited')
    q = read_jsonl(DATASET/f'queries_{split}.jsonl')
    rows = [r for r in read_jsonl(DATASET/f'templates_{split}.jsonl') if r['extent_m']==40]
    # L0 RGB is identical across anchor contexts. Prior metrics take max across
    # those duplicates then sort np.unique(scene_key). Preserve that order.
    unique = {}
    for r in rows:
        if r['scene_key'] in unique:
            assert r['image_path']==unique[r['scene_key']]['image_path']
        unique[r['scene_key']] = r
    c = [unique[k] for k in sorted(unique)]
    return q,c


def recover_poses(split, queries):
    if split not in ALLOWED:
        raise ValueError('split prohibited')
    trajectories = json.loads((ROOT/f'data/processed_citynav/citynav_{split}.json').read_text())
    episodes = {}
    for ix,tr in enumerate(trajectories):
        m=f"{tr['area']}_block_{tr['block']}"
        key=opaque_key(split,m,int(tr['object_ids'][0]),int(tr['ann_ids'][0]),ix)
        episodes[key]=tr
    poses=[]
    for q in queries:
        tr=episodes[q['episode_key']]
        p=tr['trajectory'][q['trajectory_index']]
        xy=np.asarray(p[:2],float)
        # Offline verification only. Target never goes to visual methods.
        assert abs(np.linalg.norm(xy-np.asarray(q['target_xy']))-q['distance_to_goal_m'])<1e-5
        poses.append(xy)
    return np.asarray(poses)


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
