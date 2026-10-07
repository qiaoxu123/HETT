#!/usr/bin/env python3
"""Run Full RGB retrieval only after a calibrated real-trajectory query manifest exists."""
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from multiagent.visual_goal.query_manifest import load_verified_queries
from multiagent.visual_goal.encoder import HFImageEncoder
from multiagent.visual_goal.retrieval import cosine_matrix
from multiagent.visual_goal.metrics import retrieval_metrics

p=argparse.ArgumentParser(); p.add_argument("--query-manifest",type=Path,required=True); p.add_argument("--query-root",type=Path,required=True)
p.add_argument("--templates",type=Path,required=True); p.add_argument("--template-root",type=Path,required=True)
p.add_argument("--encoder",required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
queries=load_verified_queries(a.query_manifest,a.query_root)
templates=[json.loads(x) for x in a.templates.read_text().splitlines() if x.strip()]
qimg=[cv2.cvtColor(cv2.imread(str(a.query_root/r["image"])),cv2.COLOR_BGR2RGB) for r in queries]
timg=[cv2.cvtColor(cv2.imread(str(a.template_root/r["image"])),cv2.COLOR_BGR2RGB) for r in templates]
enc=HFImageEncoder(a.encoder); q=enc.encode(qimg); t=enc.encode(timg); sim=cosine_matrix(q,t)
for size in (40,80,120):
    cols=[j for j,r in enumerate(templates) if r["scene_size_m"]==size]
    index={templates[j]["episode_id"]:k for k,j in enumerate(cols)}
    selected=[i for i,r in enumerate(queries) if r["episode_id"] in index]
    s=sim[np.ix_(selected,cols)]; pos=np.asarray([index[queries[i]["episode_id"]] for i in selected])
    same=np.asarray([[templates[cols[j]]["map_name"]==queries[i]["map_name"] for j in range(len(cols))] for i in selected])
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/f"full_rgb_{size}m.json").write_text(json.dumps(retrieval_metrics(s,pos,same),indent=2))
print(json.dumps(enc.provenance))
