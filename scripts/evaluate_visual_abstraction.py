#!/usr/bin/env python3
"""Evaluate L0-L9 against fixed real query RGB after its provenance gate passes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.visual_goal.query_manifest import load_verified_queries
from multiagent.visual_goal.encoder import HFImageEncoder
from multiagent.visual_goal.abstraction import abstract
from multiagent.visual_goal.retrieval import cosine_matrix
from multiagent.visual_goal.metrics import retrieval_metrics
import cv2
import numpy as np

p=argparse.ArgumentParser(); p.add_argument("--query-manifest",type=Path,required=True); p.add_argument("--query-root",type=Path,required=True)
p.add_argument("--templates",type=Path,required=True); p.add_argument("--template-root",type=Path,required=True)
p.add_argument("--encoder",required=True); p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
try:
    rows=load_verified_queries(a.query_manifest,a.query_root)
    templates=[json.loads(x) for x in a.templates.read_text().splitlines() if x.strip()]
    encoder=HFImageEncoder(a.encoder)
    query=[cv2.cvtColor(cv2.imread(str(a.query_root/r["image"])),cv2.COLOR_BGR2RGB) for r in rows]
    output={"gate":"REAL_QUERY_PROVENANCE_VALID","encoder":encoder.provenance,"metrics":{}}
    for metres in (40,80,120):
        ts=[r for r in templates if r["scene_size_m"]==metres]
        positive={r["episode_id"]:i for i,r in enumerate(ts)}
        keep=[i for i,r in enumerate(rows) if r["episode_id"] in positive]
        qrows=[rows[i] for i in keep]
        if not qrows or not ts: continue
        pos=np.asarray([positive[r["episode_id"]] for r in qrows])
        same=np.asarray([[t["map_name"]==r["map_name"] for t in ts] for r in qrows])
        for level in (f"L{i}" for i in range(10)):
            views=[]
            for template in ts:
                image=cv2.cvtColor(cv2.imread(str(a.template_root/template["image"])),cv2.COLOR_BGR2RGB)
                with np.load(a.template_root/template["masks"]) as masks:
                    regions={k:masks[k] for k in masks.files}
                views.append(abstract(image,level,regions))
            qfeat=encoder.encode(query)
            tfeat=encoder.encode(views)
            metrics=retrieval_metrics(cosine_matrix(qfeat,tfeat)[keep],pos,same)
            output["metrics"][f"{metres}m/{level}"]=metrics
    result=output
except Exception as exc:
    result={"gate":"BLOCKED_INVALID_OR_UNAVAILABLE_QUERY_RGB","reason":str(exc),"full_rgb_metrics":None,"abstraction_metrics":None,"action":"stop after Gate 1; do not train/generate images"}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
