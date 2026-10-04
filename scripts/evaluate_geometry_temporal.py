#!/usr/bin/env python3
"""Temporal results are produced by evaluate_geometry_reasoner.py; extract them."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);v=json.loads(a.metrics.read_text());(a.output/'metrics.json').write_text(json.dumps({'temporal':v['temporal'],'test_unseen_read':False,'future_pose_used':False},indent=2)+'\n')
