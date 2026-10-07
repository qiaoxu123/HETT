#!/usr/bin/env python3
"""Information-factor ablation protocol; consumes image/region masks from the dataset manifest."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.visual_goal.abstraction import information_table
p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({"status":"requires Gate 1 success","factors":["color","texture","geometry","context","anchor"],"levels":information_table()},indent=2))
