#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.visual_goal.dataset import build_dataset

p=argparse.ArgumentParser(description="Build north-up 40/80/120m CityNav goal templates")
p.add_argument("--data-root",type=Path,default=Path("data")); p.add_argument("--output",type=Path,required=True)
p.add_argument("--splits",nargs="+",default=["val_seen","val_unseen"]); p.add_argument("--max-per-split",type=int)
a=p.parse_args(); rows=build_dataset(a.data_root.resolve(),a.output.resolve(),tuple(a.splits),a.max_per_split)
print(f"built {len(rows)} top-down templates; no oblique/FPV views generated")
