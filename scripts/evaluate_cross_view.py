#!/usr/bin/env python3
"""Report cross-view availability; no viewpoint warp is used to synthesize FPV."""
import json,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index("--output")+1]); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"status":"unavailable","template_view":"north-up orthophoto","query_views":["high-altitude top-down","low-altitude top-down","oblique","FPV"],"reason":"dataset does not include calibrated real UAV RGB; 2.5-D warp is not accepted as FPV"},indent=2))
