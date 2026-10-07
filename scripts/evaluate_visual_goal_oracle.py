#!/usr/bin/env python3
import json,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index("--output")+1]); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"status":"unavailable","oracles":["A full RGB","B target+anchor","C minimal abstract","D geometry/identity"],"reason":"requires validated query observations and belief candidate sets"},indent=2))
