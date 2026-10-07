#!/usr/bin/env python3
import json,sys
from pathlib import Path
out=Path(sys.argv[sys.argv.index("--output")+1]); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"status":"skipped","reason":"Gate 1 did not pass; belief Top-K candidates require validated query RGB and candidate templates"},indent=2))
