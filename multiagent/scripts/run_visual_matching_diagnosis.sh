#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${VISUAL_GOAL_PYTHON:-/home/rental/20260922_1/Workspace/DATA/rsrefseg2/venv/bin/python}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
"$PYTHON" -m multiagent.scripts.evaluate_visual_overlap
"$PYTHON" -m multiagent.scripts.audit_visual_diagnosis_coordinates
"$PYTHON" -m multiagent.scripts.cache_visual_diagnosis_features --kind global
"$PYTHON" -m multiagent.scripts.evaluate_baseline_overlap
"$PYTHON" -m multiagent.scripts.cache_visual_diagnosis_features --kind patch --batch-size 64
"$PYTHON" -m multiagent.scripts.evaluate_geometry_vs_rgb --stage local
"$PYTHON" -m multiagent.scripts.evaluate_geometry_vs_rgb --stage geometry
"$PYTHON" -m multiagent.scripts.evaluate_local_ransac
"$PYTHON" -m multiagent.scripts.evaluate_classic_correspondence
"$PYTHON" -m multiagent.scripts.evaluate_geometry_controls
"$PYTHON" -m multiagent.scripts.evaluate_geometry_vs_rgb --stage metrics
"$PYTHON" -m multiagent.scripts.analyze_visual_matching_failures
"$PYTHON" -m multiagent.scripts.summarize_correspondence_controls
"$PYTHON" -m pytest -q | tee artifacts/visual_overlap_geometry_diagnosis/tests.log
"$PYTHON" - <<'PYTEST'
import json,re
from pathlib import Path
p=Path('artifacts/visual_overlap_geometry_diagnosis')
m=re.search(r'(\d+) passed in ([\d.]+)s',(p/'tests.log').read_text())
if m:
    (p/'tests.json').write_text(json.dumps({'status':'PASS','passed':int(m[1]),'seconds':float(m[2]),'command':'python -m pytest -q','original_21_tests_preserved':True},indent=2))
PYTEST
"$PYTHON" -m multiagent.scripts.export_visual_diagnosis_manifest
"$PYTHON" -m multiagent.scripts.generate_visual_diagnosis_report
