#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/multiagent"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache}"
PYTHON="${PYTHON:-/home/rental/20260922_1/Workspace/DATA/rsrefseg2/venv/bin/python}"

"$PYTHON" scripts/build_visual_goal_dataset.py \
  --split train_seen val_seen val_unseen \
  --output-dir ../artifacts/visual_goal_abstraction/dataset_v2

"$PYTHON" scripts/finalize_visual_goal_dataset.py \
  --dataset-dir ../artifacts/visual_goal_abstraction/dataset_v2

"$PYTHON" scripts/audit_visual_goal_split_leakage.py \
  --dataset-dir ../artifacts/visual_goal_abstraction/dataset_v2

"$PYTHON" scripts/finetune_partial_siglip2.py \
  --dataset-dir ../artifacts/visual_goal_abstraction/dataset_v2 \
  --output ../artifacts/visual_goal_abstraction/models/siglip2_partial_trainseen.pt

"$PYTHON" scripts/evaluate_full_goal_retrieval.py \
  --dataset-dir ../artifacts/visual_goal_abstraction/dataset_v2 \
  --output-dir ../artifacts/visual_goal_abstraction/eval \
  --abstraction --ablation

"$PYTHON" scripts/generate_visual_goal_report.py \
  --dataset-dir ../artifacts/visual_goal_abstraction/dataset_v2 \
  --eval-json ../artifacts/visual_goal_abstraction/eval/full_goal_retrieval.json \
  --output ../VISUAL_GOAL_ABSTRACTION_REPORT.md
