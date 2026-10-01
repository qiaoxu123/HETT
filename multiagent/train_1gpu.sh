#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-8}
GRID=${GRID:-5}
SEED=${SEED:-0}
RUN_NAME=${RUN_NAME:-performance_test}
REPO_ROOT=$(cd .. && pwd)

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HETT_CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/${RUN_NAME}"
mkdir -p "${HETT_CHECKPOINT_DIR}"

exec python main.py \
  --world_size 1 \
  --seed "${SEED}" \
  --feedback student \
  --mode train \
  --altitude 50 \
  --learning_rate 1e-4 \
  --batch_size "${BATCH}" \
  --train_trajectory_type mturk \
  --log_every 1 \
  --eval_every 1 \
  --epochs "${EPOCHS}" \
  --save_every 1 \
  --log_dir "log/${RUN_NAME}" \
  --move_iteration 10 \
  --max_action_len 20 \
  --grid_size "${GRID}" \
  "$@"
