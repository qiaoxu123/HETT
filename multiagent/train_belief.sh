#!/usr/bin/env bash
# Multi-landmark target-belief training.  The released train.sh remains unchanged.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python}"

"$PYTHON_BIN" main.py \
    --mode train \
    --world_size 1 \
    --seed 0 \
    --feedback student \
    --altitude 50 \
    --learning_rate 1e-4 \
    --batch_size 2 \
    --grad_accum 4 \
    --optim adamW \
    --train_trajectory_type mturk \
    --epochs 20 \
    --log_every 1 \
    --eval_every 5 \
    --save_every 1 \
    --log_dir log_multilandmark_belief \
    --move_iteration 10 \
    --max_action_len 20 \
    --grid_size 5 \
    --target_belief_head \
    --belief_grid_size 41 \
    --belief_radius_m 20 \
    --max_landmarks 9 \
    --normalize_rollout_loss
