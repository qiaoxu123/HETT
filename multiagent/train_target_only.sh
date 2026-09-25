#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$(dirname "$0")"

"${PYTHON:-/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python}" main.py \
    --mode train \
    --world_size 1 \
    --seed 0 \
    --feedback student \
    --altitude 50 \
    --learning_rate 1e-4 \
    --target_head_learning_rate 1e-3 \
    --batch_size 2 \
    --grad_accum 4 \
    --optim adamW \
    --train_trajectory_type mturk \
    --epochs 3 \
    --eval_every 1 \
    --save_every 1 \
    --grid_size 5 \
    --target_belief_head \
    --target_only_keyframes \
    --target_keyframe_mode final \
    --target_eval_view final \
    --target_near_min_m 25 \
    --target_near_max_m 60 \
    --belief_grid_size 41 \
    --belief_radius_m 20 \
    --max_landmarks 9 \
    --landmark_boundary_points 16 \
    --normalize_rollout_loss \
    "$@"
