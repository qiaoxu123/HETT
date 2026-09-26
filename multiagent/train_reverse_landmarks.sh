#!/usr/bin/env bash
# Reverse human Teacher + forward multi-landmark belief Student.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$(dirname "$0")"

"${PYTHON:-/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python}" main.py \
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
    --epochs 3 \
    --log_every 1 \
    --eval_every 1 \
    --save_every 1 \
    --move_iteration 10 \
    --max_action_len 20 \
    --grid_size 5 \
    --reverse_human_teacher \
    --reverse_teacher_weight 0.20 \
    --reverse_visual_align_weight 0.10 \
    --reverse_goal_direction_weight 0.10 \
    --reverse_target_views 3 \
    --reverse_visual_temperature 0.07 \
    --reverse_freeze_text_targets \
    --target_belief_head \
    --belief_grid_size 41 \
    --belief_radius_m 20 \
    --max_landmarks 9 \
    --landmark_match_min_similarity 0.70 \
    --normalize_rollout_loss \
    --progress_normalization initial_distance \
    --stage1_arrival_distance_m 5 \
    --stage2_replan_distance_m 15 \
    --progress_stop_threshold 0.95 \
    --stop_goal_distance_m 10 \
    --goal_stability_distance_m 5 \
    --goal_stability_steps 2 \
    --save_validation_predictions \
    --output_dir checkpoints/reverse_human_landmarks_3ep \
    "$@"
