#!/usr/bin/env bash
# 单卡评测（官方协议：feedback student / move_iteration 10 / max_action_len 20 / grid 5）
# 默认 batch_size 2，与发布代码评测配置一致。
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")"

"${PYTHON:-python}" main.py \
    --mode eval \
    --checkpoint checkpoints/multi/best_val_unseen \
    --feedback student \
    --batch_size 2 \
    --grid_size 5 \
    --move_iteration 10 \
    --max_action_len 20 \
    --world_size 1 \
    --altitude 50 \
    --seed 0 \
    --log_dir log \
    "$@"
