#!/usr/bin/env bash
# 单卡训练（适配本机 1× RTX 5090 32GB）
# 原版为 4 卡 DDP（每卡 batch 2）；单卡版用 batch 2 + --grad_accum 4。
# 累积梯度取平均；不保证与 DDP 的 BatchNorm/随机数轨迹逐位等价。
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")"

"${PYTHON:-python}" main.py \
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
    --eval_every 1 \
    --save_every 1 \
    --log_dir log \
    --move_iteration 10 \
    --max_action_len 20 \
    --grid_size 5 \
    "$@"    # 追加参数放最后，可覆盖上面默认值（argparse 后出现者胜）
            # 例：断点续训 ./train.sh --checkpoint checkpoints/multi/latest --resume_optimizer
            #     冒烟测试 ./train.sh --epochs 1 --max_episodes 32 --eval_first
