#!/usr/bin/env bash
# Single-GPU adaptation of the upstream train.sh (which targets 4-GPU DDP).
#
# Upstream train.sh spawns `torch.distributed.run --nproc_per_node=4`; this box
# has one GPU. main.py already skips init_distributed when world_size == 1, so
# world_size=1 + a plain `python main.py` is sufficient -- no code change needed.
#
# Output is written to GOAL_PREDICTOR_CHECKPOINT_DIR = ../checkpoints/multi
# (see multiagent/defaultpaths.py), i.e. inside this clone. Nothing is shared
# with hett-crotonyl.
#
# Overridable: EPOCHS BATCH GRID SEED CUDA_VISIBLE_DEVICES
set -euo pipefail

cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-2}
GRID=${GRID:-5}
SEED=${SEED:-0}

# bert-base-uncased and its tokenizer are already in the local HF cache; keep
# the run offline so it cannot stall on a hub round-trip.
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

flag="--world_size 1
      --seed ${SEED}
      --feedback student
      --mode train
      --altitude 50
      --learning_rate 1e-4
      --batch_size ${BATCH}
      --train_trajectory_type mturk
      --log_every 1
      --eval_every 1
      --epochs ${EPOCHS}
      --save_every 1
      --log_dir log
      --move_iteration 10
      --max_action_len 20
      --grid_size ${GRID}
      "

echo "[train_1gpu] epochs=${EPOCHS} batch=${BATCH} grid=${GRID} seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES}"
python main.py $flag
