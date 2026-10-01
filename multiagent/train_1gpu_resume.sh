#!/usr/bin/env bash
# Continue training run 2 (the 25 m stage-1 switch config) beyond its 2 epochs.
#
# Invocation is IDENTICAL to train_1gpu.sh -- same flags, same values -- plus
# --checkpoint/--resume_optimizer, so the extra epochs are a straight
# continuation of run 2 and not a new configuration.
#
#   run 2  epochs 0,1  ->  archived at checkpoints/run2_switch25m/
#   this   epochs 2,3  ->  resumes from that run's `latest`
#
# --epochs is the TOTAL epoch count, not "how many more": main.py resumes at
# start_epoch and loops range(start_epoch, args.epochs). run 2's `latest` was
# saved at idx 1, and save() stores epoch+1, so start_epoch = 2 and
# range(2, 4) runs exactly two more epochs.
#
# --resume_optimizer restores the AdamW moments saved in the checkpoint. Without
# it the optimizer restarts cold, which is a real (if mild) discontinuity.
#
# Output -> ../checkpoints/multi (GOAL_PREDICTOR_CHECKPOINT_DIR, hardcoded).
# NOTE: this directory is shared with any other run of this clone.
# Overridable: EPOCHS BATCH GRID SEED CUDA_VISIBLE_DEVICES CKPT
set -euo pipefail

cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-4}
BATCH=${BATCH:-8}
GRID=${GRID:-5}
SEED=${SEED:-0}
CKPT=${CKPT:-../checkpoints/run2_switch25m/latest}

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ ! -e "$CKPT" ]; then
    echo "[resume] ABORT: checkpoint not found: $CKPT" >&2
    exit 1
fi

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
      --checkpoint ${CKPT}
      --resume_optimizer
      "

echo "[train_1gpu_resume] total_epochs=${EPOCHS} batch=${BATCH} grid=${GRID} seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES}"
echo "[train_1gpu_resume] resuming from ${CKPT} with optimizer state"
python main.py $flag
