#!/usr/bin/env bash
set -euo pipefail

# Schedule a clean, 20-epoch heatmap run only after the named main-HETT run
# finishes all epochs and both validation splits. The expected code revision is
# supplied at launch so later edits cannot silently change the experiment.
EXPECTED_HEAD=${1:?expected heatmap git HEAD required}
MAIN_PID=1498720
MAIN_RECORD=/home/rental/20260922_1/Workspace/hett-main-system-opt/checkpoints/main_system_teacherfix_b8_adam_2ep/train.txt
HEATMAP_ROOT=/home/rental/20260922_1/Workspace/hett-heatmap-system-opt
RUN_NAME=heatmap_system_after_main_b8_adam_20e
RUN_LOG="${HEATMAP_ROOT}/logs/${RUN_NAME}.log"
RUN_CHECKPOINTS="${HEATMAP_ROOT}/checkpoints/${RUN_NAME}"

printf 'Waiting for main-HETT PID %s; heatmap HEAD %s\n' "$MAIN_PID" "$EXPECTED_HEAD"
while ps -p "$MAIN_PID" -o args= | grep -Fq -- '--checkpoint /home/rental/20260922_1/Workspace/hett-main-system-opt/checkpoints/main_system_teacherfix_b8_adam_2ep/epoch2_before_continuation'; do
  sleep 120
done

if ! grep -q '^EPOCH_TIMING epoch=19 ' "$MAIN_RECORD" \
  || ! grep -q '^VALIDATION_TIMING epoch=19 split=val_seen ' "$MAIN_RECORD" \
  || ! grep -q '^VALIDATION_TIMING epoch=19 split=val_unseen ' "$MAIN_RECORD" \
  || ! grep -q '^epoch 19$' "$MAIN_RECORD"; then
  printf 'Main-HETT exited without a complete epoch 19 and both validations; heatmap run NOT started.\n' >&2
  exit 1
fi

if [[ $(git -C "$HEATMAP_ROOT" rev-parse HEAD) != "$EXPECTED_HEAD" ]]; then
  printf 'Heatmap code HEAD changed; heatmap run NOT started.\n' >&2
  exit 1
fi
if [[ $(git -C "$HEATMAP_ROOT" branch --show-current) != experiment/heatmap-system-optimization-20e ]]; then
  printf 'Heatmap branch changed; heatmap run NOT started.\n' >&2
  exit 1
fi
if ! git -C "$HEATMAP_ROOT" diff --quiet HEAD --; then
  printf 'Heatmap tracked files changed; heatmap run NOT started.\n' >&2
  exit 1
fi
if [[ -e "$RUN_LOG" || -e "$RUN_CHECKPOINTS" ]]; then
  printf 'Heatmap output already exists; refusing to overwrite: %s\n' "$RUN_NAME" >&2
  exit 1
fi

export PATH="/home/20260922_1/miniconda3/envs/AirVLN39/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME EPOCHS=20 BATCH=8 GRID=5 SEED=0
printf 'Main-HETT complete. Starting heatmap run %s at %s\n' "$RUN_NAME" "$(date -Is)"
if bash "${HEATMAP_ROOT}/multiagent/train_1gpu.sh" --optim adam >"$RUN_LOG" 2>&1; then
  printf 'Heatmap training process exited successfully at %s\n' "$(date -Is)"
else
  status=$?
  printf 'Heatmap training failed with exit status %s; inspect %s\n' "$status" "$RUN_LOG" >&2
  exit "$status"
fi

if ! grep -q '^EPOCH_TIMING epoch=19 ' "${RUN_CHECKPOINTS}/train.txt" \
  || ! grep -q '^VALIDATION_TIMING epoch=19 split=val_unseen ' "${RUN_CHECKPOINTS}/train.txt"; then
  printf 'Heatmap process exited but full 20-epoch record is missing.\n' >&2
  exit 1
fi
printf 'Verified heatmap 20 epochs and final validation.\n'
