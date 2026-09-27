#!/usr/bin/env bash
# Stage-2 experiment: frozen candidate predictor + top-K target-conditioned control.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 TARGET_CHECKPOINT [extra train.sh arguments...]" >&2
    exit 2
fi

target_checkpoint="$1"
shift
cd "$(dirname "$0")"

./train.sh \
    --training_stage action \
    --feedback student \
    --target_representation candidates \
    --candidate_grid_size 8 \
    --candidate_topk 4 \
    --action_controller residual \
    --use_stop_head \
    --stop_distance_m 15 \
    --stop_threshold 0.8 \
    --progress_loss_weight 1.0 \
    --checkpoint "$target_checkpoint" \
    --reset_epoch_on_load \
    --output_dir checkpoints/candidate_action \
    "$@"
