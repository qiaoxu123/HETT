#!/usr/bin/env bash
# Landmark Arrival Understanding: RGB + language + depth + coarse Top-K evidence.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 COARSE_CHECKPOINT [extra train.sh arguments...]" >&2
    exit 2
fi

coarse_checkpoint="$1"
shift
cd "$(dirname "$0")"

./train.sh \
    --training_stage arrival \
    --feedback teacher \
    --target_representation candidates \
    --candidate_grid_size 8 \
    --candidate_topk 4 \
    --arrival_near_m 20 \
    --arrival_positive_min_m 5 \
    --arrival_far_min_m 40 \
    --arrival_far_max_m 100 \
    --arrival_positive_fraction 0.34 \
    --arrival_far_fraction 0.33 \
    --arrival_match_loss_weight 1.0 \
    --arrival_near_loss_weight 1.0 \
    --arrival_loss_weight 1.0 \
    --arrival_depth_size 64 \
    --checkpoint "$coarse_checkpoint" \
    --reset_epoch_on_load \
    --output_dir checkpoints/landmark_arrival \
    "$@"
