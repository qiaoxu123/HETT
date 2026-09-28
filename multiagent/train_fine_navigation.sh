#!/usr/bin/env bash
# Near-goal fine navigation: balanced one-state RGB-language supervision.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 COARSE_CHECKPOINT [extra train.sh arguments...]" >&2
    exit 2
fi

coarse_checkpoint="$1"
shift
cd "$(dirname "$0")"

./train.sh \
    --training_stage fine \
    --feedback teacher \
    --target_representation candidates \
    --candidate_grid_size 8 \
    --candidate_topk 4 \
    --action_controller fine_waypoint \
    --use_stop_head \
    --stop_distance_m 20 \
    --stop_threshold 0.5 \
    --fine_random_start \
    --fine_start_min_m 20 \
    --fine_start_max_m 40 \
    --fine_random_yaw \
    --fine_waypoint_m 20 \
    --fine_distance_loss_weight 1.0 \
    --fine_stop_pos_weight 1.0 \
    --fine_one_state_supervision \
    --fine_stop_positive_fraction 0.30 \
    --fine_positive_min_m 5 \
    --direction_loss_weight 1.0 \
    --progress_loss_weight 1.0 \
    --checkpoint "$coarse_checkpoint" \
    --reset_epoch_on_load \
    --output_dir checkpoints/fine_navigation \
    "$@"
