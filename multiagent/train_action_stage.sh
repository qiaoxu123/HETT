#!/usr/bin/env bash
# Stage 2: frozen target predictor plus target-conditioned action fine-tuning.
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
    --checkpoint "$target_checkpoint" \
    --reset_epoch_on_load \
    --output_dir checkpoints/two_stage_action \
    "$@"
