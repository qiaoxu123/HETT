#!/usr/bin/env bash
# Stage 1: deployment-input target localisation only.
set -euo pipefail

cd "$(dirname "$0")"

./train.sh \
    --training_stage target \
    --feedback teacher \
    --output_dir checkpoints/two_stage_target \
    "$@"
