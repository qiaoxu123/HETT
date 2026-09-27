#!/usr/bin/env bash
# Fast Stage-1 experiment: relation-aware sparse candidate grounding.
set -euo pipefail

cd "$(dirname "$0")"

./train.sh \
    --training_stage target \
    --feedback teacher \
    --target_representation candidates \
    --candidate_grid_size 8 \
    --candidate_topk 4 \
    --candidate_hidden_dim 256 \
    --candidate_attention_heads 4 \
    --candidate_classification_weight 1.0 \
    --candidate_offset_weight 1.0 \
    --target_consistency_loss_weight 0 \
    --target_distance_loss_weight 0 \
    --target_bearing_loss_weight 0 \
    --freeze_target_backbones \
    --output_dir checkpoints/candidate_target_fast \
    "$@"
