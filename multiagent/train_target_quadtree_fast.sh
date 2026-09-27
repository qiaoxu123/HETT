#!/usr/bin/env bash
# Fast Stage-1 experiment: sparse quadtree target belief with frozen visual/language backbones.
set -euo pipefail

cd "$(dirname "$0")"

./train.sh \
    --training_stage target \
    --feedback teacher \
    --target_representation quadtree \
    --quadtree_depth 5 \
    --quadtree_topk 4 \
    --quadtree_hidden_dim 256 \
    --quadtree_loss_weight 1.0 \
    --target_consistency_loss_weight 0 \
    --freeze_target_backbones \
    --output_dir checkpoints/quadtree_target_fast \
    "$@"
