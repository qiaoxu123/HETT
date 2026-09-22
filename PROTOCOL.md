# Progress stop threshold test

Question: HETT supervises `progress = clip(1 - target_distance_m / 100, 0, 1)`, while success is target distance <= 20 m. Does changing only the post-landmark stop threshold from 0.95 (5 m semantics) to 0.80 (20 m semantics) improve validation navigation?

## Fixed controls

- Checkpoint: landmark-clean Epoch 7 `best_val_unseen` (seed 0)
- Splits: `val_seen`, `val_unseen`; no `test_unseen`
- Same source, data, controller, budget, and model weights
- `--disable_task_interaction --teacher_trajectory_mode landmark_clean`

## Arms

1. Control: `--progress_stop_threshold 0.95`
2. Aligned: `--progress_stop_threshold 0.80`

Compare SR, SPL, NE, oracle SR, path length, stopping-step distribution, and paired final-distance changes. This is a single-seed diagnostic, not deployment evidence.
