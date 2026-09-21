# Landmark-aware teacher trajectory cleanup

- Separate worktree and branch; no source, run, checkpoint, or dataset in the original worktree is modified.
- Opt-in only: `--teacher_trajectory_mode landmark_clean`; default remains `original`.
- Cleanup is applied only to `train_seen`. Validation keeps the original reference trajectories and metrics.
- `test_unseen` requires explicit `--include_test_unseen` and is excluded from development runs.
- Stable arrival means two consecutive human path samples within 20 m of a supplied landmark contour.
- Eligible routes are split at stable arrival. Coarse navigation is simplified to at most 10 moves; local exploration keeps up to 10 spatially distributed observations and their interpolated human yaw.
- Repeated points and spatial loops are removed. The original endpoint is preserved exactly.
- Routes without a stable arrival keep the original teacher representation.

Primary implementation check: full offline data audit, unit checks, then a two-episode train/eval smoke. No full-result claim is made from the smoke run.

