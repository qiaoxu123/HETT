# Human vs straight teacher: seed-0 paired screen

Only this worktree is modified. Parent: corrected teacher-index baseline commit 507355b; both arms retain that same fix. Parent weights: teacher_fix_full_s0/checkpoints/latest, completed epoch 7. Each arm independently reloads these weights with a fresh optimizer, then trains through epoch 8 (one additional epoch).

- Human arm: original human-derived teacher path.
- Straight arm: exact planar start-to-target path with <=5m waypoint spacing, unchanged altitude and original initial pose. Subsequent waypoint yaw faces the target. Existing teacher view-direction augmentation, stride-index sampling, stage transition, losses and student rollout remain unchanged.
- This tests replacement of the teacher demonstration including its heading/waypoint schedule, not an isolated geometric-length effect.
- Only train_seen prepares a separate teacher path; only the teacher action branch consumes it. Original episodes, student reference trajectories, validation paths and evaluation SPL reference are retained.
- 512 raw trajectories per split, deterministic seed-0 per-map shuffled round-robin selection before the unchanged episode eligibility filter. Both arms use identical subsets. No test_unseen, extra seeds or full training are launched.
- Seed0, batch2, grad_accum4, 20 steps per rollout, move_iteration10, same losses and learning rate. Equal episodes/updates, not equal traveled distance or wall-clock time.
- Primary comparison: same-task val_unseen SR within20m. Also report NE, SPL, gained/lost successes and object-cluster bootstrap uncertainty. Bootstrap is not additional training seeds. A >=3pp increase is only a preliminary practical signal, not evidence of statistical significance or final from-scratch performance.
- Oracle target is permitted only to construct the training teacher and evaluation labels, never to select student evaluation actions. Straight flight ignores unmodeled real-world obstacles and does not establish safe physical navigation.

The systemd service hett-straight-teacher-screen-s0-20260915 serially runs both supervised train/eval jobs with the shared GPU lock. Each arm archives source snapshot, command lines, input hashes, logs, GPU telemetry, checkpoints and per-task predictions. Final report: runs/paired_screen_s0/comparison.json and comparison.png. Failure stops the queue and is recorded in status.json.

Checks: four tests cover path endpoints/spacing/altitude, zero distance, deterministic balanced sampling, training-only activation and preservation of original reference trajectories. Python compilation and git diff whitespace checks passed.
