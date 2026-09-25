# Relation-conditioned exploration validation TODO

Run exactly one step at a time.  Stop after every step and record whether the
gate passed before changing the next component.

- [x] Step 1: Validate target-free spatial fields for strict directional and
  near-only relations on `val_seen`, including flipped-relation controls.
  **Executed; overall gate FAILED.** `near` passed, while directional
  correct-vs-flipped consistency was 59.04% against the fixed 60% gate.
- [ ] Step 2: Add and validate soft/topological fields for `along`, `on`,
  `off`, `corner`, `edge`, `between`, `across`, and `opposite`.
- [ ] Step 3: Build the deterministic exploration-condition preprocessor and
  verify strong/soft/ambiguous calibration without training Diffusion.
- [ ] Step 4: Generate low/medium/high-exploration trajectory families and
  compare fixed exploration, uncertainty-conditioned exploration, and
  best-of-eight Oracle selection under the frozen 20-action budget.  Do not
  use a shortest path as the trajectory backbone.
- [ ] Step 5: Train exploration-conditioned full-trajectory Diffusion.  The
  exploration condition controls trajectory shape and diversity; endpoint,
  20-action budget, loop/overshoot penalties, and arrival error remain hard
  constraints.  Shortest distance is an efficiency metric/penalty only, not
  generation supervision.
- [ ] Step 6: Add cleaned `landmark_clean` trajectories grouped by exploration
  level.  Learn useful human search shapes while filtering loops, repeated
  points, and post-arrival motion; do not reduce supervision to offsets from a
  shortest-path skeleton.
- [ ] Step 7: Add visibility-conditioned RGB belief updates and learned stop.
- [ ] Step 8: Freeze all weights and thresholds, run `val_unseen` once, and
  leave `test_unseen` untouched until the final paper evaluation.

Current state: **Stopped after Step 1 as requested.  Step 2 has not started.**
