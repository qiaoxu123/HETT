# Relation-conditioned exploration validation TODO

Run exactly one step at a time.  Stop after every step and record whether the
gate passed before changing the next component.

- [x] Step 1: Validate target-free spatial fields for strict directional and
  near-only relations on `val_seen`, including flipped-relation controls.
  **Executed; overall gate FAILED.** `near` passed, while directional
  correct-vs-flipped consistency was 59.04% against the fixed 60% gate.
- [x] Step 2 / Online Todo 1: Build and validate an online semantic-RGB fusion sample.  At
  every time step it must combine language/landmark relation tokens, current
  RGB evidence, pose-conditioned visibility, previous belief, and remaining
  budget.  Missing or occluded visual evidence must remain unknown rather than
  become negative evidence. **PASSED on an 8 train_seen + 8 val_seen pilot;
  82 causal frames passed alignment, budget, visibility and leakage checks.**
- [ ] Step 2b / Online Todo 2: Train the minimal recurrent semantic-RGB fusion
  locator with the exact continuous target position as training supervision.
  Use a coarse belief cell plus a continuous within-cell offset so the 20x20
  map does not quantize supervision to roughly 20.5 m cells.  The 20 m radius
  is evaluation-only (`Hit@20`/SR), never the training mask or model input.
- [ ] Step 3 / Online Todo 3: Learn a continuous exploration-value head from the fused state.
  Do not assign hand-written strong/soft/ambiguous classes.  Supervise the head
  with the measured downstream advantage of exploratory trajectories over
  direct trajectories, and output both a scalar exploration value and a
  spatial exploration-value map.
- [ ] Step 4: Generate continuously conditioned trajectory families and
  compare no exploration, fixed exploration, learned exploration, and
  best-of-eight Oracle selection under the frozen 20-action budget.  Do not
  use a shortest path as the trajectory backbone or discretize exploration
  into semantic keyword classes.
- [ ] Step 5: Train online-fusion-conditioned full-trajectory Diffusion.  The
  current semantic-RGB fused state and continuous exploration value control
  trajectory shape/diversity; endpoint, 20-action budget, loop/overshoot
  penalties, and arrival error remain constraints.  Shortest distance is an
  efficiency metric/penalty only, not generation supervision.
- [ ] Step 6: Add cleaned `landmark_clean` trajectories with continuous
  exploration-value supervision.  Learn useful human search shapes while
  filtering loops, repeated points, and post-arrival motion; do not reduce
  supervision to offsets from a shortest-path skeleton.
- [ ] Step 7: Add evidence-driven learned stop on top of the same online fused
  state; do not create a separate RGB branch with inconsistent belief state.
- [ ] Step 8: Freeze all weights and thresholds, run `val_unseen` once, and
  leave `test_unseen` untouched until the final paper evaluation.

Current state: **Stopped after Online Todo 1 as requested.  The fusion model
and exploration-value head have not started.**
