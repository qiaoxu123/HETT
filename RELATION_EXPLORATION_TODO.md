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
- [ ] Step 4: Compare direct, fixed-exploration, dynamic-exploration, and
  best-of-eight Oracle rule trajectories under the frozen 20-action budget.
- [ ] Step 5: Condition endpoint-residual Diffusion V2 on exploration features
  and require SR/SPL non-regression against frozen V1.
- [ ] Step 6: Add `landmark_clean` human lateral-offset supervision and test
  whether it improves exploration without increasing redundant path length.
- [ ] Step 7: Add visibility-conditioned RGB belief updates and learned stop.
- [ ] Step 8: Freeze all weights and thresholds, run `val_unseen` once, and
  leave `test_unseen` untouched until the final paper evaluation.

Current state: **Stopped after Step 1 as requested.  Step 2 has not started.**
