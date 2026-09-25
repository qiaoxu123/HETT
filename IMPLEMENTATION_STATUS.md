# Implementation status

## 2026-09-23: fine-stage-only altitude refinement

- Replaced the intended full-trajectory altitude intervention with an
  evaluation-only fine-refinement feature flag.
- Coarse navigation remains at canonical 50 m; only RGB camera height may use
  40/30 m probes.
- Canonical map coverage, low-altitude global-memory exclusion, frozen coarse
  goals, containment gating, recovery ascent and altitude diagnostics are
  implemented.
- Development mode excludes `test_unseen` while the feature flag is enabled.
- The implementation and exact ablation order are recorded in
  `FINE_ALTITUDE_REFINEMENT.md`.
- No GPU evaluation has been launched; the prior baseline run was stopped at
  the user's request.

## 2026-09-23: Gate 0 pure-logic validation

- The complete local test suite passes: 20 tests.
- Added runtime composition and per-episode state-isolation coverage.
- Added auxiliary-head/loss forward and backward coverage.
- Added randomized controller safety-invariant coverage.
- The exact environment, scope and remaining limitations are recorded in
  `GATE0_VALIDATION.md`.
- Passing Gate 0 does not establish a navigation improvement; Gate 1 oracle
  ceiling measurements are still required before learned-policy integration.

## 2026-09-23: paired visual validation

- Added `scripts/build_altitude_view_benchmark.py` for same-pose multi-height
  crops, exact clipped footprint masks and digital-zoom controls.
- Added strict common-visibility semantic annotations so field-of-view changes
  cannot be confused with recognition improvement.
- Added an isolated RSRefSeg2 environment, evaluation config, per-sample metric
  and single-GPU supervisor.  Official/model assets are external to the repo.
- Pilot protocol and first geometry result are recorded in
  `ALTITUDE_VIEW_PROTOCOL.md`.
- The formal 128-episode-per-split geometry result and semantic environment
  validation are recorded in `ALTITUDE_VIEW_RESULTS.md`.
- The active-altitude policy remains unwired to HETT rollout until the frozen
  semantic comparison passes.

## 2026-09-24: target grounding and fine-view semantic checkpoint

- The accepted untouched-split Top-5 scorer remains frozen relation/type
  fusion at 44.15% `test_unseen` Hit@20. Top-5 oracle is 74.23%.
- Post-test exploratory full-instruction/RGB appearance fusion reaches 48.49%;
  it is not promoted to a blind benchmark because its hypothesis followed
  test-error inspection.
- Rejected ranking variants: 40/20 m embedding fusion (no map-holdout gain),
  nonlinear score calibration (overfit), and frozen-feature semantic-distance
  regression (about 19.2 m component-distance MAE).
- An annotation-only semantic-component diagnostic reaches 52.41% without
  target identity or target coordinates. It is an input upper bound, not a
  deployable RGB result.
- A lightweight RGB U-Net trained only on `train_seen` maps reaches 52.77%
  foreground mIoU on unseen maps at a 40 m view, confirming that lower altitude
  improves dense semantic perception. Its candidate fusion reaches only 48.89%.
- A center heatmap is valid for compact cars (64.47% F1 and about 0.81 m
  localization error), but polygon centers are not a stable local target for
  buildings, ground, or parking. The class-specific hybrid reaches 48.99% and
  is also rejected as a ranking upgrade.
- The current bottleneck is binding a particular object instance to each named
  landmark relation, not recognizing the coarse target class.
- Directly snapping grid candidates to the nearest annotated instance reduced
  the ideal-component fusion on both development and test, confirming a
  representation mismatch with the grid-trained relation model. Retraining the
  relation model on snapped instance coordinates produced 37.82% relation-only
  `test_unseen` Hit@20 and 52.60% ideal-component fusion. The latter is +0.89
  over the matched old-relation fusion, but `val_unseen` moved in the opposite
  direction (34.52% to 33.89%), so it is not accepted as a stable improvement.
- An interpretable cue-by-cue relation scorer was trained to make left/right,
  above/below, near, between, across, and front/behind geometry explicit. Its
  controlled removals confirm that it genuinely uses those cues, but its
  standalone `test_unseen` Hit@20 is only 35.85%. Adding it to the exploratory
  relation/type/appearance scorer with weights selected only on `val_unseen`
  changes Hit@20 from 48.15% to 48.32% (+0.17 point). This branch is retained
  only as a diagnostic, not promoted as the main scorer.
- A shape-aware named-landmark-mask ranker then retained each landmark's full
  oriented extent instead of only its centroid and nearest boundary point.
  Relation-only `val_unseen` Hit@20 is 23.77%, slightly below the accepted
  24.04% relation model. As a complementary branch it changes the exploratory
  full fusion from 32.37% to 32.41% on `val_unseen`, and from 48.13% to 48.70%
  on `test_unseen`. The development gain is too small to accept the larger test
  change as stable; this representation is retained for instance-binding
  diagnostics only.
- An ideal candidate-object-instance diagnostic bound every Top-5 point to the
  nearest annotated component of the language-predicted target class, then
  scored that component's center and boundary against each named landmark.
  It reached 24.13% relation-only `val_unseen` Hit@20 versus 24.04% for the
  accepted relation model. Wrong relations and wrong landmarks reduced it to
  23.38% and 23.51%, confirming real binding, but the +0.09 point result is
  negligible. On unseen maps the hard association moves candidates by 20.52 m
  on average, so nearest-instance snapping is too discontinuous; future work
  should use soft evidence over multiple candidate instances instead.
- A follow-up soft-set ranker retained the two nearest annotated instances of
  every semantic class and weighted them by text-class probability and spatial
  distance before relation scoring. Across three seeds it collapsed exactly to
  candidate rank 1 on `val_unseen` (20.76% Hit@20); wrong relation, wrong
  landmark, and no-relation controls produced the same ranking. The nearest
  component still dominates all candidate sets, so this formulation is
  rejected. Future instance evidence must be candidate-specific RGB/semantic
  likelihood, not map-wide proximity alone.
- Full HETT baseline training remains stopped by explicit user request.

## Present in this worktree

- Pure, per-episode active-altitude controller.
- Four-level discrete altitude and phase definitions.
- Multi-scale target-belief fusion.
- Privileged teacher target builder with soft human-height supervision.
- Independent temporally confirmed stop rule.
- Optional confidence-gated semantic-prior interface suitable for frozen
  RSRefSeg2 outputs.
- Bounded quantized observation LRU design.
- Auxiliary PyTorch heads and masked loss definitions.
- Unit-test specifications for the above components.

## Deliberately not done

- No changes to the released HETT rollout, environment or checkpoint format.
- No RSRefSeg2 model dependency or weights copied into this repository.
- No HETT training has been launched from this worktree.
- Frozen RSRefSeg2 semantic inference is pending external asset download and
  release of the shared GPU; only the CPU geometry/image pilot is complete.
- No runtime-improvement claim; navigation claims are limited to the recorded
  frozen full-split comparison below.

## 2026-09-25: target-free fixed-height candidate observations

- Added `scripts/evaluate_target_free_candidate_views.py`.  It renders real
  CityNav orthomosaic RGB at 50 m from the predicted candidates themselves;
  view positions and order never use the target or the human trajectory.
- Hyperparameters were selected on all 2,470 `val_seen` episodes, then frozen
  for all 2,697 `val_unseen` episodes.  Two views and prior weight 0.3 were
  selected.
- Geometric candidate Top-1 is 21.95% Hit@20.  The frozen target-free RGB
  fusion reaches 22.80% (+0.85 point).  The one-view diagnostic is 23.29%, but
  was not selected on `val_seen` and is therefore not promoted.
- Wrong target type falls to 18.95%; shuffled episode evidence falls to
  20.69%.  This supports genuine class-conditioned RGB use, but the selected
  fusion's mean error is 56.02 m versus 55.67 m for geometry alone.  It is not
  yet accepted as the navigation default.
- On the same full split, the epoch-2 structural-ablation HETT target predictor
  is 10.86% Hit@20 and 55.46 m mean error. It is not the paper-quality HETT
  baseline. Target-free fusion is +11.94 Hit@20 points against this diagnostic,
  but its mean error is 0.55 m worse. This is target selection, not SR/SPL.
- On the earlier matched 128-episode subset, that same epoch-2 predictor is 11.72%
  Hit@20 and 53.19 m mean error.  Human-trajectory temporal replay reaches
  28.91% and 38.39 m, but that replay remains an upper-bound-style diagnostic
  because its observation poses were selected by target-distance bands.
- Corrected navigation comparison: published HETT* is 19.10% SR / 16.70% SPL /
  51.30 m NE on refined `val_unseen`; the earlier local 20-epoch training run's
  best checkpoint is 19.54% / 15.73% / 50.52 m. The independent static system
  reaches 23.80% / 22.96% / 44.19 m, a gain of 4.26 SR points, 7.23 SPL points,
  and 6.33 m NE over the strongest local baseline. No complete `test_unseen`
  comparison exists yet.

The fixed rank-order limitation is evaluated in the next section. The RGB
evidence remains too weak to promote into the autonomous controller yet.

## 2026-09-25: fixed-height observation selection

- Exhaustively rendered eight target-free view options per episode and tested
  fixed order, a geometric information-gain proxy, an adaptive challenger, and
  a learned single-view value model.
- Hand-written information gain (22.40% Hit@20) and adaptive challenger
  (22.54%) do not beat fixed rank order (22.80%) on full `val_unseen`.
- A geometry/type-only view-value model trained on `val_seen` reaches 23.66%,
  +0.85 point over fixed order and +1.71 over geometry alone. This is a
  feasibility result; it is not promoted until trained on `train_seen`.
- The privileged single-view oracle is 32.81%, while the two-view noisy-OR
  oracle falls to 30.52%. View choice has useful headroom, but positive-only
  evidence accumulation introduces false positives.
- Full protocol and interpretation are in `OBSERVATION_SELECTION_RESULTS.md`.

## 2026-09-25: point-cloud active-view oracle pilot

- Identified 763 `val_unseen` episodes where Top-5 contains a 20-m success but
  rank 1 is wrong; evaluated a map-balanced 64-episode pilot.
- Enumerated 24 target-free cluster views from eight directions and heights
  20/35/50 m. Point-cloud teacher semantics retain roads and footpaths.
- Relation-conditioned teacher selection corrects 39.06% of initial failures;
  wrong target type corrects 10.94%. Privileged best-view selection reaches
  85.94%, confirming substantial view-choice headroom.
- This is a sampled teacher upper bound, not RGB navigation. Remaining controls
  and `train_seen` view-value training are required before integration.
- Protocol and results are in `POINTCLOUD_ACTIVE_VIEW_ORACLE.md`.
