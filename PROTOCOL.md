# RSRefSeg2-inspired goal grounding: single-seed protocol

## Objective

Test whether a HETT-native coarse-to-fine goal head improves instruction-conditioned
target localization and navigation without embedding the full RSRefSeg2/SAM stack.
The experiment is based on `codex/landmark-teacher-cleanup` so that per-episode stage
state, teacher trajectories, and landmark-arrival gating are already corrected.

## Development split and seed

- Seed: 0 only for this screening experiment.
- Training: `train_seen`.
- Development evaluation: `val_seen` and `val_unseen` only.
- `test_unseen` is excluded until the design and thresholds are frozen.

## Model change

1. Preserve the map encoder's 15 x 15 spatial feature grid instead of flattening it
   before goal prediction.
2. Predict a language-conditioned coarse spatial distribution.
3. Predict a bounded offset inside the selected cell and derive the final normalized
   goal from coarse cell plus offset.
4. Keep direction/action outputs checkpoint-compatible where practical.
5. Add a separate target-arrival/stop confidence output only after the grounding head
   passes its standalone localization screen. Diagnostic logging must not participate
   in stopping control flow.

## Required comparisons

1. Parent checkpoint, unchanged inference.
2. Coarse-to-fine grounding head.
3. Coarse-to-fine head plus semantic stopping, only if comparison 2 improves unseen
   target localization.

All comparisons use the same episode lists and report paired deltas.

## Metrics

- Goal Hit@10 m and Hit@20 m, median goal error, and P90 goal error.
- SR, SPL, navigation error, oracle SR, and stage-transition statistics.
- Stop precision/recall inside 20 m, premature-stop rate, and timeout rate.
- Language sensitivity on same-map counterfactual instruction pairs.

## Screening gates

- Grounding gate: improve `val_unseen` Goal Hit@20 without materially degrading
  `val_seen`.
- Navigation candidate gate: `val_unseen` SR improves by about 3 percentage points
  and SPL does not materially regress.
- A single seed is evidence for screening only, not a final paper claim.

## Runtime estimate on the local RTX 5090

- Implementation and CPU/unit tests: 4-6 hours.
- Small GPU smoke run: 0.5-1 hour.
- Full training: approximately 2.7-3.0 hours per epoch.
- Three-epoch screening fine-tune: 8-9 hours.
- Full `val_seen` plus `val_unseen` evaluation: 0.6-0.8 hours per checkpoint.
- Grounding-only plus full-model single-seed screen: approximately 18-30 GPU hours,
  depending on whether the grounding gate passes.

Existing GPU work must finish first. Checkpoints must not be written to the root
filesystem; the run starts only after sufficient space is available on `dataext`.
