# Scene-evidence grounding benchmark

This branch asks whether RGB captured at the published CityNav teacher poses
contains instruction-specific local scene evidence. It does not train actions,
run a controller, or perform a navigation rollout.

## Data contract

- Only `train_seen`, `val_seen`, and `val_unseen` are accepted.
- RGB is reconstructed through the existing `cropclient` at the exact saved
  trajectory `(x, y, z, direction)`; candidate-centred cameras are forbidden.
- At most one frame is sampled from each `0–20`, `20–40`, `40–80`, and `>80 m`
  goal-distance bucket.
- Region proposals are CityRefer polygons intersecting the current FOV. They
  localize content in the real observation and do not move the camera.
- Splits are audited by episode, target object, and generated sample ID.

## Gates

Phase 4 requires all of the following on `val_seen`: near-minus-far score above
0.02, same-map instruction hard-negative accuracy above 55%, and mean
per-episode score/distance Spearman below -0.1. A distance curve by itself is
not sufficient because altitude and generic scene confidence co-vary with
trajectory progress.

If the Gate fails, `train_scene_matcher.py` and
`evaluate_scene_belief_update.py` record a skipped result. The Oracle evaluator
is still run because it distinguishes insufficient perception from an
intrinsically non-discriminative scene representation.

Every experiment is launched through `scripts/supervise_experiment.py` and
records a source snapshot, commands, provenance, status, and phase metrics.
