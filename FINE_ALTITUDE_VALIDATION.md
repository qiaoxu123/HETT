# Fine-altitude paired validation

## Scope

This is a development smoke test, not a final benchmark. It evaluates the
released epoch-26 checkpoint on the first 32 episodes of each development
split, with identical seed, episode order, batch size and task-interaction
setting. `test_unseen` remains sealed.

## Results

| Split | Policy | SR (%) | SPL (%) | NE (m) | Mean altitude (m) | Low-alt frames (%) |
|---|---|---:|---:|---:|---:|---:|
| val_seen | fixed 50 | 31.25 | 25.89 | 41.39 | 50.00 | 0.00 |
| val_seen | 50 -> 45 | 31.25 | 24.50 | 40.96 | 47.64 | 47.17 |
| val_seen | 50 -> 40 | 25.00 | 19.50 | 42.00 | 45.27 | 47.28 |
| val_unseen | fixed 50 | 0.00 | 0.00 | 87.43 | 50.00 | 0.00 |
| val_unseen | 50 -> 45 | 0.00 | 0.00 | 88.98 | 47.69 | 46.13 |
| val_unseen | 50 -> 40 | 0.00 | 0.00 | 88.50 | 45.39 | 46.13 |

All 64 paired stage-1 trajectories are exactly identical for each dynamic
variant, including pose and altitude. The matching stage-1 length and NE are
therefore an implementation invariant rather than only an aggregate match.

## Decision

The architecture now correctly isolates altitude changes to fine navigation,
but the released fixed-50 m checkpoint does not show consistent benefit from
lower RGB views. The 40 m view clearly regresses `val_seen`; 45 m is safer but
does not improve both splits. Keep this feature evaluation-only and do not run
`test_unseen` until a low-altitude-aware fine-stage model passes a larger paired
development test.

The next useful experiment is fine-stage adaptation: keep coarse navigation
and the global map frozen at 50 m, and train only the visual fine-navigation
path with 45/40 m crops plus an altitude/scale embedding. A dual-scale 50 m +
low-altitude fusion is preferable to replacing the 50 m observation outright.
