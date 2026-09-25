# Unified Belief Transformer: stage 1

## Scope

The first stage uses only inference-visible language relation features,
independent named-landmark instances, and a named-landmark BEV. RGB, point
clouds, active views, altitude, stop prediction and closed-loop control are
deliberately excluded.

The model forward method has no target-coordinate input. Target coordinates
are used only to rasterize the 20 m training label and to calculate metrics.
Training uses `train_seen`; checkpoint selection and hyperparameters use
`val_seen`. No `val_unseen` data is loaded by the training entry point.

## Gate 1: small-set overfitting

On 128 fixed `train_seen` samples after 500 optimization epochs:

| Check | Hit@20 |
|---|---:|
| Correct inputs | 100.00% |
| Opposite directional relation | 60.16% |
| Wrong same-map landmark binding | 10.94% |
| Shuffled landmark order | 100.00% |

The largest logit difference after permuting landmark order was
`1.68e-5`. This passes the capacity, landmark-dependence and set-invariance
checks. The full repository test suite passed: 101 tests.

## Gate 2: three-epoch `train_seen` run

Validation Hit@20 improved monotonically:

| Epoch | val_seen Hit@20 |
|---|---:|
| 1 | 18.38% |
| 2 | 25.30% |
| 3 | 29.47% |

Best epoch 3 metrics:

| Check | val_seen Hit@20 | val_seen mean NE |
|---|---:|---:|
| Correct inputs | 29.47% | 37.77 m |
| Opposite directional relation | 28.95% | 38.62 m |
| Wrong same-map landmark binding | 16.96% | 83.12 m |
| Shuffled landmark order | 29.47% | 37.77 m |

The previous relation candidate ranker averaged 30.92% Hit@20 on the same
split (three seeds); the strongest shape-aware run reached 33.00%. Therefore
v1 is trainable and strongly landmark-dependent, but it has not yet preserved
the established relation performance. The small opposite-relation drop also
shows that precise directional grounding is still weak.

## Next controlled change

Keep the same inputs, supervision and split protocol. Add explicit
grid-cell-to-landmark relative geometry inside the unified transformer instead
of requiring generic self-attention to infer subtraction and direction from
two absolute coordinate tokens. Repeat the 128-sample and three-epoch gates
before adding RGB or any other subsystem.

## Relative-geometry v2 result

The controlled change above was implemented and rerun with the same seed,
splits, model width and three-epoch budget. `val_seen` Hit@20 increased from
29.47% to 38.26% (+8.79 points). The established three-seed relation-ranker
mean was 30.92%, so v2 exceeds it by 7.34 points on the same split.

The v2 validation trajectory was 31.21%, 36.32%, and 38.26%; epoch 3 remained
the best checkpoint. Its mean navigation error was 34.98 m.

Strict inference-time controls on `val_seen`:

| Input condition | Hit@20 | Change |
|---|---:|---:|
| Correct independent landmarks | 38.26% | -- |
| No language features or cues | 18.99% | -19.27 |
| Wrong same-map landmarks | 18.54% | -19.72 |
| First landmark only | 27.57% | -10.69 |
| Landmarks averaged into one | 26.60% | -11.66 |
| Landmark order shuffled | 38.26% | 0.00 |

The benefit is concentrated where independent instances matter:

| Original instruction | Independent instances | Averaged instance |
|---|---:|---:|
| One landmark | 31.31% | 31.31% |
| Two landmarks | 42.95% | 23.38% |
| Three or more | 55.50% | 15.18% |

These controls show that the gain is not explained by an absolute map prior:
language removal and wrong bindings both destroy roughly half the performance,
while landmark permutation leaves the result numerically unchanged. Stage 1
therefore passes and the next allowed increment is RGB projection plus a
visibility-gated signed temporal evidence update.
