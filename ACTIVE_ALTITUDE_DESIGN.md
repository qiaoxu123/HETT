# Uncertainty-Guided Active Altitude Navigation

## Status and scope

This worktree contains a design implementation only.  It has not been wired
into the legacy HETT rollout and no training, evaluation, smoke test, or unit
test has been run.  This separation is intentional: the released fixed-50 m
path must remain unchanged until each component passes its own offline gate.

Branch: `codex/active-altitude`

Base: `main` at `51a1828`

## Research hypothesis

Altitude is an active-perception action that trades field of view for spatial
resolution.  The student should therefore choose a small discrete sensing
scale from uncertainty, landmark agreement and temporal stability, rather than
regress an arbitrary human altitude.

The proposed policy has four phases and four AGL levels:

| Phase | AGL | Horizontal step | Purpose |
|---|---:|---:|---|
| Search | 120 m | 50 m | broad, low-confidence discovery |
| Coarse approach | 80 m | 20 m | select a target region |
| Landmark approach | 60 m | 10 m | resolve language/landmark constraints |
| Refinement | 30 m | 5 m | localize and verify arrival |

The learned vertical action space is `ASCEND`, `HOLD`, `DESCEND`.  A state
machine limits movement to one altitude level per step and uses temporal
confirmation to prevent oscillation.  High uncertainty may reopen the previous
phase and ascend one level.

## Code topology

- `multiagent/active_altitude/controller.py`: deployable per-episode phase,
  altitude, horizontal-step and stop controller.  It accepts prediction-only
  signals and cannot access target ground truth.
- `multiagent/active_altitude/belief.py`: log-space fusion of target heatmaps.
  High-altitude evidence is broader and weaker than low-altitude evidence.
- `multiagent/active_altitude/teacher.py`: privileged distance and human-height
  labels.  Keeping this in a separate module creates an explicit train/eval
  boundary.
- `multiagent/active_altitude/semantics.py`: optional frozen RSRefSeg2-style
  semantic prior.  It is confidence-gated and late-fused.
- `multiagent/active_altitude/runtime.py`: per-episode composition layer.  It
  prevents the legacy batch-shared stage-state bug.
- `multiagent/active_altitude/cache.py`: bounded on-demand cache keyed by
  quantized map, XY, yaw and altitude level.  It does not attempt exhaustive
  offline view generation.
- `multiagent/models/active_altitude_heads.py`: altitude-action, phase, stop and
  goal-uncertainty heads.
- `multiagent/active_altitude/losses.py`: masked auxiliary losses.

## Stop-control invariant

Stopping is independent of diagnostic logging and phase-transition branches.
It requires all of the following for multiple consecutive observations:

1. refinement phase;
2. predicted distance at most 20 m;
3. calibrated stop probability above threshold;
4. stable predicted goal across recent observations.

Ground-truth target distance is used only to form teacher labels.  The runtime
controller never receives it.

## Human-height teacher

Human altitude is a noisy soft prior, not an oracle action.  The teacher first
creates a phase target from privileged distance, then softly mixes the nearest
human altitude level.  Human influence decays exponentially when it disagrees
with the phase-derived level and can never exceed 0.5.  This prevents an
operator's incidental flight habit from dominating the sensing policy.

## RSRefSeg2 integration decision

RSRefSeg2 is worth testing for language-conditioned target selection, but not as
a direct navigation or stop controller.  Existing local evidence shows genuine
instruction sensitivity after CityRefer fine-tuning, while localization remains
too weak on large regions to be trusted alone.

The proposed integration is therefore:

1. run RSRefSeg2 offline or in a separate preprocessing job;
2. convert its mask/box evidence into a calibrated grid heatmap;
3. store the result by map and instruction ID;
4. reject missing, malformed and low-confidence priors;
5. late-fuse accepted logits into HETT's target belief with a capped weight;
6. never let the prior directly issue movement, altitude, or stop actions.

This gives a clean `semantic_prior=off/on` ablation and avoids placing a heavy
segmentation model inside every closed-loop rollout step.

## Observation rendering and caching

Pre-rendering only four heights is insufficient because a view also depends on
continuous XY and yaw.  Exhaustively materializing all combinations would be
large and mostly unused.  The implementation instead defines an on-demand LRU
keyed by `(map, x_cell, y_cell, yaw_bin, altitude_level)`.

Recommended engineering order after profiling:

1. keep the source orthomosaics resident and build four scale-pyramid levels;
2. batch perspective transforms with CUDA `grid_sample`;
3. add the bounded LRU for repeated quantized views or frozen visual features;
4. precompute immutable landmark contours, centers and map-cell assignments;
5. vectorize dynamic map updates before increasing cache coverage.

Fixed-height and dynamic-height measurements had similar per-episode throughput,
so view generation is not yet demonstrated to be the main bottleneck.  No large
offline cache should be generated before a stage profiler proves its value.

## Staged integration plan

### Gate 0: pure logic

Review and test controller, belief fusion, teacher targets, semantic gating and
cache behavior without loading HETT, CityNav or CUDA.

### Gate 1: oracle ceiling

Wire only logging and oracle controls.  Compare fixed altitude against:

- ground-truth phase/altitude with learned horizontal control;
- learned altitude with ground-truth stopping;
- ground-truth target controller with learned altitude.

If the altitude oracle adds less than roughly 2--3 unseen SR points, stop this
line of work before full training.

### Gate 2: altitude without semantics

Attach `ActiveAltitudeHeads` to the fused motion token.  Add normalized current
altitude to the pose input, preserve per-episode controller state, and supervise
altitude action plus phase.  Keep the published target and stop behavior fixed
for causal attribution.

### Gate 3: multi-scale belief

Preserve the spatial map feature grid, update `MultiScaleBelief` at every step,
and drive phase transitions from entropy, top-2 margin and temporal stability.

### Gate 4: independent stop head

Enable the stop head with identical stopping logic in every comparison.  Report
precision/recall inside 20 m, premature stops and timeouts in addition to SR/SPL.

### Gate 5: RSRefSeg2 prior

Enable only the frozen late-fusion provider.  First require better unseen
Hit@20 and same-map instruction-switch sensitivity.  Only then evaluate
navigation metrics.

## Required ablations

1. released fixed-50 m HETT;
2. four-level observations with a fixed scripted schedule;
3. learned discrete altitude actions;
4. uncertainty-driven phase transitions;
5. altitude-aware belief fusion;
6. independent semantic stop head;
7. full system without RSRefSeg2;
8. full system with frozen RSRefSeg2 prior.

Use `val_seen` and `val_unseen` during development.  Do not use `test_unseen`
until thresholds and the model are frozen.  A candidate requires roughly three
unseen SR points with no material SPL regression and confirmation across seeds.

## Metrics beyond SR/SPL

- Goal Hit@10/20, median and P90 target error.
- Stop precision/recall, premature-stop rate and timeout rate.
- Phase confusion and phase-transition distance.
- Altitude action accuracy and oscillation count.
- Belief entropy and calibration error by altitude.
- Same-map counterfactual instruction sensitivity.
- Wall time split into rendering, map update, model forward/backward and sync.
