# Fine-altitude refinement

## Decision

Dynamic altitude is restricted to fine visual refinement.  The released HETT
coarse-navigation path remains at 50 m AGL because it is driven primarily by
the language-selected landmark map, current/explored coverage and historical
grid memory.  Changing coarse altitude would alter all of those input
distributions before demonstrating a navigation benefit.

The default evaluation-only policy uses a conservative 50 and 45 m AGL step.
40 and 30 m remain explicit experimental options:

- 50 m: canonical coarse navigation and default RGB observation;
- 45 m: default local visual probe after the coarse goal is stable and
  contained; the smaller scale change limits checkpoint distribution shift;
- 40 m: stronger experimental refinement; paired smoke testing reduced
  `val_seen` success, so it is no longer the default;
- 30 m: optional final probe only when explicitly requested and the predicted
  goal remains contained. The first paired smoke test showed degradation at
  this level, so it is not enabled by default.

No learned altitude head, RSRefSeg2 prior, horizontal-step change or new stop
rule is enabled in this first causal test.

## Isolation invariants

When `--fine_altitude_refinement` is disabled, the fixed-50 m rollout remains
the default.

When it is enabled:

1. Only the RGB camera pose changes altitude.
2. The three-channel navigation map is always updated using a canonical 50 m
   pose with the real XY and yaw.
3. Low-altitude visual features are marked invalid for global historical-grid
   aggregation.
4. The coarse goal is frozen while the controller is in refinement.
5. Existing horizontal movement, rotation, progress stopping and navigation
   metrics remain unchanged.
6. High entropy, unstable goal predictions or a large predicted distance cause
   one-level-at-a-time recovery toward 50 m.
7. Development evaluation loads only `val_seen` and `val_unseen`; `test_unseen`
   remains sealed.

## Entry and containment

The default controller requires a three-observation stable goal, sufficiently
concentrated candidate probabilities and two confirming decisions.  It enters
refinement only within 40 m and only if the predicted goal fits inside the next
lower view after a 5 m safety margin.

The same containment rule prevents an unconditional drop to 30 m. With an
explicit `50 40 30` schedule, a predicted target must be within 25 m before a
30 m probe is allowed.

## Evaluation command template

Do not launch until the shared GPU/memory-throughput condition is healthy.
Use the same checkpoint, seed, batch size and student-evaluation protocol for
both arms.

Baseline arm:

```bash
python multiagent/main.py --mode eval --feedback student --altitude 50 \
  --checkpoint CHECKPOINT
```

Fine-altitude arm:

```bash
python multiagent/main.py --mode eval --feedback student --altitude 50 \
  --checkpoint CHECKPOINT --fine_altitude_refinement \
  --fine_altitude_levels 50 45
```

The result now reports mean camera altitude, low-altitude observation fraction
and mean altitude changes per episode in addition to SR, SPL and NE.

## Ordered ablation

1. Fixed 50 m baseline.
2. Conservative 50/45 m.
3. 50/40 m and containment-gated 50/40/30 m (experimental; not currently
   supported by the released fixed-height checkpoint).
4. Only after a positive paired result, add a local language-vision refinement
   head while keeping the coarse model frozen.
5. Train a height selector only after the scripted policy establishes an
   empirical ceiling.
6. Evaluate an adapted semantic prior last and separately.

The first acceptance criterion is unchanged: approximately three val_unseen SR
points without a material SPL regression, followed by confirmation across
seeds.
