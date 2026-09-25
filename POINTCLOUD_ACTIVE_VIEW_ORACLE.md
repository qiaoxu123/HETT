# Relation-conditioned point-cloud active-view oracle

## Question

Can active observation repair the specific failure where the independent
generator's Top-5 contains a point within 20 m of the target but rank 1 is
wrong?

There are 826 such `val_seen` episodes and 763 such `val_unseen` episodes. The
pilot balances 64 episodes across the four unseen maps. This subset is selected
with target coordinates, but view generation, selection and posterior updates
do not receive target coordinates or target identity.

## Candidate views

The computationally bounded first gate uses one common candidate-cluster orbit:

- eight horizontal directions;
- heights 20, 35 and 50 m;
- 30 m orbit radius;
- camera points toward the Top-5 cluster center;
- 24 views per episode at 48 x 48 resolution;
- native SensatUrban classes retain roads and footpaths.

Each view records teacher semantic visibility around every candidate, named
landmark visibility, visible road pixels in candidate-landmark corridors,
valid-pixel fraction and travel cost. Candidate pairs inside 20 m receive no
discrimination reward. Utility and posterior weights are selected on a separate
balanced 64-episode `val_seen` sample, then frozen.

## Pilot result

| Balanced `val_unseen` diagnostic subset | Hit@20 | mean NE |
|---|---:|---:|
| Initial Top-1 | 0.00% | not applicable |
| Wrong target-type control | 10.94% | 29.24 m |
| **Relation-conditioned point-cloud teacher** | **39.06%** | **23.76 m** |
| Privileged best rendered view | 85.94% | 11.81 m |
| Top-5 oracle | 100.00% | 10.04 m |

The learned utility chooses 20 m, 35 m and 50 m in 21.88%, 48.44% and 29.69%
of cases. No single fixed height dominates; 35 m is most common.

The result supports continuing the direction. Teacher semantic visibility at a
selected view corrects 25 of 64 initial ranking failures, while changing only
the requested target type reduces correction to 7 of 64. The privileged-view
gap (39.06% versus 85.94%) shows that predicting view value is now the main
active-perception bottleneck in this oracle setting.

## Limits and next gate

- This is a balanced 64-episode pilot, not the full 763-episode population.
- It uses point-cloud teacher semantics and visibility, not RGB predictions.
- The first gate orbits the shared candidate cluster. Per-candidate 120-view
  search passed a four-episode smoke test but is too expensive for the first
  scale-up.
- Explicit wrong-landmark, shuffled-pose, random-view and fixed-height controls
  remain required before accepting the planner.
- The current joint relation signal confirms co-visibility and road corridors;
  finer left/right relation parsing is not yet part of the view utility.
- It is neither autonomous navigation nor an SR/SPL result.

The next gate should cache the 24 teacher observations per episode, add the
remaining controls without rerendering, and train a view-value model on
`train_seen`. Point-cloud outputs remain supervision only; runtime evaluation
must replace them with RGB-predicted semantic/visibility likelihoods.
