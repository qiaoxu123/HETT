# Multi-trajectory feasibility gate

Date: 2026-09-25

## Question

Before training a diffusion policy, test whether several target-free trajectory
hypotheses contain substantially more navigation success than the current
Top-1 route.  If a privileged selector cannot improve official SR, a learned
generator cannot solve the current bottleneck.

## Protocol

- split: full `val_seen` (2,470 episodes);
- fixed altitude: 50 m;
- official outer-action budget: 20;
- official movement iterations per action: 10;
- success radius: 20 m;
- hypotheses: spatially distinct modes of the frozen unified belief map;
- minimum separation: 25 m;
- path: deterministic official actions from the start pose to each mode;
- target use: only the privileged selector and final metrics;
- `val_unseen` and `test_unseen`: not loaded.

The path generator has no target-coordinate argument.  The oracle receives all
completed paths and chooses the one whose final pose is closest to the metric
target.  It is an upper bound, not a deployable policy result.

## Results

| Selector | Paths | Endpoint Hit@20 | Strict SR | Strict SPL | NE |
|---|---:|---:|---:|---:|---:|
| Current belief Top-1 | 1 | 38.26 | 36.92 | 35.56 | 35.02 m |
| Privileged oracle | 2 | 54.74 | 53.44 | 51.51 | 26.84 m |
| Privileged oracle | 3 | 64.94 | 63.89 | 61.72 | 22.83 m |
| Privileged oracle | 5 | **76.28** | **75.63** | **73.01** | **18.43 m** |

Top-5 improves over Top-1 by **+38.70 SR** and **+37.45 SPL** points.  The
Top-5 oracle uses 5.63 actions on average, at most 12, and no episode exceeds
the 20-action budget.  Its selected-rank histogram is 815 / 578 / 439 / 347 /
291 for ranks 1 through 5, so the upper bound is not produced only by a few
rank-2 corrections.

## Interpretation

The gate passes decisively: the belief map already contains diverse useful
goal hypotheses, and official-action paths to those hypotheses remain cheap.
The immediate bottleneck is therefore trajectory **selection**, not trajectory
generation.  A Conditional 1D U-Net is justified only after a non-privileged
trajectory scorer is established; otherwise diffusion would generate more
routes without solving which route is correct.

The next model should score each target-free trajectory using language-to-
landmark relation consistency, online RGB verifiability, belief support and
remaining action budget.  It must be evaluated against wrong-relation,
wrong-landmark and wrong-pose controls.  Rolling replanning should execute only
the first one or two actions before updating belief and rescoring.

Diffusion Policy is the appropriate later implementation reference because its
official code provides `ConditionalUnet1D` for observation-conditioned action
sequences.  MotionDiffuser contributes the idea of differentiable constrained
sampling, but its full multi-agent architecture is not needed here.

## Artifacts

- evaluator: `scripts/evaluate_multitrajectory_oracle.py`
- Top-2: `runs/multitrajectory_oracle/val_seen_top2_full.json`
- Top-3: `runs/multitrajectory_oracle/val_seen_top3_full.json`
- Top-5: `runs/multitrajectory_oracle/val_seen_top5_full.json`

