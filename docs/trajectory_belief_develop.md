# Optional Trajectory Belief in develop

Trajectory generation is integrated as an **optional** branch so it can be
ablated independently from the corrected heatmap + landmark baseline.

## Representation

The original 7x7 HETT/heatmap path is left intact. From the same conditioned
7x7 candidate features, an auxiliary decoder upsamples to a 28x28 latent field.
Every dense location predicts one joint trajectory-belief vector:

```
[mode score | dx1 dy1 | ... | dx5 dy5]
```

The geometry channels are zero-initialized, so the initial trajectories are
stable straight anchors.

## Fixed physical horizons

The first four waypoints have stable metric semantics:

```
10 m / 25 m / 50 m / 100 m / final endpoint
```

Teacher targets are sampled from the human CityFlight teacher path by continuous
polyline projection, using the same physical horizons plus the final goal.
Unavailable long horizons near the goal are masked instead of duplicating the
goal target.

## Dense belief and sparse proposals

The 28x28 field is supervised by a Gaussian goal belief. Inference performs
GPU max-pool NMS and retains Top-K proposals (default K=8). Only those K
trajectories are materialized for proposal logging/execution.

## Controller isolation

Default behavior remains the existing Two-Stage controller.

Enable trajectory learning only:

```bash
--enable_trajectory_belief
```

This trains/logs trajectory proposals but still executes the original heatmap
Two-Stage controller.

Enable receding-horizon trajectory execution:

```bash
--enable_trajectory_belief --trajectory_execution
```

Then the student executes only the first 10 m-horizon waypoint of the highest
belief trajectory and replans from the next observation. This bypasses the
Stage-1/Stage-2 switch for student execution, while the teacher rollout remains
unchanged.

## Recommended ablation order

1. corrected 7x7 heatmap baseline;
2. + zero-gated global landmark prior;
3. + trajectory belief supervision, Two-Stage execution;
4. + trajectory receding-horizon execution.

Referenced landmark centroid tokens remain off by default and should be tested
separately.
