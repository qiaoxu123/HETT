# Simplified Trajectory Belief in develop

The trajectory path now reuses the original 7x7 HETT candidate grid directly.
There is no 7x7 -> 28x28 dense upsampling stage.

## Representation

For each of the 49 HETT candidate regions, the existing heatmap head supplies
the mode score. An optional geometry head predicts:

```
endpoint offset:      dx, dy
trajectory residual:  5 x (dx, dy)
```

So each candidate represents:

```
region belief
+ continuous endpoint refinement
+ fixed-horizon trajectory residual
```

The endpoint offset is bounded to half a 7x7 cell in each axis, so every mode
refines continuously inside its own coarse spatial region.

## Fixed physical horizons

Each trajectory uses stable metric semantics:

```
10 m / 25 m / 50 m / 100 m / final endpoint
```

The base anchor is the straight fixed-horizon path from the current UAV
position toward the refined continuous endpoint. The residual head is
zero-initialized, so the initial prediction is exactly this stable anchor.

## Training

There is only one spatial classification loss: the existing 7x7 heatmap loss.

Trajectory geometry is supervised on the GT 7x7 region only:

```
L = L_HETT
  + lambda_heatmap * L_heatmap
  + lambda_endpoint * L_endpoint
  + lambda_trajectory * L_trajectory
```

`L_endpoint` refines the GT region center to the continuous target coordinate.
`L_trajectory` compares the predicted fixed-horizon trajectory in that matched
mode against the CityFlight human teacher path sampled at
10/25/50/100 m plus the final goal.

Long horizons that do not exist near the destination are masked.

This removes the previous duplicated 28x28 belief loss and avoids supervising
dozens of unrelated spatial modes toward the same trajectory.

## Top-K proposals

Inference applies local NMS directly on the 7x7 heatmap probabilities, then
keeps Top-K candidate regions (default K=8, NMS kernel=3). Their learned endpoint
offsets convert the coarse regions into continuous endpoints before trajectory
anchors are built.

## Execution

Default behavior is unchanged:

```
--enable_trajectory_belief
```

trains/logs trajectory proposals while the original Two-Stage controller still
executes navigation.

Optional receding-horizon execution:

```
--enable_trajectory_belief --trajectory_execution
```

selects the highest-score trajectory, executes only its first approximately
10 m waypoint, observes again, and replans.

## Intentionally unchanged

The existing teacher + student rollout schedule is preserved.

BERT and DarkNet also remain trainable exactly as in the current develop
baseline.
