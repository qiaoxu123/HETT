# Direct Trajectory Navigation in develop

The learned student policy no longer uses the Stage-1 / Stage-2 controller.
Navigation is driven directly by trajectory hypotheses built on the original
7x7 HETT candidate grid.

## Representation

For each of the 49 HETT candidate regions, the existing heatmap head provides
the trajectory-mode score. A geometry head predicts:

```
endpoint offset:      dx, dy
trajectory residual:  5 x (dx, dy)
```

Each candidate therefore represents:

```
region belief
+ continuous endpoint refinement
+ fixed-horizon trajectory residual
```

The endpoint offset is bounded to half a 7x7 cell in each axis, so each mode
refines continuously inside its coarse region.

## Fixed physical horizons

Each trajectory uses:

```
10 m / 25 m / 50 m / 100 m / final endpoint
```

A straight fixed-horizon anchor is built from the current UAV position toward
the refined endpoint. The residual head is zero-initialized, so early training
starts from stable straight anchors.

## Training

There is one spatial mode-classification objective: the existing 7x7 heatmap
loss.

Trajectory geometry is supervised only on the GT 7x7 mode:

```
L = L_HETT
  + lambda_heatmap * L_heatmap
  + lambda_endpoint * L_endpoint
  + lambda_trajectory * L_trajectory
```

The CityFlight human teacher path is projected continuously from the current
state and sampled at 10/25/50/100 m plus the final goal. Invalid long horizons
near the destination are masked.

The teacher + student rollout schedule is intentionally unchanged. BERT and
DarkNet also remain trainable.

## Student execution

Student navigation is trajectory-only:

```
7x7 trajectory-mode scores
        ↓
3x3 NMS + Top-K
        ↓
continuous endpoints
        ↓
fixed-horizon trajectories
        ↓
select Top-1 trajectory
        ↓
execute only first ~10 m waypoint
        ↓
observe again and replan
```

There is no learned-policy Stage-1 / Stage-2 switch, recovery threshold, or
fine-navigation branch in student execution.

The legacy Stage-1 / Stage-2 code remains reachable only inside the preserved
teacher rollout path so that the requested teacher training behavior is not
changed.

## Key simplification

Previous:

```
7x7 heatmap -> Stage-1 -> Stage-2
                    or
7x7 -> 28x28 dense belief -> trajectory
```

Current:

```
7x7 HETT candidates
        ↓
score + endpoint offset + trajectory residual
        ↓
Top-K trajectories
        ↓
first waypoint
        ↓
replan
```
