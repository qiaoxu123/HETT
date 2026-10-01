# Trajectory Belief Prototype

This branch replaces HETT's hard coarse-to-fine execution switch with a
receding-horizon trajectory-belief controller inspired by the representation
ideas in HOME, MultiPath, and SBFNav.

## Core idea

For every one of the 7x7 spatial candidates, the same language-conditioned
candidate token now predicts:

1. one heatmap logit (where the target is likely to be), and
2. a short 2D future trajectory (how to move if that spatial hypothesis is the
   correct mode).

The trajectory is represented as a learned residual around a straight-line
anchor from the current UAV position to the candidate-cell center. The final
residual layer is zero-initialized, so before learning the new branch reduces to
stable straight-line anchors rather than random waypoints.

At inference:

```
new observation
  -> language-conditioned heatmap
  -> one trajectory hypothesis per heatmap cell
  -> choose the highest-probability heatmap mode
  -> execute only the first waypoint
  -> observe again and replan
```

There is no Stage-1/Stage-2 distance switch in this controller. The original
direction head is kept as an auxiliary learning signal, while progress remains
the stopping signal.

## Supervision

At every training state, the current pose is projected to the nearest point on
the teacher trajectory. The remaining teacher path is sampled into
`trajectory_steps` arc-length-uniform future waypoints.

The future-trajectory loss is computed for all spatial modes and weighted by the
same Gaussian target distribution used by the heatmap:

```
L = L_direction
  + 0.1 L_progress
  + 2 L_goal
  + lambda_h L_heatmap
  + lambda_t sum_k Q(k) L_traj(k)
```

This keeps the endpoint/spatial uncertainty and trajectory hypotheses coupled
instead of collapsing the heatmap to a single goal before planning.

## Default prototype parameters

- grid: 7x7
- trajectory waypoints per mode: 5
- trajectory residual range: 0.10 normalized map units
- trajectory loss weight: 1.0
- execution: first waypoint only, then replan
- progress stopping threshold: unchanged from HETT (0.95)

## Intended ablation

1. Original HETT hard two-stage controller.
2. Gaussian heatmap + hard 25 m switch.
3. Language-conditioned heatmap + hard 25 m switch.
4. Trajectory belief + receding-horizon execution (this branch).

The first experiment should focus on whether removing the hard switch improves
SR/SPL/NE and reduces the oracle-to-final success gap before adding Top-K
trajectory selection or a learned selector.
