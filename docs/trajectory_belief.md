# Dense Trajectory Belief Prototype

This branch replaces HETT's hard coarse-to-fine execution switch with a dense,
receding-horizon trajectory-belief controller inspired by HOME, MultiPath/MTR,
and SBFNav.

## Design principle

HETT already performs language/vision/history fusion in its cross-modal
Transformer. The trajectory-belief module therefore starts directly from the
multimodal candidate features; it does **not** add a second language-attention
block.

The original HETT history representation remains 7x7 for compatibility and
efficiency. A separate dense belief decoder lifts the 7x7 multimodal candidate
map to a 28x28 spatial field.

```
Language + RGB + History/Map
          |
 HETT cross-modal Transformer
          |
     7x7 candidate features
          |
     dense belief decoder
          |
     28x28 latent field
          |
 Trajectory Belief Head
 [score | trajectory residual]
          |
       NMS Top-K
          |
  K continuous endpoints
          |
 K trajectory hypotheses
          |
 execute first waypoint
          |
 observe and replan
```

## Why the grids are decoupled

A 7x7 grid over a 410 m map has cells roughly 58.6 m wide, so using cell
centers directly as trajectory modes introduces substantial quantization error.
The 28x28 belief field reduces the cell width to about 14.6 m while avoiding
784 expensive controller branches.

The dense grid describes **where probability lives**. It does not define how
many trajectories are executed. NMS extracts only the most relevant K modes
(default K=8).

## Continuous endpoint proposals

NMS first finds separated peaks on the 28x28 field. Each peak is then refined by
a local 3x3 soft-argmax, producing a continuous normalized endpoint rather than
using the discrete cell center directly.

Default proposal settings:

- dense belief: 28x28
- NMS Top-K: 8
- NMS kernel: 5
- local endpoint refinement: 3x3 soft-argmax
- future waypoints per trajectory: 5

## Joint trajectory-belief representation

The dense decoder is followed by a **single joint head**, not two independent
task heads. At every spatial location u it outputs:

```
B_u = [s_u | Delta tau_u]
```

where `s_u` is the trajectory-mode score and `Delta tau_u` contains the
future waypoint residuals. The score and geometry therefore describe one
trajectory hypothesis and are trained with separate belief/trajectory losses
only because their supervision differs.

For each selected endpoint g_k, a straight anchor trajectory is constructed
from the current UAV position x_t:

```
A_k = Interpolate(x_t, g_k)
tau_k = A_k + Delta tau_k
```

The final residual convolution is zero-initialized, so training starts from
stable straight anchors instead of random trajectories.

The same residual field is supervised densely during training, but only the
Top-K hypotheses are instantiated for navigation analysis and execution.

## Training

At each training state, the current pose is projected onto the teacher
trajectory and the remaining route is sampled into five arc-length-uniform
future waypoints.

The 28x28 Gaussian target keeps the same approximate physical spread as the
original 7x7 heatmap by scaling sigma with the grid-resolution ratio.

```
L = L_direction
  + 0.1 L_progress
  + 2 L_goal
  + lambda_h L_heatmap
  + lambda_t sum_u Q(u) L_traj(u)
```

Direction remains an auxiliary learning signal. Progress remains the stopping
signal. Neither is used for a coarse/fine stage switch.

## Inference

At every navigation step:

1. predict the 28x28 spatial belief;
2. apply NMS and keep Top-K peaks;
3. refine peaks to continuous endpoints;
4. form K residual-corrected trajectory hypotheses;
5. execute the first waypoint of the highest-belief trajectory;
6. observe again and replan.

There is no Stage-1/Stage-2 distance switch or recovery state machine.

## Recommended ablation

1. HETT hard two-stage controller.
2. 7x7 heatmap + hard 25 m switch.
3. 7x7 trajectory-belief prototype.
4. 28x28 dense belief + Top-K trajectory proposals (this version).
5. Optional later comparison: data-driven/K-means intention anchors.

The immediate test should verify whether the finer field improves SR/SPL/NE,
candidate coverage, and the oracle-to-final-success gap without destabilizing
trajectory learning.
