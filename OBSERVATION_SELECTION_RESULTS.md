# Fixed-50-m observation selection

## Protocol

- Candidate set: five independent language/landmark-geometry proposals.
- View options: start, each candidate center, candidate centroid, and
  prior-weighted centroid.
- Runtime selector inputs: candidate belief and geometry, predicted target
  type, start/current pose, and fixed camera footprint.
- RGB is consumed only after a view is selected. Target coordinates and target
  identity are never selector inputs.
- Prior weight 0.3 and two-view budget are inherited from the preceding
  `val_seen` gate. Geometry utility weights are selected on `val_seen`; final
  metrics use all 2,697 `val_unseen` episodes.

## Results

| Method | Hit@20 | mean NE |
|---|---:|---:|
| Geometry candidate rank 1 | 21.95% | 55.67 m |
| Fixed candidate ranks 1 then 2 | 22.80% | 56.02 m |
| Geometric information-gain proxy | 22.40% | 56.34 m |
| Adaptive visual challenger | 22.54% | 55.93 m |
| **Learned single-view value** | **23.66%** | **55.95 m** |
| Privileged single-view oracle | 32.81% | 50.31 m |
| Privileged two-view oracle | 30.52% | 51.12 m |
| Top-5 candidate oracle | 50.24% | 39.19 m |

The hand-written geometric selector is rejected. In a fixed-altitude top-down
orthomosaic, translating the crop does not create parallax, and centering all
candidates often merges their semantic evidence. The adaptive challenger also
fails to beat the fixed order.

The learned view-value feasibility model is trained on `val_seen`, with an
object-grouped 4/1 internal train/selection split, and evaluated on unseen maps.
It uses no future RGB. It improves Hit@20 by 0.85 point (23 episodes) over the
fixed two-view policy and by 1.71 points over geometry alone. This is a modest
feasibility gain, not a final accepted model: training must move to
`train_seen`, with `val_seen` used only for selection, before integration.

The oracle result is more important than the small learned gain. One well-chosen
view can reach 32.81%, but adding a second view lowers the oracle to 30.52% under
positive noisy-OR fusion. Therefore false-positive evidence accumulation is a
second bottleneck. The next version should choose one view first and request a
second view only when calibrated posterior entropy remains high; evidence must
use signed likelihood ratios rather than positive-only noisy-OR.

## Point-cloud scope

Point-cloud supervision is not used in this gate or in the current independent
navigation result. It has separately improved perspective foreground mIoU from
21.18% to 42.84% and orthomosaic named-landmark binding from 79.32% to 85.44%,
but those gains have not yet been connected to candidate ranking, view value,
or closed-loop SR/SPL.
