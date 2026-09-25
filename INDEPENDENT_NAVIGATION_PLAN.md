# Independent navigation plan and feasibility gates

This branch builds a navigation stack that does not call the HETT goal model,
candidate generator, policy, or action generator. HETT is retained only as a
frozen comparison baseline.

## Dependency boundary

The system is a new project architecture named **Landmark Belief Navigation
(LBN)**. GeoNav is used only as literature-level motivation for decomposing
navigation into search and localization. LBN does **not** import or modify
GeoNav code, checkpoints, model definitions, prompts, perception interfaces, or
action policies. It also does not use HETT components at runtime.

Allowed dependencies are general-purpose numerical and deep-learning libraries
such as PyTorch, OpenCV, and rasterio. In the finished system, every task-specific
runtime model will be trained from the current project's data and owned by
`independent_nav`.

## LBN system

```text
instruction
  -> named landmarks, per-landmark relations, target type
  -> map polygons for each named landmark
  -> static relation probability map

RGB + camera pose + altitude
  -> semantic masks
  -> camera-to-map projection
  -> association with named map landmarks
  -> visual target evidence and observed-empty evidence

static prior + accumulated visual evidence
  -> persistent log-odds target belief map
  -> next-best-view planner
  -> independent discrete action controller
  -> evidence-based stop controller
```

The belief map is the central state. The model does not predict a new unrelated
goal from every frame. Each observation adds positive evidence where the target
type is seen and negative evidence inside visible regions where it is absent.

## Newly designed modules

1. `InstructionGraphEncoder` converts the instruction into target-type and
   `(landmark, relation)` nodes. Two landmarks remain two independent nodes.
2. `RelationFieldPrior` represents each map cell by its distance and angle to
   every named polygon, then uses masked cross-attention to create a static goal
   heatmap. Multiple constraints are combined as a product of evidence, not by
   averaging landmark centers.
3. `ProjectSemanticNet` is our own encoder-decoder trained with Building, Car,
   Ground, and Parking map labels. It predicts both class probability and pixel
   uncertainty. No external open-vocabulary detector or segmentation model is
   required by the base system.
4. `PoseProjector` is deterministic camera geometry. It places semantic
   probabilities and uncertainty into world-map cells.
5. `MapLandmarkBinder` associates visual components with named map polygons using
   projected overlap, position, size, and type. Appearance is an optional learned
   residual and never overrides an incompatible geographic position.
6. `LogOddsBelief` maintains persistent target state:

   ```text
   belief_t = belief_(t-1)
              + relation evidence
              + visible target-type evidence
              - confidently observed empty-space evidence
   ```

   Observation uncertainty controls the size of each update so a poor frame
   cannot erase reliable history.
7. `InformationViewPlanner` creates viewpoints around the top belief regions and
   scores `expected uncertainty reduction - travel cost - revisit cost`.
8. `GeometricGoalController` converts the selected viewpoint into 30-degree turns,
   5-metre forward moves, vertical moves, and stop.
9. `EvidenceStopController` stops only when probability mass is concentrated in a
   20-metre circle, stable across frames, target type is supported, and stated
   landmark relations remain valid.

## Training strategy

Training remains modular so each claimed improvement is measurable:

1. Train `RelationFieldPrior` on `train_seen` target heatmaps with target
   coordinates used only as supervision.
2. Train `ProjectSemanticNet` from map-derived semantic masks; select checkpoints
   only on `val_seen`.
3. Replay training trajectories to train uncertainty calibration and temporal
   belief updates. At inference, the updater receives no target coordinates.
4. First use the deterministic information-gain planner and controller. Only if
   this plateaus, train a planner by imitation from target-free information-gain
   search traces; reinforcement learning is not required for the first version.
5. Keep `val_unseen` for final evaluation and anti-cheating controls.

## Completed feasibility gate: RGB to named landmark

The semantic detector was trained on `train_seen`, selected on `val_seen`, and
evaluated on `val_unseen` once. It did not receive target coordinates or
navigation success labels.

At 50 m in the current top-down crop environment:

| Metric | val_seen | val_unseen |
|---|---:|---:|
| Building detected at >=25% coverage | 91.19% | 88.18% |
| Named Top-1, conditional on detection | 66.90% | 70.95% |
| End-to-end named Top-1 | 61.01% | 62.56% |
| Median projected component error | 21.24 m | 16.33 m |
| Wrong-pose Top-1 control | 26.21% | 15.08% |
| Wrong-name Top-1 control | 10.79% | 11.17% |

The camera/world transform round-trip error was below 0.0001 m. Therefore the
remaining position error comes mainly from merged or fragmented segmentation
components, not from the homography implementation. The large drop under wrong
pose and wrong name controls is evidence that matching uses the intended inputs.

Limitation: this gate uses exact top-down orthomosaic crops. It proves feasibility
inside the current benchmark observation generator, not robustness to oblique or
photorealistic AirSim camera images.

## Completed feasibility gate: independent static navigation

The existing independent relation generator produced one goal using only language
and landmark geometry. A new controller then generated every action from start
pose and that predicted goal. Target coordinates were read only after the entire
trajectory had been generated.

On 2,697 `val_unseen` episodes:

| System | SR | SPL | final NE |
|---|---:|---:|---:|
| Frozen HETT baseline | 9.86% | 8.06% | 62.56 m |
| Independent model + independent controller | **23.80%** | **22.96%** | **44.20 m** |

This uses HETT's exact evaluation convention: both trajectory and goal are reduced
to XY before the 20-metre success check. SR improves by **13.94 points**, SPL by
**14.90 points**, and navigation error falls by **18.36 m**. Height affects the
quality of visual observations, but it is not included directly in SR or SPL.

## Next validation gates

1. Replay sequences and update the belief map over time. Compare static relation
   prior, one-frame RGB fusion, and temporal fusion on the same episodes.
2. Add wrong-pose, wrong-name, flipped-relation, and shuffled-time controls to the
   temporal result. A valid model must deteriorate under each relevant control.
3. Add next-best-view selection and run a closed loop at fixed 50 m. This isolates
   the value of active observation from height.
4. Only then expose altitude actions to the planner. Descend when belief location
   is concentrated but target-type evidence is weak; otherwise remain high.
5. Compare the complete independent system with the frozen HETT baseline using
   exactly the same episode IDs and XY SR/SPL implementation.

The immediate next implementation is the landmark binder, followed by the
persistent belief updater—not a larger vision backbone. The detector already
finds most visible buildings; the main loss is ambiguous component-to-name
association and lack of temporal evidence.

## Locked next step after point-cloud gate

The next trainable module is `MapLandmarkBinder`. Before implementing temporal
belief updates, replace whole-component nearest-centroid matching with per-pixel
soft assignment to named map polygons:

```text
RGB pixel feature + predicted depth + camera pose
  -> world XYZ distribution
  -> affinity to every visible named polygon
  -> per-pixel landmark identity probability
  -> per-landmark evidence map
```

Point-cloud `world_xyz`, semantic class, landmark ID, validity, and occlusion are
training labels only. Runtime inputs remain RGB, source map, instruction, and
camera pose. A merged two-building mask may therefore be separated by projected
world position instead of requiring the 2D segmenter to split it first.

The binder must pass an oracle ladder before entering the belief updater:

1. static LBN prior and independent controller;
2. plus oracle point-cloud depth and full semantics;
3. plus oracle landmark ID, validity, and occlusion;
4. replace each oracle channel with the RGB model prediction.

All four variants use identical episode IDs. This identifies whether the next
bottleneck is geometry, RGB perception, or control/stop rather than attributing
all changes to one end-to-end number.
