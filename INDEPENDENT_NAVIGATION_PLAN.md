# Independent navigation plan and feasibility gates

This branch builds a navigation stack that does not call the HETT goal model,
candidate generator, policy, or action generator. HETT is retained only as a
frozen comparison baseline.

## Proposed system

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

## Modules

1. `LanguageMapPrior`: keeps each landmark separate and produces a static target
   heatmap from its polygon and language relation.
2. `GeoSemanticPerception`: segments RGB into building, vehicle, ground, parking,
   and later road; projects masks into world coordinates with pose and altitude.
3. `LandmarkAssociator`: assigns projected building components to named polygons,
   with geography weighted more strongly than appearance.
4. `BeliefUpdater`: fuses static relation evidence, temporal RGB evidence, and
   already-observed empty space in log-odds form.
5. `ViewPlanner`: maximizes expected uncertainty reduction minus travel cost;
   altitude is one of its actions rather than an always-on heuristic.
6. `GeometricGoalController`: converts the chosen observation point into 30-degree
   turns, 5-metre forward moves, vertical moves, and stop.
7. `StopController`: stops only when the 20-metre mass is concentrated, the
   estimate is stable across frames, and target-type and relation evidence agree.

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

The immediate next implementation is the persistent belief updater, not a larger
vision backbone. The current detector already finds most visible buildings; the
main loss is ambiguous component-to-name association and lack of temporal evidence.
