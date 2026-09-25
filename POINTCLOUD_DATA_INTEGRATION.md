# SensatUrban perspective data integration assessment

## Decision

Use the point cloud data as **geometry-aware auxiliary supervision**, not as a
replacement for simulator RGB. It can strengthen LBN's projector, semantic
encoder, landmark binder, temporal belief updater, and observation-quality
model. Low-altitude rendered RGB must not be treated as photorealistic training
data because missing facades and occlusions create large holes.

This pipeline remains independent of GeoNav and HETT models.

## Verified data coverage

The local SensatUrban PLY collection contains 38 blocks. Coverage against the
current CityNav splits is:

| split | map coverage | episode coverage | recorded frames |
|---|---:|---:|---:|
| train_seen | 24/24 | 21,878/21,878 | 1,922,122 |
| val_seen | 23/23 | 2,470/2,470 | 217,576 |
| val_unseen | 4/4 | 2,697/2,697 | 188,333 |
| test_unseen | 2/6 | 732/5,281 (13.86%) | 375,233 total |

Therefore it is suitable for training and val_unseen ablations, but cannot be
the only data source for a complete test_unseen claim.

The inspected Cambridge block 4 PLY contains XYZ, RGB, and semantic class for
each point. The 28-frame reproduction uses recorded pose and direction and does
not steer the camera with the target coordinate. However,
`target_in_camera_frustum_frames` proves only that the target point projects
inside the image rectangle; it does not test z-buffer occlusion or whether the
target has valid rendered pixels.

## New render outputs required

For every sampled viewpoint, render aligned tensors rather than only a filled
RGB image:

1. `rgb`: nearest visible point colour;
2. `semantic`: semantic class of the same z-buffer-selected point;
3. `depth`: camera depth of that point;
4. `valid`: whether a real point produced the pixel;
5. `world_xyz`: reconstructed from depth, intrinsics, and pose when needed;
6. `landmark_id`: map-polygon association computed from `world_xyz`;
7. camera intrinsics, pose, altitude, and source trajectory ID.

Invalid pixels must use an ignore mask. Sky-coloured holes and 3x3 dilated pixels
must never be supervised as real background or empty space.

For storage, render selected keyframes at the model's working resolution and
save RGB, float16 depth, uint8 semantic/valid masks, and uint16 landmark IDs.
Do not save a dense float32 XYZ image for every frame because XYZ is exactly
recoverable from depth and camera metadata.

No target marker is written into model input. Target coordinates may be used as
training labels after rendering, but may not change camera orientation, frame
selection, validity, candidate generation, or inference-time belief updates.

## How LBN should use the data

Keep the native 13-class SensatUrban label during teacher-data generation.
Runtime heads may group rare classes, but road, footpath, parking, vegetation,
wall/fence, and water must not be collapsed into Background before supervision.
Road and footpath topology directly supervise relations such as `across`,
`corner`, `along`, and `on the road`; vegetation and walls are useful occluders.

### 1. PoseProjector calibration

Project predicted pixels back to the map and compare them with the stored
`world_xyz`. This directly tests oblique-view projection error without relying
on target coordinates. It is the safest and highest-priority use.

### 2. ProjectSemanticNet pretraining

Train RGB-to-semantic prediction only on valid pixels, with depth and point
density used as confidence. Use medium/high-quality frames for the main loss;
retain sparse low views only for uncertainty calibration. Fine-tune later on
real or orthomosaic RGB to reduce the synthetic-view domain gap.

### 3. MapLandmarkBinder supervision

Use projected `world_xyz` and named map polygons to build positive and negative
visual-component/landmark pairs. Train position, overlap, size, and type terms;
appearance remains a residual term. This teaches oblique visual components to
bind to names without asking RGB to read landmark names.

Do not assign one identity to an entire connected component. Generate a
`landmark_id` for each valid projected point and train soft per-pixel assignment.
This lets a geometrically merged visual component contribute to several named
polygons while retaining a separate confidence for each.

### 4. Temporal LogOddsBelief training

Replay several frames from the same trajectory and require world-map semantic
predictions to be consistent. Add negative evidence only where `valid=1` and
the semantic prediction is confident. Holes contribute zero evidence.

### 5. Observation quality and height

Use valid-pixel ratio, point density, depth discontinuity, and semantic entropy
to predict whether an observation is useful. This can teach the planner that a
lower viewpoint is not automatically better. Do not train the height policy to
copy the recorded human altitude directly.

For active views, optimize candidate discrimination rather than generic image
quality. The point-cloud teacher can estimate how different the expected
observation would be for the top two goal candidates. A useful score is:

```text
candidate separation - travel cost - predicted invalid/occluded fraction
```

This directly asks whether a new position or altitude will resolve the current
ambiguity.

## Belief-map dimensionality

Use the point cloud as a 3D teacher but keep the online state as a 2.5D world
grid unless experiments prove voxels are necessary. Each XY cell stores semantic
probabilities, surface height/depth uncertainty, landmark identity, observed and
occluded state, observation confidence, and target probability. CityNav's
relations and 20-m success metric are primarily planar, so a full dense voxel map
would add substantial memory and learning complexity without a demonstrated gain.

## Paired simulator RGB requirement

The current rendered RGB and labels come from the same point cloud. They prove
that perspective supervision is useful, but they do not by themselves train a
model to transfer from photorealistic simulator RGB. Before claiming that
capability, capture simulator RGB at the exact same pose, direction, FOV, and
resolution and pair it with point-cloud depth/semantic/landmark labels. Reject
or down-weight pairs whose silhouettes or coordinate calibration disagree.

## What the data should not be used for

- replacing final simulator RGB evaluation;
- learning low-altitude texture or facade recognition from hole-filled images;
- treating an in-frustum target as a visible target;
- treating invalid pixels as observed empty space;
- evaluating autonomous viewpoint selection on the same human poses used as
  input replay—the replay evaluates perception and belief updates only;
- reporting complete test_unseen performance, because four of six blocks lack
  local PLY coverage.

## Minimal validation experiment

1. Select trajectory keyframes using movement, yaw change, and altitude bins,
   rather than rendering all highly correlated frames.
2. Train only on train_seen maps; select settings on val_seen; evaluate
   val_unseen once.
3. Compare four variants on identical frames:
   - current orthomosaic-only semantic model;
   - point-cloud semantic pretraining;
   - pretraining plus landmark-binding loss;
   - pretraining, binding, and temporal consistency.
4. Report valid-pixel semantic IoU, world projection error, named-landmark Top-1,
   belief-map Hit@20, and calibration versus altitude/validity.
5. Run wrong-pose, wrong-name, shuffled-time, and invalid-mask removal controls.

Only after point-cloud pretraining improves val_unseen belief-map Hit@20 should
it be connected to the closed-loop view planner. Navigation SR/SPL is a later
test and must not be attributed to this data before that ablation passes.
