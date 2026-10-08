# SBF-inspired relative geometry for HETT dense heatmap

Base: `experiment/heatmap-system-optimization-20e`. Do not merge into `develop` before an equal-protocol evaluation.

## Verified pre-change gap

The original controller consumes current `[sin(yaw), cos(yaw), normalized_x, normalized_y]` as a direction token, but its dense heatmap calls `CompactSpatialBelief(maps, emb_lang, lang_mask)`. Thus the dense belief branch does **not** directly consume the current pose or yaw. It uses only current/explored/global-landmark/referenced-landmark masks and language.

## Change

- A vectorized 13-channel candidate geometry grid contains east/north displacement, range, absolute bearing sin/cos and egocentric bearing sin/cos relative to the current UAV, plus reference-mask-centroid displacement, distance, bearing and reference validity.
- CityNav normalized y points south; yaw is in radians CCW from east. Correct the y sign before computing bearings.
- Reference centroid is estimated ONLY from the instruction-referenced landmark mask. Target/GT is never read by the geometry encoder.
- Add a trainable zero-initialized geometry gate before language cross-attention, preserving legacy heatmap outputs at initialization. Disable through `--no_heatmap_relative_geometry`.
- Default NMS proposals: 20 rather than 16. Append coverage@5 and coverage@20 without removing existing coverage@1/@4/@8/@16.
- This is NOT a full SBFNav recreation: the reference mask is reduced to one centroid, so multi-landmark pairwise relations are not fully modeled.

## Run (from multiagent directory)

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=10 BATCH=8 GRID=7 RUN_NAME=heatmap_geometry_10e bash train_1gpu.sh
```

Use a separate `RUN_NAME` for the map-only control with `--no_heatmap_relative_geometry`. Keep seed, episodes, steps, map_meters, kernel, threshold and train budget identical.

## Primary metrics

Report `coverage@5` and `coverage@20` at `success_dist` meters, separately for train_seen/val_seen/val_unseen. The current code logs these as valid **step-wise** coverage metrics, not episode-wise recall and not navigation SR. Add separate episode-level retrieval evaluation for a complete comparison with SBFNav; select the same GT definition, all/map-matched candidates, target radius and sampling protocol. Report NMS kernel and candidate diversity. Top-K metrics increase with K by construction; compare at equal K.

## Status

Code and unit tests are committed, but no CUDA training or local pytest execution was performed through this GitHub-only connector session. Model parity with SBFNav is an experiment target, not a confirmed result.
