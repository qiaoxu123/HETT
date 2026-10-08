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
- **Updated:** with `--heatmap_multi_landmark` enabled (default), HETT now preserves each instruction-mentioned landmark independently and uses name-aligned BERT contextual vectors plus a cross-landmark relation head. The old anonymous reference centroid is disabled in this mode, retaining only UAV pose geometry in the 13-channel grid.
- The relation head models candidate-to-each-landmark (east/north offset, range, bearing, contour extent) and exchanges evidence among all named landmarks using multihead self-attention. Joint mean and soft-min aggregate multiple constraints.
- This is still a new learned architecture inspired by SBFNav, not an exact reproduction of SBFNav's implementation or published metrics.
- Dataset annotations use existing `description_landmarks` and the matched landmark contours from `LandmarkNavMap`; neither goal ground truth nor future observations are fed to inference.

## Multi-landmark implementation

- `multiagent/env.py` attaches independent `reference_landmarks` name, normalized contour center and bounding-box extent to each observation.
- `multiagent/models/multi_landmark.py` aligns each name to an exact BERT token subsequence in the instruction, without another language backbone pass. Unmatched names retain geometry but use global language context, and the match rate is logged as `landmark_refs_name_matched` / `landmark_refs_total`. Exceeding `--heatmap_max_landmarks` (default 16) is logged via `landmark_refs_truncated`.
- Train and val/test use precisely the same anchor construction. The per-episode alignment is cached once per rollout, unlike the current UAV pose.
- `--no_heatmap_multi_landmark` disables joint relation scoring (original centroid geometry remains), and `--no_heatmap_relative_geometry` separately disables the UAV 13-channel pose geometry. Use a fresh run directory for each ablation.
- Old model checkpoint parameter loading uses the existing missing-key-tolerant loader. When loading an old checkpoint do **not** use `--resume_optimizer` because the added head changes optimizer parameter groups.

## Run (from multiagent directory)

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=10 BATCH=8 GRID=7 RUN_NAME=heatmap_geometry_10e bash train_1gpu.sh
```

Use a separate `RUN_NAME` for the map-only control with `--no_heatmap_relative_geometry`. Keep seed, episodes, steps, map_meters, kernel, threshold and train budget identical.

## Primary metrics

Report `coverage@5` and `coverage@20` at `success_dist` meters, separately for train_seen/val_seen/val_unseen. The current code logs these as valid **step-wise** coverage metrics, not episode-wise recall and not navigation SR. Add separate episode-level retrieval evaluation for a complete comparison with SBFNav; select the same GT definition, all/map-matched candidates, target radius and sampling protocol. Report NMS kernel and candidate diversity. Top-K metrics increase with K by construction; compare at equal K.

## Status

Code and unit tests are committed, but no CUDA training or local pytest execution was performed through this GitHub-only connector session. Model parity with SBFNav is an experiment target, not a confirmed result.


## Verify on the GPU host

```bash
python -m unittest tests.test_multi_landmark_relations tests.test_heatmap_relative_geometry tests.test_dense_spatial_belief -v
```

Compare (1) map-only, (2) UAV-relative geometry with one aggregate centroid, and (3) UAV-relative geometry + all named landmarks using identical seeds, samples, and train epochs. A zero-initialized `multi_landmark_gate` means new head parameters enter the loss gradually; checkpoint the gate and verify gradients. Check name token match/truncation counters before treating results as a full semantic reasoning test.
