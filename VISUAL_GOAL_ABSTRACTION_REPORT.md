# Visual Goal Abstraction / Minimal Sufficient Visual Template

## Status

**Pre-Gate blocked: calibrated UAV trajectory RGB is unavailable.** The offline goal-template phase completed. The Full RGB retrieval and every later metric are `NOT RUN`, not zero and not a failed retrieval. No controller, policy, diffusion model, or navigation rollout was trained or changed. `test_unseen` was not read.

The released inputs contain CityNav trajectories, CityRefer geometry, orthophoto RGB, and DSM. They do not contain authentic UAV camera observations. The previous external AirSim frame cache is not admissible: its prior renderer audit found a near-uniform direct-pose rendering, constant depth, and no validated CityRefer/map registration. An exploratory translation was explicitly uncalibrated. The 2.5-D orthophoto renderer also cannot serve as query RGB. Therefore none of these frames were used.

## Phase 1: goal templates

Built 100 fixed-hash sampled episodes per evaluation split, `val_seen` and `val_unseen`, at 40 m, 80 m, and 120 m. This produced 600 north-up orthophoto templates plus target and referenced-anchor contour masks. Target coordinates were used offline to center these reference crops. The templates contain real orthophoto pixels and CityRefer target/anchor vector contours; they are not oblique or FPV imagery.

| Split | Episodes | 40 m | 80 m | 120 m | Other views |
|---|---:|---:|---:|---:|---|
| val_seen | 100 | 100 | 100 | 100 | unavailable |
| val_unseen | 100 | 100 | 100 | 100 | unavailable |

Artifacts: `runs/visual_goal_abstraction_v1_complete/metadata/templates.jsonl` and `runs/visual_goal_abstraction_v1_complete/templates/`. Run outputs are gitignored by repository convention.

## A. Full RGB Gate

| Encoder | R@1 | R@5 | Same-map hard-negative accuracy | Margin | Status |
|---|---:|---:|---:|---:|---|
| Existing SigLIP | N/A | N/A | N/A | N/A | not run: no valid query images |
| Partial-tuned SigLIP2 | N/A | N/A | N/A | N/A | not run: no valid query images |
| DINOv2-small | N/A | N/A | N/A | N/A | not run: no valid query images |

The gate validator was run and blocked on the missing real-trajectory query manifest. No same-map accuracy, margin, AUROC, or similarity-vs-distance conclusion can be drawn. Thus the required question “can a current UAV RGB identify the goal scene?” remains unanswered. Do not interpret the data block as `Real visual place matching itself is unreliable.`

## B. Progressive abstraction L0–L9

Retrieval metrics for every level are N/A because Gate 1 did not produce a valid query set. The implementation describes the intended transformations; no level was scored.

| Level | Texture | Color | Background | Semantic | Geometry | Retrieval |
|---|---|---|---|---|---|---|
| L0 Full RGB | yes | yes | yes | implicit | yes | not run |
| L1 Background blur | local detail | yes | blurred | implicit | yes | not run |
| L2 Target + anchor | anchor detail | yes | removed | implicit | masks | not run |
| L3 Low frequency | removed | coarse | yes | implicit | coarse | not run |
| L4 Posterized | retained | quantized | yes | implicit | yes | not run |
| L5 Grayscale | retained | removed | yes | implicit | yes | not run |
| L6 Semantic regions | removed | class palette | coarse | yes | yes | not run |
| L7 Contour + color | removed | coarse | removed | partial | contours | not run |
| L8 Pure contour | removed | removed | removed | removed | contours/skeleton | not run |
| L9 Anchor + target geometry | removed | removed | removed | anchor/target only | relative footprint masks | not run |

L6–L8 require semantic-region and road-skeleton masks beyond the target/anchor contours available in this dataset. Those masks must come from a validated annotation or segmentation source before the corresponding level can be treated as an actual scene abstraction. They are not inferred from color or claimed as existing labels.

## C. Minimal sufficient representation

**Unknown.** There is no performance curve, so no minimum level can be selected and no 0.9×Full-RGB or −5 percentage-point criterion can be evaluated.

## D. Target, anchor, and context

**Not evaluated.** Target and referenced-anchor masks were created from CityRefer contours. A controlled Target-only / Anchor-only / Target+Anchor / +Context retrieval comparison requires valid query RGB and Gate 1 success. Context labels for roads, parking, vegetation, and open areas are also absent from the current template dataset.

## E. Cross-view robustness

**Unavailable.** North-up top-down orthophoto templates exist. High/low-altitude top-down trajectory images, calibrated oblique views, and authentic FPV query images do not. No rotation, perspective warp, or 2.5-D proxy is presented as real FPV.

## F. Belief Top-K reranking

**Skipped by the Gate 1 rule.** No belief Top-K visual reranking or Top-1 distance metric was computed.

## G. Oracle upper bound

**Unavailable.** Oracle A–D need aligned query observations and belief candidate sets. Computing an upper bound without those would not answer whether the current observation contains enough identifying information.

## Leakage and controls

- `test_unseen` is rejected by the dataset builder and query validator and was not opened.
- The query validator requires a calibrated renderer provenance, a 5-value trajectory pose, and `image_origin=uav_trajectory_pose`; it rejects candidate-centered crops and split overlap for the same episode/object.
- Evaluation encodes only image pixels. GT coordinates appear only in offline template construction and distance labels.
- `val_unseen` was used only as a fixed evaluation split; no tuning was done.
- Altitude, camera scale, brightness, and map-ID shortcut controls remain untested because no retrieval was run.

## CASE decision and next step

No scientific CASE A/B/C/D can be assigned before a valid Full RGB result exists. Operationally, stop the visual-imagination pipeline at the same boundary as CASE D: do not proceed to abstraction modeling, candidate reranking, or diffusion. This is a data/renderer block, **not evidence for CASE D's claim that real visual place matching fails**.

Next required input: an original CityFlight observation cache or a per-map calibrated Unreal/AirSim-to-orthophoto registration validated against RGB, depth, and/or segmentation overlays. Then build a small paired val_seen query audit; only after that passes should the Full RGB Gate run. `Instruction + Landmark Geometry → Imagined Template` is not justified yet.

## Reproduction

```bash
python scripts/build_visual_goal_dataset.py \
  --splits val_seen val_unseen --max-per-split 100 \
  --output runs/visual_goal_abstraction_v1_complete
python scripts/evaluate_visual_abstraction.py \
  --query-manifest <verified-trajectory-query.jsonl> \
  --query-root <query-image-root> \
  --templates runs/visual_goal_abstraction_v1_complete/metadata/templates.jsonl \
  --template-root runs/visual_goal_abstraction_v1_complete \
  --encoder <frozen-encoder-id> \
  --output runs/visual_goal_abstraction_v1_complete/results/abstraction.json
```

The second command validates query provenance before loading the image encoder. It is not an evaluation command until the validated query manifest exists.
