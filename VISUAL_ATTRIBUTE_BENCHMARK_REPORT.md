# Visual Attribute Observability Benchmark for CityNav

## Executive result

This benchmark does not train a controller and does not predict a goal coordinate
from RGB. It measures whether explicit instruction attributes can be recovered
from UAV imagery and then used to re-rank Static B0 candidates.

The attribute task succeeds, but the current candidate-verification interface
does not. Partial-tuned SigLIP2 is the best attribute encoder; coarse semantic,
size, and roof presence are the three strongest retained attributes. Multi-view
fusion improves every tested attribute. However, explicit attribute scores do
not improve Static B0 on `val_unseen`: val-seen tuning selects zero visual weight
for every variant except context, and context slightly hurts unseen performance.
Controller integration is therefore not justified.

No `test_unseen` data was loaded or evaluated.

## 1. Census and dataset audit

The canonical existing source census records **32,326** released refined
instructions. Development policy forbids reading `test_unseen`, so the new
detailed taxonomy census covers the **27,045** permitted instructions:

| Split | Instructions | ≥1 lexical visual category | ≥2 |
|---|---:|---:|---:|
| train_seen | 21,878 | 99.91% | 98.86% |
| val_seen | 2,470 | 99.92% | 98.87% |
| val_unseen | 2,697 | 99.93% | 98.03% |

These are lexical coverage rates, not observability claims. Category presence is:

| Category | Instructions | Rate |
|---|---:|---:|
| color | 23,516 | 86.95% |
| size | 5,970 | 22.07% |
| shape | 2,139 | 7.91% |
| roof | 5,800 | 21.45% |
| context | 22,361 | 82.68% |
| semantic | 26,278 | 97.16% |

Using the three empirically strongest categories (coarse semantic, size, roof),
**98.17%** of permitted instructions contain at least one provisional reliable
attribute. This must not be read as 98.17% of instructions being visually solved:
the semantic head has only four metadata-supported coarse classes
(`Building/Car/Ground/Parking`), not landmark identity such as church or library.

The image dataset contains 4,900 target objects and 19,600 centered top-down
views at 20/40/60/80m. Object and episode overlap are exactly zero across all
three development splits. The height-distance extension contains 49,000 views.

Full counts, combinations, split distributions, and referenced-landmark
co-occurrence are in `VISUAL_ATTRIBUTE_CENSUS.md` and
`visual_attribute_stats.json`.

## 2. Frozen encoder comparison

Best linear/MLP `val_unseen` Macro-F1 on the same crop dataset:

| Encoder | Params | color | size | shape | coarse semantic | road context | roof |
|---|---:|---:|---:|---:|---:|---:|---:|
| HETT DarkNet | 41.1M | .346 | .681 | .397 | .692 | .583 | .666 |
| DINOv2-small | 22.1M | .343 | .670 | .418 | .664 | **.610** | .678 |
| existing SigLIP | 203.2M | .367 | .710 | **.448** | **.767** | .592 | **.692** |
| SigLIP2 Base | 375.2M | **.400** | **.719** | .429 | .759 | .591 | .684 |

Frozen existing SigLIP is strongest overall; SigLIP2 leads color and size, and
DINOv2 leads road context. Feature extraction over 19,600 crops took 19.4s,
22.9s, 37.5s, and 31.9s respectively. Peak allocated GPU memory was 3.53,
0.36, 1.01, and 1.57 GiB. The machine exposes one RTX 5090, so all GPU jobs were
run serially.

Zero-shot SigLIP/SigLIP2 is materially weaker and prompt-sensitive. Best
SigLIP2 zero-shot F1 is .332 color, .432 size, .156 shape, .217 semantic,
.407 context, and .461 roof. A trained head is necessary for this domain.

## 3. Partial fine-tuning and multi-seed stability

Only the last vision block and multi-task head were trained (7.29M trainable
parameters, 3 epochs). SigLIP2 is the best final model:

| Attribute | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---:|---:|---:|---:|
| color | .504 | .515 | .511 | .510 ± .005 |
| size | .754 | .763 | .760 | .759 ± .004 |
| shape | .462 | .490 | .499 | .484 ± .016 |
| coarse semantic | .810 | .831 | .805 | .815 ± .011 |
| road context | .601 | .593 | .593 | .596 ± .004 |
| roof presence | .737 | .733 | .739 | .737 ± .002 |

Seed-0 SigLIP2 training took 39.6s, used 1.81 GiB peak allocated GPU memory,
and has 375.4M total parameters. Fine-tuning is useful: every category improves
over frozen SigLIP2, most clearly color, semantic, and roof.

## 4. Height and distance

The available raster renderer produces genuine north-up top-down observations.
No local CityFlight/AirSim executable was found, so oblique and FPV results are
**not available**; no rotated orthophoto was used as a fake substitute.

Frozen SigLIP2 Macro-F1 on offset top-down views:

| Attribute | 20m high | 40m | 80m | 140m | 0–20m distance | 80m+ |
|---|---:|---:|---:|---:|---:|---:|
| color | .406 | .374 | .368 | .315 | .371 | .331 |
| size | .744 | .734 | .688 | .744 | .717 | .767* |
| shape | .392 | .416 | .419 | .421 | .419 | .426 |
| coarse semantic | .809 | .779 | .713 | .679 | .756 | .684 |
| road context | .572 | .581 | .589 | .588 | .588 | .600 |
| roof | .710 | .687 | .670 | .664 | .684 | .658 |

`*` Size benefits from the known-object contour crop and should not be treated as
pure appearance-at-distance evidence. Low top-down is best for color, semantic,
and roof; high top-down remains useful for footprint shape and broad context.

## 5. Whole image, crop, and segmentation

| Representation | color | size | shape | semantic | road context | roof |
|---|---:|---:|---:|---:|---:|---:|
| whole image | .253 | .492 | .311 | .534 | **.616** | .671 |
| contour crop | **.400** | **.719** | .429 | .759 | .591 | **.684** |
| oracle contour mask | .380 | .693 | **.600** | **.852** | .549 | .672 |

Oracle isolation strongly helps shape and semantic class but removes useful
context. It is an upper bound, not a learned segmentation result.

The paired SAM ViT-B experiment uses the same 500/200/200 objects. SAM's mean
IoU against the CityRefer contour is only .505 despite a mean self-reported mask
quality of .899. Compared with crop, SAM masking changes semantic `.707→.749`
and roof `.635→.680`, but color `.389→.359`, road context `.576→.514`, and shape
`.426→.422`. Learned segmentation is therefore selectively useful, not a
significant general improvement.

## 6. Multi-view and active-view upper bound

| Attribute | single observations | mean fusion | max-confidence | oracle active view |
|---|---:|---:|---:|---:|
| color | .353 | .424 | .392 | .754 |
| size | .726 | .814 | .782 | .996 |
| shape | .417 | .482 | .424 | .677 |
| coarse semantic | .726 | .760 | .783 | .962 |
| road context | .586 | .611 | .635 | .907 |
| roof | .675 | .689 | .720 | .951 |

Multi-view is consistently better than single-view. The large oracle gap makes
offline Active Perception promising, especially for color, shape, context, and
semantic evidence. This is an upper bound only; no controller was trained.

## 7. Provisional observability

The requested score combines multi-seed partial-tuned unseen F1 and calibration
with frozen-model cross-observation/cross-height agreement. It is therefore a
top-down provisional score, not a complete cross-pitch/FPV score:

| Attribute | Provisional score | val_unseen F1 | Interpretation |
|---|---:|---:|---|
| coarse semantic | .849 | .815 | reliable for four coarse classes |
| size | .780 | .759 | reliable, but geometry/crop leakage matters |
| roof presence | .776 | .737 | reliable coarse presence, not roof type |
| road context | .736 | .596 | stable but only moderate discrimination |
| shape | .644 | .484 | consistent yet weakly discriminative |
| color | .572 | .510 | view/distance sensitive |

Fine semantic landmark identity (church/library/college/etc.) is not validated;
metadata did not provide a clean target label for that taxonomy.

## 8. Candidate verification against Static B0

The evaluation takes the existing seed-0 referenced-landmark B0 Top-16, maps
each cell to world coordinates, finds the nearest CityRefer object, renders its
top-down crop, predicts attributes, and tunes a non-negative explicit weight on
`val_seen` only. The unmodified cache reproduces:

| val_unseen | R@1/20 | R@4/20 | R@8/20 | R@16/20 | R@8/40 | R@16/40 | Top-1 distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static B0 | 21.95% | 48.20% | 67.85% | 81.46% | 89.14% | 94.96% | 49.17m |

Color, size, shape, semantic, all-visual, geometry, visual+geometry, and every
leave-one-out variant select `lambda=0` on val_seen and exactly reproduce B0.
Context selects `lambda=.25`, but unseen R@1/20 and R@4/20 fall to 21.54% and
47.68%; Top-1 distance rises to 49.40m. Thus visual attribute verification does
**not** significantly improve Static B0 in the current candidate-to-object
grounding. Likely failure sources are nearest-object assignment error, weak
language labels, and using orthophoto crops instead of actual candidate views.

## 9. Answers to the research questions

- **A. Reliable coverage:** 98.17% contain at least one of the provisional top
  three categories, but this is dominated by coarse semantic words and is not a
  solved-instruction rate.
- **B. Three most stable:** coarse semantic, size, roof presence.
- **C. Reliable on unseen:** the same three; road context is stable but only
  moderate, color and shape are not yet reliable.
- **D. View roles:** low top-down favors color/semantic/roof; high top-down favors
  footprint and broad context. Oblique/FPV are untested because the renderer is
  unavailable.
- **E. Best model:** partial-tuned SigLIP2. Frozen existing SigLIP is the strongest
  frozen model overall; DINOv2 leads only road context; hand-crafted features are
  competitive for geometry but weaker overall.
- **F. Fine-tuning:** yes, partial last-block tuning is consistently beneficial;
  full fine-tuning is unnecessary for the observed gains.
- **G. Segmentation:** oracle isolation helps shape/semantic greatly; actual SAM
  gives only selective modest gains and harms context/color.
- **H. Multi-view:** yes, mean/max fusion improves every category.
- **I. Static B0:** no significant gain; most tuned weights collapse to zero.
- **J. Keep three:** coarse semantic, size, roof presence, with the stated label
  and geometry caveats.
- **K. Active Perception:** worth an offline next step because oracle view choice
  is much stronger than fixed/naive fusion, but not yet controller integration.

## 10. Artifacts and reproducibility

Each phase has a standalone `metrics.json`, `commands.json`, source snapshot,
git diff, status, and telemetry under:

```text
/mnt/windows-data/hett-visual-attributes/
  phase1_handcrafted_s0_20261004_r3/
  phase2_frozen_s0_20261004/
  phase2_zeroshot_20261004/
  phase3_finetune_s0_20261004/
  phase4_topdown_view_distance_s0_20261004/
  phase5_isolation_s0_20261004/
  phase5_sam_paired_s0_20261004/
  phase6_multiview_s0_20261004/
  phase7_candidate_verification_s0_20261004_r2/
  phase8_siglip2_s1_20261004/
  phase8_siglip2_s2_20261004/
```

The benchmark used the isolated Transformers 4.57.1 dependency directory at
`/mnt/windows-data/hett-visual-attributes/pydeps`; the HETT environment itself was
not replaced. No navigation rollout, action prediction, controller training, or
`test_unseen` evaluation was performed.
