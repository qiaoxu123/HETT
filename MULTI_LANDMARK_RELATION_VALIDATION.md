# Multi-landmark relation fusion validation

## Scope

This experiment changes only target-point selection. Each instruction landmark is
kept independently (up to nine), and a frozen relation ranker is fused with the
original HETT target logits:

`final_goal_logits = hett_goal_logits + alpha * hett_scale * relation_logits`

RGB semantic segmentation, road layers, candidate type matching, and dynamic
altitude are disabled. All comparisons load the same released HETT epoch-2
checkpoint. Target-point experiments use one rollout step so they measure target
selection rather than downstream motion.

## Alpha selection on val_seen

| alpha | overall Hit@20 | multi-landmark Hit@20 |
|---:|---:|---:|
| A / off | 19.35% | 19.07% |
| 0.10 | 18.30% | 22.85% |
| 0.25 | 22.91% | 28.37% |
| **0.50** | **24.74%** | **30.02%** |
| 1.00 | 20.36% | 26.00% |

`alpha=0.5` was selected before inspecting the val_unseen control results.

## Held-out val_unseen target selection

| experiment | overall Hit@20 | multi-landmark Hit@20 | goal NE |
|---|---:|---:|---:|
| A: current HETT | 10.86% | 14.14% | 55.46 m |
| B: separate landmarks, no relation | 16.76% | 21.17% | 80.75 m |
| C: separate landmarks + relation | **17.58%** | **24.33%** | 70.25 m |
| C with wrong relation | 16.98% | 22.51% | 70.49 m |
| C with wrong landmark | 9.01% | 12.32% | 79.87 m |

C improves multi-landmark Hit@20 by 3.16 points over B and 10.19 points over A.
Wrong relations reduce it by 1.82 points, while wrong landmarks reduce it by
12.01 points. The relation effect is real but smaller than the independent
landmark-geometry effect.

## Full val_unseen navigation

| experiment | SR | SPL | NE |
|---|---:|---:|---:|
| A: current HETT | 9.86% | 8.06% | **62.56 m** |
| B: separate landmarks, no relation | 9.75% | 8.12% | 85.91 m |
| C: separate landmarks + relation | **11.27%** | **9.49%** | 79.27 m |

C improves SR by 1.41 points and SPL by 1.43 points over A. However, mean NE is
still worse, showing that the branch creates more successful predictions but
also some large misses. This is a positive validation, not a finished model.

## Relation-only ablation

The strict `relation_only` mode removes the HETT target heatmap entirely and
scores all 400 map points using only per-landmark geometry and relation language.

| split | overall Hit@20 | multi-landmark Hit@20 | goal NE |
|---|---:|---:|---:|
| val_seen | 1.38% | 2.13% | 247.91 m |
| val_unseen | 0.56% | 0.95% | 226.17 m |

This is decisively worse than both HETT and fused prediction, so a full
navigation run is not warranted. The offline relation ranker was trained to
order a small set of plausible candidates; it is not a global target proposal
model. HETT remains necessary to provide the global location prior, while the
relation branch is useful as a residual correction.

## HETT-independent proposal generator

A separate supervised geometry generator creates five candidates directly from
instruction text and per-landmark centers/boundaries. It never reads HETT goal
logits. The frozen relation-ranker ensemble then selects one of those candidates.

| val_unseen target method | overall Hit@20 | multi-landmark Hit@20 | goal NE |
|---|---:|---:|---:|
| independent generator Top-1 | 20.76% | 28.44% | 45.47 m |
| independent Top-5 oracle | 50.76% | 62.80% | 29.63 m |
| **generator + relation ranker** | **24.29%** | **34.20%** | **44.17 m** |
| wrong relation control | 23.32% | 32.15% | 44.73 m |
| wrong landmark control | 18.50% | 25.43% | 48.75 m |

The proposal generator and ranker use no HETT candidates. Targets stored in the
offline evaluation JSON are read only after scores are fixed; the runtime goal
artifact contains episode IDs and predictions, not targets.

Using the independent prediction as the navigation goal while retaining HETT
only for movement, turning, and stopping gives:

| full val_unseen navigation | SR | SPL | final NE |
|---|---:|---:|---:|
| A: current HETT | 9.86% | 8.06% | 62.56 m |
| HETT + relation heatmap | 11.27% | 9.49% | 79.27 m |
| **independent goal + HETT executor** | **14.91%** | **13.56%** | **54.85 m** |

The independent run has stage-1 SR 23.99%, but final SR falls to 14.91% after
HETT fine navigation. Thus the next largest navigation loss is no longer global
proposal generation: it is movement away from a good predicted goal during the
fine-navigation/stop phase.

## Leakage and control audit

The runtime feature artifact contains only episode IDs, landmark centers,
landmark boundaries, valid masks, masked relation-language embeddings, and
structured relation cues. It contains no target coordinate, target identity,
RGB, success label, or trajectory outcome.

Wrong controls preserve map and landmark count and select a different tensor
whenever the group permits it. Of 27,045 episodes, 26,966 receive a different
relation tensor and 26,933 receive different landmark geometry. The unavoidable
unchanged cases are groups with no distinct same-map/count alternative.

## Result files

- Alpha sweep: `runs/alpha_val_seen_*`
- Held-out controls: `runs/control_v2_val_unseen_*` and
  `runs/control_val_unseen_{B_geometry,C_relation}`
- Full navigation: `runs/full_val_unseen_alpha050_*`
