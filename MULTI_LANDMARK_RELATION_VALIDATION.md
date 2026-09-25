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

