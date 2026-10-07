# Spatial canonicalization Gate F

## Decision
**FAIL** under the required conjunction. The measured geometry now includes trajectory windows, named RoadRegions, and explicit second anchors; failure is no longer attributed to those three columns being absent.

## Input and parser
- Legacy diagnostic samples: 2,290. Extended clause rows: 6,230 from 2,416 unique episodes; evaluable rows: 3,263.
- Full trajectory: 6,230/6,230. Named RoadRegion hypothesis: 2,536/6,230. Explicit bound second-anchor between rows: 264 (annotation path 117).
- DeepSeek paired sample: 108 instructions; valid A/B 100.0%/100.0%; exact agreement 40.7%; target agreement 80.6%; clause Jaccard 0.656; UNKNOWN 1.9%. Its 287 clauses have a separate geometry path (180 bound; 27 canonical UNKNOWN).

## Gate conditions

| Condition | Met |
|---|---|
| three_classes_over_shuffle | False |
| two_classes_over_opposite | False |
| probabilistic_over_old | False |
| no_systemic_sign_flip | False |
| view_dependent_stable_or_unknown | False |

## F0–F4 comparison

F0 executes the existing hard geometry ontology for near and between; unsupported words receive a neutral score. The learned old teacher/DeepSeek pipeline is not reproduced on these exact clause rows. F1 uses one phrase-level program; F2 adds anchor context; F3 marginalizes programs; F4 applies val_seen reliability. Protocols have different negative difficulty and are shown separately.

| Split | Protocol | Variant | n | Top-1 | Top-4 | MRR | AUC | Margin |
|---|---|---|---|---|---|---|---|---|
| train_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F0 | 146 | 0.100 | 0.400 | 0.293 | 0.500 | 0.000 |
| train_seen | legacy_hard_negative | F0 | 1453 | 0.189 | 0.630 | 0.416 | 0.567 | 14.657 |
| train_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F1 | 146 | 0.481 | 0.856 | 0.652 | 0.834 | 31.182 |
| train_seen | legacy_hard_negative | F1 | 1236 | 0.236 | 0.717 | 0.461 | 0.630 | 23.814 |
| train_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F2 | 146 | 0.481 | 0.856 | 0.652 | 0.834 | 31.182 |
| train_seen | legacy_hard_negative | F2 | 1415 | 0.217 | 0.696 | 0.444 | 0.619 | 24.502 |
| train_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F3 | 146 | 0.478 | 0.824 | 0.640 | 0.818 | 1.280 |
| train_seen | legacy_hard_negative | F3 | 1453 | 0.197 | 0.635 | 0.418 | 0.572 | 0.217 |
| train_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F4 | 146 | 0.478 | 0.824 | 0.640 | 0.818 | 0.969 |
| train_seen | legacy_hard_negative | F4 | 1453 | 0.203 | 0.628 | 0.421 | 0.571 | 0.061 |
| val_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F0 | 129 | 0.100 | 0.400 | 0.293 | 0.500 | 0.000 |
| val_seen | legacy_hard_negative | F0 | 866 | 0.192 | 0.616 | 0.411 | 0.555 | 19.554 |
| val_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F1 | 129 | 0.515 | 0.929 | 0.693 | 0.884 | 35.694 |
| val_seen | legacy_hard_negative | F1 | 754 | 0.208 | 0.708 | 0.443 | 0.624 | 29.575 |
| val_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F2 | 129 | 0.515 | 0.929 | 0.693 | 0.884 | 35.694 |
| val_seen | legacy_hard_negative | F2 | 845 | 0.202 | 0.724 | 0.445 | 0.634 | 31.107 |
| val_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F3 | 129 | 0.498 | 0.924 | 0.683 | 0.879 | 1.417 |
| val_seen | legacy_hard_negative | F3 | 866 | 0.159 | 0.608 | 0.388 | 0.548 | 0.112 |
| val_seen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F4 | 129 | 0.498 | 0.924 | 0.683 | 0.879 | 1.081 |
| val_seen | legacy_hard_negative | F4 | 866 | 0.159 | 0.620 | 0.391 | 0.554 | 0.039 |
| val_unseen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F0 | 126 | 0.100 | 0.400 | 0.293 | 0.500 | 0.000 |
| val_unseen | legacy_hard_negative | F0 | 489 | 0.200 | 0.659 | 0.433 | 0.597 | 33.884 |
| val_unseen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F1 | 126 | 0.401 | 0.835 | 0.607 | 0.828 | 26.652 |
| val_unseen | legacy_hard_negative | F1 | 390 | 0.221 | 0.790 | 0.475 | 0.674 | 52.879 |
| val_unseen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F2 | 126 | 0.401 | 0.835 | 0.607 | 0.828 | 26.652 |
| val_unseen | legacy_hard_negative | F2 | 480 | 0.189 | 0.737 | 0.442 | 0.642 | 49.408 |
| val_unseen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F3 | 126 | 0.393 | 0.841 | 0.604 | 0.825 | 1.231 |
| val_unseen | legacy_hard_negative | F3 | 489 | 0.193 | 0.702 | 0.434 | 0.622 | 0.270 |
| val_unseen | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F4 | 126 | 0.393 | 0.841 | 0.604 | 0.825 | 0.951 |
| val_unseen | legacy_hard_negative | F4 | 489 | 0.195 | 0.692 | 0.432 | 0.613 | 0.074 |

## Paired variant comparisons on val_unseen

Each row uses identical candidate sets for both variants; the sign test clusters steps by episode.

| Source | Protocol | Variant vs F0 | n clusters | W/L/T | AUC Δ | p |
|---|---|---|---|---|---|---|
| annotation | legacy_hard_negative | F1 | 184 | 70/43/71 | 0.042 | 0.007 |
| annotation | legacy_hard_negative | F2 | 218 | 84/56/78 | 0.034 | 0.011 |
| annotation | legacy_hard_negative | F3 | 221 | 85/68/68 | 0.019 | 0.098 |
| annotation | legacy_hard_negative | F4 | 221 | 80/68/73 | 0.009 | 0.183 |
| parser | legacy_hard_negative | F1 | 167 | 68/28/71 | 0.068 | 2.73e-05 |
| parser | legacy_hard_negative | F2 | 215 | 92/52/71 | 0.049 | 5.41e-04 |
| parser | legacy_hard_negative | F3 | 220 | 84/69/67 | 0.020 | 0.129 |
| parser | legacy_hard_negative | F4 | 220 | 75/62/83 | 0.014 | 0.153 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F1 | 125 | 115/9/1 | 0.328 | 7.23e-25 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F2 | 125 | 115/9/1 | 0.328 | 7.23e-25 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F3 | 125 | 114/11/0 | 0.326 | 4.82e-23 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | F4 | 125 | 114/11/0 | 0.326 | 4.82e-23 |
| deepseek | legacy_hard_negative | F1 | 23 | 10/7/6 | 0.086 | 0.315 |
| deepseek | legacy_hard_negative | F2 | 23 | 10/7/6 | 0.086 | 0.315 |
| deepseek | legacy_hard_negative | F3 | 23 | 11/6/6 | 0.112 | 0.166 |
| deepseek | legacy_hard_negative | F4 | 23 | 8/4/11 | 0.086 | 0.194 |

| Source | Protocol | n clusters | F2 over F1 W/L/T | AUC Δ | p |
|---|---|---|---|---|---|
| annotation | legacy_hard_negative | 184 | 16/16/152 | 0.005 | 0.570 |
| parser | legacy_hard_negative | 166 | 8/6/152 | 0.005 | 0.395 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | 125 | 0/0/125 | 0.000 | — |
| deepseek | legacy_hard_negative | 23 | 0/0/23 | 0.000 | — |

## Controls

Paired p-values use a one-sided sign test over episode clusters. Shuffled anchors use a deterministic unrelated same-kind map entity. These controls test geometry discrimination but are sensitive to the quality of binding.

| Source | Protocol | Phrase | Control | n clusters | W/L/T | AUC Δ | p |
|---|---|---|---|---|---|---|---|
| annotation | legacy_hard_negative | next to | shuffled_relation | 32 | 27/2/3 | 0.377 | 8.12e-07 |
| annotation | legacy_hard_negative | next to | opposite_relation | 32 | 29/1/2 | 0.686 | 2.89e-08 |
| annotation | legacy_hard_negative | next to | shuffled_anchor | 32 | 27/3/2 | 0.418 | 4.22e-06 |
| parser | legacy_hard_negative | next to | shuffled_relation | 30 | 25/2/3 | 0.369 | 2.82e-06 |
| parser | legacy_hard_negative | next to | opposite_relation | 30 | 27/1/2 | 0.693 | 1.08e-07 |
| parser | legacy_hard_negative | next to | shuffled_anchor | 30 | 23/2/5 | 0.413 | 9.72e-06 |
| annotation | legacy_hard_negative | on | shuffled_relation | 67 | 26/11/30 | 0.094 | 0.010 |
| annotation | legacy_hard_negative | on | shuffled_anchor | 67 | 44/12/11 | 0.129 | 1.04e-05 |
| parser | legacy_hard_negative | on | shuffled_relation | 68 | 27/12/29 | 0.109 | 0.012 |
| parser | legacy_hard_negative | on | shuffled_anchor | 68 | 44/13/11 | 0.129 | 2.36e-05 |
| annotation | legacy_hard_negative | left of | shuffled_relation | 14 | 5/6/3 | -0.038 | 0.726 |
| annotation | legacy_hard_negative | left of | opposite_relation | 14 | 6/5/3 | 0.083 | 0.500 |
| annotation | legacy_hard_negative | left of | shuffled_anchor | 14 | 4/0/10 | 0.048 | 0.062 |
| parser | legacy_hard_negative | left of | shuffled_relation | 13 | 3/8/2 | -0.150 | 0.967 |
| parser | legacy_hard_negative | left of | opposite_relation | 13 | 7/3/3 | 0.205 | 0.172 |
| parser | legacy_hard_negative | left of | shuffled_anchor | 13 | 0/0/13 | 0.000 | — |
| annotation | legacy_hard_negative | behind | shuffled_relation | 37 | 14/20/3 | 0.013 | 0.885 |
| annotation | legacy_hard_negative | behind | opposite_relation | 37 | 12/20/5 | -0.138 | 0.945 |
| annotation | legacy_hard_negative | behind | shuffled_anchor | 37 | 13/19/5 | -0.017 | 0.892 |
| parser | legacy_hard_negative | behind | shuffled_relation | 36 | 15/19/2 | 0.034 | 0.804 |
| parser | legacy_hard_negative | behind | opposite_relation | 36 | 15/18/3 | -0.059 | 0.757 |
| parser | legacy_hard_negative | behind | shuffled_anchor | 35 | 15/15/5 | 0.035 | 0.572 |
| annotation | legacy_hard_negative | between | shuffled_relation | 24 | 4/4/16 | -0.005 | 0.637 |
| annotation | legacy_hard_negative | between | shuffled_anchor | 24 | 15/1/8 | 0.216 | 2.59e-04 |
| annotation | legacy_hard_negative | between | shuffled_second_anchor | 24 | 9/4/11 | 0.131 | 0.133 |
| parser | legacy_hard_negative | between | shuffled_relation | 27 | 7/3/17 | 0.035 | 0.172 |
| parser | legacy_hard_negative | between | shuffled_anchor | 27 | 19/1/7 | 0.252 | 2.00e-05 |
| parser | legacy_hard_negative | between | shuffled_second_anchor | 27 | 11/3/13 | 0.187 | 0.029 |
| annotation | legacy_hard_negative | near | shuffled_relation | 6 | 6/0/0 | 0.454 | 0.016 |
| annotation | legacy_hard_negative | near | opposite_relation | 6 | 6/0/0 | 0.611 | 0.016 |
| annotation | legacy_hard_negative | near | shuffled_anchor | 6 | 5/0/1 | 0.370 | 0.031 |
| annotation | legacy_hard_negative | right of | shuffled_relation | 16 | 3/8/5 | -0.163 | 0.967 |
| annotation | legacy_hard_negative | right of | opposite_relation | 16 | 7/5/4 | 0.076 | 0.387 |
| annotation | legacy_hard_negative | right of | shuffled_anchor | 16 | 4/0/12 | 0.042 | 0.062 |
| parser | legacy_hard_negative | right of | shuffled_relation | 15 | 3/8/4 | -0.163 | 0.967 |
| parser | legacy_hard_negative | right of | opposite_relation | 15 | 8/3/4 | 0.119 | 0.113 |
| parser | legacy_hard_negative | right of | shuffled_anchor | 15 | 3/0/12 | 0.033 | 0.125 |
| annotation | legacy_hard_negative | in front of | shuffled_relation | 14 | 6/4/4 | 0.034 | 0.377 |
| annotation | legacy_hard_negative | in front of | opposite_relation | 14 | 5/6/3 | -0.048 | 0.726 |
| annotation | legacy_hard_negative | in front of | shuffled_anchor | 14 | 0/0/14 | 0.000 | — |
| parser | legacy_hard_negative | in front of | shuffled_relation | 13 | 5/4/4 | 0.017 | 0.500 |
| parser | legacy_hard_negative | in front of | opposite_relation | 13 | 5/7/1 | -0.103 | 0.806 |
| parser | legacy_hard_negative | in front of | shuffled_anchor | 13 | 0/0/13 | 0.000 | — |
| parser | legacy_hard_negative | near | shuffled_relation | 5 | 5/0/0 | 0.444 | 0.031 |
| parser | legacy_hard_negative | near | opposite_relation | 5 | 5/0/0 | 0.556 | 0.031 |
| parser | legacy_hard_negative | near | shuffled_anchor | 5 | 4/0/1 | 0.311 | 0.062 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | off | shuffled_relation | 40 | 13/19/8 | 0.058 | 0.892 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | off | shuffled_anchor | 40 | 30/6/4 | 0.325 | 3.48e-05 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | along | shuffled_relation | 40 | 23/6/11 | 0.292 | 0.001 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | along | shuffled_anchor | 40 | 33/4/3 | 0.343 | 5.42e-07 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | on | shuffled_relation | 39 | 24/8/7 | 0.383 | 0.004 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | on | shuffled_anchor | 39 | 29/5/5 | 0.323 | 1.93e-05 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | across | shuffled_relation | 6 | 2/3/1 | -0.019 | 0.812 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | across | shuffled_anchor | 6 | 4/1/1 | 0.324 | 0.188 |
| raw_parser | 9 deterministic map distractors plus target; separate from legacy hard-negative protocol | across | shuffled_second_anchor | 6 | 1/3/2 | 0.056 | 0.938 |
| deepseek | legacy_hard_negative | behind | shuffled_relation | 5 | 4/1/0 | 0.275 | 0.188 |
| deepseek | legacy_hard_negative | behind | opposite_relation | 5 | 4/1/0 | 0.567 | 0.188 |
| deepseek | legacy_hard_negative | behind | shuffled_anchor | 5 | 2/0/3 | 0.067 | 0.250 |
| deepseek | legacy_hard_negative | off | shuffled_relation | 3 | 1/1/1 | 0.111 | 0.750 |
| deepseek | legacy_hard_negative | off | shuffled_anchor | 3 | 1/2/0 | 0.046 | 0.875 |
| deepseek | legacy_hard_negative | in front of | shuffled_relation | 2 | 1/1/0 | 0.111 | 0.750 |
| deepseek | legacy_hard_negative | in front of | opposite_relation | 2 | 1/1/0 | 0.000 | 0.750 |
| deepseek | legacy_hard_negative | in front of | shuffled_anchor | 2 | 0/0/2 | 0.000 | — |
| deepseek | legacy_hard_negative | on | shuffled_relation | 4 | 3/0/1 | 0.375 | 0.125 |
| deepseek | legacy_hard_negative | on | shuffled_anchor | 4 | 4/0/0 | 0.354 | 0.062 |
| deepseek | legacy_hard_negative | next to | shuffled_relation | 4 | 3/0/1 | 0.312 | 0.125 |
| deepseek | legacy_hard_negative | next to | opposite_relation | 4 | 4/0/0 | 0.792 | 0.062 |
| deepseek | legacy_hard_negative | next to | shuffled_anchor | 4 | 3/1/0 | 0.479 | 0.312 |

## Failure attribution

Frame ambiguity and language semantics instability are the dominant failures. The true ternary between program is not recoverable beyond the pairwise control with the current anchor geometry. Anchor resolution remains incomplete; road geometry is available for named regions, while across-road interpretation often lacks an independent second reference.

## Interpretation

- Behind/front and left/right show val_seen-to-val_unseen sign flips in key contexts; no stable common reference frame is established.
- On-road association has evidence above chance, but the road program family does not establish the full cross-relation gate. Off/along use a separately sampled candidate protocol.
- True ternary between does not beat the old pairwise interpretation on val_unseen. Its shuffled-second control is weaker, which shows the second anchor carries information without establishing the chosen ternary score as best.
- F3 gains over F0 need to be read by covered and uncovered relation separately; unsupported F0 words are neutral by definition.

## Leakage and split control

Text-only parser requests contain no target IDs, positions, candidate ranks or answer fields. Graph binding retains all name hypotheses. GT target type and index are read only by the offline evaluator for same-class candidate filtering and metrics. Full trajectory is diagnostic; final approach would not be available at instruction time. Program choice, probabilities and rho use train_seen plus val_seen only.

## One next step

Stop the fine-grained relation line and archive these diagnostics, including the limited proximity and named-road association signals.
