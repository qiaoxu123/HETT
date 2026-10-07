# Explicit layout Gate G

**Gate G: FAIL under the specified conjunction.** The failure is a coverage and grounding limitation, not proof that all layout language lacks information. No visual fusion was attempted.

Full corpus: 27,045 instructions; explicit layout 3,506 (13.0%); explicit frame cue 3,117 (11.5%); both 1,535 (5.7%). True target-type counts for layout instructions: Building 278, Car 3,225, Ground 2, Parking 1.
Group A/B/C/other text counts: 1,535/1,971/19,834/3,705. Group C has generic relations but no explicit layout/frame and is not scored by L0–L8. On val_unseen Group A has only 19 detected instructions; the evaluable L3 count is shown below.

## Ablations: val seen

| Arm | n | Top1 | Top4 | MRR | Pairwise AUC | Mean rank |
|---|---:|---:|---:|---:|---:|---:|
| L0 Anchor prior | 42 | 0.024 | 0.119 | 0.097 | 0.402 | 64.833 |
| L1 Global layout | 42 | 0.054 | 0.147 | 0.117 | 0.457 | 61.774 |
| L2 PCA layout | 42 | 0.012 | 0.063 | 0.063 | 0.386 | 60.464 |
| L3 Instruction frame | 30 | 0.013 | 0.055 | 0.061 | 0.630 | 63.083 |
| L4 Instruction + anchor | 30 | 0.000 | 0.000 | 0.030 | 0.415 | 77.833 |
| L5 Soft ordinal | 30 | 0.039 | 0.124 | 0.112 | 0.633 | 49.717 |
| L6 Shuffled index | 30 | 0.067 | 0.167 | 0.135 | 0.637 | 49.367 |
| L7 Flipped frame | 30 | 0.000 | 0.033 | 0.035 | 0.289 | 90.133 |
| L8 Shuffled reference | 30 | 0.020 | 0.047 | 0.070 | 0.694 | 40.783 |

## Ablations: val unseen

| Arm | n | Top1 | Top4 | MRR | Pairwise AUC | Mean rank |
|---|---:|---:|---:|---:|---:|---:|
| L0 Anchor prior | 12 | 0.250 | 0.500 | 0.422 | 0.626 | 5.000 |
| L1 Global layout | 12 | 0.208 | 0.521 | 0.372 | 0.623 | 5.708 |
| L2 PCA layout | 12 | 0.104 | 0.515 | 0.306 | 0.589 | 5.333 |
| L3 Instruction frame | 0 | — | — | — | — | — |
| L4 Instruction + anchor | 0 | — | — | — | — | — |
| L5 Soft ordinal | 0 | — | — | — | — | — |
| L6 Shuffled index | 0 | — | — | — | — | — |
| L7 Flipped frame | 0 | — | — | — | — | — |
| L8 Shuffled reference | 0 | — | — | — | — | — |

The rows above have different coverage. Geometry scores use the rule parser; DeepSeek is a separate consistency diagnostic. Only the following matched-pair tests support direct arm comparisons. All p-values are one-sided exact sign tests over episode clauses; no val_unseen result was used to select a frame, threshold, or weight.

| Split | Pair | n | Wins / losses / ties | Mean AUC Δ | p |
|---|---|---:|---:|---:|---:|
| val_seen | L3 vs L1 | 30 | 19/10/1 | 0.196 | 0.068 |
| val_seen | L3 vs shuffled ordinal | 30 | 5/10/15 | -0.007 | 0.941 |
| val_seen | L3 vs flipped frame | 30 | 22/8/0 | 0.341 | 0.008 |
| val_seen | L3 vs shuffled reference | 30 | 15/15/0 | -0.064 | 0.572 |
| val_unseen | L3 vs L1 | 0 | 0/0/0 | — | — |
| val_unseen | L3 vs shuffled ordinal | 0 | 0/0/0 | — | — |
| val_unseen | L3 vs flipped frame | 0 | 0/0/0 | — | — |
| val_unseen | L3 vs shuffled reference | 0 | 0/0/0 | — | — |

Behavioral controls over 236 resolved-frame rows: index changes scores in 236, 180° frame flip in 236, reference replacement in 236. These are behavior checks, not significance tests.

Gate criteria fail because: (1) val_unseen has only 0 resolved Group A L3 parking rows, so stable transfer cannot be established; (2) ordinal shuffle does not consistently lose on val_seen; (3) two high-frequency families cannot be validated on unseen hard negatives; (4) DeepSeek parser exact agreement is low; (5) most candidate scopes are ambiguous or miss GT. The primary failure categories are parser instability, frame sign ambiguity, object grouping/scope binding, ordinal start ambiguity, and insufficient unseen coverage.

The parking candidate set is same-lot and includes row neighbors. Road first/last/corner clauses abstain when the start end is unbound. These abstentions prevent a misleading oracle-selected success.
