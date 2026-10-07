# Between: true ternary relation

Only annotation clauses with explicit `between A and B` and both bound anchors enter B0–B4. Annotation names are offline oracle evidence. B3 replaces B with a deterministic unrelated same-kind map entity; B4 uses proximity to A.


## train_seen
n = 60

| Control | Pairwise AUC | Top-1 | MRR |
|---|---|---|---|
| B0_distance | 0.720 | 0.217 | 0.466 |
| B1_pairwise | 0.789 | 0.367 | 0.572 |
| B2_ternary | 0.730 | 0.283 | 0.508 |
| B3_shuffle_second | 0.642 | 0.119 | 0.367 |
| B4_shuffle_relation | 0.720 | 0.217 | 0.466 |

| B2 versus | Clustered wins/losses/ties | Mean AUC difference | One-sided paired p |
|---|---|---|---|
| B0_distance | 22/21/9 | -0.003 | 0.500 |
| B1_pairwise | 9/21/22 | -0.057 | 0.992 |
| B3_shuffle_second | 27/11/13 | 0.085 | 0.007 |
| B4_shuffle_relation | 22/21/9 | -0.003 | 0.500 |

## val_seen
n = 31

| Control | Pairwise AUC | Top-1 | MRR |
|---|---|---|---|
| B0_distance | 0.758 | 0.194 | 0.482 |
| B1_pairwise | 0.838 | 0.452 | 0.651 |
| B2_ternary | 0.828 | 0.323 | 0.585 |
| B3_shuffle_second | 0.551 | 0.032 | 0.292 |
| B4_shuffle_relation | 0.758 | 0.194 | 0.482 |

| B2 versus | Clustered wins/losses/ties | Mean AUC difference | One-sided paired p |
|---|---|---|---|
| B0_distance | 14/4/12 | 0.074 | 0.015 |
| B1_pairwise | 2/5/23 | -0.010 | 0.938 |
| B3_shuffle_second | 25/2/3 | 0.282 | 2.82e-06 |
| B4_shuffle_relation | 14/4/12 | 0.074 | 0.015 |

## val_unseen
n = 26

| Control | Pairwise AUC | Top-1 | MRR |
|---|---|---|---|
| B0_distance | 0.738 | 0.192 | 0.466 |
| B1_pairwise | 0.748 | 0.154 | 0.463 |
| B2_ternary | 0.746 | 0.115 | 0.438 |
| B3_shuffle_second | 0.637 | 0.077 | 0.355 |
| B4_shuffle_relation | 0.738 | 0.192 | 0.466 |

| B2 versus | Clustered wins/losses/ties | Mean AUC difference | One-sided paired p |
|---|---|---|---|
| B0_distance | 9/9/6 | 0.006 | 0.593 |
| B1_pairwise | 3/4/17 | -0.002 | 0.773 |
| B3_shuffle_second | 10/7/7 | 0.109 | 0.315 |
| B4_shuffle_relation | 9/9/6 | 0.006 | 0.593 |

A useful ternary program must beat the pairwise and shuffled-second controls on val_unseen. That condition is checked directly, rather than inferred from its raw AUC.
