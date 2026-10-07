# Parking layout reasoning

Cars are collected only inside a text-selected CityRefer Parking polygon (2 m tolerance); negatives are all other cars in that same lot. Grouping and ordering use map centers before the answer is consulted. This is a hard same-lot candidate set, not random negatives. It can still contain multiple physical subgrids.

Evaluated 384 samples. Abstentions: ambiguous parking region 2402, selected region misses GT 332, too few cars 319, non-car target 69.

## Val seen

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

## Val unseen

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


## val_seen layout families

| Family | Global n / Top1 / MRR / rank | Instruction n / Top1 / MRR / rank |
|---|---|---|
| NTH_FROM_RIGHT | 13 / 0.000 / 0.044 / 92.385 | 12 / 0.008 / 0.045 / 67.333 |
| NTH_FROM_LEFT | 12 / 0.100 / 0.149 / 45.583 | 8 / 0.033 / 0.114 / 49.875 |
| ROW_BOTTOM | 12 / 0.000 / 0.034 / 77.458 | 9 / 0.006 / 0.047 / 66.444 |
| NTH_FROM_BOTTOM | 9 / 0.046 / 0.119 / 46.667 | 6 / 0.001 / 0.029 / 74.500 |
| ROW_INDEX | 9 / 0.037 / 0.079 / 85.278 | 8 / 0.000 / 0.022 / 73.750 |
| COLUMN_INDEX | 7 / 0.000 / 0.017 / 70.643 | 7 / 0.000 / 0.026 / 75.500 |
| NTH_FROM_TOP | 3 / 0.111 / 0.209 / 85.333 | 2 / 0.003 / 0.032 / 74.250 |
| ROW_TOP | 3 / 0.333 / 0.496 / 5.833 | 2 / 0.062 / 0.211 / 8.500 |
| COLUMN_LEFT | 2 / 0.000 / 0.011 / 97.000 | 2 / 0.007 / 0.038 / 73.500 |

## val_unseen layout families

| Family | Global n / Top1 / MRR / rank | Instruction n / Top1 / MRR / rank |
|---|---|---|
| NTH_FROM_LEFT | 4 / 0.625 / 0.714 / 3.250 | 0 / — / — / — |
| COLUMN_RIGHT | 3 / 0.000 / 0.292 / 4.500 | 0 / — / — / — |
| COLUMN_LEFT | 2 / 0.000 / 0.184 / 6.250 | 0 / — / — / — |
| NTH_FROM_TOP | 2 / 0.000 / 0.375 / 2.750 | 0 / — / — / — |
| ROW_INDEX | 2 / 0.000 / 0.088 / 12.000 | 0 / — / — / — |
| COLUMN_INDEX | 1 / 0.000 / 0.190 / 5.500 | 0 / — / — / — |
| NTH_FROM_BOTTOM | 1 / 0.000 / 0.250 / 4.000 | 0 / — / — / — |
| ROW_BOTTOM | 1 / 0.000 / 0.106 / 9.500 | 0 / — / — / — |

Same-row neighbor AUC (val_seen): 0.822 over 25 evaluable clauses. This compares the GT with cars assigned to the same inferred row.

Same-row neighbor AUC (val_unseen): — over 0 evaluable clauses. This compares the GT with cars assigned to the same inferred row.

Rows and columns use PCA or a resolved instruction frame followed by gap clustering. This first pass has no validated lot subdivision, so very large parking polygons can create incorrect row numbers. L1/L2 assume bottom-first row order as a baseline; L3 applies ROW_INDEX only when the instruction states its row start, otherwise that clause abstains.
