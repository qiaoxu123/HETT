# CityNav spatial lexicon (diagnostic draft)

Probabilities are exploratory weights from train_seen and val_seen only. The sampled geometry file does not support the road, route, final approach, or ternary programs, so these are not validated lexicon entries.

| Phrase | Anchor type | n train / val seen / val unseen | Best supported program | val seen AUC | val unseen AUC | Reliability |
|---|---|---:|---|---:|---:|---:|
| behind | unknown | 42 / 28 / 11 | BEHIND_CURRENT_AGENT | 0.499 | 0.567 | 0.00 |
| in front of | unknown | 32 / 26 / 6 | FRONT_GLOBAL | 0.588 | 0.481 | 0.18 |
| next to | unknown | 37 / 19 / 26 | CENTER_DISTANCE | 0.466 | 0.609 | 0.00 |
| left of | unknown | 18 / 15 / 5 | LEFT_GLOBAL | 0.656 | 0.678 | 0.00 |
| next to | building | 20 / 15 / 6 | CENTER_DISTANCE | 0.555 | 0.583 | 0.00 |
| left of | building | 8 / 13 / 6 | LEFT_GLOBAL | 0.703 | 0.444 | 0.00 |
| behind | building | 32 / 12 / 24 | BEHIND_CURRENT_AGENT | 0.603 | 0.583 | 0.00 |
| near | road | 14 / 11 / 1 | CENTER_DISTANCE | 0.515 | 0.889 | 0.00 |
| near | unknown | 18 / 11 / 3 | CENTER_DISTANCE | 0.477 | 0.278 | 0.00 |
| right of | unknown | 20 / 10 / 4 | RIGHT_ANCHOR | 0.596 | 0.542 | 0.00 |
| left of | road | 4 / 9 / 5 | LEFT_GLOBAL | 0.765 | 0.350 | 0.00 |
| in front of | road | 4 / 7 / 3 | FRONT_START_AGENT | 0.675 | 0.556 | 0.00 |
| right of | building | 15 / 7 / 11 | RIGHT_ANCHOR | 0.702 | 0.561 | 0.00 |
| beside | building | 3 / 5 / 0 | CENTER_DISTANCE | 0.711 | — | 0.00 |
| right of | road | 9 / 5 / 3 | RIGHT_START_AGENT | 0.692 | 0.593 | 0.00 |
| beside | unknown | 12 / 4 / 5 | CENTER_DISTANCE | 0.333 | 0.422 | 0.00 |
| close to | unknown | 2 / 4 / 0 | CENTER_DISTANCE | 0.392 | — | 0.00 |
| in front of | building | 25 / 4 / 11 | FRONT_START_AGENT | 0.722 | 0.551 | 0.00 |
| near | building | 5 / 4 / 2 | CENTER_DISTANCE | 0.341 | 0.722 | 0.00 |
| next to | road | 28 / 4 / 4 | CENTER_DISTANCE | 0.667 | 0.500 | 0.00 |
| beside | road | 5 / 3 / 0 | CENTER_DISTANCE | 0.324 | — | 0.00 |
| behind | road | 15 / 2 / 10 | BEHIND_CURRENT_AGENT | 0.500 | 0.540 | 0.00 |
