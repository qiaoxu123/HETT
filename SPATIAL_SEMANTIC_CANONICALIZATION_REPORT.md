# Spatial semantic canonicalization report

## Status

Gate F: **FAIL / unestablished**. The current diagnostic cannot establish a stable language-to-geometry gain. Visual fusion should not start.

## Corpus and parser
- Raw episodes: 27,045; matched phrase occurrences: 66,832; distinct matched phrases: 27.
- Short prepositions, especially `on` and `in`, include lexical false positives. Anchor types in the full corpus audit are inferred from text only.
- DeepSeek consistency check: 27 instructions, two ontology orders each; valid JSON A/B: 100.0%/100.0%; exact agreement: 44.4%; UNKNOWN rate: 3.7%.

## Geometry diagnostic

The existing 2,290 sampled rows contain global, agent and anchor projections, plus center distance. They omit road geometry and full trajectories. The diagnostic chooses the nearest named anchor using the GT target solely for offline evaluation. These numbers cannot be used as deployable scores.

| Phrase | Anchor type | n val seen | Best available program | val seen AUC | val unseen AUC |
|---|---|---:|---|---:|---:|
| behind | unknown | 28 | BEHIND_CURRENT_AGENT | 0.499 | 0.567 |
| in front of | unknown | 26 | FRONT_GLOBAL | 0.588 | 0.481 |
| next to | unknown | 19 | CENTER_DISTANCE | 0.466 | 0.609 |
| left of | unknown | 15 | LEFT_GLOBAL | 0.656 | 0.678 |
| next to | building | 15 | CENTER_DISTANCE | 0.555 | 0.583 |
| left of | building | 13 | LEFT_GLOBAL | 0.703 | 0.444 |
| behind | building | 12 | BEHIND_CURRENT_AGENT | 0.603 | 0.583 |
| near | road | 11 | CENTER_DISTANCE | 0.515 | 0.889 |
| near | unknown | 11 | CENTER_DISTANCE | 0.477 | 0.278 |
| right of | unknown | 10 | RIGHT_ANCHOR | 0.596 | 0.542 |
| left of | road | 9 | LEFT_GLOBAL | 0.765 | 0.350 |
| in front of | road | 7 | FRONT_START_AGENT | 0.675 | 0.556 |
| right of | building | 7 | RIGHT_ANCHOR | 0.702 | 0.561 |
| beside | building | 5 | CENTER_DISTANCE | 0.711 | — |
| right of | road | 5 | RIGHT_START_AGENT | 0.692 | 0.593 |
| beside | unknown | 4 | CENTER_DISTANCE | 0.333 | 0.422 |
| close to | unknown | 4 | CENTER_DISTANCE | 0.392 | — |
| in front of | building | 4 | FRONT_START_AGENT | 0.722 | 0.551 |
| near | building | 4 | CENTER_DISTANCE | 0.341 | 0.722 |
| next to | road | 4 | CENTER_DISTANCE | 0.667 | 0.500 |

## Missing evidence

- Full DeepSeek parsing of all instructions; the 27 instruction consistency sample is insufficient for corpus parse success.
- True ternary between, road association/across/along, route and final approach evaluation.
- O0–O4 comparison, correct versus shuffled/opposite, paired p-values, reliability calibration, candidate-order invariance.
- Therefore no supported answer yet to whether final approach is useful or probabilistic canonicalization beats the old ontology.

## Next step

Extend the diagnostic rows with trajectory windows and explicit named road/anchor geometry, then rerun train_seen fitting, val_seen selection and sealed val_unseen reporting.
