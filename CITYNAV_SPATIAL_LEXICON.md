# CityNav spatial lexicon: Gate F evidence

Probabilities and reliability are fitted from train_seen plus val_seen only. The unseen column is a post-selection stability check. UNKNOWN remains a program hypothesis. Annotation rows are offline oracle diagnostics; raw_parser rows are instruction-only and use easier deterministic map distractors, so their rho is not comparable with legacy hard-negative rows.


| Phrase | Semantic family | Anchor | Source | n train/seen/unseen | Selected program | Frame | P(selected) | P(UNKNOWN) | rho | Seen AUC | Unseen AUC | Stable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| on | ROAD_ASSOCIATION | road_region | annotation | 292/195/70 | ROAD_ASSOCIATION | ROAD_REGION | 0.238 | 0.153 | 0.111 | 0.555 | 0.640 | yes |
| on | ROAD_ASSOCIATION | road_region | parser | 253/172/68 | ROAD_ASSOCIATION | ROAD_REGION | 0.237 | 0.157 | 0.103 | 0.551 | 0.644 | yes |
| on | ROAD_ASSOCIATION | road_region | raw_parser | 40/40/40 | ROAD_ASSOCIATION | ROAD_REGION | 0.249 | 0.013 | 0.800 | 0.900 | 0.861 | yes |
| off | ROAD_ASSOCIATION | road_region | raw_parser | 40/40/40 | ROAD_ASSOCIATION | ROAD_REGION | 0.465 | 0.024 | 0.767 | 0.883 | 0.772 | yes |
| along | ROAD_ALONG | road_region | raw_parser | 40/40/40 | NEAR_ROAD_REGION | ROAD_REGION | 0.465 | 0.027 | 0.781 | 0.890 | 0.865 | yes |
| behind | VIEW_DEPENDENT_DIRECTION | building | annotation | 72/36/39 | BACK_ANCHOR_MAJOR | ANCHOR_MAJOR | 0.111 | 0.089 | 0.265 | 0.632 | 0.456 | no |
| next to | PROXIMITY | building | annotation | 59/33/26 | FOOTPRINT_DISTANCE | NONE | 0.512 | 0.085 | 0.448 | 0.724 | 0.874 | yes |
| behind | VIEW_DEPENDENT_DIRECTION | building | parser | 60/32/35 | BACK_ANCHOR_MAJOR | ANCHOR_MAJOR | 0.094 | 0.091 | 0.263 | 0.632 | 0.478 | no |
| next to | PROXIMITY | building | parser | 59/29/23 | FOOTPRINT_DISTANCE | NONE | 0.482 | 0.089 | 0.421 | 0.711 | 0.879 | yes |
| in front of | VIEW_DEPENDENT_DIRECTION | building | annotation | 52/25/17 | FRONT_FINAL3 | FINAL3 | 0.184 | 0.071 | 0.239 | 0.619 | 0.440 | no |
| left of | VIEW_DEPENDENT_DIRECTION | building | annotation | 24/24/10 | LEFT_GLOBAL | GLOBAL | 0.155 | 0.079 | 0.167 | 0.584 | 0.533 | yes |
| in front of | VIEW_DEPENDENT_DIRECTION | building | parser | 45/22/16 | FRONT_FINAL3 | FINAL3 | 0.179 | 0.074 | 0.197 | 0.610 | 0.449 | no |
| left of | VIEW_DEPENDENT_DIRECTION | building | parser | 25/22/8 | LEFT_FINAL20 | FINAL20 | 0.079 | 0.079 | 0.000 | 0.583 | 0.646 | yes |
| between | BETWEEN | building | parser | 36/16/12 | PAIRWISE_OLD | NONE | 0.384 | 0.037 | 0.632 | 0.895 | 0.833 | yes |
| between | BETWEEN | building | annotation | 38/16/12 | PAIRWISE_OLD | NONE | 0.475 | 0.035 | 0.611 | 0.882 | 0.833 | yes |
| right of | VIEW_DEPENDENT_DIRECTION | building | annotation | 27/15/16 | RIGHT_GLOBAL | GLOBAL | 0.088 | 0.080 | 0.154 | 0.602 | 0.556 | yes |
| right of | VIEW_DEPENDENT_DIRECTION | building | parser | 26/14/12 | RIGHT_GLOBAL | GLOBAL | 0.078 | 0.078 | 0.000 | 0.615 | 0.528 | yes |
| near | PROXIMITY | building | annotation | 17/11/5 | FOOTPRINT_DISTANCE | NONE | 0.496 | 0.130 | 0.185 | 0.668 | 0.811 | yes |
| between | BETWEEN | road_region | annotation | 10/9/2 | PAIRWISE_OLD | NONE | 0.322 | 0.318 | 0.257 | 0.786 | 0.667 | yes |
| left of | VIEW_DEPENDENT_DIRECTION | unknown | parser | 18/9/3 | LEFT_FINAL3 | FINAL3 | 0.053 | 0.393 | 0.156 | 0.694 | 0.750 | yes |
| left of | VIEW_DEPENDENT_DIRECTION | road_region | parser | 4/9/3 | LEFT_FINAL20 | FINAL20 | 0.096 | 0.425 | 0.057 | 0.563 | 0.389 | no |
| near | PROXIMITY | building | parser | 15/9/4 | FOOTPRINT_DISTANCE | NONE | 0.280 | 0.465 | 0.168 | 0.687 | 0.806 | yes |
| across | ROAD_ACROSS | road_region | raw_parser | 26/9/6 | LINE_CROSSES_ROAD | ROAD_REGION | 0.357 | 0.566 | 0.256 | 0.784 | 0.722 | yes |
| between | BETWEEN | road_region | parser | 8/8/2 | DISTANCE_PRIOR | NONE | 0.091 | 0.323 | 0.221 | 0.777 | 0.833 | yes |
| near | PROXIMITY | road_region | annotation | 9/8/1 | CENTER_DISTANCE | NONE | 0.099 | 0.734 | 0.000 | 0.833 | 0.778 | yes |
| left of | VIEW_DEPENDENT_DIRECTION | road_region | annotation | 4/8/6 | LEFT_FINAL3 | FINAL3 | 0.058 | 0.430 | 0.000 | 0.667 | 0.500 | yes |
| beside | PROXIMITY | building | annotation | 15/8/4 | FOOTPRINT_DISTANCE | NONE | 0.439 | 0.332 | 0.289 | 0.861 | 0.785 | yes |
| beside | PROXIMITY | building | parser | 14/8/4 | FOOTPRINT_DISTANCE | NONE | 0.368 | 0.439 | 0.289 | 0.861 | 0.785 | yes |
| on | ROAD_ASSOCIATION | road_region | deepseek | 19/8/4 | ALONG_ROAD | ROAD_REGION | 0.098 | 0.569 | 0.133 | 0.667 | 0.729 | yes |
| near | PROXIMITY | road_region | parser | 7/6/0 | CENTER_DISTANCE | NONE | 0.107 | 0.787 | 0.000 | 0.734 | — | yes |
| in front of | VIEW_DEPENDENT_DIRECTION | unknown | parser | 12/5/0 | FRONT_FINAL5 | FINAL5 | 0.140 | 0.273 | 0.050 | 0.667 | — | yes |
| left of | VIEW_DEPENDENT_DIRECTION | unknown | annotation | 2/5/0 | LEFT_FINAL3 | FINAL3 | 0.058 | 0.425 | 0.150 | 0.800 | — | yes |
| in front of | VIEW_DEPENDENT_DIRECTION | road_region | annotation | 1/5/3 | FRONT_ROUTE_END | ROUTE_END | 0.189 | 0.369 | 0.083 | 0.667 | 0.667 | yes |
| right of | VIEW_DEPENDENT_DIRECTION | unknown | parser | 12/5/2 | RIGHT_FINAL10 | FINAL10 | 0.112 | 0.343 | 0.090 | 0.681 | 0.611 | yes |
| to the left of | VIEW_DEPENDENT_DIRECTION | building | deepseek | 4/5/2 | LEFT_GLOBAL | GLOBAL | 0.094 | 0.399 | 0.035 | 0.569 | 0.500 | yes |
| near | PROXIMITY | building | deepseek | 3/5/0 | CENTER_DISTANCE | NONE | 0.253 | 0.493 | 0.119 | 0.738 | — | yes |

A probability is an exploratory geometric calibration, not DeepSeek confidence. Low counts and unseen sign flips preclude deploying a phrase even when its val_seen AUC exceeds chance.
