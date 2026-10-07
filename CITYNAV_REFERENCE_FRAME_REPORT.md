# CityNav reference frame report

Selection uses train_seen statistics and val_seen AUC. The val_unseen column is sealed reporting. FINAL windows use XY motion from the window start to the trajectory end; route uses start-to-end or PCA motion. Full future trajectory is diagnostic evidence about the annotator viewpoint, not an instruction-time feature.

The anchor frame is geometric footprint orientation and does not identify a facade. Road frame uses a local RoadRegion tangent. Invalid motion windows have no score.

| Final window | Valid legacy rows | Valid rate | Mean step-heading agreement |
|---|---|---|---|
| 3 | 1913 | 0.835 | 0.999 |
| 5 | 1937 | 0.846 | 0.809 |
| 10 | 2124 | 0.928 | 0.733 |
| 20 | 2202 | 0.962 | 0.635 |

## behind

Anchor type: building; train/val seen/val unseen clause counts: 72/36/39. Best: `BACK_ANCHOR_MAJOR`; reliability 0.265; unseen sign flip: True.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.421 | 0.028 | 0.248 | 30.550 | 0.502 |
| START | 0.485 | 0.056 | 0.303 | 38.056 | 0.484 |
| FINAL3 | 0.416 | 0.000 | 0.246 | 13.086 | 0.452 |
| FINAL5 | 0.451 | 0.000 | 0.267 | 18.115 | 0.429 |
| FINAL10 | 0.444 | 0.000 | 0.259 | 14.789 | 0.354 |
| FINAL20 | 0.427 | 0.000 | 0.251 | 13.213 | 0.324 |
| ROUTE_END | 0.420 | 0.028 | 0.258 | 12.719 | 0.362 |
| ROUTE_PCA | 0.434 | 0.028 | 0.260 | 13.368 | 0.344 |
| ROAD | — | — | — | — | — |
| ANCHOR_MAJOR | 0.632 | 0.111 | 0.403 | 54.251 | 0.456 |
| ANCHOR_MINOR | 0.535 | 0.083 | 0.324 | 38.285 | 0.611 |

Anchor type: road_region; train/val seen/val unseen clause counts: 6/3/6. Best: `BACK_ROAD`; reliability 0.088; unseen sign flip: True.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.317 | 0.000 | 0.189 | -0.520 | 0.708 |
| START | 0.571 | 0.000 | 0.333 | -7.999 | 0.556 |
| FINAL3 | 0.365 | 0.000 | 0.200 | -35.430 | 0.578 |
| FINAL5 | 0.365 | 0.000 | 0.200 | -34.885 | 0.648 |
| FINAL10 | 0.365 | 0.000 | 0.200 | -34.579 | 0.463 |
| FINAL20 | 0.310 | 0.000 | 0.189 | -36.026 | 0.519 |
| ROUTE_END | 0.365 | 0.000 | 0.206 | -27.062 | 0.491 |
| ROUTE_PCA | 0.365 | 0.000 | 0.206 | -36.315 | 0.491 |
| ROAD | 0.794 | 0.000 | 0.444 | 5.734 | 0.426 |
| ANCHOR_MAJOR | — | — | — | — | — |
| ANCHOR_MINOR | — | — | — | — | — |

Anchor type: unknown; train/val seen/val unseen clause counts: 11/3/0. Best: `BACK_ROAD`; reliability 0.100; unseen sign flip: False.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.389 | 0.000 | 0.217 | -11.935 | — |
| START | 0.556 | 0.000 | 0.278 | 9.439 | — |
| FINAL3 | 0.278 | 0.000 | 0.214 | -11.104 | — |
| FINAL5 | 0.278 | 0.000 | 0.214 | -11.007 | — |
| FINAL10 | 0.222 | 0.000 | 0.206 | -8.572 | — |
| FINAL20 | 0.278 | 0.000 | 0.198 | -7.449 | — |
| ROUTE_END | 0.278 | 0.000 | 0.198 | -6.370 | — |
| ROUTE_PCA | 0.333 | 0.000 | 0.206 | -5.526 | — |
| ROAD | 1.000 | 1.000 | 1.000 | 19.490 | — |
| ANCHOR_MAJOR | 0.611 | 0.000 | 0.306 | 8.917 | — |
| ANCHOR_MINOR | 0.444 | 0.000 | 0.244 | -11.869 | — |

## in front of

Anchor type: building; train/val seen/val unseen clause counts: 52/25/17. Best: `FRONT_FINAL3`; reliability 0.239; unseen sign flip: True.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.578 | 0.040 | 0.306 | 36.093 | 0.595 |
| START | 0.527 | 0.000 | 0.283 | 30.898 | 0.503 |
| FINAL3 | 0.619 | 0.000 | 0.319 | 30.662 | 0.440 |
| FINAL5 | 0.554 | 0.000 | 0.271 | 28.852 | 0.500 |
| FINAL10 | 0.501 | 0.000 | 0.262 | 35.000 | 0.531 |
| FINAL20 | 0.490 | 0.000 | 0.263 | 35.120 | 0.448 |
| ROUTE_END | 0.513 | 0.080 | 0.301 | 35.717 | 0.575 |
| ROUTE_PCA | 0.532 | 0.120 | 0.331 | 35.930 | 0.572 |
| ROAD | — | — | — | — | — |
| ANCHOR_MAJOR | 0.453 | 0.040 | 0.254 | 32.961 | 0.539 |
| ANCHOR_MINOR | 0.496 | 0.040 | 0.285 | 30.762 | 0.536 |

Anchor type: road_region; train/val seen/val unseen clause counts: 1/5/3. Best: `FRONT_ROUTE_END`; reliability 0.083; unseen sign flip: False.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.400 | 0.000 | 0.230 | -4.390 | 0.389 |
| START | 0.500 | 0.000 | 0.267 | 3.013 | 0.444 |
| FINAL3 | — | — | — | — | — |
| FINAL5 | 0.533 | 0.200 | 0.390 | -2.389 | 0.500 |
| FINAL10 | 0.467 | 0.000 | 0.257 | -1.951 | 0.500 |
| FINAL20 | 0.533 | 0.000 | 0.283 | 6.243 | 0.444 |
| ROUTE_END | 0.667 | 0.000 | 0.367 | 9.706 | 0.667 |
| ROUTE_PCA | 0.600 | 0.000 | 0.317 | 8.441 | 0.667 |
| ROAD | 0.333 | 0.000 | 0.212 | -2.510 | 0.667 |
| ANCHOR_MAJOR | — | — | — | — | — |
| ANCHOR_MINOR | — | — | — | — | — |

Anchor type: unknown; train/val seen/val unseen clause counts: 8/3/0. Best: `FRONT_ROAD`; reliability 0.000; unseen sign flip: False.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.384 | 0.000 | 0.159 | -6.628 | — |
| START | 0.505 | 0.000 | 0.198 | 7.381 | — |
| FINAL3 | 0.444 | 0.000 | 0.171 | -8.620 | — |
| FINAL5 | 0.444 | 0.000 | 0.171 | -8.690 | — |
| FINAL10 | 0.463 | 0.000 | 0.181 | -4.913 | — |
| FINAL20 | 0.426 | 0.000 | 0.170 | 5.252 | — |
| ROUTE_END | 0.463 | 0.000 | 0.181 | 1.240 | — |
| ROUTE_PCA | 0.463 | 0.000 | 0.181 | 2.460 | — |
| ROAD | 0.542 | 0.000 | 0.214 | 3.810 | — |
| ANCHOR_MAJOR | 0.394 | 0.000 | 0.173 | -1.075 | — |
| ANCHOR_MINOR | 0.384 | 0.000 | 0.159 | -5.784 | — |

## left of

Anchor type: building; train/val seen/val unseen clause counts: 24/24/10. Best: `LEFT_GLOBAL`; reliability 0.167; unseen sign flip: False.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.584 | 0.000 | 0.308 | 45.295 | 0.533 |
| START | 0.536 | 0.042 | 0.296 | 42.392 | 0.581 |
| FINAL3 | 0.534 | 0.087 | 0.316 | 45.721 | 0.460 |
| FINAL5 | 0.445 | 0.000 | 0.222 | 37.549 | 0.460 |
| FINAL10 | 0.500 | 0.000 | 0.263 | 43.438 | 0.710 |
| FINAL20 | 0.570 | 0.042 | 0.312 | 45.080 | 0.683 |
| ROUTE_END | 0.533 | 0.083 | 0.319 | 44.665 | 0.562 |
| ROUTE_PCA | 0.536 | 0.042 | 0.302 | 44.470 | 0.529 |
| ROAD | — | — | — | — | — |
| ANCHOR_MAJOR | 0.496 | 0.042 | 0.277 | 41.351 | 0.537 |
| ANCHOR_MINOR | 0.578 | 0.125 | 0.354 | 46.022 | 0.561 |

Anchor type: road_region; train/val seen/val unseen clause counts: 4/8/6. Best: `LEFT_FINAL3`; reliability 0.000; unseen sign flip: False.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.448 | 0.000 | 0.267 | -0.892 | 0.528 |
| START | 0.479 | 0.000 | 0.252 | -0.875 | 0.486 |
| FINAL3 | 0.667 | 0.333 | 0.500 | 3.992 | 0.500 |
| FINAL5 | 0.472 | 0.000 | 0.260 | -2.612 | 0.433 |
| FINAL10 | 0.595 | 0.000 | 0.333 | 1.065 | 0.583 |
| FINAL20 | 0.500 | 0.000 | 0.275 | -1.311 | 0.583 |
| ROUTE_END | 0.542 | 0.000 | 0.290 | 0.495 | 0.521 |
| ROUTE_PCA | 0.542 | 0.000 | 0.290 | 0.819 | 0.521 |
| ROAD | 0.500 | 0.000 | 0.275 | 0.195 | 0.556 |
| ANCHOR_MAJOR | — | — | — | — | — |
| ANCHOR_MINOR | — | — | — | — | — |

Anchor type: unknown; train/val seen/val unseen clause counts: 2/5/0. Best: `LEFT_FINAL3`; reliability 0.150; unseen sign flip: False.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.333 | 0.000 | 0.213 | -4.537 | — |
| START | 0.567 | 0.200 | 0.397 | 5.669 | — |
| FINAL3 | 0.800 | 0.200 | 0.533 | 15.269 | — |
| FINAL5 | 0.667 | 0.200 | 0.467 | 10.658 | — |
| FINAL10 | 0.767 | 0.000 | 0.433 | 11.785 | — |
| FINAL20 | 0.667 | 0.200 | 0.480 | 4.339 | — |
| ROUTE_END | 0.633 | 0.000 | 0.357 | 3.294 | — |
| ROUTE_PCA | 0.733 | 0.000 | 0.417 | 6.402 | — |
| ROAD | — | — | — | — | — |
| ANCHOR_MAJOR | 0.633 | 0.000 | 0.317 | 1.352 | — |
| ANCHOR_MINOR | 0.233 | 0.000 | 0.183 | -1.361 | — |

## right of

Anchor type: building; train/val seen/val unseen clause counts: 27/15/16. Best: `RIGHT_GLOBAL`; reliability 0.154; unseen sign flip: False.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.602 | 0.133 | 0.366 | 29.916 | 0.556 |
| START | 0.468 | 0.000 | 0.253 | 40.265 | 0.458 |
| FINAL3 | 0.557 | 0.000 | 0.279 | 44.848 | 0.466 |
| FINAL5 | 0.549 | 0.000 | 0.281 | 43.242 | 0.457 |
| FINAL10 | 0.479 | 0.000 | 0.247 | 36.084 | 0.530 |
| FINAL20 | 0.556 | 0.000 | 0.270 | 39.010 | 0.493 |
| ROUTE_END | 0.558 | 0.000 | 0.280 | 47.671 | 0.483 |
| ROUTE_PCA | 0.566 | 0.067 | 0.314 | 49.087 | 0.462 |
| ROAD | — | — | — | — | — |
| ANCHOR_MAJOR | 0.517 | 0.000 | 0.258 | 41.006 | 0.549 |
| ANCHOR_MINOR | 0.589 | 0.067 | 0.330 | 35.264 | 0.493 |

Anchor type: road_region; train/val seen/val unseen clause counts: 3/4/2. Best: `RIGHT_ROUTE_PCA`; reliability 0.133; unseen sign flip: False.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.417 | 0.000 | 0.237 | -11.064 | 0.417 |
| START | 0.792 | 0.500 | 0.646 | 20.042 | 0.667 |
| FINAL3 | 0.500 | 0.333 | 0.464 | 1.461 | 0.583 |
| FINAL5 | 0.389 | 0.000 | 0.242 | -10.216 | 0.667 |
| FINAL10 | 0.833 | 0.500 | 0.688 | 32.795 | 0.667 |
| FINAL20 | 0.833 | 0.500 | 0.688 | 30.867 | 0.667 |
| ROUTE_END | 0.833 | 0.500 | 0.688 | 29.614 | 0.583 |
| ROUTE_PCA | 0.833 | 0.500 | 0.688 | 28.523 | 0.583 |
| ROAD | 0.125 | 0.000 | 0.170 | -39.906 | 0.167 |
| ANCHOR_MAJOR | — | — | — | — | — |
| ANCHOR_MINOR | — | — | — | — | — |

Anchor type: unknown; train/val seen/val unseen clause counts: 7/2/0. Best: `RIGHT_ROUTE_END`; reliability 0.000; unseen sign flip: False.

| Frame | Val seen AUC | Top-1 | MRR | Margin | Val unseen AUC |
|---|---|---|---|---|---|
| GLOBAL | 0.292 | 0.000 | 0.171 | -40.765 | — |
| START | 0.625 | 0.000 | 0.292 | 36.182 | — |
| FINAL3 | 0.750 | 0.500 | 0.625 | 27.078 | — |
| FINAL5 | 0.750 | 0.500 | 0.625 | 27.078 | — |
| FINAL10 | 0.667 | 0.500 | 0.600 | 20.341 | — |
| FINAL20 | 0.750 | 0.500 | 0.625 | 23.324 | — |
| ROUTE_END | 0.833 | 0.500 | 0.667 | 48.771 | — |
| ROUTE_PCA | 0.833 | 0.500 | 0.667 | 48.228 | — |
| ROAD | 0.312 | 0.000 | 0.188 | -38.307 | — |
| ANCHOR_MAJOR | 0.312 | 0.000 | 0.188 | -38.372 | — |
| ANCHOR_MINOR | 0.375 | 0.000 | 0.196 | -32.831 | — |

View-dependent relations are not stable when a val_seen-positive frame reverses on val_unseen. Reported unseen results never alter frame selection or probabilities.
