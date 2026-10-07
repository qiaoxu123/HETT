# Road relation canonicalization

RoadRegion is a named connected component. Candidate geometry uses the nearest local part of an ordered segment-centre polyline; local tangent, normal, signed side, projection and corridor distance are available. The polyline is an approximation, so hairpin roads can have imperfect local ordering. Across-road crossing and opposite side remain undefined where the instruction lacks an independent reference entity on the other side.

Legacy diagnostic road graph summary: 25 maps; 329 RoadRegions. The supplemental raw sample has 516 clauses across on/off/along/across; it uses a separate random map-distractor protocol.

## on + road

Source `annotation`; n train/val seen/val unseen = 292/195/70; selected `ROAD_ASSOCIATION`; reliability 0.111; unseen sign flip False.

| Program | Val seen AUC | Val unseen AUC |
|---|---|---|
| ALONG_ROAD | 0.539 | 0.633 |
| IN_ROAD_BUFFER | 0.526 | 0.588 |
| NEAR_ROAD_REGION | 0.540 | 0.637 |
| ROAD_ASSOCIATION | 0.555 | 0.640 |

Source `parser`; n train/val seen/val unseen = 253/172/68; selected `ROAD_ASSOCIATION`; reliability 0.103; unseen sign flip False.

| Program | Val seen AUC | Val unseen AUC |
|---|---|---|
| ALONG_ROAD | 0.536 | 0.636 |
| IN_ROAD_BUFFER | 0.521 | 0.589 |
| NEAR_ROAD_REGION | 0.536 | 0.641 |
| ROAD_ASSOCIATION | 0.551 | 0.644 |

Source `raw_parser`; n train/val seen/val unseen = 40/40/40; selected `ROAD_ASSOCIATION`; reliability 0.800; unseen sign flip False.

| Program | Val seen AUC | Val unseen AUC |
|---|---|---|
| ALONG_ROAD | 0.889 | 0.864 |
| IN_ROAD_BUFFER | 0.856 | 0.815 |
| NEAR_ROAD_REGION | 0.896 | 0.863 |
| ROAD_ASSOCIATION | 0.900 | 0.861 |

Source `deepseek`; n train/val seen/val unseen = 19/8/4; selected `ALONG_ROAD`; reliability 0.133; unseen sign flip False.

| Program | Val seen AUC | Val unseen AUC |
|---|---|---|
| ALONG_ROAD | 0.667 | 0.729 |
| IN_ROAD_BUFFER | 0.594 | 0.729 |
| NEAR_ROAD_REGION | 0.646 | 0.729 |
| ROAD_ASSOCIATION | 0.646 | 0.729 |

## off + road

Source `raw_parser`; n train/val seen/val unseen = 40/40/40; selected `ROAD_ASSOCIATION`; reliability 0.767; unseen sign flip False.

| Program | Val seen AUC | Val unseen AUC |
|---|---|---|
| NEAR_ROAD_REGION | 0.881 | 0.785 |
| OFF_ROAD_BUFFER | 0.668 | 0.714 |
| ROAD_ASSOCIATION | 0.883 | 0.772 |

Source `deepseek`; n train/val seen/val unseen = 2/1/3; selected `OFF_ROAD_BUFFER`; reliability 0.000; unseen sign flip True.

| Program | Val seen AUC | Val unseen AUC |
|---|---|---|
| NEAR_ROAD_REGION | 0.167 | 0.704 |
| OFF_ROAD_BUFFER | 0.333 | 0.546 |
| ROAD_ASSOCIATION | 0.167 | 0.704 |

## along + road

Source `raw_parser`; n train/val seen/val unseen = 40/40/40; selected `NEAR_ROAD_REGION`; reliability 0.781; unseen sign flip False.

| Program | Val seen AUC | Val unseen AUC |
|---|---|---|
| ALONG_ROAD_TANGENT | 0.883 | 0.865 |
| NEAR_ROAD_REGION | 0.890 | 0.865 |

## across + road

Source `raw_parser`; n train/val seen/val unseen = 26/9/6; selected `LINE_CROSSES_ROAD`; reliability 0.256; unseen sign flip False.

| Program | Val seen AUC | Val unseen AUC |
|---|---|---|
| LINE_CROSSES_ROAD | 0.784 | 0.722 |
| OPPOSITE_SIDE_OF_ROAD | 0.593 | 0.333 |

## across from + road
No evaluable rows.

A road name alone cannot define which side an independent landmark occupies. The across controls therefore abstain when a second anchor is missing.
