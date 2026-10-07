# Explicit layout language audit

CityNav train_seen, val_seen, val_unseen: 27,045 instructions. Text-only regex parsing; target IDs are read only afterward to aggregate actual target types.

| Measure | Count | Share |
|---|---:|---:|
| Explicit layout program | 3,506 | 12.96% |
| Explicit frame cue | 3,117 | 11.53% |
| Frame cue plus layout | 1,535 | 5.68% |
| Group A: frame plus layout | 1,535 | 5.68% |
| Group B: layout only | 1,971 | 7.29% |
| Group C: generic relation only | 19,834 | 73.34% |
| Other/no matched relation | 3,705 | 13.70% |

## Layout families

| Family | Count | Actual target type | Splits | Context |
|---|---:|---|---|---|
| ROW_INDEX | 1732 | {'Car': 1729, 'Building': 2, 'Parking': 1} | {'train_seen': 1559, 'val_seen': 162, 'val_unseen': 11} | The gray car that is 1st from the right in the 12th row of cars in the large parking lot behind the One Stop Building, when viewed aerially with the long side of the One Stop Build |
| NTH_FROM_LEFT | 1280 | {'Building': 55, 'Car': 1225} | {'train_seen': 1140, 'val_seen': 98, 'val_unseen': 42} | The brown house that is 2nd from the left in the row of brown houses behind the long row of green trees along Wellington Road. |
| NTH_FROM_RIGHT | 928 | {'Building': 42, 'Car': 885, 'Ground': 1} | {'train_seen': 815, 'val_seen': 82, 'val_unseen': 31} | Gray building that is second from the right on the row of buildings above Willmore Road, when viewed aerially with Wellington road to the right. |
| NTH_FROM_BOTTOM | 441 | {'Car': 415, 'Building': 26} | {'train_seen': 377, 'val_seen': 44, 'val_unseen': 20} | On Wellington Road north of Willmore Road and to the left of St Teresa's Court, it is a white car, fourth car from the bottom on the left. |
| NTH_FROM_TOP | 339 | {'Building': 26, 'Car': 312, 'Ground': 1} | {'train_seen': 294, 'val_seen': 33, 'val_unseen': 12} | On the right side of Willmore Road, this building has a gray roof and is third one from the top. |
| ROW_TOP | 275 | {'Car': 275} | {'train_seen': 256, 'val_seen': 12, 'val_unseen': 7} | The black car that is 4th from the left in the top row of cars in the parking lot of the St. Teresa of the Child Jesus Catholic Church, when viewed aerially with Wellington Road to |
| ROW_BOTTOM | 226 | {'Car': 225, 'Building': 1} | {'train_seen': 196, 'val_seen': 25, 'val_unseen': 5} | The black car that is 3rd from the left in the bottom row of cars in the brown parking lot off of Wellington Road, in front of a gray and white building. |
| NTH_FROM_CORNER | 100 | {'Building': 60, 'Car': 40} | {'train_seen': 82, 'val_seen': 10, 'val_unseen': 8} | A brick building with two residents each with a blue and gray trash bin back to back with each other in the front yard area. The building is facing Willmore Road and is the third b |
| COLUMN_INDEX | 92 | {'Car': 92} | {'train_seen': 68, 'val_seen': 10, 'val_unseen': 14} | With the One Stop building on the left, it is the silver car parked in the first column, counting from the bottom. The column is perpendicular to the majority of the parked cars. T |
| ROAD_SEQUENCE_ORDER | 78 | {'Building': 59, 'Car': 19} | {'train_seen': 63, 'val_seen': 8, 'val_unseen': 7} | The last building on Willmore Road that has a small gray porch on the back and the next building has a green and black trampoline. |
| COLUMN_RIGHT | 52 | {'Car': 52} | {'train_seen': 43, 'val_unseen': 9} | The gray car that is 3rd from the bottom in the rightmost column of cars in the parking lot of the One Stop Building, when viewed aerially with Walsall Road below. |
| COLUMN_LEFT | 47 | {'Car': 47} | {'train_seen': 39, 'val_seen': 6, 'val_unseen': 2} | The gray car that is 7th from the bottom in the leftmost column of cars in the parking lot behind the brown building at the intersection of Birchfield Road and Wellington Road, whe |
| HALFWAY_ALONG | 13 | {'Building': 7, 'Car': 6} | {'train_seen': 8, 'val_seen': 4, 'val_unseen': 1} | The building halfway down Thornbury Road that has no car in front of it and the start of the zigzag yellow line across the street at the school. |

## Frame cues

| Axis cue | Count | Example |
|---|---:|---|
| view_cue | 1784 | Gray building that is second from the right on the row of buildings above Willmore Road, when viewed aerially with Wellington road to the right. |
| building_long_axis | 778 | The gray car that is 1st from the right in the 12th row of cars in the large parking lot behind the One Stop Building, when viewed aerially with the long side of the One Stop Build |
| road_axis | 644 | Perpendicular to Wellington Road and at the end of a string of houses with red roofs, this smaller, square buildinghas a gray roof and is adjacent to St Teresa of the Child Jesus C |
| row_axis | 3 | In the One Stop parking lot, from the perspective of the long side of the building on the left and the short side at the top and the first row of cars being at the bottom and the f |

Target types are offline aggregate metadata from CityRefer object IDs; the parser and frame constructor receive instruction text only. Group C contains no layout/frame cue and at least one Gate F parsed generic spatial clause. Other instructions are outside A/B/C. Regex recall and precision are not assumed perfect.
