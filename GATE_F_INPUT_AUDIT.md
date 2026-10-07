# Gate F input audit

Inspected 2,290 existing diagnostic rows by joining `split + episode_index` to raw CityNav episodes and `object_ids[0] + ann_ids[0]` to CityRefer objects/processed descriptions.

| Checked source value | Valid rows |
|---|---:|
| Episode index resolves | 2,290 |
| Instruction exact match | 2,290 |
| Map/block exact match | 2,290 |
| Target ID exact match (offline only) | 2,290 |
| Trajectory has >1 steps, six columns | 2,290 |
| Saved UAV XY matches trajectory step | 2,290 |
| Annotation ID resolves | 2,290 |
| Annotation description exact match | 2,243 |
| Annotation description whitespace-normalized match | 2,290 |
| Finite target object position | 2,290 |
| Target contour with >=3 points | 2,290 |
| Processed description has landmarks | 2,289 |
| Processed description has >=2 landmarks | 1,130 |

CityNav trajectory columns are verified from `citynav._load_split_episodes`: `(x,y,z,dx,dy,dz)`. The last three are recorded look directions; motion headings are computed separately from successive XYZ positions.

RoadRegion is built from CityRefer TrafficRoad objects by name and connected component in `spatial_graph.road_region`. Named road binding must retain every matching region. Building contours come from the same CityRefer object loader.

A second anchor is **not** implied by two landmarks alone. It must appear in a `between A and B` clause and both names must bind. Target positions and IDs are read only in this audit and offline evaluation.

## Extended rows
The 2,290 legacy samples produced 5,427 annotation/parser clause rows before supplementation. 3,708 bound at least one anchor; 1,938 bound a named RoadRegion; 256 between clause rows had a text-supported second anchor. These counts include both parsing paths.
A separate, randomized raw-corpus road sample added 516 rows with deterministic map distractors. Its ranking difficulty differs from the legacy candidate lists, so its metrics are reported separately.
The controlled DeepSeek sample added 287 text-derived clause rows; 180 bound at least one named map hypothesis. These share the legacy candidate protocol and are reported as a separate source.
All paths together contain 6,230 clause rows from 2,416 unique episodes: 6,230 carry trajectory, 2,536 have a named RoadRegion hypothesis, and 264 have a bound second anchor for between (117 annotation-derived). Schema omissions and duplicate sample IDs: 0/0.
Annotation-derived anchor names come from the target object and are an **offline diagnostic oracle**. Parser-derived anchors come only from instruction text. Neither path uses GT position to pick an anchor or road segment.
