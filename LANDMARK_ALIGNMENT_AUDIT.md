# Landmark name-to-image alignment audit

## Question

Does a landmark name in an instruction identify the correct annotated map
object, and can the current frozen visual model recognize that object in RGB?
These are separate checks: annotation lookup can be correct even when the
visual model cannot infer a proper name from aerial appearance.

## Symbolic name-to-map alignment

Across train_seen, val_seen and val_unseen, 99.46% of 42,177 landmark mentions
normalize to an exact object name. Only 227 mentions require fuzzy matching;
148 mentions (0.35% overall) have either a score below 0.8 or a top-two margin
below 0.05.

On val_unseen, 4,162/4,165 mentions (99.93%) are exact. Two of the three fuzzy
cases are risky: `Mallory Court` resolves to `Mallory Court, NW Range`, while
`The Quadrangle` incorrectly resolves with low confidence to
`The Museum of Cambridge`.

Exact names are not always spatially unique. On val_unseen, 32.61% of mentions
have multiple contours with the same normalized name. All such duplicates are
roads, footpaths, bridges or water features; no duplicated building names were
found. On val_seen the ambiguous rate is 47.85%, also dominated by segmented
roads and footpaths, with 37 duplicated-building mentions. The current lookup
selects only the last matching contour. Linear landmarks should instead union
all same-name contours.

Every selected val_unseen landmark centroid falls inside its georeferenced RGB
raster. This confirms coordinate-system consistency, but not semantic visual
recognition.

## Frozen SigLIP2 visual retrieval

For every referenced landmark, a 160 m RGB crop was centered on its annotated
contour. Each landmark text query was compared only against candidate landmark
crops from the same map. Duplicate contours with the queried name were all
accepted as correct.

| Split / query | Queries | Top-1 | Random Top-1 | MRR |
|---|---:|---:|---:|---:|
| val_seen / proper name | 449 | 5.57% | 5.12% | 14.55% |
| val_unseen / proper name | 99 | 3.03% | 4.04% | 13.32% |
| val_seen / annotated appearance (diagnostic oracle) | 255 | 12.55% | 2.23% | 22.29% |
| val_unseen / annotated appearance (diagnostic oracle) | 57 | 28.07% | 2.68% | 42.34% |

Proper-name retrieval is at chance on seen maps and below chance on unseen
maps. The current frozen SigLIP2 encoder therefore does not reliably associate
names such as `Pepys Library` or `One Stop` with their aerial RGB appearance.

The appearance-description diagnostic is substantially above chance. This
shows that the crops contain useful visual evidence and that the encoder can
recognize some roof, shape, color and surrounding-context language. These
human object descriptions are not part of the deployed input and must not be
used as test-time side information; the result is diagnostic only.

## Decision

The current system has a mostly reliable symbolic name-to-annotation mapping
for buildings, but it does **not** have reliable name-to-RGB grounding. Treat
the annotated landmark masks as the authoritative landmark locations. Do not
ask the visual model to rediscover a proper-named landmark from its name alone.

Before further target-location training:

1. union all same-name road/path/water contours rather than selecting one;
2. reject or manually alias low-confidence fuzzy matches such as
   `The Quadrangle`;
3. use each known landmark mask to extract visual appearance features;
4. train target matching from observable appearance phrases and relative
   geometry, while retaining landmark names only as symbolic identifiers.

## Implemented follow-up

The resolver now returns all exact same-name components and rejects fuzzy
matches below 0.58 similarity or below a 0.05 top-two margin. Multi-component
centroids are area weighted and 16 boundary samples are distributed over the
combined perimeter. This removes the previous last-contour-wins behavior while
leaving proper names as symbolic identifiers.

With map-level out-of-fold candidate generation, the fix reduces val_unseen
top-1 mean error from 55.67 m to 45.47 m and raises Hit@50 from 63.63% to
65.70%. Top-5 oracle mean error falls from 39.19 m to 29.63 m and Top-5 Hit@50
rises from 79.31% to 82.83%. Tight top-1 Hit@20 falls from 21.95% to 20.76%, so
fine candidate ordering remains unresolved.
