# Road sequence reasoning

Attempted 191 text-detected road sequence instructions. Status: {'start_or_corner_unresolved': 40, 'road_unresolved': 116, 'scope_miss': 7, 'scope_too_small': 28}.

A named road must resolve to one connected RoadRegion. Candidate objects are selected by text-inferred class within 15 m of that region. A curvilinear coordinate is computed from projection onto the region polyline. The GT is consulted only after the candidate set exists.

A road polyline has two ends. “First”, “last”, or “from the corner” cannot choose an end without a text-bound start/corner. These clauses abstain; selecting the end nearest the GT would leak the answer. HALFWAY_ALONG is orientation invariant and can be evaluated when a named road and candidate class bind.

| Split | Evaluable halfway clauses | Top1 | MRR | AUC |
|---|---:|---:|---:|---:|
| train_seen | 0 | — | — | — |
| val_seen | 0 | — | — | — |
| val_unseen | 0 | — | — | — |
