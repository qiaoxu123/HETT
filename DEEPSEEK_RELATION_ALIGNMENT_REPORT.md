# Phases 4 and 5: mapping language onto the ontology

Branch `2027-CVPR/hierarchical-spatial-graph`. Source:
`scripts/run_deepseek_parse.py` → `artifacts/spatial_graph/deepseek_parse_val_unseen*.json`
over all 400 `val_unseen` instructions of the relation round, same instructions,
same targets, same candidate order.

The division of labour is the round's point, and it is enforced rather than
described. DeepSeek decides **which relation the sentence used**. It is never
asked which of a road's 9 segments is meant, never shown the target, and never
shown a candidate. The binding to map entities happens afterwards, in this
repository, against the graph.

## The model sees the sentence and the ontology, and nothing else

`build_request` is the only place a request is assembled, so a single test
guards every call: `check_no_leakage` rejects `target_id`, `target_position`,
`is_target`, `candidate_id`, `candidate_index`, `object_ids`, `ann_ids`,
`description_landmarks` and the rest. The guard caught its own first version
being wrong — it listed the English word "answer", which the system prompt uses
legitimately when telling the model it is not being shown one — and now lists
identifier-shaped tokens only.

The key is read from the environment (`DEEPSEEK_API_KEY`, or this machine's
`ANTHROPIC_AUTH_TOKEN`), goes only in the `Authorization` header, and appears in
neither the cache nor the repository; a test asserts the cache holds no
credential, and `artifacts/` is gitignored.

## Results, 400 instructions

| | |
|---|---:|
| valid parses | **400 / 400** |
| relations emitted | 857 |
| mapped to the ontology | 466 (**54.4%**) |
| **UNKNOWN** | **370 (43.2%)** |
| ASSOCIATED_WITH | 21 (2.5%) |
| anchors bound to a node | 608 / 840 (72.4%) |
| of those, bound to a **RoadRegion** | 236 |
| ambiguous bindings | 201 (23.9%) |
| prompt / completion tokens | 428,118 / 142,111 |

## The finding: 43% of the corpus's relations are not mappable

**What comes back UNKNOWN is almost exactly the vocabulary the ontology
deliberately excludes.**

| raw phrase | count | | raw phrase | count |
|---|---:|---|---|---:|
| behind | 54 | | bordered by | 6 |
| in front of | 44 | | surrounded by | 6 |
| to the left of | 13 | | connected to | 5 |
| in | 11 | | surrounding | 5 |
| on the right side of | 8 | | parked in front of | 4 |
| off | 8 | | on the left side of | 3 |
| to the right of | 7 | | at the right side of | 3 |

Two independent routes reach the same conclusion, and they share no information.
Last round's **geometry audit** measured the corpus's own usage and found that
left/right and front/behind select no consistent geometric frame. **DeepSeek**,
reading the same sentences with no map, no target and no candidates, declines to
map those same words onto well-defined geometry. One is a measurement of the
corpus; the other is a language model's judgement about the words. They agree.

What *does* map is dominated by proximity and betweenness:

| relation | count | | relation | count |
|---|---:|---|---|---:|
| near | 162 | | opposite_side_of_road | 23 |
| between | 126 | | north_of | 10 |
| along_road | 84 | | east_of | 10 |
| near_intersection | 30 | | south_of / west_of | 12 |

The practical consequence is stated plainly because it decides what the next
phase can possibly show: **`near` alone is 35% of the mapped relations and
`between` another 27%.** A language-conditioned model evaluated on this corpus
is mostly being asked whether it can read "near" and "between".

## Consistency under reordering

Section 19 asks for the same instruction to be parsed twice with the ontology
list presented in a different order. A model reading the sentence returns the
same relations; one reading position does not.

| | |
|---|---:|
| relation sequences identical | 0.7525 |
| relation Jaccard | 0.8873 |
| anchor Jaccard | 0.8902 |

High, but the 25% that change are a real sensitivity to presentation order and
are reported rather than rounded away. Nothing downstream depends on a single
parse being stable: the reasoner marginalises over anchors, and an unstable
parse contributes an unstable column rather than a wrong answer.

## Two ways this report could have been wrong, and is not

**The parser could have been reading the answer.** It is not: the request
contains the instruction text and a fixed ontology listing, verified by assertion
on every call, and the model has no access to `description_landmarks` or any
other annotation, all of which are target-keyed.

**The binding could have been picking the target.** It is not: a name binds to
every node carrying it — regions first — and keeps all of them as hypotheses.
Where a name matched several nodes (23.9% of anchors) nothing picks a winner.

## What this phase does and does not establish

It establishes that a hosted model can be held to a fixed ontology, that it will
decline to force a mapping rather than guess, and that its refusals coincide with
what the geometry audit found independently. It does **not** establish that the
mapped half is enough: 43% of the corpus's relations are outside the ontology by
construction, and the largest words inside it are the two the previous round
already found carry a proximity signal that a distance prior reproduces.

## Reproducing

```bash
export DEEPSEEK_API_KEY=...      # never committed
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY -m pytest tests/test_deepseek_parser.py -q     # 14 tests, no network
$PY scripts/run_deepseek_parse.py                  # ~25 min, cached
$PY scripts/run_deepseek_parse.py                  # ~5 s, all cache hits
```
