# Phases 2 and 3: clean supervision, and the gate that judged it

Branch `2027-CVPR/hierarchical-spatial-graph`, on `f79f64f`. The previous round
ended with a reasoner at chance and a diagnosis that could not be tested from
the outside: the corpus's relation words do not select any geometry, so training
on them teaches nothing and fails invisibly. This round changes what the labels
are made of. The labels are the map's own geometry, so a model that fails to fit
them is failing at something demonstrably learnable — and that is a difference
the gate can see.

**The short answer: Gate A passes, at 0.9573 against 0.1738 for the opposite
relation.** The same architecture that scored 0.125 against 0.10 chance last
round scores 0.9573 here. The failure was the supervision, not the geometry and
not the model.

## What the labels are

`sensaturban_fpv/spatial_graph/` defines an ontology geometrically — no English
in the loop — and generates training pairs from the map alone, reading no
instruction text anywhere:

```
north_of south_of east_of west_of      near far      between
same_side_of_road opposite_side_of_road across_road   along_road
near_intersection    aligned_with parallel_to perpendicular_to
```

Two exclusions, both deliberate. **No viewpoint-dependent relations**: `behind`,
`in front of`, `left of`, `right of` each need a reference frame, the previous
round's audit found no frame the corpus agrees on, and admitting them would
smuggle that failure back in at the point where it could no longer be seen. **No
contact relations**: `on` is the corpus's most frequent relation word and the
audit could not pin it to any geometry.

Each relation returns a **signed margin in metres**, not a boolean, so a
threshold can be swept without redefining the geometry and so "barely holds" is
distinguishable from "clearly holds".

## The split is by map, not by CityNav split

Measured, not assumed: **`val_seen` shares all 23 of its maps with
`train_seen`**. Only `val_unseen`'s four maps are disjoint. For instruction-level
work that is a defensible split; for supervision derived from the map's own
geometry it is not, because a held-out *episode* on a map the model has already
seen is not held out at all. So the 30 maps outside `val_unseen` divide by map:

| | maps | examples | negative pairs |
|---|---:|---:|---:|
| synthetic-train | 20 | 249,282 | 1,230,492 |
| synthetic-val | 10 | 60,942 | 309,299 |
| synthetic-test | 4 | 31,415 | 160,916 |

The test maps — `birmingham_block_5`, `cambridge_block_10`,
`cambridge_block_2`, `cambridge_block_3` — appear in no training or
threshold-selection set. All 15 relations are represented, from 10,053
(`near_intersection`) to 30,247 (`same_side_of_road`) training examples.

## Hard negatives are matched on everything except the relation

A negative drawn at random lets a model separate positive from negative on
distance, entity kind or local density without ever reading the relation — which
is what the previous round's reasoner did. Each negative is therefore drawn from
the **same kind group** and the **same distance band** (0.5×–2× the positive's
centre distance) as the positive, and differs only in failing the relation.

## Thresholds: two criteria were built and both were degenerate

Section 11 forbids inventing thresholds. Three attempts were needed, and the two
failures are worth recording because each looked reasonable:

| criterion | what it did | why |
|---|---|---|
| maximise the number of examples | walked every threshold to its **loosest** | loosening admits positives faster than it removes the negatives they are compared against, so the count is monotone. Result: "everything is north of everything". |
| maximise the total margin retained | walked every threshold to its **tightest** | each surviving example then contributes more, so the sum is monotone the other way. Result: `near_m == far_m == 140`, a distance that is simultaneously near and far. |
| **match a target selectivity** | picks an interior value | the relation should hold for a stated fraction of nearby pairs |

Neither direction of the first two is a fact about the corpus; both are artefacts
of the objective. The third is a **design decision and is reported as one** — no
data in this project says how far north "north of" starts. What is
data-determined is the grid it is chosen from, which comes from the maps' own
distance quantiles, and the rate is evaluated on the val maps.

| threshold | value | | threshold | value |
|---|---:|---|---|---:|
| `cardinal_margin` | 42 m | | `intersection_m` | 11 m |
| `near_m` | 12 m | | `along_cos` | 0.97 |
| `far_m` | 147 m | | `align_cos` | 0.97 |
| `between_perp_m` | 9 m | | `perp_cos` | 0.35 |
| `between_t_margin` | 0.05 | | `side_min_m` | 2.0 m |

`near_m` 12 < `far_m` 147 is the check that matters: the second criterion had set
both to 140.

## Gate A

Synthetic-test, 31,415 sets, each one positive against up to eight matched
negatives. Teacher: 44,289 trainable parameters, an MLP over the 32-column edge
vector conditioned on a 48-dimensional relation embedding. Chosen on
synthetic-val (`lr` 1e-2, hidden 128, 18 epochs).

| arm | Top-1 | Top-4 | MRR | median margin |
|---|---:|---:|---:|---:|
| **correct relation** | **0.9573** | 0.9885 | 0.9726 | +16.87 |
| opposite relation | **0.1738** | 0.5770 | 0.3906 | −8.86 |
| shuffled relation | 0.3433 | 0.7121 | 0.5303 | −2.59 |
| zero relation | 0.3085 | 0.7215 | 0.5129 | −1.54 |
| shuffled anchor | 0.3376 | 0.7007 | 0.5262 | −3.17 |

| criterion | required | measured |
|---|---|---|
| A1 correct > shuffled relation, p < 0.05 | yes | 19,542 vs 255, **p = 0.0** ✓ |
| A2 correct > opposite relation, p < 0.05 | yes | 24,888 vs 274, **p = 0.0** ✓ |
| A3 zeroing the relation hurts by > 0.10 | yes | 0.957 − 0.309 = **0.649** ✓ |
| A4 shuffling the anchor hurts by > 0.10 | yes | 0.957 − 0.338 = **0.620** ✓ |
| A5 counterfactual AUC > 0.5 | yes | **0.616** ✓ |
| **Gate A** | **PASS** | **PASS** |

Pairwise AUC on the ranking is **0.9753**. Every control was verified to move the
scores before its numbers were reported (86–169 in absolute score) — a control
that silently no-ops reads as a pass, and two earlier attempts at an
anchor-shuffle control in this project were exactly that.

### The gate earned its place: it found two relations the model could not *see*

The first run passed overall at 0.7725, and the per-relation breakdown is what
made that pass worth having:

| relation | first run | after the fix | opposite |
|---|---:|---:|---:|
| **between** | 0.2399 | **0.9913** | 0.357 |
| **parallel_to** | 0.3481 | **0.9985** | 0.173 |
| **perpendicular_to** | 0.3065 | **0.9979** | 0.186 |

Neither was a relation the model failed to learn. Both were relations it was
**never shown**:

* `parallel_to` and `perpendicular_to` depend on the angle between two entities'
  footprint axes. The schema had only `anchor_major_coord` and
  `anchor_minor_coord`, which project the *displacement* onto the anchor's axes —
  a positional fact, not an orientation one. `aligned_with` reads the
  displacement and scored 0.96; the two that read the axes scored chance. Two
  columns added: `axis_parallel`, `axis_perp`.
* `between` is a ternary factor and its three columns were always zero, because
  `pair_facts` never passed a second anchor into the edge context. An all-zero
  feature column is unlearnable by construction, and scored exactly chance — the
  correct score for a relation that cannot be seen.

Between now gets its own generation pass, because the positive and every negative
must be read against the **same** second anchor or the comparison is between two
different questions. The second anchors are chosen by geometry alone and never by
which one best brackets the target, which would make the relation true by
construction.

### One relation got worse and is reported as such

`near_intersection` fell from 0.5553 to **0.4238**, and its opposite and shuffled
arms are identical at 0.1818. The intersection threshold moved from 12 m to 11 m
between runs and the test set holds only 682 examples of it, so this is a
relation that the current graph supports weakly — 596 intersection nodes over 34
blocks is sparse, and a "near the junction" instruction may be about a junction
the annotation does not mark. It is the weakest relation in the ontology and the
report does not claim otherwise.

## What this settles

The same question the previous round could not answer is answered here. A
relation-conditioned scorer given the right anchor and the right relation ranks
the target at **0.9573** on maps it has never seen, and the **opposite** relation
scores 0.1738 — the model has learned the *direction*, not merely that some
relation holds. The previous round's 0.125 was not a limit of the architecture or
of the geometry. It was the labels.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY -m pytest tests/ -q                     # 226 tests
$PY scripts/run_synthetic_relations.py      # ~20 min
$PY scripts/train_relation_teacher.py       # ~5 min
```

Artefacts: `artifacts/spatial_graph/{synthetic_relations_*.jsonl,
synthetic_relations_stats.json, synthetic_teacher_gate.json}`. The pre-fix gate
is kept as `synthetic_teacher_gate_prerefix.json`, since it is the evidence for
the two schema bugs.
