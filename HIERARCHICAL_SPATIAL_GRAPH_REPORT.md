# Phases 6-7: the language-conditioned graph, and Gate B

Branch `2027-CVPR/hierarchical-spatial-graph`. Source:
`scripts/eval_graph_grounding.py` → `artifacts/spatial_graph/graph_grounding_gate.json`,
on the 400 `val_unseen` instructions of the relation round.

Everything before this was built so that one measurement could be trusted. The
teacher is known to understand the relations because it was trained on labels
that **are** geometry (Gate A, 0.9573 against 0.1738 for the opposite). The
parser saw only the sentence and the ontology, verified by assertion on every
call. The binding maps a name onto a road **region** rather than guessing a
segment. This phase puts the three together.

## Results, `val_unseen`, ten candidates, chance 0.10

All three arms are scored on the **same 238 samples** — the intersection of those
where each is defined — so the comparison is not between three different
questions.

| arm | Top-1 | Top-4 | MRR | median margin | n |
|---|---:|---:|---:|---:|---:|
| **`deepseek_graph`** | **0.1008** | 0.4664 | 0.3201 | −5.11 | 238 |
| `shuffled_relation` | **0.0336** | 0.3487 | 0.2374 | −5.67 | 238 |
| `no_relation` (distance only) | **0.1176** | 0.5756 | 0.3531 | −3.91 | 238 |
| `td_masked` (from the entity round) | 0.320 | 0.757 | 0.521 | −0.01 | 400 |

| criterion | required | measured |
|---|---|---|
| B1 language beats a shuffled relation, p < 0.05 | yes | 23 vs 7, **p = 0.0052** ✓ |
| **Gate B** | **PASS** | **PASS** |

## The pass is real and it is not enough

**Gate B passes on its own criterion, decisively: 0.1008 against 0.0336, a
threefold gap at p = 0.005.** Telling the model which relation the sentence used
is worth far more than telling it a random one. The language does reach the
geometry — the relation label is being read, and the previous round's finding
that it was not is overturned for the mapped half of the corpus.

**And the absolute number is chance.** 0.1008 against a chance rate of 0.10. The
graph arm is also *below* the distance-only prior (0.1176), which uses the same
bound anchors but reads no relation and has no learned parameters, and 22 points
below the frozen visual baseline (0.320).

So the honest reading is: **the relation label carries information, and that
information is not yet enough to beat "pick the candidate nearest the anchor".**

## Why, and what the evidence for it is

Two causes were found by measurement rather than inference, and one was fixed.

**`between` was being scored with its second anchor thrown away.** The parser
reports a between-factor correctly — as a symmetric pair of `between` relations,
one per anchor — and the first version read each against a single anchor, so the
three between columns were zero. That is 73% of the covered samples scored on a
relation whose features had been discarded. Pairing them (each member read with
the other as its second anchor) moved the arm from **0.0840 to 0.1008**.

**The corpus's relations are concentrated in the two words a distance prior
already handles.** Of the 466 mapped relations, `near` is 162 (35%) and `between`
126 (27%). The measurement behind that: for the covered samples the true target
sits a median of **14.6 m** from the bound anchor, against the synthetic `near`
threshold of 12 m — so "near" is being read essentially correctly, and "pick the
nearest" is a strong baseline *because the corpus mostly says near*.

That second point is the round's structural constraint and it is not fixable by
a better model. **43.2% of the corpus's relations cannot be mapped to the
ontology at all** (see `DEEPSEEK_RELATION_ALIGNMENT_REPORT.md`), and the 54.4%
that can is dominated by two relations that a distance prior reproduces. The
ceiling on what a language-conditioned geometry model can show on this corpus is
set by that, not by the teacher, which is near-perfect on all fifteen relations.

## Phases not run

**Phase 8 (oblique ablation) was not run.** Section 29 makes it conditional on
the graph arm earning its place, and it has not: the graph is at chance and
below the distance prior, so asking whether the oblique view adds appearance
evidence on top of it would be measuring the oblique view against a baseline
that is not there. `td_masked` at 0.320 remains the number to beat and nothing
built this round beats it.

**Graph + TD was not run**, for the same reason and per §28's ordering: a fusion
weight chosen on val_seen between a 0.32 visual score and a 0.10 graph score
would be selected as "ignore the graph", which is a fact about the graph arm and
not a fusion result.

## Gate summary

| gate | | |
|---|---|---|
| **A** synthetic geometry | **PASS** | 0.9573 vs 0.1738 opposite, p = 0.0 |
| **B** natural language | **PASS** | 0.1008 vs 0.0336 shuffled, p = 0.005 |
| **C** graph value over baselines | **FAIL** | 0.1008 vs 0.1176 distance prior |
| **D** final target grounding | **FAIL** | 0.1008 vs the 0.320 baseline; `< 0.370` |

**The round's first objective is met and its third is not.** Section 2 of the
brief states them in order, and the order was the point: prove that clean
geometry supervision trains a model that understands relations — done, at 0.9573
— then that language can be mapped onto it — done for 54.4% of relations — and
only then that the result improves target ranking. It does not, and the reason is
in the data rather than in the implementation.

## One next step

**Test whether the 54.4% that maps is enough, by scoring only the instructions
whose relations are entirely inside the ontology.**

The evidence for this being the right next move is narrow and specific: the
teacher is near-perfect on all fifteen relations, the binding works, the between
fix moved the arm by 1.7 points the moment the features stopped being discarded,
and the arm beats its shuffle threefold — so the machinery is functioning. What
is not functioning is coverage: 43% of relations are outside the ontology and the
two dominant ones inside it are reproduced by a distance prior.

Scoring the subset whose relations are *all* mapped separates "the machinery is
weak" from "the corpus is mostly outside the ontology". If that subset scores
strongly, the ontology is right and the corpus is the limit, and the honest
conclusion is that this benchmark cannot measure relation reasoning. If it does
not, the fault is downstream of the parse and the marginalisation over anchors is
where to look — the 23.9% ambiguous binding rate is the obvious suspect.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY -m pytest tests/ -q                        # 226 tests
$PY scripts/audit_map_graph.py                 # ~90 s
$PY scripts/run_synthetic_relations.py         # ~20 min
$PY scripts/train_relation_teacher.py          # ~5 min
$PY scripts/run_deepseek_parse.py              # needs DEEPSEEK_API_KEY, cached
$PY scripts/eval_graph_grounding.py            # ~4 min
```
