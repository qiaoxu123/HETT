# Gate E: what the graph is worth where the language is inside the ontology

Branch `2027-CVPR/ontology-covered-subset`, on `44b6031`. Source:
`scripts/eval_ontology_subset.py` → `artifacts/spatial_graph/ontology_subset_gate.json`.

Last round measured the language-conditioned graph at 0.1008 against a distance
prior's 0.1176 and could not say why. Two explanations called for opposite
decisions, and this round separates them:

* **the corpus is mostly outside the ontology**, so nothing downstream could have
  shown anything, or
* **the mapping is formal** — it succeeds without carrying target-discriminative
  information even where it succeeds.

Nothing is retrained. The teacher is the one Gate A judged, the parses are the
cached ones, no threshold or weight is fitted here, and the subsets are defined
by the parse and the map, both of which are fixed before the answer is consulted.

## Answer

**Relation used, no incremental value — and it is not the binding's fault.**

On the subset where the ontology covers every spatial relation *and* every anchor
resolves to exactly one node, the graph scores **0.0545** against a distance
prior's **0.1273**. It loses, and it is *worst* precisely where the binding is
cleanest. Binding ambiguity is therefore not what suppresses it: the ambiguous
subset scores **0.2174**, the graph's best result anywhere in this project.

The relation is nonetheless demonstrably being read: on the same subset, showing
the model the sentence's own relation beats showing it the opposite one, on the
true target, **23 times out of 32, p = 0.020**, with mean scores of −0.60 against
−41.53.

## The subsets

400 `val_unseen` instructions. A colour or a size word is not a spatial relation
and never disqualifies a sample; `ASSOCIATED_WITH` asserts no geometry and does
not either.

| parse class | n | share |
|---|---:|---:|
| **ONTOLOGY_COVERED** — every spatial relation maps | **138** | 34.5% |
| PARTIAL_COVERED — some map, at least one does not | 129 | 32.3% |
| UNKNOWN_ONLY — none map | 126 | 31.5% |
| NO_SPATIAL_RELATION | 7 | 1.8% |

Binding, within the covered 138. Only anchors that a *mapped* relation uses are
counted — an anchor introduced by an UNKNOWN relation never scores, so calling a
sample ambiguous because of it would measure the wrong thing:

| binding class | n | share of covered |
|---|---:|---:|
| **BINDING_CLEAN** — one node per anchor | **55** | 39.9% |
| BINDING_AMBIGUOUS — at least one resolves to several | 46 | 33.3% |
| BINDING_UNRESOLVED — at least one resolves to none | 37 | 26.8% |

A unique **RoadRegion** counts as one node, since the region is the entity the
name denotes. A name that reaches two *disconnected* regions of the same road is
ambiguous, and 59 such names exist in the corpus.

## The core table

Every arm on the same samples — the intersection of those where each is defined —
so the rows are comparable rather than three different questions. Chance is 0.10.

| subset | n | distance prior | shuffled relation | **DeepSeek + graph** |
|---|---:|---:|---:|---:|
| ONTOLOGY_COVERED (all) | 133 | 0.1203 | 0.0677 | 0.1053 |
| **ONTOLOGY_COVERED + BINDING_CLEAN** | **55** | **0.1273** | **0.0182** | **0.0545** |
| ONTOLOGY_COVERED + BINDING_AMBIGUOUS | 46 | 0.1522 | 0.0217 | **0.2174** |
| NON_DISTANCE (no `near`/`far`) | 77 | 0.1429 | 0.0779 | 0.0909 |
| MULTI_ANCHOR | 102 | 0.1176 | 0.0588 | 0.1078 |
| PARTIAL_COVERED | 105 | 0.1143 | 0.0476 | 0.0952 |

`TD_masked` remains **0.320** as a reference line; no fusion is attempted here.

### The three numbers §17 asks for

| | |
|---|---:|
| **A.** BINDING_CLEAN, DeepSeek + graph | **0.0545** |
| **B.** BINDING_CLEAN, distance prior | **0.1273** |
| **C.** BINDING_CLEAN, shuffled relation | **0.0182** |

Graph > shuffled, but graph ≤ distance. That is §18's second case:
**RELATION USED BUT NO INCREMENTAL VALUE.**

## Paired tests on the two clean subsets

| comparison | subset | a only | b only | p |
|---|---|---:|---:|---:|
| graph vs shuffled | ONTOLOGY_COVERED | 12 | 7 | **0.359** |
| graph vs distance | ONTOLOGY_COVERED | 11 | 13 | **0.839** |
| graph vs shuffled | BINDING_CLEAN | 3 | 1 | **0.625** |
| graph vs distance | BINDING_CLEAN | 2 | 6 | **0.289** |

**None of the four reaches significance.** And there is a result here that
matters as much as the headline: on the covered subset the graph-versus-shuffled
comparison is p = 0.359, where on the full 238 samples last round it was
p = 0.005. Two things change together — the covered subset is smaller (133 vs
238, so fewer discordant pairs to test with) and the ratio weakens (12:7 against
23:7). **The previous round's significant advantage over the shuffled control
does not survive restriction to the samples whose language is inside the
ontology**, which is a caution about that result rather than a vindication of
this one.

## Counterfactual: the relation is read, and it is not enough

On ONTOLOGY_COVERED + BINDING_CLEAN, the true target scored under the sentence's
own relation against the same target under the relation's opposite:

| | |
|---|---:|
| n | 32 |
| mean score, correct relation | **−0.60** |
| mean score, opposite relation | **−41.53** |
| win rate | **0.719** (23 wins, 9 losses) |
| paired p (binomial) | **0.020** |

This is the round's clearest positive. The model is not merely producing a
constant: it separates the correct relation from its opposite on real language,
at the target, on samples where the ontology covers the sentence and the anchors
are unambiguous. §12's requirement is met.

It is also why the verdict is "used but no incremental value" rather than
"language alignment fail". The relation reaches the geometry. What it does not do
is rank candidates better than the distance prior does.

## Per relation, on ONTOLOGY_COVERED

| relation | n | distance | shuffled | graph | graph MRR |
|---|---:|---:|---:|---:|---:|
| near | 55 | 0.0909 | 0.0182 | **0.1273** | 0.368 |
| along_road | 44 | **0.1591** | 0.0227 | 0.0455 | 0.263 |
| between | 34 | **0.1765** | 0.1176 | 0.1176 | 0.289 |
| near_intersection | 15 | 0.2000 | 0.0000 | **0.2667** | 0.493 |
| opposite_side_of_road | 11 | 0.0000 | 0.0909 | **0.1818** | 0.434 |
| north_of / south_of / east_of / west_of | 8 / 3 / 7 / 2 | — | — | 0.000 | 0.15–0.38 |
| far, same_side, across, parallel, perpendicular | 1–2 each | — | — | — | — |

Three readings.

**`near` is where the graph beats the prior** (0.1273 against 0.0909). So the
graph is not simply worse than distance everywhere; it is better exactly on the
relation a distance prior approximates, which is consistent with the teacher
having learned `near` from geometry.

**`along_road` and `between` are where it loses badly.** `between` at 0.1176 ties
the shuffled arm exactly, on the relation §11 singles out as the important
multi-anchor case. The second anchor is now paired correctly — components
sections below — and it still does not beat the prior.

**The directional relations have 2–8 samples each.** Last round established from
the corpus side that the compass words occur 26, 10, 0 and 1 times; DeepSeek maps
30 of them across 400 instructions. No claim about them is supportable at this n,
and the table says so rather than reporting a zero.

## Multi-anchor subset

102 samples (n=102 scored): distance **0.1176**, shuffled 0.0588, graph
**0.1078**. The graph does not beat the prior here either.

§11 asks, if `between` does not improve under clean binding, whether the two
anchors are reaching the factor. They are — and the check is worth reporting
because the previous round got it wrong. The parser reports a between-factor as a
**symmetric pair** of `between` relations, one per anchor. Components read each
against a single anchor, so the three between columns were zero for **73%** of
the samples it scored. Pairing them moved the full-set arm from 0.0840 to 0.1008
last round. `tests/test_ontology_subset.py` now asserts both directions: one
relation has no second anchor and leaves the column at zero, and the symmetric
pair populates it. So the mechanism is verified and the result stands on its own:
`between` does not beat the distance prior with the factor wired correctly.

## Ambiguity: marginalisation beats both commitment and the oracle

On BINDING_AMBIGUOUS (n=46):

| variant | Top-1 | Top-4 | MRR |
|---|---:|---:|---:|
| hard top-1 binding | 0.1522 | 0.587 | 0.354 |
| **uniform marginalisation (deployable)** | **0.2174** | 0.478 | 0.397 |
| ORACLE anchor (diagnostic only) | 0.1522 | 0.565 | 0.344 |

**Marginalising over the anchor hypotheses beats committing to one, and beats
being handed the right one.** That is the same shape the language-anchor round
found and it is worth stating plainly: with several hypotheses to average over,
a reasoner that is sometimes wrong is usable and one that is forced to choose is
not. The ORACLE row is diagnostic, uses the true target to pick the hypothesis,
and is never mixed into a deployable table.

The graph's best number anywhere in this project — 0.2174 — is on the subset
whose binding is *ambiguous*, which is the direct answer to the round's secondary
question: **binding ambiguity is not what suppresses the graph.**

## Leakage controls

* No arm reads `target_position`, `object_ids`, `candidate_index` or any GT
  target field to choose an anchor; a test asserts `binding_status` has no such
  name in its source, and `graph_scores` likewise.
* **Verified: in 0 of 138 covered samples is the target among the bound anchors
  of a used relation.** The ambiguous-subset result is therefore not a degenerate
  self-anchor effect.
* `val_unseen` is evaluated once; no threshold, weight or subset boundary is
  fitted on it — the subsets come from the parse and the map.
* The ORACLE anchor arm is named as such, reported only in its own table, and
  kept out of every deployable row.

## Failure analysis

§8 anticipated three outcomes. The third is what happened, with a qualification:

* *Graph ≈ shuffled* would mean language alignment fails outright. It does not:
  the counterfactual separates correct from opposite at p = 0.020, so the words
  reach the geometry.
* *Graph >> distance* would mean the graph method stands and the headline 0.10
  was coverage and ambiguity. It does not: the graph loses to the prior on the
  cleanest subset, and the ambiguous subset — not the clean one — is where it
  does best.
* **What holds is the middle case**: the relation is read, and it carries no
  information the candidate's distance to the anchor does not already carry.
  The reason is visible in the per-relation table — the graph's one win is on
  `near`, which is the relation a distance prior *is*, and it loses on
  `along_road` and `between`, which are the relations it would have to win.

## Gate E

| criterion | required | measured |
|---|---|---|
| graph > shuffled relation, significantly | yes | 3 vs 1, p = 0.625 ✗ |
| graph ≥ distance prior + 3 points | yes | 0.0545 vs 0.1273, **−7.3** ✗ |
| MRR improves over the prior | yes | 0.2949 vs the prior's **0.3601** ✗ |
| correct relation > opposite relation | yes | 23/32, **p = 0.020** ✓ |
| **Gate E** | | **RELATION USED BUT NO INCREMENTAL VALUE** |

## One next step

**Stop building on this ontology and test whether the graph's one win — `near`,
0.1273 against the prior's 0.0909 — survives at a scale where it can be
measured.**

The evidence is narrow and specific. The graph loses on the cleanest subset, so
the problem is not coverage or ambiguity; it loses on `along_road` (44 samples)
and `between` (34), so it is not a single broken relation; and it *wins* on
`near` (55) and `near_intersection` (15), which is small but is the only place in
this project where a relation-conditioned model has beaten a geometric prior on
language. 55 samples is ten correct answers either way, so that win is not yet
established and should not be reported as one.

That makes `near` the only live thread, and it is worth one round: take the
`near` cases across all 400 instructions rather than the 55 inside the covered
subset, and ask whether the graph's advantage is real. If it is, the graph's
value is as a *proximity* model with better structure than a raw distance, and
the honest framing of the whole line is that language contributes proximity and
not topology. If it is not, the graph adds nothing on this corpus at any coverage
level and the line should be closed rather than refined.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY -m pytest tests/ -q                      # 251 tests
$PY scripts/eval_ontology_subset.py          # ~3 min, no training
```
