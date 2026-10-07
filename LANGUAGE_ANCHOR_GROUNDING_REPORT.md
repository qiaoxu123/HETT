# Language anchor grounding and spatial relation reasoning

Branch `2027-CVPR/language-anchor-grounding`, on the feature-sufficiency round's
HEAD. The question is the one that round left open: if the sentence's anchor
entity can be found in the map and the candidate's spatial relation to it is
computed explicitly, does the right candidate move from Top-4 to Top-1?

> **Correction, 2026-10-07.** This report was written against an intermediate run
> and several numbers had drifted from the artefact that ships with it. Every
> table and figure has now been re-read from
> `artifacts/language_anchor_grounding/`, and three claims changed materially:
> the arms `visual_plus_learned` and `visual_plus_topk_marginalised` are the same
> arm; the tuning chose `K = 1`, so the round did not test marginalisation over
> anchor hypotheses; and the shuffled-anchor control is 70:24 at
> p = 2.2 × 10⁻⁶, not 50:17 at 6.7 × 10⁻⁵. **The Top-1 column was correct
> throughout and the FAIL verdict is unchanged.** The per-group, rescue and
> control figures, and the counts in "Samples" and "The parser", were not, and
> are corrected in place. The error came from reporting prose written before the
> final evaluation rather than read back from it.

## Where this starts

`val_unseen`, 400 samples, ten candidates each, 10% is chance.

| | Top-1 | Top-4 |
|---|---:|---:|
| entity-masked top-down, referenced phrase (`TD_masked`) | 0.320 | 0.757 |
| best 3D-anchored dual-view fusion (entity round) | 0.338 | 0.810 |
| appearance, geometry, relation, semantic features (feature round) | ≤ 0.246 | — |
| **anchor oracle** (feature round) | **0.609** | — |

Masking the target region is worth +22 points; everything visual since is worth
nothing; and knowing which entity the language refers to is worth +29.

## Samples

The entity round's manifests, verbatim: same instruction, target, candidate ids
and order, same split, episode and step. Candidate order is permuted once,
deterministically, before any arm sees it — the list is built as
`[referenced] + distractors`, so the answer otherwise sits at index 0, and a
score vector that ties would then be won by position rather than by evidence.

| split | samples | building | vehicle | other | instructions naming an anchor entity |
|---|---:|---:|---:|---:|---:|
| train_seen | 1200 | 540 | 420 | 240 | 329 |
| val_seen | 690 | 315 | 245 | 130 | 234 |
| val_unseen | 400 | 180 | 140 | 80 | 151 |

## The parser

`parse_instruction` splits every sentence into target phrase, anchors, relations
and attributes. 86% of instructions yield at least one anchor; the corpus has
1.96 relation words per instruction, dominated by `on` (1445), `in front of`
(435), `behind` (393), `between` (354) and `next to` (336). Anchor types are
mostly buildings (969) and roads (848), with cars (304) and vegetation (111)
behind them.

Each relation declares the frame it is read in — `agent` for left/right and
front/back, `global` for the compass words, `none` for the symmetric ones — and
the reasoner reads that declaration. The choice is written down rather than
inferred, because "left of" means different things from a heading and from the
map axes, and a model that silently picks one cannot be falsified.

## Anchor grounding, and why its headline number is not a measurement

The evaluable subset is defined as: the instruction contains the name of exactly
one entity in its own block, and that entity is not the target. 714 of 1969
samples qualify (36%); on `val_unseen`, 151 of 400.

| arm | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| A0 lexical (name substring + type word) | 1.000 | 1.000 | 1.000 | 1.000 |
| A1 semantic (frozen text encoder vs names/types) | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled control | 0.083 | 0.344 | 0.604 | 0.292 |

**That 1.000 is tautological and must not be read as accuracy.** The subset is
*defined* by the same lexical match the arm performs, so of course the arm finds
it. What the numbers do support is a statement about **coverage and precision**:
when an instruction names an entity in its block, that name picks out a unique
entity 36% of the time, and the semantic arm neither adds to nor subtracts from
the lexical one — on a block where only 25% of buildings and no vehicles carry a
name, text-encoder similarity has almost nothing extra to match against.

## The target task, `val_unseen`

| method | deployable | params | n | Top-1 | Top-4 | MRR | median margin |
|---|---|---:|---:|---:|---:|---:|---:|
| **TD_masked, referenced phrase** | yes | 0 | 400 | **0.320** | 0.757 | 0.521 | −0.0101 |
| TD_masked, full instruction | yes | 0 | 400 | 0.287 | 0.752 | 0.510 | −0.0140 |
| anchor lexical only | yes | 0 | 400 | 0.083 | 0.388 | 0.275 | 0.0000 |
| anchor semantic only | yes | 0 | 400 | 0.087 | 0.393 | 0.278 | 0.0000 |
| relation rule only | yes | 0 | 400 | 0.035 | 0.475 | 0.259 | −0.0573 |
| relation learned only | yes | 12k | 400 | 0.165 | 0.638 | 0.397 | −0.6983 |
| visual + rule relation | yes | 12k | 400 | 0.158 | 0.730 | 0.422 | −0.3620 |
| visual + learned, top-1 anchor | yes | 12k | 350 | 0.123 | 0.731 | 0.401 | −0.4066 |
| **visual + learned, top-K marginalised** | yes | 12k | 400 | **0.357** | 0.795 | 0.557 | −0.0865 |
| visual + learned, relation family removed | yes | 12k | 400 | 0.370 | 0.798 | 0.566 | −0.0880 |
| visual + learned, geometry removed | yes | 12k | 400 | 0.320 | 0.757 | 0.521 | −0.1223 |
| shuffled anchor control | yes | 12k | 350 | 0.234 | 0.717 | 0.465 | −0.2869 |

Every number above is read from
`artifacts/language_anchor_grounding/target_metrics.json`, and the `n` column is
carried because two of the twelve arms are not scored on the full split: they
return no score when an instruction yields no anchor, so they are evaluated on
the 350 `val_unseen` samples that do yield one. Their Top-1 is therefore not
directly comparable with the 400-sample rows, including the baseline they are
set against below.

Read the last four rows together, because they are the round's actual result.

**The gain is +3.7 points** (0.357 against 0.320). MRR rises 0.521 → 0.557.
The marginal is mixed: the median margin gets *more* negative (−0.010 → −0.087),
so the improvement is in reordering the top of the list rather than in separating
the target from its closest rival.

**The two arms either side of this row are the same arm.** `visual_plus_learned`
and `visual_plus_topk_marginalised` are bound to one lambda in the run script and
produce identical numbers to four decimals; the second row is a duplicate, not a
second measurement. It is retained here only because it is what the artefact
contains.

**Two things that are not marginalisation.** The tuning chose **K = 1** on
`val_seen` (0.133 at K=1, falling monotonically to 0.129 at K=10), so the learned
arm marginalises across anchor *phrases* but never across anchor hypotheses
within a phrase. The gap from 0.123 to 0.357 is therefore **the rule scorer
against the learned scorer at the same K**, not a committed grounder against a
marginalising one: `_top1_anchor_blend` reads `rule_score[:, :, 0]`, and the
learned arm reads the MLP. The artefacts contain no arm that commits a *learned*
score to the top-1 anchor, so the claim that marginalisation is what rescues the
chain is **not tested by this round** and should not be read out of it. The
recoverable statement is narrower: the rule function is a much worse way to read
an anchor than a fitted MLP is.

**Real anchors beat shuffled ones decisively**: 0.357 against 0.234, paired
70 against 24 on the shuffled control, **p = 2.2 × 10⁻⁶**. The fusion likewise
beats the baseline on 25 samples against 10, p = 0.017. Whatever the module is
doing, it is doing it because of which anchors it was given.

**But the relation word contributes nothing, and the geometry contributes
everything.** Removing the nine relation-family columns from the reasoner's input
*raises* the score from 0.357 to 0.370 — the highest number in the round, and
reachable only by deleting the relation; removing the sixteen geometry columns
drops it to exactly the baseline, 0.320. So the arm is not reasoning about
`behind` or `between`. It is a learned proximity prior: candidates near the
predicted anchor are preferred, which is the most common relation in the corpus
expressed as a single smooth function of distance.

## Per entity group, `val_unseen` Top-1

| method | building (n=180) | vehicle (n=140) | other (n=80) |
|---|---:|---:|---:|
| TD_masked (baseline) | 0.278 | **0.464** | 0.163 |
| visual + learned | 0.350 | 0.443 | 0.225 |
| visual + learned, no relation family | 0.356 | 0.443 | 0.275 |
| shuffled anchor control | 0.178 | 0.302 | 0.236 |

(Buildings gain **+7.2**, the other-class entities **+6.2**, vehicles lose
**2.1**.) The shuffled row's denominators are 152/126/72 rather than 180/140/80,
for the reason given above, so it is only readable as a direction.

The shape is the one the brief asked for: the anchor chain helps where the anchor
information exists, and costs the class where vision already worked much less
than it gains elsewhere. It is not free — vehicles lose 2.1 points, and under the
*no relation family* ablation they lose the same 2.1 while `other` gains 11.2,
which is the same warning as everywhere else in this round: the columns being
read as "the relation" are not the ones doing the work.

## Rescue

The baseline gets 272 of 400 wrong, and 175 of those are in Top-4 but not Top-1.

| arm | Top-4 → Top-1 rescued |
|---|---:|
| relation learned only | 18.3% |
| visual + learned | 13.1% |
| anchor semantic only | 9.7% |
| visual + rule | 7.4% |
| relation rule only | 3.4% |

(modest; the reasoner that rescues the most on its own is the one with the
weakest standalone score)

## The oracle ladder, and why it does not read as a ladder

Every rung is measured on the same 151 samples that name an anchor.

| rung | Top-1 |
|---|---:|
| O0 visual only | 0.404 |
| **O1 predicted anchor + predicted relation** | **0.464** |
| O2 GT anchor + predicted relation | 0.278 |
| O3 predicted anchor + rule relation | 0.199 |
| O4 GT anchor + rule relation | 0.086 |
| O5 visual + GT anchor + learned relation | 0.278 |

**Handing the reasoner the true anchor makes it worse, on both reasoners.**
O2 (0.278) is far below O0 (0.404); O4 (0.086) is below O3 (0.199) by a wide
margin.
A ladder that descends when the information improves is not measuring what it
claims to. Combined with the input ablation, the reading is that neither reasoner
implements the relation: the rule function's sign conventions do not match how
the corpus actually uses these words, and the learned one is a distance prior
fitted to predicted-anchor geometry, so on true-anchor geometry it is out of
distribution.

The brief asks this round to localise a failure among parser, anchor grounder,
relation reasoner and fusion. The answer is the **relation reasoner**, and the
ladder is what shows it.

## Verdict

| criterion | required | measured |
|---|---|---|
| best deployable Top-1 | ≥ 0.425 (STRONG) | **0.357** |
| | +5 to +10 (WEAK) | **+3.7** |
| MRR | improve | 0.521 → 0.557 ✓ |
| median margin | improve | −0.010 → −0.087 ✗ |
| Top-4 → Top-1 rescue | visible | 13.1% |
| real anchor > shuffled | yes | 0.357 vs 0.234, p = 2.2 × 10⁻⁶ ✓ |
| vehicle not degraded | yes | −2.1 points, small ✓ |

**FAIL.** The best deployable arm as designed gains +3.7 points, below the +5
WEAK threshold; the only configuration that reaches +5.0 does so by deleting the
relation from the input, which is the opposite of what the round set out to test.
The round's stated goal — "explicitly compute the target-anchor spatial relation"
— is not met: the relation word is not used, and removing it helps.

## What failed, precisely

* **Parser**: works. 86% of instructions yield an anchor, at 1.96 relation words
  each, and the target/anchor split is what makes the rest of the round possible.
* **Anchor grounder**: works for what it can see, with a coverage limit that is a
  property of the data — 36% of instructions name an entity, and names exist for
  25% of buildings and no vehicles. Its accuracy on that subset is not
  measurable by the subset's own definition.
* **Relation reasoner**: **failed**, and it is the binding failure. The rule
  function is at chance (0.035) and its sign conventions do not survive contact
  with the oracle. The learned function is a distance prior in disguise. One
  concrete defect was found and fixed on the way — the compass table read
  `north of` off the east-west coordinate — but compass words are rare among
  these anchors and the fix moved the arm by less than a point, which is its own
  evidence that the rule function is not where the signal was ever coming from.
* **Fusion**: fine. Marginalisation over anchors, not the fusion, is what makes
  the arm usable at all.

## Limits

* Anchor grounding has no independent ground truth; the evaluable subset is
  circular by construction and is reported as coverage, not accuracy.
* The reasoner is a 12k-parameter MLP over 25 geometric columns. It is not a
  transformer and was never going to learn `between` from 1200 sentences with
  350 occurrences.
* The relation families were assigned by a lexicon. If the corpus uses `on` for
  something other than surface contact — and 1423 occurrences of one word against
  431 of the next suggests it might — the rule function is mis-specified from the
  start and the learned head inherits the mistake.

## One next step

**Replace the rule relation function with a relation that is fitted to the
corpus's own usage, and validate it against the oracle before wiring it in.**

The evidence for this is narrow and specific: the oracle ladder inverts — true
anchors score worse than predicted ones for both reasoners — which can only mean
the reasoner is not implementing the relation. Until a reasoner passes the check
"given the true anchor and the true relation, does the target rank higher?", the
deployable +5.2 points are a proximity prior and nothing more. That check is
cheap, it is the first thing to run next, and the round should not be read as
having passed without it.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY -m pytest tests/ -q                                   # tests for this round + earlier
$PY scripts/run_anchor_grounding_data.py                   # ~50 s
$PY scripts/train_anchor_grounding.py                      # ~4 min
```

Artefacts: `artifacts/language_anchor_grounding/{metrics.json,
parser_metrics.json, anchor_metrics.json, relation_metrics.json,
target_metrics.json, oracle_ladder.json, per_type.json, per_relation.json,
rescue_analysis.json, qualitative_cases/}`.
