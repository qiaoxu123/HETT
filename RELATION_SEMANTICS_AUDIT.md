# How this corpus actually uses its relation words

Branch `2027-CVPR/reference-binding-relation-v2`. Audit only; nothing is
trained. Source: `scripts/audit_relation_semantics.py` →
`artifacts/relation_v2/relation_distributions.json`. Frame selection uses
`train_seen` + `val_seen`; `val_unseen` is reported but never used to choose.

The previous round hard-coded one frame and one sign per relation family, and
its oracle showed the convention was wrong: handing the rule reasoner the *true*
anchor made it score **worse** than handing it a guess, which can only happen if
the function does not mean what the corpus means. This audit replaces that
assumption with a measurement over the 14 relation phrases that have enough
samples to be scored.

## What was measured

For each sample: the target's geometry relative to the anchor, against the same
geometry for every same-class distractor in the candidate list. Six directions
are available and each is tried with both signs —

`global_x`, `global_y`, `agent_ahead`, `agent_lateral`, `anchor_along`,
`anchor_perp`

— and scored by paired AUC: within a sample, does the true target sit further
along that direction than each same-class distractor? The anchor entity used is
the one *nearest the target*, which is the most favourable reading available and
so cannot be used to make the corpus look worse than it is.

## The controls

| | paired AUC |
|---|---:|
| best direction **per relation**, fitted | **0.597** |
| best direction after **shuffling the relation labels** | **0.587** (sd 0.016) |
| the single prior "the answer lies ahead of the anchor", no word at all | **0.488** |

**Fitting a separate direction to each relation word does no better than fitting
one to randomly relabelled relations.** The observed mean exceeds the shuffled
mean by 0.010 against a control spread of 0.016 — inside one standard deviation.
Whatever these words are doing, it is not selecting a geometric direction.

The heading prior is *at chance*, 0.488. That is a correction to an earlier
version of this audit, which reported 0.561 and read it as the trajectory prior
accounting for most of the effect. Both that number and the per-phrase table
below were computed before the candidate order was permuted, and the earlier
figures were contaminated by it -- see "A confound that had to be removed"
below.

## Per relation: the frame the lexicon declared, against the axis the data picks

| phrase | n | declared | measured | AUC | unseen |
|---|---:|---|---|---:|---:|
| adjacent to | 16 | none | anchor_along **−** | 0.750 | 0.519 |
| left of | 67 | agent | global_x **−** | 0.647 | 0.488 |
| facing | 22 | none | anchor_along + | 0.639 | 0.458 |
| north of | 26 | global | global_x + | 0.633 | 0.472 |
| across from | 54 | none | agent_lateral − | 0.615 | 0.525 |
| beside | 32 | none | agent_ahead − | 0.611 | 0.533 |
| by the | 15 | none | agent_lateral − | 0.600 | 0.479 |
| next to | 123 | none | global_y − | 0.576 | 0.512 |
| between | 158 | none | agent_lateral + | 0.570 | 0.466 |
| near | 63 | none | anchor_along + | 0.560 | 0.556 |
| in front of | 98 | agent | global_y + | 0.550 | 0.472 |
| behind | 131 | agent | anchor_along + | 0.550 | 0.444 |
| right of | 66 | agent | global_x − | 0.538 | 0.617 |
| on | 697 | none | agent_ahead − | 0.513 | 0.479 |

Three things are wrong with this table at once.

**The declared frame matches the measured one nowhere.** Not for the agent-frame
words (`left of`, `right of`, `behind`, `in front of`) and not for the global one
(`north of`, which lands on the agent's lateral axis at n = 26).

**Opposite words do not pick opposite axes.** `left of` and `right of` both
choose the global x axis — at least one axis between them — but `behind` chooses
the anchor's long axis while `in front of` chooses the global y axis, so that
pair shares no axis at all. If the corpus used these words with any consistent
geometry, each pair would be forced onto one axis with two signs.

**No phrase reaches a reliable effect.** The largest-n phrase in the corpus
(`on`, n = 697) sits at 0.513, and the best value anywhere (0.750) is at n = 16.
The `unseen` column is worse: ten of the fourteen fall below 0.55 out of sample,
and no phrase exceeds 0.62 there.

## The signal that is actually there is proximity

| phrase | n | target median distance | distractor median | proximity AUC |
|---|---:|---:|---:|---:|
| on | 697 | 33.1 m | 37.4 m | 0.561 |
| next to | 123 | 33.5 m | 37.6 m | 0.548 |
| right of | 66 | 43.8 m | 42.5 m | 0.536 |
| left of | 67 | 39.8 m | 41.0 m | 0.524 |
| in front of | 98 | 40.4 m | 41.6 m | 0.521 |
| behind | 131 | 34.3 m | 37.3 m | 0.508 |
| near | 63 | 43.4 m | 43.1 m | 0.490 |
| between | 158 | 43.7 m | 43.4 m | 0.459 |

**There is no proximity signal either.** Everything sits between 0.459 and 0.561
— chance, with `between`, `near` and `behind` actually *below* it, meaning that on
those words the target is if anything farther from the anchor than its
distractors are.

This is the correction that matters most, because an earlier version of this
document reported a strong proximity effect (`between` at 0.735 with the target
27.3 m from the anchor against distractors at 50.5 m) and concluded that the
corpus's relation words were "a proximity signal and nothing directional". On the
corrected data that reading does not survive: there is no proximity signal
either. The whole measurable content of these words, against same-class
distractors drawn from the same neighbourhood, is at chance.

## What this implies for the rest of the round

The brief asks for the round to stop and report if the relation reasoner cannot
be shown to use the relation. This audit is the first place that question gets
an answer, and the answer is preparatory and negative:

* A scorer given the relation word cannot beat one given a random word, so there
  is no directional target for it to learn on this corpus.
* The words that the brief names specifically — north/south/east/west — occur 26,
  10, 0 and 1 times. `east of` cannot be evaluated at all. Any claim about
  compass reasoning in this corpus would be unsupported.
* Neither is there a proximity signal to fall back on: the previous round's
  "learned proximity prior" was, on the corrected candidate order, itself partly
  an artefact of the confound below.

The self-test in `scripts/train_relation_v2.py` therefore has a specific,
pre-registered prediction to fail against, and its gate is reported whether or
not it passes.

## A confound that had to be removed

The candidate list is built as `[referenced] + distractors`, so the answer sits
at **index 0** in the stored files. The previous round permuted that order in its
training script, at load time; the extractor this audit reads from did not, and
the first version of these measurements therefore ran with the answer always
first. That is not a leak a model could merely learn — it decides ties by
position, and position 0 is always the answer.

Removing it changed the numbers substantially. The heading prior fell from 0.561
to **0.488** — chance — and the proximity table inverted: `between` goes from an
apparent 0.735 to 0.459. `target_index` is now
uniform over the ten positions, and the extraction applies the permutation once
and deterministically so that no downstream consumer has to remember.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY scripts/audit_relation_semantics.py     # ~2 min
```
