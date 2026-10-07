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
| best direction **per relation**, fitted | **0.592** |
| best direction after **shuffling the relation labels** | **0.603** (sd 0.023) |
| the single prior "the answer lies ahead of the anchor", no word at all | **0.561** |

**Fitting a separate direction to each relation word does no better than fitting
one to randomly relabelled relations.** The observed mean (0.592) sits *below*
the shuffled mean (0.603). Whatever these words are doing, it is not selecting a
geometric direction: a model that is told which word was used cannot predict the
target's position any better than one that is told nothing.

The 0.561 from the heading alone is worth reading carefully. The agent flies
toward the goal, so "the answer is ahead of the anchor" is true under *every*
word, and it accounts for most of the 0.59. This is the proximity-and-heading
prior that the previous round found at the end, here measured directly and
separately from the words.

## Per relation: the frame the lexicon declared, against the axis the data picks

| phrase | n | declared | measured | AUC | unseen |
|---|---:|---|---|---:|---:|
| beside | 32 | none | agent_lateral **+** | 0.659 | 0.422 |
| near | 63 | none | agent_ahead + | 0.634 | 0.444 |
| facing | 22 | none | global_y + | 0.617 | 0.542 |
| by the | 15 | none | anchor_perp + | 0.607 | 0.547 |
| behind | 131 | agent | global_y + | 0.605 | 0.486 |
| in front of | 98 | agent | agent_ahead + | 0.604 | 0.522 |
| across from | 54 | none | global_x **−** | 0.603 | 0.379 |
| adjacent to | 16 | none | global_x − | 0.587 | 0.389 |
| north of | 26 | global | global_y **−** | 0.577 | 0.514 |
| left of | 67 | agent | anchor_perp + | 0.576 | 0.412 |
| right of | 66 | agent | agent_ahead + | 0.568 | 0.620 |
| between | 158 | none | agent_ahead + | 0.562 | 0.542 |
| on | 697 | none | agent_ahead + | 0.554 | 0.562 |
| next to | 123 | none | agent_lateral − | 0.535 | 0.434 |

Three things are wrong with this table at once.

**The declared frame is almost never the measured one.** Of the five phrases the
lexicon declared to be in the agent frame, one (`in front of`) matches. `behind`
wants the global y axis; `left of` wants the anchor's own perpendicular; `right
of` wants the agent's heading. `north of` is declared global and does pick the
global y axis — **with the opposite sign**, meaning that on this corpus a
building "north of" the anchor is, if anything, to its south. At n = 26 that is
weak, but it is not the direction the rule assumed.

**Opposite words do not pick opposite axes.** `behind` and `in front of` land on
different axes rather than on the same axis with opposite signs; so do `left of`
and `right of`. If the corpus used these words with any consistent geometry, the
pairs would be forced to mirror each other. They are not, so there is no frame
in which both members of a pair are satisfied at once.

**No phrase reaches a reliable effect.** The best is 0.659 at n = 32. The six
phrases above n = 60 all sit between 0.554 and 0.634. And the `unseen` column is
worse still — five of the fourteen fall **below 0.45** on `val_unseen`, i.e. the
axis reverses out of sample. `across from` is 0.603 on train and 0.379 on unseen.

## The signal that is actually there is proximity

| phrase | n | target median distance | distractor median | proximity AUC |
|---|---:|---:|---:|---:|
| between | 158 | 27.3 m | 50.5 m | **0.735** |
| in front of | 98 | 24.6 m | 46.8 m | 0.707 |
| behind | 131 | 27.9 m | 40.4 m | 0.677 |
| near | 63 | 35.5 m | 50.7 m | 0.669 |
| next to | 123 | 30.5 m | 40.1 m | 0.642 |
| on | 697 | 30.1 m | 37.8 m | 0.628 |
| right of | 66 | 31.8 m | 47.7 m | 0.621 |
| left of | 67 | 37.9 m | 44.5 m | 0.579 |

For nearly every phrase, **plain distance to the anchor separates target from
distractor better than the best directional axis does** — `between` 0.735 by
distance against 0.562 by direction, `in front of` 0.707 against 0.604. The
target is simply nearer the anchor than the distractors are, under every word
including the ones that assert the opposite of nearness.

This is the whole finding in one line: **on this corpus the relation words carry
a proximity signal and almost nothing directional.** It is exactly the
"learned proximity prior" the previous round inferred from its ablation, here
confirmed by direct measurement of the corpus rather than by an ablation on a
model.

## What this implies for the rest of the round

The brief asks for the round to stop and report if the relation reasoner cannot
be shown to use the relation. This audit is the first place that question gets
an answer, and the answer is preparatory and negative:

* A scorer given the relation word cannot beat one given a random word, so there
  is no directional target for it to learn on this corpus.
* The words that the brief names specifically — north/south/east/west — occur 26,
  10, 0 and 1 times. `east of` cannot be evaluated at all. Any claim about
  compass reasoning in this corpus would be unsupported.
* The one signal that is reliably present is proximity, and a proximity scorer
  does not need the relation word.

The self-test in `scripts/train_relation_v2.py` therefore has a specific,
pre-registered prediction to fail against, and its gate is reported whether or
not it passes.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY scripts/audit_relation_semantics.py     # ~2 min
```
