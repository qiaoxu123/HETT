# What CityNav actually hands over as "referenced landmarks"

Branch `2027-CVPR/reference-binding-relation-v2`. Audit only; nothing is trained
here. Source: `scripts/audit_reference_binding.py` →
`artifacts/relation_v2/reference_audit.json`, over the entity round's 2290 kept
samples (train_seen 1200 / val_seen 690 / val_unseen 400).

## The premise, checked

The round was briefed to stop searching the block for anchors and instead take
the anchor candidate set from the referenced-landmark field the data provides.
That field does not exist in the form the brief assumes, and the audit is the
place to say so rather than the place to quietly work around it.

**`object_ids` is the target.** Across all 27,045 records in the three splits,
`|object_ids| == 1`, and in 27,045 of 27,045 cases that object's position equals
one of the episode's `target_positions`. The upstream code takes the same view:
`MTurkTrajectory.object_id` returns `object_ids[0]` (there is a separate
`desc_id` for `ann_ids[0]`), and `gsamllavanav/dataset/generate.py` constructs
`Episode(objects[map][object_id], ...)` — the object handed to the episode *is*
the referent. **Taking `object_ids` as an anchor candidate set would be the
target-id leak the brief forbids**, so this round does not read it.

**What is available is the landmark *names*.** `CityReferObject` carries
`processed_descriptions[i]` with `target`, `landmarks` and `surroundings`, and
upstream exposes them as `Episode.description_landmarks` — a first-class
property, not a scrape. Its content is a list of anchor name strings and it does
not name the target. That is what this round consumes.

**The access path is target-keyed, and that is a real caveat.** Annotations are
stored under the target object's id, so `description_landmarks` is only
reachable once you know which object the sentence describes. The information
added over reading the sentence is therefore the *segmentation into anchors* —
which noun phrases are references — and nothing about the answer. Every arm in
this round is reported twice, once from parser-derived anchors and once from
annotation-derived ones, so that the difference between them is visible as the
parser's own error rather than folded into "binding".

Alignment is verified rather than assumed: the instruction text matches
`objects[map][target][descriptions][ann_id]` exactly after whitespace
normalisation in **2290 of 2290** samples (`description_text_mismatch: 0`), and
no sample is missing an annotation. The only difference between the shipped
sentence and the source file is that runs of spaces have been collapsed.

## Answers to the eight questions

**1. How many referenced landmarks per instruction?** Not one. Nearly half the
corpus names two or more.

| landmarks | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| samples | 1 | 1159 | 981 | 132 | 16 | 1 |

1130 of 2290 samples (49.3%) are multi-reference. `surroundings` are separate
and more numerous — 432 samples have none, and the tail runs to ten; they name
things that are *not* landmarks (a car park, a roof colour) and this round does
not use them.

**2. Does each reference carry id, name, type, XYZ, footprint, dimensions?**
The reference carries **a name string only**. Everything else is reached by
binding that string to a block entity, which then supplies `object_type`,
`position`, `dimension` and `contour` (the footprint polygon). So the audit's
central quantity is how well a name string identifies an entity.

**3. Do the names match metadata?** Yes, almost always, on both counts.

| check | rate |
|---|---:|
| annotation name matches some entity's name exactly (normalised) | 3573 / 3586 = **99.6%** |
| annotation name appears verbatim in its own instruction | 3219 / 3586 = **89.8%** |

The 10% that do not appear verbatim are the interesting ones: the instruction
says "the national probation service building" where the annotation says
"national probation service building", or inflects a road name. This is the
parser's problem, not the binder's, and it is why the round keeps the two paths
separate.

**4–5. Multi-reference structure.** 1130 samples; in 443 of them every reference
resolves to the same entity type, and in 300 at least two reference names share a
content word.

**6. Is there an explicit phrase → id binding?** **No. CASE A is empty, by
construction** — the annotation gives names, so even the "explicit" path is a
name lookup. This matters for how the results are read: there is no rung of the
ladder where binding is free.

**7. Is the reference list in mention order?** Usually. Of the 1048 multi-
reference samples where every name could be located in the sentence, **913
(87.1%)** list them in the order the sentence mentions them. High enough to be a
useful prior, not high enough to rely on — and the round does not use it as one.

**8. How many samples need real semantic binding?** **Ten.** Four CASE C and six
CASE D, out of 2290.

## CASE A–E

| case | meaning | samples | share |
|---|---|---:|---:|
| A | explicit phrase → landmark id in the data | **0** | 0% |
| B | no explicit binding, but the name matches directly | **2279** | 99.5% |
| C | name not identical, needs semantic matching | 4 | 0.17% |
| D | multiple references whose candidate sets can be swapped | 6 | 0.26% |
| E | information insufficient to bind | **0** | 0% |
| — | instruction names no landmark at all | 1 | 0.04% |

By split: `val_seen` is 690/690 CASE B. `val_unseen` is 399 B and one sample
that names no landmark. The four C and six D cases are all in `train_seen` —
which means **the binding evaluation that matters has no hard cases in it**, and
a binding accuracy of ~100% on `val_seen`/`val_unseen` would be a statement about
the data's name vocabulary, not about a method. The round reports it that way.

## The finding that actually constrains the round

Binding is nearly always *possible* and rarely *unique*.

| | |
|---|---:|
| landmark phrases | 3586 |
| bind to exactly one entity by name | 2081 (58.0%) |
| bind to **more than one** entity by name | 1492 (41.6%) |
| need fuzzy matching to bind at all | 13 (0.4%) |
| samples containing a one-to-many name | 1254 (54.8%) |
| mean candidate entities per sample | 3.54 |

The one-to-many names are roads. A road is stored as many segments that all
carry the same name. In a block the largest such group is **15** segments
(`walsall road`, birmingham_block_5); `aldridge road` reaches 9 and
`wellington road` 3. So "the building on Aldridge Road" names several anchors
rather than one, and the binding problem is **which segment**, not **which
name**.

An earlier version of this audit gave 180, 103 and 88 for those three names.
Those figures were counts of *samples* in which the name was ambiguous, taken
from a table whose rows were samples; they are not entity counts and overstate
the group size by up to twentyfold. The correct per-block maxima are in
`artifacts/spatial_graph/road_region_audit.json`. The conclusion is unchanged --
a name still resolves to a set rather than to an entity, mean 4.01 and median 3
entities per ambiguous bind, maximum 18 -- but the numbers quoted for it were
wrong and are corrected here.

That reframes what a reference binder has to do here. It is not a semantic
matching problem — a text encoder has almost nothing to add when 99.6% of names
match exactly. It is a **disambiguation** problem: given a name that picks out a
set of entities, and a relation, choose among them. A reasoner that can read
"on Aldridge Road" and "beside Aldridge Road" differently is worth more than a
better name matcher, which is the opposite of where the previous round's effort
went.

The six CASE D samples are the road-junction ones, and they are genuinely
ambiguous by construction — the landmark name is the concatenation of two other
landmark names:

```
['Wellington Rd', 'Grosvenor Rd', 'Wellington Rd / Grosvenor Rd']
['Maltings Yard', 'New Park Street', 'Bishop Bateman Court (Trinity Hall)', 'Trinity Hall']
```

## What this changes for the round

* The anchor set is the **annotation's landmark name list**, treated as a legal
  input with the target-keyed access path stated in the report. Parser-derived
  anchors are carried alongside it as a second path.
* Binding is **one-to-many**, so the binder's output is a set of candidate
  entities per phrase, carrying a confidence — which is what the reasoner
  marginalises over.
* Because 99.5% of bindings are exact-name and only ten samples are hard,
  **binding is not where this round can succeed or fail**. The measurement that
  decides the round is the relation self-test, and the report is structured so
  that binding errors cannot hide relation errors inside one number.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY scripts/audit_reference_binding.py      # ~4 min
```
