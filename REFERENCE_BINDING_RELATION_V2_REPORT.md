# Reference binding and relation semantics, second attempt

Branch `2027-CVPR/reference-binding-relation-v2`, on `c390d28`. The previous
round ended with a specific charge against its own reasoner: handing it the
*true* anchor made it score **worse** than handing it a guess, which can only
happen if the function does not mean what the corpus means. The brief for this
round is therefore ordered deliberately — first prove that a reasoner given the
right anchor and the right relation ranks the target, and only then consider
fusing it with the visual channel.

**The short answer: it cannot, and the round stops at the gate.** The reasoner
lands at 0.1055 against 0.10 chance; the opposite relation scores higher than the
correct one; a randomly relabelled relation scores identically; and the loss term
written specifically to force the word to matter was rejected by the `val_seen`
tuning. Per §15 no visual fusion was run, and §16–20 are reported as not run
rather than run-and-quietly-dropped.

## A. Reference input audit

Summary of `REFERENCE_BINDING_AUDIT_V2.md`; full source in
`artifacts/relation_v2/reference_audit.json`.

The round was briefed to take the anchor candidate set from the data's
referenced-landmark field rather than searching the block. **That field does not
hold what the brief assumes.** `object_ids` has exactly one entry in all 27,045
records across the three splits, and in 27,045 of them that entity's position is
one of the episode's `target_positions` — it is the *target*. Upstream agrees:
`MTurkTrajectory.object_id` returns `object_ids[0]` and `generate.py` builds
`Episode(objects[map][object_id], ...)`. Reading it as an anchor set would be the
target-id leak §22 forbids, so this round does not.

What is legitimately available is the landmark **names**, through
`CityReferObject.processed_descriptions` and the upstream
`Episode.description_landmarks` property. Its access path is keyed on the target
object, which is stated here rather than worked around, and every arm was carried
twice — parser-derived anchors and annotation-derived anchors — so the parser's
own error is visible separately.

Alignment was verified, not assumed: the instruction matches
`objects[map][target][descriptions][ann_id]` exactly after whitespace
normalisation in **2290 of 2290** samples, with no sample missing an annotation.

## B. Binding accuracy and ambiguity

| case | meaning | samples | share |
|---|---|---:|---:|
| A | explicit phrase → landmark id | **0** | 0% |
| B | no explicit binding, name matches directly | **2279** | 99.5% |
| C | name not identical, needs semantic matching | 4 | 0.17% |
| D | multiple references whose candidate sets can be swapped | 6 | 0.26% |
| E | insufficient information to bind | **0** | 0% |

Landmark strings matching an entity name exactly: **3573 / 3586 = 99.6%**.
Appearing verbatim in their own instruction: 3219 / 3586 = 89.8%.

**Binding is therefore not where this round can succeed or fail**, and all ten
CASE C/D samples sit in `train_seen`, so the evaluation subsets contain no hard
cases at all. The number that does matter is ambiguity:

| | |
|---|---:|
| landmark phrases | 3586 |
| bind to exactly one entity by name | 2081 (58.0%) |
| bind to more than one entity | 1492 (41.6%) |
| samples containing such a name | 1254 (54.8%) |
| mean candidate entities per sample | 3.54 |

The shared names are roads. `aldridge road` is 180 separate segments, `wellington
road` 103, `birchfield road` and `aston lane` 88 each. So binding here is
disambiguation, not matching — which is why every score marginalises over the
anchor set with a `logsumexp` rather than committing to one entity.

**49.3% of samples name two or more references** (1159 name one, 981 name two,
132 name three, 16 name four, 1 names five).

## C. Relation semantic distributions

Full source in `RELATION_SEMANTICS_AUDIT.md` and
`artifacts/relation_v2/relation_distributions.json`; frame selection on
`train_seen` + `val_seen` only.

**54.7% of anchor phrases carry a relation word at all** (1960 of 3585). The
other 45.3% are co-referenced features — "the building with the Gym, Nike
Factory Store and The Food Warehouse" names landmarks that stand in no spatial
relation to the target — and a relation self-test run over them would be
measuring geometry with no word to test. The evaluable set is the 1436 samples
with at least one real relation word: train_seen 727, val_seen 453, **val_unseen
256**.

## D. Frame selection

For each phrase, each of six directions was tried with both signs and scored by
paired AUC of the target against same-class distractors, using the anchor entity
nearest the target.

| | paired AUC |
|---|---:|
| best direction **per relation**, fitted | **0.597** |
| best direction after **shuffling the relation labels** | **0.587** (sd 0.016) |
| the prior "the answer lies ahead of the anchor", no word at all | **0.488** |

The fitted direction beats the shuffled control by 0.010 against a control spread
of 0.016 — inside one standard deviation. **A model told which word was used
cannot place the target relative to the anchor any better than one told
nothing.** The declared frame matches the measured axis for **none** of the 14
scorable phrases, and no opposite pair lands on one axis with two signs, so there
is no frame in which both members of a pair are satisfied at once. The largest-n
phrase, `on` (n = 697), sits at 0.522; seven of fourteen fall below 0.50 on
`val_unseen`. The compass words are 26, 10, 0 and 1 occurrences.

## E. The relation self-test

`scripts/train_relation_v2.py`. A relation-conditioned scorer over the 18 raw
geometry columns in three frames, 13,313 trainable parameters, fitted on
`train_seen` with a listwise ranking loss over candidates plus a margin term
putting the sentence's own relation above its opposite for the same target and
anchor. Hyper-parameters chosen on `val_seen`.

`val_unseen`, 256 samples, ten candidates, chance 0.10.

| arm | val_seen | Top-1 | Top-4 | MRR |
|---|---:|---:|---:|---:|
| **correct relation** | 0.1148 | **0.1055** | 0.3750 | 0.2860 |
| opposite relation | 0.0971 | **0.1250** | 0.3984 | 0.3069 |
| shuffled relation | 0.1015 | 0.1055 | 0.3672 | 0.2854 |
| no relation word | 0.1170 | 0.0938 | 0.4297 | 0.3013 |
| zeroed relation | 0.0927 | 0.0938 | 0.4531 | 0.3007 |
| shuffled anchor | 0.0817 | 0.0938 | 0.3594 | 0.2814 |

**The correct relation does not win.** It lands half a point above chance, and
the *opposite* relation scores higher still (0.1250 against 0.1055). Relabelling
every sample's relation at random reproduces the correct arm to four decimals.
Deleting the word entirely leaves 0.0938 — below chance. Nothing here is
distinguishable from noise: every paired comparison gives p ≥ 0.54, and
correct-versus-shuffled-relation is 10 against 10.

## F. The counterfactual test, and the term that was rejected

§13 asks for a loss term that forces the model to use the relation word. It was
implemented — the same target against the same anchor, scored under the
sentence's own relation and under its opposite, with a margin — and **the
`val_seen` tuning rejected it**:

| lr | cf_weight | val_seen Top-1 |
|---|---:|---:|
| 3e-3 | **0.0** | **0.1192** |
| 3e-3 | 1.0 | 0.1104 |
| 3e-3 | 3.0 | 0.1082 |
| 1e-2 | 1.0 | 0.1104 |

Adding the term that was supposed to make the relation matter costs about a
point on held-out seen data at every weight tried, and the final model was fitted
with `cf_weight = 0.0`. The margin is smaller than the 4 points an earlier run of
this pipeline showed, but the ordering is the same at every setting: the
supervision designed to teach the word never helped. That is the round's clearest
single statement — not that the model failed to learn the relation, but that
being *made* to use it is worse than being left free to ignore it.

## G–I. Target grounding, per type, oblique ablation — **not run**

§15 is explicit that a reasoner failing its self-test must not be fused, and the
reason is not caution. A fusion number computed on top of a reasoner that cannot
rank the target given the right anchor measures the visual baseline it was added
to and attributes the total to a module that contributed nothing. That is
precisely the failure mode of the previous round, whose +3.7 points turned out
to be a proximity prior. So §16–20 are not reported. The gate's own verdict is
the deliverable.

The visual baseline these would have been measured against is unchanged from last
round: `TD_masked` on the referenced phrase, **0.320** on `val_unseen`.

## J. Failure analysis

§29 asks which link of the chain failed. The answer is **relation semantics**,
and it fails upstream of the model.

* **Reference binding**: not the failure, and not testable here — 99.5% of
  bindings are exact-name matches and only ten samples in 2290 need semantic
  matching. The real property of this corpus is one-to-many naming (41.6% of
  phrases), which marginalisation handles.
* **Relation semantics**: **the failure.** Measured directly against the corpus,
  no relation word selects a geometric direction better than a randomly
  assigned label does (0.597 fitted against 0.587 shuffled, control sd 0.016),
  and the words that a rule would most obviously need — north/south/east/west —
  occur 26, 10, 0 and 1 times. Neither is there a proximity fallback: every
  phrase's distance separation sits between 0.459 and 0.561.
* **Relation model**: not indictable separately. A 13k-parameter scorer cannot be
  blamed for failing to learn a signal that the audit finds is not in the data.
  The counterfactual term's rejection by tuning is the same fact from the
  training side.
* **Final fusion**: not reached, by design.

## What this changes about the previous round

The previous round reported a rule reasoner whose oracle ladder inverted, and
inferred that its hand-written sign convention did not match the corpus. That
inference was right. This round measures the corpus directly and finds the
stronger statement: **there is no consistent convention to match.** It also
removes a confound the previous round's own diagnostic inherited — the answer sat
at candidate index 0 in the stored files, so any tie broke toward the answer —
which on the corrected order reverses the direction of the "proximity prior"
finding. `REFERENCE_BINDING_AUDIT_V2.md` and `RELATION_SEMANTICS_AUDIT.md` record
this rather than replacing the earlier numbers silently.

## Verdict

| criterion | required | measured |
|---|---|---|
| relation self-test passes (§15) | yes | **no — 3 of 5 conditions fail** |
| correct > shuffled relation, p < 0.05 | yes | 0.1055 vs 0.1055, p = 1.00 ✗ |
| correct > opposite relation, p < 0.05 | yes | 0.1055 vs 0.1250, p = 0.54 ✗ |
| dropping the relation hurts by > 0.01 | yes | +0.012 ✓ (barely) |
| shuffled anchor hurts, p < 0.05 | yes | 0.1055 vs 0.0938, p = 0.77 ✗ |
| reasoner above chance | yes | 0.1055 vs 0.10 ✓ |
| **§15 gate** | **PASS** | **FAIL — fusion not run** |

**FAIL**, at the gate, on the round's own primary objective.

## The one next step

**Establish whether the spatial relations in this corpus are recoverable at all,
before building another reasoner.**

The evidence is now specific rather than suggestive: fitting a direction per
relation word does no better than fitting one to shuffled labels, and the
supervision designed to force the word to matter was rejected by held-out
tuning. Either the words are being parsed into the wrong phrases, or the
annotators used them loosely, or the anchors are unresolved — and the audit
cannot separate those, because it measures the target against the anchor entity
nearest the target and so already assumes the anchor is right.

The cheap experiment that separates them: take the 256 `val_unseen` samples with
a relation word, hand-annotate the true anchor entity for a subset, and re-run
the frame measurement against it. If no direction appears even then, the corpus's
relation words carry no usable spatial content and the whole anchor line of work
should be dropped rather than refined. If one does appear, the frame exists and
the failure is in the binding of phrase to anchor entity — which the audit has
already shown is a one-to-many disambiguation problem, not a matching one.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY -m pytest tests/ -q                          # 181 tests
$PY scripts/audit_reference_binding.py           # ~4 min
$PY scripts/run_relation_v2_data.py              # ~15 s
$PY scripts/audit_relation_semantics.py          # ~2 min
$PY scripts/train_relation_v2.py                 # ~2 min
```

Artefacts: `artifacts/relation_v2/{reference_audit.json, relation_samples.jsonl,
relation_distributions.json, relation_self_test.json, relation_data_stats.json}`.
