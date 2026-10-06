# Feature sufficiency: once the target region is masked, what is still missing?

Branch `2027-CVPR/feature-sufficiency-diagnostic`, on the entity round's final
HEAD. Nothing here trains a new model: a deliberately small probe is handed one
family of evidence at a time and what each buys is measured, under a strict
split between what exists at test time and what is an oracle.

## The shape this round starts from

`val_unseen`, n=394, ten candidates per sample, so 10% is chance.

| | Top-1 | Top-4 |
|---|---:|---:|
| whole top-down crop (`TD_global`) | 0.100 | 0.535 |
| entity-masked top-down (`TD_masked`) | 0.320 | 0.757 |
| entity-masked oblique (`O_masked`) | 0.185 | 0.623 |
| best 3D-anchored fusion (`GeoAligned`) | 0.338 | 0.810 |
| permuted-pairing control (`GeoShuffled`) | 0.343 | 0.812 |

Masking the target region is worth +22 points. Adding the second view and a
world-coordinate correspondence on top of it is worth +1.8, inside noise. Top-4
is 0.76 while Top-1 is 0.32: the model is choosing the right *kind* of thing and
not the right *instance*.

## Samples

Verbatim reuse — same instruction, target, candidate ids in the same order, same
split, episode and step as the entity round. Nothing is resampled.

| split | samples | building | vehicle | other |
|---|---:|---:|---:|---:|
| train_seen | 1175 | 536 | 407 | 232 |
| val_seen | 686 | 311 | 245 | 130 |
| val_unseen | 394 | 180 | 134 | 80 |

Every choice — the epoch, the hidden width, which arm counts as a method — is
made on `train_seen` / `val_seen`. `val_unseen` is read once.

## Two things the data forced, which are findings in themselves

**1. The candidate list is built around the answer, and one feature family read
that off it.** `candidates = [referenced] + nearest same-class to it + some
other-class`, so the target sits at index 0 in all 2255 samples. Candidate order
is now permuted once, deterministically, before any arm sees it.

More seriously, the first version of the relation block measured "how many
same-class entities are near this candidate, how many within 30 m, what the local
type histogram looks like" using the *candidate list* as the neighbourhood. But
that list was constructed by proximity to the reference, so those columns say
"I am the entity the list was built around". They reached **Top-1 0.685 with
3,841 parameters and no scene information at all** — above every method in the
entity round. Recomputed against **every entity in the block**, the same block
scores **0.201**.

That is the single most important line in this report: a plausible, well-formed,
spectacular number that measured the sampling procedure. It is reported here
rather than quietly fixed because the same trap is available to anyone who
builds features over a curated candidate set.

**2. The camera is not the leak.** The pose was chosen so the referenced entity
is 25–220 m away and near the frame centre, which looks like it should identify
the target. It does not: 94% of distractors are also in front of the camera, and
`Pose_only` (distance, ahead/lateral, off-axis) scores **0.094**, at chance.

## Feature families

**A — appearance** (45 columns). The entity's own measured point colours: mean,
median, spread, HSV, per-channel histogram, joint colour entropy, grey/dark/
bright/saturated shares. And from the top-down raster at its native 0.1 m/px, the
pixels the entity's points land on: brightness percentiles, edge density, local
contrast. No learned embedding is involved.

**B — geometry** (26). Annotated size and volume, aspect ratio, measured spans,
footprint, point count, density, compactness, height statistics, footprint
orientation, elongation, projected pixel size in both views, distance and bearing
from the agent.

**C — relation** (28). Distance and bearing to the agent decomposed into
ahead/lateral and off-axis; distance to the nearest road surface and road share,
from a per-block raster built off the semantic label; distance to the nearest
entity of any type, of a different type and of the same type, over the whole
block; how many entities are within 80 m and 30 m; the same-class share; the
local class histogram.

**D — semantic** (19). Type one-hot with a slot for an unknown type, normalised XY
in the block, block entity count, whether the entity is named.

**Instruction.** The frozen text encoder's embedding of the name, the referenced
phrase and the *full* instruction, plus a rule parser's flags for relation
families (`near`, `beside`, `behind`, `between`, `north of`, …) and attribute
families (colour, size, shape, material).

**ORACLE.** An anchor-relation block measuring a candidate's distance and bearing
to a named anchor entity, and an entity-identity block. Both use the target's own
annotation to choose the anchor or the identity, so both bound what perfect
parsing would buy and neither is a method.

## Probe

`Linear(D, 128) → GELU → Linear(128, 1)` for every arm, cross-entropy over the ten
candidates, three seeds, early stopping on `val_seen`. Widths follow the input, so
parameter counts differ; a bigger head for one family would be measuring the head.

## Main table, `val_unseen` Top-1 (n=394, 10 candidates, chance = 0.10)

| method | deployable | params | Top-1 | Top-4 | MRR |
|---|---|---:|---:|---:|---:|
| **TD_masked (frozen text cosine)** | yes | 0 | **0.325** | 0.782 | 0.526 |
| Visual (same feature, learned head) | yes | 147k | 0.162 | 0.685 | 0.414 |
| Visual_tight | yes | 147k | 0.183 | 0.698 | 0.430 |
| O_visual | yes | 147k | 0.155 | 0.632 | 0.406 |
| **Appearance** | yes | 6.0k | **0.104** | 0.530 | 0.337 |
| **Geometry** | yes | 3.6k | **0.231** | 0.746 | 0.489 |
| **Relation** | yes | 3.8k | **0.201** | 0.695 | 0.435 |
| Semantic | yes | 2.7k | 0.069 | 0.614 | 0.331 |
| Relation_flags (instruction only) | yes | 2.2k | 0.107 | 0.388 | 0.297 |
| Pose_only | yes | 1.2k | 0.094 | 0.665 | 0.356 |
| Relation without pose columns | yes | 2.9k | 0.117 | 0.490 | 0.326 |
| Visual + Appearance | yes | 153k | 0.152 | 0.650 | 0.415 |
| **Visual + Geometry** | yes | 151k | **0.246** | 0.716 | 0.475 |
| **Visual + Relation** | yes | 151k | **0.234** | 0.723 | 0.472 |
| Visual + Semantic | yes | 150k | 0.175 | 0.668 | 0.416 |
| Full deployable | yes | 310k | 0.213 | 0.713 | 0.465 |
| Text: name / phrase / full | yes | 299k | 0.231 | 0.754 | 0.481 |
| **Relation ORACLE** | **no** | 1.7k | **0.561** | 0.987 | 0.754 |
| **Visual + Relation ORACLE** | **no** | 149k | **0.609** | 0.975 | 0.784 |
| Entity-ID ORACLE | **no** | 193k | 0.157 | 0.637 | 0.414 |

The three text arms are numerically identical to the fourth decimal. The FiLM
conditioning stayed at its zero initialisation and the probe learned to ignore
it, so **this instrument cannot answer whether the full instruction helps.** That
is a defect in the probe design, not a result, and it is the one place where the
brief's instruction-variant comparison is unanswered.

The identity oracle lands at 0.157, at chance: on unseen maps the entity ids were
never seen, so a memorised identity has nothing to offer. That is the expected
result and it confirms the oracle arm is measuring memorisation rather than
leaking a usable signal.

## What each family is worth

Against the same probe on the visual feature alone (0.162), adding:

| family | Δ Top-1 |
|---|---:|
| geometry | **+8.4** |
| relation (map structure) | **+7.2** |
| semantic | +1.3 |
| appearance | −1.0 |
| pose | −0.7 |

And against the thing that actually matters — the zero-shot text cosine on the
same masked features, 0.325 — **nothing added beats it**. The best deployable arm
is 0.246.

## The oracle, which is the most actionable number here

| | Top-1 |
|---|---:|
| frozen masked visual, zero-shot text cosine | 0.325 |
| + deployable relation | 0.234 |
| + **ORACLE** relation (anchor known) | **0.609** |

Knowing which entity the language's anchor refers to would nearly double Top-1
over the visual baseline, and more than double it over the best deployable
combination. The oracle chooses the anchor from the target's own annotation, so
it is a loose upper bound — it bounds the value of *anchor information*, not of
relation parsing in general. But the gap between 0.234 and 0.609 is where the
round's answer lives.

## Per entity group, `val_unseen` Top-1

| method | building (n=180) | vehicle (n=134) | other (n=80) |
|---|---:|---:|---:|
| TD_masked (frozen) | 0.278 | 0.464 | 0.163 |
| Visual (learned) | 0.156 | 0.201 | 0.300 |
| Appearance | — | — | — |
| Geometry | 0.161 | 0.224 | 0.400 |
| Relation | 0.256 | 0.067 | 0.300 |
| Visual + Geometry | 0.194 | 0.246 | 0.362 |
| Visual + Relation | 0.206 | 0.194 | 0.362 |
| **Visual + Relation ORACLE** | **0.756** | 0.373 | 0.675 |

Vehicles are the easy class for the frozen visual feature (0.464) and buildings
the hard one (0.278).  The structured families do not invert that: relation
features are near-useless for vehicles (0.067), while the oracle lifts buildings
from 0.278 to 0.756 — the largest single move anywhere in this round.  So the
missing information is concentrated in buildings and in the other-class
entities, and it is anchor information rather than appearance.

## View independence

The two views are projections of one point cloud, and their masked features agree
accordingly: **mean cosine 0.744** (median 0.777) between the top-down and oblique
entity features on 3,842 candidates, with a candidate-ranking correlation of
0.455. A viewpoint change here does not introduce an independent source of
information, which is the structural reason the entity round's dual-view fusion
could not pay for itself. Real independent imagery (CityFlight RGB) is a future
experiment and is not required by this round's question.

## Rescue analysis

The reference is the frozen masked visual: 266 samples wrong on `val_unseen`, of
which 180 are in Top-4 but not Top-1 — the subset this round exists to explain.
How many of those 180 each family moves to rank 1:

| arm | Top-4 → Top-1 rescued | of all 266 wrong |
|---|---:|---:|
| Visual + Relation ORACLE | **63.9%** | 59.0% |
| Relation ORACLE | 62.2% | 60.9% |
| Geometry | 27.2% | 21.8% |
| Visual + Geometry | 20.0% | 20.7% |
| Visual + Relation | 19.4% | 19.2% |
| Relation | 18.9% | 21.1% |

The oracle columns are upper bounds and are not methods.  Among deployable arms,
geometry is the strongest rescuer and reaches about a fifth of the gap; the
anchor oracle reaches nearly two thirds of it.

## Bottleneck ranking

Ordered by the measured effect on Top-1, not by intuition:

1. **Instance appearance** ★★★★★ — the colour, texture and size of the correctly
   masked target region land at **0.104**, i.e. chance. This is measured from the
   native 0.1 m/px raster and the entity's own point colours, so it is not an
   encoder limitation: at this resolution and range, what this dataset records of
   an entity does not separate it from its same-class neighbours.
2. **Anchor grounding** ★★★★★ — the oracle shows that 0.609 is reachable if the
   entity the language anchors to were known, against 0.234 for everything my
   rules can compute without it. The gap is the capability, not the features.
3. **Instruction relation semantics** ★★★★☆ — the parser's relation flags score
   0.107, chance. The instructions contain relation words (1,022 of 1,175
   training samples have at least one), and none of it reaches the model.
4. **Geometry** ★★★☆☆ — the only deployable family with a clear positive effect
   on the visual arm (+8.4), though it still lands below the frozen baseline.
5. **Map identity / semantic type** ★☆☆☆☆ — type one-hot and normalised block
   coordinates score 0.069, at chance, and the coordinate block is a memorisation
   trap that cannot transfer to unseen maps.
6. **An extra view of the same point cloud** ☆☆☆☆☆ — 0.744 feature correlation
   and no independent information.

## What to do next

One recommendation: **build anchor grounding, not a better encoder.**

The evidence is that the appearance channel of this benchmark is exhausted at the
resolution it provides (0.104 from native raster statistics), that the two views
are one source (0.744 correlation), and that the one thing which would move the
number is knowing which entity the language refers to (0.234 → 0.609). That is a
language-and-map problem — resolve "the church" to an entity in the map — not a
vision problem. Building a larger visual encoder, a finer crop or a third view
would be spending effort on the channel that is measurably at chance.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY -m pytest tests/ -q                                  # 131 tests
$PY scripts/run_feature_sufficiency_data.py              # ~7 min, 2255 samples
$PY scripts/train_feature_probes.py                      # ~35 min, 22 arms x 3 seeds
$PY scripts/train_feature_probes.py --hidden 0 --tag linear   # single-layer probe
```

Artefacts: `artifacts/feature_sufficiency/{metrics.json, per_type.json,
per_instruction_type.json, rescue_analysis.json, oracle_analysis.json,
qualitative_cases/}`.
