# CityRefer grounding model validation

Date: 2026-09-23. The new reproduction uses development splits only.

## Question

Can an open-source remote-sensing vision-language model, or a conditional
spatial diffusion model, localize the target described by a CityRefer
instruction more reliably than the current implicit HETT goal head?

## RSRefSeg2 RGB + language reproduction

The open-source RSRefSeg2 implementation was initialized from its official
RefSegRS checkpoint. Samples are map-balanced. Each sample contains a
512-by-512 RGB orthomosaic ROI covering the target and instruction landmarks;
the CityRefer target footprint is the supervision mask. This is an oracle-ROI
grounding test, not global search or navigation evaluation.

Pilot data:

- train_seen: 512 samples from 24 seen maps;
- val_unseen: 256 samples from four unseen maps;
- optimizer: AdamW, learning rate 1e-5, bf16;
- adaptation: one epoch, batch size 2;
- evaluation: identical val_unseen samples, batch size 16.

| Model | Box Acc@0.25 | Box Acc@0.50 | Mean Box IoU | Center hit | Mask gIoU | Non-empty |
|---|---:|---:|---:|---:|---:|---:|
| Official RefSegRS zero-shot | 3.52% | 1.17% | 4.41% | 24.61% | 1.80% | 100.00% |
| CityRefer 512 / 1 epoch | 1.56% | 0.00% | 3.32% | 8.59% | 1.48% | 81.25% |

This small adaptation regresses. It demonstrates that a few hundred examples
are insufficient and that pixel accuracy is misleading because background
dominates the ROI.

An earlier completed experiment used 4,096 map-balanced train_seen samples and
two epochs. On its frozen complete unseen evaluation, Box Acc@0.25 increased
from 1.38% to 29.98%, Box Acc@0.50 from 0.11% to 8.46%, and center hit from
20.66% to 33.21%. Same-image instruction switching also became more responsive,
but both instructions were correctly localized at IoU 0.25 in only 8.85% of
pairs. Thus adequate CityRefer adaptation learns real language sensitivity but
is not yet a reliable endpoint detector.

## Landmark map + language models

The existing pre-integration experiment trained on all 21,878 train_seen
trajectories and evaluated only val_seen/val_unseen. All models receive a BERT
instruction vector, a landmark union raster and the start position.

| Model | val_unseen mean error | Hit@10 | Hit@20 | Hit@50 |
|---|---:|---:|---:|---:|
| Point regression | 63.58 m | 3.11% | 11.05% | 44.05% |
| 20x20 heatmap | 72.95 m | 1.93% | 8.05% | 45.98% |
| Coordinate diffusion medoid | 61.69 m | 3.74% | 12.27% | 49.05% |

For conditional diffusion, 20 samples have an average spatial spread of
28.64 m. Its selected medoid is weak, but the best of the 20 samples reaches
Hit@20 = 52.98% with a mean oracle error of 32.48 m. The model is therefore
more useful as a multimodal candidate generator than as a single-goal decoder.

## Named-landmark ablation

The follow-up ablation keeps the spatial decoder, frozen BERT features,
positional channels, optimizer and data identical. The baseline collapses all
landmark polygons into one binary plane. The named model retains one mask per
landmark and paints each mask with a learned projection of that landmark's
BERT name embedding. Both models use a language-conditioned convolutional
heatmap decoder.

Three five-epoch runs were made with seeds 0, 1 and 2. The table reports the
mean on all 2,697 val_unseen examples.

| Representation | Top-1 mean error | Top-1 Hit@20 | Top-5 Hit@20 | Top-20 Hit@20 | Top-20 Hit@50 |
|---|---:|---:|---:|---:|---:|
| Union landmark mask | 64.98 m | 16.30% | 46.17% | 74.80% | 88.35% |
| Named landmark masks | **58.94 m** | **18.11%** | 45.94% | **74.87%** | **89.99%** |

The Top-1 error improvement is consistent across all three seeds
(-6.04 m mean delta, 0.68 m standard deviation). Precise candidate coverage is
not improved: Top-20 Hit@20 changes by only +0.07 points. Named landmarks make
the model rank broad regions better; they do not solve fine target selection.

The seed-0 breakdown shows where the gain comes from:

| val_unseen subset | Count | Union Top-1 error | Named Top-1 error | Union Hit@20 | Named Hit@20 |
|---|---:|---:|---:|---:|---:|
| One landmark | 1,429 | 67.08 m | 71.70 m | 15.12% | 12.67% |
| Two landmarks | 1,075 | 67.06 m | **48.89 m** | 14.14% | **18.98%** |
| Three or more | 191 | 46.97 m | **38.54 m** | 19.37% | **35.60%** |

Identity preservation is therefore useful specifically for relational,
multi-landmark instructions. A practical model should gate between a simpler
single-landmark path and the identity-aware relational path rather than force
one representation onto every instruction.

## Frozen SigLIP2 RGB reranking

A deterministic, map-balanced set of 256 val_unseen examples was selected.
For each example, the top five named-landmark candidates were converted to
equal 160 m RGB orthomosaic crops and scored against the instruction with
frozen `google/siglip2-so400m-patch16-512`. A one-position shuffled-instruction
control tests whether changes come from the instruction rather than generic
image preference.

| Selector | Mean error | Hit@20 | Hit@50 |
|---|---:|---:|---:|
| Spatial model Top-1 | **51.50 m** | **18.75%** | **66.41%** |
| Frozen SigLIP2 + true instruction | 56.31 m | 17.19% | 60.94% |
| Frozen SigLIP2 + shuffled instruction | 57.48 m | 15.63% | 57.03% |
| Oracle best of five | 33.57 m | 51.17% | 82.81% |

True instructions perform slightly better than shuffled instructions, so the
frozen encoder is not completely language-blind. It nevertheless changes the
spatial model's first choice in 82.81% of examples and makes every main metric
worse. Frozen generic RGB-language similarity must not be connected to the
navigator as a candidate selector. The large gap to the top-five oracle shows
that candidate generation is already useful and that CityRefer-domain target
discrimination is now the main missing component.

## Explicit candidate-to-landmark geometry gate

A subsequent Gate-1 experiment removed RGB and scored every cell in the same
20-by-20 candidate grid from the instruction, UAV start, and explicit
candidate-to-landmark geometry. Geometry consists of normalized `dx`, `dy`,
log-distance, `sin(angle)` and `cos(angle)`. Polygon centroids are computed in
world coordinates and converted to the north-up map frame. Unit tests cover
the east/north convention and the zero-distance case.

Two otherwise identical rankers were trained for five epochs with three
seeds:

- anonymous geometry retains one geometry token per landmark but omits its
  name embedding;
- named geometry adds a learned projection of the frozen BERT landmark-name
  embedding to the corresponding geometry query.

| Split / model | Mean error | Hit@20 | Hit@50 | Top-5 Hit@20 | Top-20 Hit@20 |
|---|---:|---:|---:|---:|---:|
| val_seen / anonymous | 41.48 m | 26.49% | 72.85% | 57.42% | 86.94% |
| val_seen / named | **37.14 m** | **32.32%** | **77.94%** | **64.87%** | **91.50%** |
| val_unseen / anonymous | **60.87 m** | **21.10%** | 60.52% | **47.40%** | 71.88% |
| val_unseen / named | 63.82 m | 17.95% | **61.07%** | 44.06% | 71.88% |

The direct name binding is strongly beneficial on seen maps but reverses on
unseen maps. This is a failed generalization gate, not support for deploying
the named geometry ranker. It is consistent across all three seeds: on
val_unseen, named geometry loses 3.15 Hit@20 points and adds 2.95 m mean error.
The same reversal appears in both one- and two-landmark subsets.

The geometry-flip control provides a useful diagnosis. At the true target
cell, named geometry prefers the real orientation over a 180-degree-flipped
orientation by +1.15 logits on val_seen and only +0.35 on val_unseen.
Anonymous geometry has a +0.59 margin on val_seen and -0.06 on val_unseen.
The current pooled-text query therefore learns city-specific layout/name
associations more readily than transferable relation language.

The next geometry model should not inject a freely trainable landmark-name
embedding into the relation query. It should first locate each landmark span
in the instruction, derive a landmark-specific contextual relation embedding,
and bind that context to pure geometry. Direction-word-preserving rotation and
mirror augmentation, nearest-boundary geometry for large landmarks, and
explicit relation perturbation loss should be validated before RGB is added.

### Landmark-span relation follow-up

The follow-up masks each processed landmark mention inside the original
instruction and encodes the resulting context as that landmark's relation
query. The landmark name itself is no longer injected into the geometry
branch. Automatic punctuation-tolerant span matching covers 38,008 of 42,200
mentions (90.07%); the remainder contains aliases, misspellings, or semantic
references that require a better resolver.

To control for training duration, anonymous geometry, span geometry, and span
geometry plus a directional 180-degree-flip margin were each trained for one
epoch with the same three seeds.

| val_unseen model | Mean error | Hit@20 | Hit@50 | Top-5 Hit@20 | Top-20 Hit@20 |
|---|---:|---:|---:|---:|---:|
| Anonymous geometry | 63.86 m | 21.69% | 60.96% | 47.92% | 72.04% |
| Landmark-span geometry | 63.71 m | 23.08% | 61.64% | 48.40% | **72.35%** |
| Span geometry + flip loss | **62.63 m** | **23.25%** | **62.30%** | **48.55%** | 72.23% |

Relative to the controlled anonymous baseline, the best span model adds 1.56
Hit@20 points, adds 1.35 Hit@50 points, and reduces mean error by 1.23 m. The
two-landmark Hit@20 gain is 2.20 points without flip loss. This is a consistent
positive signal, but it remains below the predefined +3 point overall gate.

The flip loss also fails its intended causal check: its average true-target
logit margin against flipped geometry is -0.057 on val_unseen. Its metric gain
cannot yet be attributed to correct direction understanding. Do not add RGB or
connect this ranker to navigation yet. The next iteration must improve alias
span coverage, use token-level landmark-conditioned attention rather than a
mean-pooled masked sentence, and report perturbation margins only on explicitly
directional instructions.

### Token context, boundary geometry and soft-distance targets

The next iteration implements those data fixes. Exact, resolved-name and fuzzy
span matching raises landmark masking coverage from 90.07% to 99.25%
(41,885/42,200). Instead of mean pooling the masked sentence, it uses BERT's
contextual vector at the `[MASK]` token. Directional instructions are reported
separately.

For large or elongated landmarks, each contour is resampled to 16 boundary
points. Candidate geometry then includes both centroid-relative and
nearest-boundary-relative `dx`, `dy`, log-distance, sine and cosine. Finally,
hard single-cell cross entropy is replaced by a target distribution decaying
with metric distance using a 20 m temperature. All results below use one epoch,
three identical seeds and the same val_unseen split.

| Geometry model | Mean error | Hit@20 | Hit@50 | Top-5 Hit@20 | Top-20 Hit@20 |
|---|---:|---:|---:|---:|---:|
| Anonymous centroid baseline | 63.86 m | 21.69% | 60.96% | 47.92% | 72.04% |
| Token-context centroid | 60.19 m | **22.79%** | 62.27% | 48.92% | 72.76% |
| Token-context centroid + boundary | 58.35 m | 21.76% | 62.54% | **51.03%** | 75.94% |
| Boundary + soft-distance target | **55.99 m** | 22.06% | **63.80%** | 50.71% | **75.97%** |

Relative to the controlled anonymous baseline, the final model reduces mean
error by 7.87 m (12.3%), adds 2.84 Hit@50 points, adds 2.79 Top-5/20m points,
and adds 3.93 Top-20/20m points. On two-landmark instructions, mean error drops
from 71.27 m to 52.49 m. The predefined candidate-coverage gate is therefore
passed.

Top-1 Hit@20 improves by only 0.37 points. Direction understanding also remains
unproven: on explicitly directional unseen instructions, the final model's
true-versus-180-degree-flipped target margin is -0.011 logits. The geometry
model should now be treated as a candidate generator, not an endpoint
selector. The next justified experiment is a candidate-level RGB/semantic
ranker trained on these hard candidates; frozen zero-shot SigLIP2 remains
disallowed because it previously degraded ranking.

## Candidate-level RGB/language ranker

The boundary/soft-distance geometry model generated five hard candidates per
instruction. A pilot used 4,096 map-balanced train_seen instructions and 512
examples from each development split. When no training group contained a
candidate within 20 m, the true location was injected and the five candidates
were shuffled; this occurred in 1,607/4,096 groups. Validation candidates were
never modified.

SigLIP2 image and text encoders remained frozen. A small projection/ranking
head was trained for 30 epochs and selected by val_seen loss. Three seeds used
the same cached features. A geometry-rank weight was selected on val_seen and
then fixed for val_unseen.

| val_unseen selector | Mean error | Hit@20 | Hit@50 |
|---|---:|---:|---:|
| Geometry first candidate | 49.24 m | 19.14% | 64.26% |
| RGB + true instruction | 49.28 m | 21.42% | 64.97% |
| Geometry + RGB + instruction | **48.79 m** | **21.94%** | **65.89%** |
| RGB + shuffled instruction | 50.77 m | 19.40% | 63.15% |
| Oracle best of five | 32.20 m | 50.78% | 83.40% |

The fused ranker adds 2.80 Hit@20 points and 1.63 Hit@50 points over geometry
alone. True instructions beat shuffled instructions by 2.54 Hit@20 points, so
the trained head uses language rather than only generic image appearance. The
three seeds agree in direction, although the result remains just below the
predefined +3 point deployment gate.

This is a positive pilot, not deployment evidence. Training candidates were
generated in-sample rather than with map-level out-of-fold geometry models,
39.2% of training groups required a positive injection, and only 512 unseen
examples were evaluated. The next required validation is out-of-fold candidate
generation followed by the same frozen-feature ranker on the complete
val_unseen split.

### Map-level out-of-fold confirmation

The required leakage-controlled run does not confirm the pilot. The 24
training maps were assigned to four disjoint folds. Every training candidate
was produced by a geometry model that had not trained on that candidate's map.
The four held-out folds contained 5,417, 5,462, 5,429 and 5,570 examples.
The four val_unseen maps do not occur in train_seen or val_seen.

The RGB ranker used 4,096 map-balanced training groups, all 2,470 val_seen
groups and all 2,697 val_unseen groups. Target injection and candidate
shuffling were restricted to training; 1,850 training groups needed an
injected positive. Model checkpoint and geometry-fusion weight selection used
val_seen only. Validation candidates were neither injected nor shuffled.

| Full val_unseen selector (3-seed mean) | Mean error | Hit@20 | Hit@50 |
|---|---:|---:|---:|
| Geometry first candidate | **55.67 m** | **21.95%** | 63.63% |
| RGB + true instruction | 57.75 m | 19.16% | 62.29% |
| Geometry + RGB + instruction | 56.26 m | 21.53% | **63.97%** |
| RGB + shuffled instruction | 57.71 m | 19.16% | 62.39% |
| Oracle best of five | 39.19 m | 50.24% | 79.31% |

The fused selector loses 0.42 Hit@20 points and adds 0.59 m mean error versus
geometry alone. It improves only Hit@50, by 0.35 points. Only one of three
seeds beats the geometry baseline on Hit@20 and mean error. True instructions
and the shuffled-instruction control both average 19.16% Hit@20, with mixed
per-seed differences, so this run provides no evidence that the ranker
understands the target relation.

This is a failed confirmation gate. The pilot gain was not robust to
map-out-of-fold candidates and full-split evaluation. The RGB checkpoint must
not be integrated into HETT. The reported val_unseen split has also been
observed repeatedly during development, so it is development evidence rather
than a pristine final test; future architectural changes require a newly
reserved map-level holdout before making a final generalization claim.

### Same-map wrong-instruction training

A fixed follow-up adds one contrastive control to the same out-of-fold setup.
For every training group, an instruction describing a different target on the
same map is paired with the original candidate crops. The loss requires the
true instruction to score the target-nearest candidate above this switched
instruction. The loss weight (0.2) and margin (0.5) were fixed before the run;
no weight sweep was performed.

| Full val_unseen selector (3-seed mean) | Mean error | Hit@20 | Hit@50 |
|---|---:|---:|---:|
| Geometry first candidate | **55.67 m** | **21.95%** | 63.63% |
| RGB + true instruction | 57.15 m | 19.64% | 62.98% |
| Geometry + RGB + instruction | 55.68 m | 21.73% | **64.45%** |
| RGB + same-map wrong instruction | 57.41 m | 19.24% | 62.30% |

The control increases instruction sensitivity: true instructions exceed
same-map wrong instructions by 0.40 Hit@20 points, and the mean target-score
margin is +0.097 logits. This is much smaller than required and is dominated
by one seed. The fused selector still loses 0.22 Hit@20 points to geometry
alone, although Hit@50 improves by 0.83 points. This variant also fails the
localization gate.

The dataset contains a nominally untouched test_unseen split with 5,281
instructions on six new maps. Its required PNG and georeferenced TIFF imagery
is absent from the available `data/rgbd` package, so no RGB result is reported
for that split. It remains sealed rather than substituting images or reporting
a geometry-only number as an RGB test.

### Explicit candidate-to-landmark binding

The next fixed experiment restores information discarded by the original
candidate export. Each candidate now carries inference-visible landmark
centers, 16-point boundaries, resolved names and relation phrases. Target
coordinates are used only to create training loss and evaluation metrics, not
as model input. Cache alignment is checked by recomputing candidate distances
before training, preventing a stale or reordered feature cache from silently
producing invalid results.

Giving the ranker raw geometry while pairing every landmark with the same
whole-sentence SigLIP vector does not help: fused Hit@20 is 21.67%, below the
21.95% geometry baseline. This isolates the missing operation as relation-to-
landmark binding rather than merely missing coordinates.

The final variant masks each landmark mention separately, extracts frozen
BERT context at that `[MASK]` token, and pairs it only with that landmark's
centroid and nearest-boundary geometry. RGB and whole-instruction SigLIP
features remain unchanged.

| Full val_unseen selector (3-seed mean) | Mean error | Hit@20 | Hit@50 |
|---|---:|---:|---:|
| Geometry first candidate | 55.67 m | 21.95% | 63.63% |
| Relation-bound RGB + instruction | 56.51 m | 20.34% | 63.44% |
| Geometry + relation-bound RGB + instruction | **55.33 m** | **22.12%** | **64.73%** |
| Same-map wrong instruction | 57.41 m | 19.69% | 62.22% |

Binding relation context to the corresponding landmark is directionally
correct. The fused model reduces mean error by 0.33 m, adds 0.17 Hit@20 points
and adds 1.10 Hit@50 points over geometry alone. Correct instructions exceed
same-map wrong instructions by 0.66 Hit@20 points, and the target-score margin
rises from +0.098 to +0.173 logits.

This remains development evidence, not a passed gate. Two seeds improve
Hit@20, while one loses 1.22 points; the mean gain is far below the predefined
+3 point requirement. The checkpoint must not be integrated into HETT. It
does, however, validate the data path: explicit per-landmark relation binding
is more useful than either whole-sentence geometry or RGB alone.

### Multi-component landmark resolution

The name-to-map audit found that exact lookup silently selected one contour
when roads, paths or water features had several same-name components. The
resolver now unions exact same-name components, computes an area-weighted
centroid, samples their combined boundary, and rejects low-score or ambiguous
fuzzy matches. On val_unseen, all exact duplicates are linear or water
features rather than duplicated buildings.

Candidates were regenerated with the same four map-level out-of-fold protocol.
The correction substantially reduces large errors:

| val_unseen candidates | Mean error | Hit@20 | Hit@50 | Top-5 mean | Top-5 Hit@50 |
|---|---:|---:|---:|---:|---:|
| Previous resolver | 55.67 m | **21.95%** | 63.63% | 39.19 m | 79.31% |
| Grouped/rejecting resolver | **45.47 m** | 20.76% | **65.70%** | **29.63 m** | **82.83%** |

The grouped resolver cuts mean error by 10.20 m and Top-5 oracle error by
9.57 m, but loses 1.19 Hit@20 points. It improves global candidate placement
and suppresses severe mistakes, while leaving fine ordering unresolved.

The locked relation-bound RGB ranker was then retrained on the new candidate
crops:

| Full val_unseen selector (3-seed mean) | Mean error | Hit@20 | Hit@50 |
|---|---:|---:|---:|
| Grouped geometry first candidate | 45.47 m | 20.76% | 65.70% |
| Relation-bound RGB + instruction | 46.00 m | 19.64% | 65.80% |
| Geometry + relation-bound RGB + instruction | **45.03 m** | **21.55%** | **66.32%** |
| Same-map wrong instruction | 46.82 m | 18.85% | 64.45% |

The fused mean improves by 0.44 m, Hit@20 by 0.79 points and Hit@50 by 0.62
points over the corrected geometry baseline. This gain is not stable: seed 0
adds 3.19 Hit@20 points, while seeds 1 and 2 lose 0.56 and 0.26 points. The
resolver fix is accepted as a data correction; the RGB selector still fails
the stability and +3 point gates.

### Target-type alignment diagnostic

A four-class diagnostic tests the minimal semantic claim directly: can frozen
SigLIP2 features distinguish `Building`, `Car`, `Ground`, and `Parking`? Only a
linear classifier is trained. Training uses `train_seen`, checkpoint selection
uses `val_seen`, and all reported results use map-disjoint `val_unseen`.
Images are deduplicated by target object before training. Target coordinates
are never classifier or selector inputs.

| val_unseen, 3-seed mean | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Full instruction -> target type | **96.16%** | **95.61%** | **93.76%** |
| 160 m RGB crop -> center-object type | 56.60% | 54.88% | 49.84% |

The instruction type is therefore already highly recoverable. RGB type is
present but weak: mean per-class recall is about 69% for buildings, 36% for
cars, 78% for ground, and 37% for parking. Small cars and parking areas are the
main failures at this crop scale.

As a strict downstream test, each instruction and each of its five candidate
crops are classified independently. The selector retains geometry order and
chooses the first candidate whose predicted RGB type matches the predicted
instruction type. It sees neither the true class nor target coordinates.

| val_unseen selector | Mean error | Hit@20 | Hit@50 |
|---|---:|---:|---:|
| Grouped geometry first candidate | **45.47 m** | **20.76%** | 65.70% |
| Predicted type match, 3-seed mean | 45.64 m | 20.73% | **65.88%** |

Type matching changes only 9--10% of selections and produces no meaningful
fine-localization gain. It should be retained as an auxiliary semantic feature,
not used as a standalone candidate selector. The next useful test is a smaller
or multi-scale crop and explicit object segmentation, especially for cars and
parking regions.

#### Fine-view scale confirmation

The same leakage-controlled experiment was repeated with 40 m and 80 m RGB
crops. A concatenated 40+80+160 m feature was also tested. The language probe
and geometry candidates were unchanged.

| RGB view | Image accuracy | Image balanced accuracy | Paired text-image agreement | Type-match Hit@20 |
|---|---:|---:|---:|---:|
| 160 m | 56.60% | 54.88% | 56.66% | 20.73% |
| 80 m | 65.95% | 62.43% | 65.62% | 21.98% |
| **40 m** | **76.73%** | **72.54%** | **75.69%** | 23.83% |
| 40+80+160 m | 72.94% | 69.07% | 72.69% | **23.94%** |

The grouped geometry baseline is 20.76% Hit@20. Type matching at 40 m adds
3.07 points, and the three-scale variant adds 3.18 points. All three seeds
improve by about three points. The 40 m view is the cleanest type classifier;
the fused view slightly improves fine candidate selection but reduces Hit@50
from 65.70% to 65.06%. This confirms basic semantic-to-RGB type matching while
showing that it should be a local fine-navigation signal rather than a global
replacement for geometry.

### Controlled spatial-relation probe

A four-way mirrored-candidate probe isolates language-to-geometry binding from
candidate generation and RGB. For every relation, the scorer sees only masked
relation text and four symmetric relative positions: true, x-mirrored,
y-mirrored and 180-degree mirrored. It has no absolute map position or
candidate-index feature. Chance accuracy is 25%.

The probe uses only direct patterns such as "to the left of [landmark]" and
requires at least 10 m separation on both axes. Even in this restricted set,
only 82/217 (37.8%) val_unseen phrases agree with the current north-up target-
minus-landmark coordinate sign. Nearest-boundary geometry does not resolve the
mismatch, indicating an ambiguous or inconsistent language reference frame
rather than only a large-landmark centroid problem.

| val_unseen, 3-seed mean | Four-way accuracy | Same-map wrong text | Target margin |
|---|---:|---:|---:|
| Canonical synthetic relation text | **99.54%** | 45.31% | **+1.28** |
| Original human instruction | 27.19% | 28.11% | **-2.36** |

The architecture can learn exact spatial binding when the language uses a
defined coordinate frame. The direct numeric comparison is conservative:
synthetic prompts state both horizontal and vertical relations, while most
selected human phrases constrain only one axis, so exact four-way target
selection over-penalizes the human set. Nevertheless, same-map wrong text is
not worse and the true-position margin is negative. The follow-up below audits
the one stated axis directly before adding visual inputs.

#### Reference-frame and landmark-mask audit

Instruction landmark names are now resolved to their annotated polygon groups
and rasterized as map-aligned masks. A visual audit overlays the resolved mask,
target polygon, landmark-to-target line and extracted direction word on the
real RGB map.

Eight fixed coordinate transforms and the UAV initial heading were evaluated
using direct one-axis phrases. The transform is selected on `train_seen` only.
With nearest-boundary geometry, a 270-degree rotation is best on train_seen
(68.00%) and remains best on val_unseen (77.88%). UAV-heading coordinates stay
near chance (51.15%). A per-map oracle reaches only 78.80% on val_unseen, less
than one point above the fixed transform, so map-specific rotations do not
justify additional complexity. The remaining roughly 22% should be treated as
ambiguous/noisy relation supervision rather than hard constraints.

A visual four-candidate probe then consumes full-map RGB, the instruction-
resolved landmark mask, a candidate marker, masked instruction text and the
fixed-frame candidate-to-boundary relation. It is trained on train_seen,
selected on val_seen and tested on 189 in-bounds val_unseen relations.

| val_unseen input/control, 3-seed mean | Four-way accuracy |
|---|---:|
| RGB + landmark mask + corrected relation | **41.27%** |
| Remove explicit corrected relation | 30.69% |
| Same-map wrong relation cue | 26.81% |
| Remove RGB | 41.62% |
| Remove landmark mask image | 41.80% |
| Same-map wrong full instruction | 42.50% |

The corrected structured relation is causal: removing it loses 10.58 points
and supplying the wrong cue falls nearly to the 25% chance level. However, the
current CNN does not use RGB, the raster mask, or the remaining free-form text;
their ablations do not reduce accuracy. Therefore the accepted component is a
soft structured geometry prior derived from the named landmark polygon and the
270-degree language-frame correction. The visual spatial branch is rejected in
its current form and must not be connected to HETT yet.

#### Twenty-meter performance by landmark count

The corrected grouped candidates and 40 m predicted type matcher were
re-evaluated using only the operational 20 m success circle.

| Resolved instruction landmarks | Count | Geometry Top-1 Hit@20 | 40 m type-match Hit@20 | Top-5 contains a Hit@20 candidate |
|---|---:|---:|---:|---:|
| One | 1,428 | 14.01% | 18.39% | 40.20% |
| Two | 1,075 | 26.70% | **28.90%** | **60.09%** |
| Three or more | 191 | 38.22% | 36.30% | 78.01% |

One-landmark instructions are not already solved. Two landmarks are materially
easier because their constraints intersect: they add 10.51 Hit@20 points over
the one-landmark type matcher, and the five candidates contain a valid 20 m
answer in 60.09% of cases. On direct direction phrases, the fixed-frame cue
agrees with nearest-boundary geometry in 73.26% of one-landmark examples and
79.63% of two-landmark examples. The two-landmark bottleneck is therefore
candidate ordering, not candidate coverage or landmark-name resolution.

#### Relation-conditioned Top-5 ranker

A candidate-level relation/geometry Transformer was trained on all 21,878
`train_seen` instructions. Its only inputs are frozen BERT embeddings of the
landmark-masked relation text, compact structured relation cues, and the
candidate-to-landmark centroid/boundary geometry after the train-selected
270-degree reference-frame correction. It receives no RGB, target type,
absolute candidate coordinate, or target coordinate. Training candidates are
map-level out-of-fold predictions; target coordinates are used only to form
the 20 m/listwise labels.

The initial model failed the anti-cheating test: switching the complete
same-map relation/landmark input improved Hit@20. Adding structured relation
cues made the language signal measurable but did not fix landmark identity.
The accepted experimental variant therefore uses same-map, same-landmark-count
mismatch negatives during training. Three random seeds were selected on
`val_seen` and evaluated on all 2,697 `val_unseen` instructions.

| `val_unseen`, 3-seed mean | Hit@20 | Mean error |
|---|---:|---:|
| Original candidate rank 1 | 20.76% | 45.47 m |
| Correct relation + correct landmarks | **24.04%** | **44.49 m** |
| Same-map wrong relation | 22.33% | 45.25 m |
| Same-map wrong landmarks | 21.53% | 45.78 m |
| Relation and landmarks both switched | 21.74% | 45.66 m |
| Left/right and above/below reversed | 22.77% | 45.00 m |
| Remove all relation inputs | 22.11% | 45.29 m |
| Top-5 oracle | 50.76% | 29.63 m |

Correct relation/landmark binding adds 3.28 absolute Hit@20 points over the
candidate-order baseline. The gain is causal under the mismatch controls, but
modest and seed-dependent: one seed performs worse than its no-relation
ablation even though wrong relation and wrong landmark controls are lower in
all seeds. This is evidence that the spatial branch is useful, not evidence
that free-form spatial language is solved.

| Resolved landmarks | Relation ranker Hit@20 | Geometry rank-1 Hit@20 | Gain |
|---|---:|---:|---:|
| One | 15.99% | 14.01% | +1.98 |
| Two | **30.20%** | 26.70% | +3.50 |
| Three or more | **48.52%** | 38.22% | +10.30 |

The benefit grows with landmark count, consistent with the intended
multi-landmark intersection behavior. The relation ranker is slightly above
the 40 m type matcher for two-landmark instructions, but below it for
one-landmark instructions. The next controlled experiment should combine the
two scores rather than replace either branch.

#### Relation and 40 m target-type score fusion

The accepted relation checkpoint and independently trained 40 m RGB/type
probes were frozen. Their per-candidate scores and the original candidate-rank
prior were normalized and fused with only two parameter groups: one-landmark
instructions and multi-landmark instructions. Fusion weights were selected on
`val_seen` only and then evaluated once on all `val_unseen` rows. All three
seeds selected type-dominant weights for one landmark and a stronger relation
contribution for multiple landmarks, matching the pre-registered rationale.

| `val_unseen`, 3-seed mean | Hit@20 | Mean error |
|---|---:|---:|
| Original candidate rank 1 | 20.76% | 45.47 m |
| Hard 40 m type match | 23.83% | 44.79 m |
| Relation ranker only | 24.04% | 44.49 m |
| **Relation + type fusion** | **27.94%** | **42.95 m** |
| Same-map wrong relation | 27.07% | 43.52 m |
| Same-map wrong landmarks | 26.60% | 43.69 m |
| Same-map switched type text | 27.59% | 43.13 m |
| Guaranteed type-class permutation | **16.54%** | 47.82 m |
| Top-5 oracle | 50.76% | 29.63 m |

Fusion adds 7.18 absolute Hit@20 points over candidate rank 1, 3.90 points
over relation alone, and 4.12 points over hard type matching. It is stable
across seeds (27.77%, 27.96%, and 28.11%). Same-map type switching causes only
a small drop because many instructions on a map share the same target class;
the guaranteed four-class permutation causes an 11.41-point collapse and
confirms that target type is causal. Wrong relations and landmarks also reduce
performance for every seed, though their effects are smaller because the
one-landmark group deliberately relies mostly on type.

| Resolved landmarks | Fusion Hit@20 | Best previous component |
|---|---:|---:|
| One | **21.08%** | 18.39% type match |
| Two | **33.30%** | 30.20% relation ranker |
| Three or more | **49.56%** | 48.52% relation ranker |

The two-landmark group now reaches the predefined 32--35% target. This is the
first candidate-ranking variant that improves both the single-landmark and
multi-landmark regimes rather than trading one for the other.

#### Fresh `test_unseen` confirmation and RGB blocker

The accepted candidate generator was retrained deterministically from
`train_seen`. Before touching `test_unseen`, its 2,697 `val_unseen` Top-5
outputs were compared with the saved candidates: all coordinates were exactly
equal, Top-1 reproduction was 100%, and maximum coordinate difference was
zero. The reproduced generator then produced candidates for all 5,281 fresh
`test_unseen` instructions without using their target coordinates.

The frozen relation checkpoints transfer to this split:

| Full `test_unseen`, 3-seed mean | Hit@20 | Mean error |
|---|---:|---:|
| Original candidate rank 1 | 35.22% | 30.61 m |
| **Correct relation + correct landmarks** | **38.17%** | **30.10 m** |
| Same-map wrong relation | 36.98% | 30.42 m |
| Same-map wrong landmarks | 35.28% | 31.26 m |
| Left/right and above/below reversed | 36.91% | 30.38 m |
| Remove all relation inputs | 35.59% | 30.64 m |
| Top-5 oracle | 74.23% | 17.01 m |

The relation branch therefore reproduces a +2.95 absolute Hit@20 gain on the
fresh split, and all controls reduce it. By resolved landmark count it reaches
30.92% for one landmark, 44.16% for two, and 50.17% for three or more.

After the six missing RGB orthomosaics were supplied, the complete frozen
fusion was run without changing the models or the weights selected on
`val_seen`.

| Full `test_unseen`, 3-seed mean | Hit@20 | Mean error |
|---|---:|---:|
| Original candidate rank 1 | 35.22% | 30.61 m |
| Hard 40 m type match | 39.17% | 30.16 m |
| Relation ranker only | 38.17% | 30.10 m |
| **Frozen relation + type fusion** | **44.15%** | **28.94 m** |
| Same-map wrong relation | 43.53% | 29.17 m |
| Same-map wrong landmarks | 42.30% | 29.58 m |
| Guaranteed type-class permutation | **27.02%** | 33.69 m |
| Top-5 oracle | 74.23% | 17.01 m |

Fusion improves Hit@20 by 8.93 absolute points over candidate rank 1, 5.98
points over relation alone, and 4.97 points over hard type matching. The three
frozen runs are stable at 43.82%, 44.01%, and 44.61%. Type remains strongly
causal: permuting its four class meanings loses 17.12 points. Wrong relations
lose 0.62 points and wrong landmarks lose 1.84 points; their smaller effect is
expected because the one-landmark group uses the previously selected
type-dominant weights.

| Resolved landmarks | Frozen fusion Hit@20 |
|---|---:|
| One | **38.88%** |
| Two | **47.77%** |
| Three or more | **56.30%** |

This confirms the development result on a genuinely fresh map split: type is
most valuable for ambiguous one-landmark instructions, while relation geometry
adds more constraint as landmark count grows.

#### Post-test ranking-error analysis and exploratory appearance fusion

After the frozen benchmark had been reported, its 1,547 consensus ranking
errors were inspected separately from the 1,361 examples where no Top-5
candidate was within 20 m. The main findings were:

- mean pairwise cosine similarity among the five 40 m candidate RGB features
  was 0.944 on ranking errors, so coarse type often cannot distinguish nearby
  candidates of the same class;
- 55.98% of ranking errors had both positive and negative candidates matching
  the predicted target type, while 28.38% had a match only on a negative;
- the relation and type branches could independently put a positive above all
  negatives in only 31.09% and 18.16% of ranking errors respectively;
- errors were concentrated in one-landmark instructions (43.01% conditional
  error rate) and building targets (44.02%), with `across/opposite` also weak
  at 44.94%;
- raw frozen SigLIP full-instruction/RGB similarity was complementary: it could
  separate 46.80% of the existing ranking errors.

An exploratory four-score fusion therefore added raw full-instruction/RGB
similarity to relation, coarse type, and candidate prior. All weights were
selected only on `val_seen`; all three seeds retained a nonzero appearance
weight. On `test_unseen` it reached 48.49% mean Hit@20 (47.74%, 48.70%, and
49.04%) and 27.72 m mean error, compared with 44.15% and 28.94 m for the
previous fusion. Because this hypothesis was formed after inspecting
`test_unseen` errors, the result is explicitly exploratory rather than a new
pristine blind benchmark. A new map-level holdout is required before accepting
it as the replacement scorer.

Two follow-up ranking variants were rejected. Adding a 20 m candidate crop to
the 40 m context crop changed the `val_seen` map-holdout Hit@20 from 31.66% to
31.60%, despite a small exploratory `test_unseen` change from 48.49% to 48.75%.
A 32-unit nonlinear score calibrator reached only 30.97% on the same map
holdout and 46.23% on exploratory `test_unseen`. Thus neither crop scale alone
nor a more flexible fusion rule addresses the remaining ambiguity.

An ideal semantic-component diagnostic then used all annotated object class
components on each map and the predicted instruction target class. It did not
use the target object id or target coordinate. Adding distance to the nearest
component of each possible target class improved the map holdout from 31.45%
to 33.65%, and exploratory `test_unseen` from 49.31% to 52.41%. Semantic input
alone reached 47.40%. This is not an RGB-only deployable result; it measures
the value available if RGB semantic segmentation can recover class components.
It supports training a candidate-centered semantic component predictor as the
next model, followed by relation scoring between those components and the
resolved landmark masks.

#### RGB semantic-component feasibility

A frozen SigLIP 40 m/20 m feature probe could not regress nearest semantic
component distances accurately enough: its cross-map distance MAE was about
19.1 m, its map-holdout fusion changed Hit@20 only from 30.03% to 30.61%, and
exploratory `test_unseen` fell from 49.31% to 49.01%. Global embeddings have
therefore discarded too much pixel location information for this task.

A small from-scratch U-Net was then trained directly on RGB tiles and masks
rasterized from all-object annotations on the 24 `train_seen` maps. On four
unseen validation maps, an 80 m / 128 px view reached 46.52% foreground mIoU.
Reducing the view to 40 m at the same pixel resolution improved foreground
mIoU to 52.77%, with class IoUs of 62.73% building, 65.93% car, 52.78% ground,
and 29.63% parking. This validates the fine-altitude hypothesis specifically
for dense semantic perception.

However, connected components extracted from the 40 m prediction were not yet
accurate enough to close the ranking gap. Their score correlated only 0.497
with the annotation-derived component score on `test_unseen`, and their Top-1
agreement was 46.53%. Predicted semantic ranking alone reached 38.93% Hit@20,
versus 47.40% for ideal annotated components. When fused with the other frozen
scores, `test_unseen` changed only from 48.72% to 48.89% (weights selected on
`val_unseen`). The segmentation model is therefore useful evidence that lower
altitude improves semantic resolution, but it is not accepted as a ranking
improvement. The next segmentation objective must optimize component center
and instance separation rather than pixel mIoU alone.

A follow-up four-class center-heatmap model showed that this objective is not
equally well-defined for every class in a local tile. It learned car centers
on unseen maps (64.47% F1, 72.14% recall, and about 0.81 m localization error),
but did not learn stable geometric centers for buildings, ground, or parking.
Those large or irregular components can extend beyond the 40 m crop and their
polygon centroid is not necessarily a visible feature. A class-balanced retry
collapsed confidence across the sparse classes and was stopped before test
evaluation.

The validated car-center channel was therefore combined with connected
components from the 40 m segmentation model for the other classes. This raised
semantic-only `test_unseen` Hit@20 from 38.93% to 40.22% and slightly improved
score correlation with ideal components from 0.497 to 0.513. The complete
fusion changed 48.72% to only 48.99%, however. This hybrid is also rejected as
a ranking upgrade. The evidence now favors class-specific representation:
point centers for compact vehicles, instance masks/components for buildings,
and region masks for ground and parking. More importantly, these components
must be bound to each instruction's landmark relation; class presence alone is
already saturated.

#### Instance-bound relation diagnostic

Using the predicted text class, each Top-5 grid point was bound to its nearest
annotated component without using the target object id or target coordinate.
Feeding these instance centers directly to the existing grid-trained relation
model reduced component fusion from 34.52% to 34.00% on `val_unseen` and from
51.71% to 50.96% on exploratory `test_unseen`, demonstrating a coordinate
representation mismatch.

The relation model was then retrained on the same instance-bound geometry,
while retaining original Top-5 target distances as labels. Relation-only
`test_unseen` reached 37.82%, slightly below the original 38.17%. With ideal
annotated components, type, and appearance, fusion reached 52.60%, versus
51.71% for the matched old-relation formulation. However, `val_unseen` moved
in the opposite direction, from 34.52% to 33.89%. The +0.89 test change is
therefore not accepted as stable. Instance binding has some complementary
signal, but the present masked-BERT/centroid geometry relation representation
does not generalize reliably across maps.

#### Explicit cue-by-cue relation diagnostic

To distinguish genuine spatial-language use from a flexible latent scorer, an
interpretable ranker was trained with one satisfaction channel per parsed
relation cue, followed by mean/min/max aggregation across named landmarks. It
was trained on `train_seen` and selected on `val_seen`; target coordinates were
used only as ranking labels and metrics.

| Model / control | `val_unseen` Hit@20 | `test_unseen` Hit@20 |
|---|---:|---:|
| Candidate prior | 20.76% | 35.22% |
| Explicit relation ranker | 21.69% | 35.85% |
| Remove all relation inputs | 17.15% | 29.49% |
| Remove structured cues | 18.32% | 30.78% |
| Reverse directional cues | 20.53% | 35.32% |

The large removal losses establish that the model really uses language and
spatial cues, but its standalone accuracy is below the accepted learned
relation ranker (24.04% / 38.17%). It was then added as a complementary score
to relation, type, 40 m appearance, and candidate prior. Base weights and the
new score weight were selected only on `val_unseen`, then frozen for
`test_unseen`:

| Fusion | `val_unseen` Hit@20 | exploratory `test_unseen` Hit@20 |
|---|---:|---:|
| Base scorer | 32.37% | 48.15% |
| + explicit relation | 32.55% | 48.32% |

The selected explicit weight is zero for one-landmark instructions and 0.25
for multiple landmarks. The +0.17-point test change is consistent with a small
multi-landmark complement, but is too small to treat as a practical upgrade.
The principal remaining error is therefore not failure to expose directional
words; it is visually binding the correct same-class object instance to those
relations.

#### Shape-aware landmark-mask binding

The next diagnostic replaced the centroid-only landmark representation with a
compact description of the full named-landmark mask: oriented support in eight
directions, width, height, area, containment, centroid offset, and nearest
boundary offset. The added support and shape descriptors are invariant to the
arbitrary first contour point. They contain no target coordinate. The relation encoder and landmark
aggregation were otherwise kept comparable to the accepted relation model.

Across three seeds, relation-only `val_unseen` Hit@20 reached 23.77%, versus
24.04% for the accepted relation ranker. Removing relation input reduced it to
19.79%; switching the relation or named landmarks reduced it to 22.70% and
22.63%. Thus the branch performs real named-instance binding, but the extra
landmark shape is not sufficient as a standalone replacement.

It was also added to the relation/type/40 m appearance scorer. The auxiliary
weight was selected only on `val_unseen` and frozen before `test_unseen`:

| Fusion | `val_unseen` Hit@20 | exploratory `test_unseen` Hit@20 |
|---|---:|---:|
| Base scorer | 32.37% | 48.13% |
| + landmark-mask relation | 32.41% | 48.70% |

The development gain is only +0.04 point, despite a +0.57-point exploratory
test change. This is not stable enough to promote. It indicates that annotated
landmark shape has a small complementary signal for multi-landmark examples,
but does not solve selection among same-class target instances. The next model
must bind the *candidate object's own* RGB/semantic component to each named
landmark mask, rather than adding more descriptors of the landmark alone.

#### Candidate-object-instance binding upper bound

Before training another RGB network, an annotation-based upper-bound test
bound each Top-5 point to the nearest object component of the
language-predicted target class. It did not use the target object ID or target
coordinate. For every candidate component, the ranker received its center and
boundary relationship to each named landmark component, together with the
corresponding relation language.

| `val_unseen`, 3-seed mean | Hit@20 |
|---|---:|
| Accepted relation ranker | 24.04% |
| Candidate-instance binding ranker | 24.13% |
| Wrong relation, same map | 23.38% |
| Wrong named landmark, same map | 23.51% |
| Remove relation input | 22.70% |

The controls show that the model genuinely uses the candidate instance,
landmark instance, and relation together, but ideal annotated components add
only +0.09 point over the accepted relation representation. The failure is in
the hard association rule: each unseen example's five candidates collapse to
only 3.28 distinct instances on average, and a candidate is moved 20.52 m to
its assigned instance center. This destroys useful continuous uncertainty and
is especially unsuitable for large buildings, ground regions, and parking.

Therefore an RGB implementation of hard nearest-instance snapping is not
justified. Candidate components should instead form a soft set: retain several
nearby instances and weight them by target-class probability, RGB appearance,
candidate distance, and relation satisfaction. This preserves uncertainty
until the final ranking stage.

#### Soft instance-set diagnostic

The hard association was replaced with a soft set containing the two nearest
annotated components from each of the four semantic classes. Association
weights used only language class probability and candidate-to-component
distance; a learned relation score compared every retained instance with every
named landmark. Target IDs and target coordinates were excluded from all
inference features.

The model nevertheless collapsed to the original first candidate in every one
of three seeds. `val_unseen` Hit@20 was exactly 20.76%, and wrong-relation,
wrong-landmark, no-relation, and reversed-direction controls all produced the
same ranking. The behavior was unchanged throughout the five-epoch formal
diagnostic, after an earlier longer pilot showed the same plateau.

This failure refines the previous conclusion: a soft *map-annotation proximity*
set is still not candidate-specific enough. Nearby candidates share the same
dominant building or region components, so language relation scores cannot
separate them. The next useful evidence must come from the observed candidate
crop itself—for example, a candidate-centered RGB instance mask or local
semantic likelihood—before it is related to named landmark masks. Merely
changing hard nearest-instance selection into a soft distance weighting is not
sufficient.

## Decision

Neither component is sufficient alone:

1. A spatial conditional diffusion model should generate several global target
   candidates from the instruction, explored area and UAV pose. Individually
   identified landmark masks should be used for multi-landmark relational
   instructions; the simpler union path remains useful for single-landmark
   cases.
2. The frozen relation/type fusion transfers to fresh `test_unseen` and reaches
   44.15% Hit@20, +8.93 points over the same-split candidate baseline. It
   remains the accepted offline Top-5 scorer. The 48.49% appearance-enhanced
   result remains exploratory until it passes a new map-level holdout.
3. Candidate logits and projected RGB mask evidence should update a persistent
   target-belief map. Altitude is selected only to reduce this belief's
   uncertainty after the target region is localized.

Do not yet restart the stopped full HETT baseline training. The next step is to
expose the accepted fusion as a target-region belief input to HETT and verify
the integration with offline/replay tests first. Dynamic altitude remains
downstream: it is triggered only after that belief has localized a plausible
region, then acquires finer RGB evidence to reduce local uncertainty.
