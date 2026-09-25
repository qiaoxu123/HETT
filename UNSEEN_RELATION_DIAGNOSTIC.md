# Unseen relation diagnostic

## Question

Does the current candidate verifier causally match unseen instructions to
named-landmark geometry, and is the main failure caused by positional encoding
or attention?

## Findings

1. The historical frame audit applies a matrix `M` to language direction
   vectors, while the geometry helper applied the same `M` to map deltas.  The
   equivalent geometry transform is `M^T`, not `M`.  On 204 broad-cue,
   relation-discriminative, unique-Hit@20 unseen pairs, the implemented
   convention agrees with the successful candidate in 41.18%; the transpose
   convention reaches 58.82%.
2. Correcting this convention alone does not improve the learned full relation
   ranker.  Three-seed mean unseen Hit@20 changes from 24.04% to 23.64%.  The
   implementation mismatch is real, but it is not the main learned-ranking
   bottleneck.
3. The broader cue parser is noisy.  Restricting evaluation to explicit forms
   such as `to the left of [landmark]` raises candidate/label consistency to
   69.51%, but covers only 10.83% of unseen instructions.
4. A strict, audit-consistent direction score was weighted only on val_seen.
   On all val_unseen rows, visual Hit@20 changes from 24.18% to 24.25%.  On the
   100 strict-relation rows with exactly one Hit@20 candidate, pair accuracy
   changes from 70% to 72%; flipping left/right and above/below reduces it to
   63%.  Thus the high-confidence relation signal is causal but sparse.

## Root cause

The relation branch is trained mainly through a 20 m navigation label and
same-map mismatch negatives.  A 20 m-positive point need not lie on the same
side of a landmark as the annotated target, and broad keyword cues introduce
additional ambiguity.  Consequently, the model can minimize navigation loss
without learning a stable mapping from relation words to geometry axes.

The primary issue is therefore **relation-supervision mismatch**, not missing
generic positional embeddings or insufficient attention depth.

## Required fix

Train a factorized relation head with direct per-landmark supervision:

- parse typed tuples `(landmark, relation, confidence)`;
- supervise directional compatibility using the training target-to-landmark
  geometry, ignoring ambiguous axes and low-margin examples;
- keep navigation Hit@20 loss separate from relation compatibility loss;
- apply relation evidence only when parser confidence and geometric margin are
  high;
- preserve the visual verifier as an independent residual evidence branch;
- require correct-vs-flipped and correct-vs-wrong-landmark gaps on frozen
  unseen diagnostic subsets before online rollout.

The next experiment should expand strict-relation coverage using supervised
paraphrase normalization while holding the current high-precision subset fixed.

## Factorized parser result

A five-way per-landmark parser (`left/right/above/below/none`) was trained from
frozen relation-language embeddings.  Direction labels were accepted only when
an explicit training phrase agreed with target-to-landmark geometry by at least
10 m.  The initial imbalanced model collapsed to `none`; inverse-frequency
class weighting and macro-accuracy checkpoint selection corrected this.

The confidence threshold (0.95) and relation fusion weight (0.1) were selected
only on val_seen.  Frozen val_unseen results are:

- visual-only Hit@20: 24.18%;
- factorized relation fusion: 24.29%;
- flipped-direction control: 23.88%;
- active episode coverage: 11.05%;
- on 102 active episodes with one Hit@20 candidate: visual 69.61%, correct
  relation 72.55%, flipped relation 61.76%.

The 10.78-point correct-vs-flipped gap establishes a causal unseen relation
signal.  The small overall gain is explained by sparse high-confidence
coverage, not by absence of any learned relation capability.  This module is
not yet promoted to online navigation; the next gate is at least 75% correct
pair accuracy while increasing coverage without reducing the flip gap.
# DeBERTa landmark-relation parser (frozen unseen gate)

The earlier factorized parser had low unseen coverage because it classified a
frozen sentence embedding.  A DeBERTa-v3-base classifier was therefore trained
on geometry-verified `train_seen` relation mentions.  Checkpoint selection used
only strict-template `val_seen`; broad-expression `val_seen` and all
`val_unseen` checks were run after freezing.

- Strict-template `val_seen`: 96.74 macro F1, 100.00 direction accuracy.
- Held-out broad-expression `val_seen`: 99.70% accuracy over 660 mentions.
- Same-instruction multi-landmark binding: 97.03%; removing the focus landmark
  reduced accuracy to 33.66%.
- Frozen `val_unseen`: 99.47% over 1,890 relation mentions.
- Frozen unseen multi-landmark binding: 99.06%; blank-focus accuracy 33.96%.

The language result does transfer to candidate ranking, but only weakly:

- `val_unseen` candidate-prior Hit@20: 20.76%.
- Prior + correct relation Hit@20: 21.13% (+0.37 points).
- Prior + flipped relation Hit@20: 20.32%.
- On episodes with one correct Top-5 candidate, relation-only pair accuracy was
  25.40%, versus 19.94% after flipping left/right and above/below.

This isolates the next bottleneck: relation parsing and landmark conditioning
are now reliable on the lexical test set, but the deterministic geometry score
is too weak to turn that understanding into robust candidate selection.  No
unseen result was used to tune the model, confidence threshold, or fusion
weight.
