# RGB temporal belief gate

## Purpose

This gate tests whether frozen RGB semantics can update the stage-1 unified
belief model without target-coordinate leakage. The static transformer is
frozen. Only a visibility-gated signed temporal evidence head is trained.

## Rejected human-trajectory experiment

The first 256/128 train/validation pilot selected three frames by trajectory
order. It appeared to increase validation Hit@20 from 35.94% to 65.62%.
Strict controls showed that this result was invalid:

| Input | val_seen Hit@20 |
|---|---:|
| RGB plus trajectory visibility | 65.62% |
| Shuffled RGB, original visibility | 60.94% |
| Visibility only, no RGB | 59.38% |

The final human trajectory pose reveals that the observer is near the target.
This experiment is retained only as a leakage diagnostic and is not an
accepted result.

## Target-free candidate-view protocol

The replacement protocol centers fixed-50-m observations on the first three
predicted relation candidates. View generation uses only predicted candidates
and named-landmark centers. It does not use human poses, target coordinates,
target distance, target identity, or target markers.

The frozen orthomosaic semantic network produces five-class RGB evidence. A
language-conditioned type head predicts Building/Car/Ground/Parking, while a
signed evidence head updates only the 20-m neighborhood around the candidate
being verified. Invalid and out-of-support cells receive exactly zero evidence.

The scaled gate uses 2,048 `train_seen` episodes and 512 `val_seen` episodes.
All model and evidence settings were fixed before this scaled run.

## Scaled result

| Method | val_seen Hit@20 | mean NE |
|---|---:|---:|
| Frozen static belief | **39.65%** | 34.89 m |
| Best temporal prefix | 39.45% | 34.79 m |
| Visibility only, no RGB | 37.30% | 35.15 m |
| Wrong target type | 37.70% | 34.94 m |
| Visibility mask removed, three frames | 30.66% | 47.94 m |

The target-type head reaches 91.99% accuracy, but target localization does not
improve. Therefore target category matching is not the missing instance-level
signal: many candidate regions contain the correct broad category. Removing
visibility gating confirms that unrestricted negative evidence is harmful.

## Decision

Do not promote the current RGB type-only temporal head. The next RGB increment
must verify relation-specific evidence: co-visibility and projected geometry of
the candidate, each named landmark, and roads. Active view training remains
blocked on that observation signature; adding more type-only epochs or views is
not justified by this gate.

## High-resolution relation-signature follow-up

The initial 20x20 RGB BEV left only about one cell inside a 20-m candidate
support. Raising only the visual projection to 64x64 improves target-type
candidate ranking from 27.54% to 29.69% Hit@20 on the same 512-episode
`val_seen` subset. The static belief grid remains 20x20.

An instance-preserving follow-up then retains up to nine separate landmark
tokens, each containing relation language, candidate-to-landmark geometry,
visual landmark confirmation and an optional road-corridor score:

| Candidate ranker | val_seen Hit@20 |
|---|---:|
| Candidate prior | 22.66% |
| Relation geometry, no RGB landmark evidence | 29.10% |
| + per-landmark visual confirmation | 29.69% |
| + per-landmark road confirmation | 27.73% |
| Top-3 oracle | 49.02% |

The per-landmark visual increment is only +0.59 point; road evidence hurts and
an opposite-relation control does not decrease accuracy. These signals are not
strong enough for integration into the unified belief model. The next justified
stage is relation-conditioned active-view supervision, with point-cloud data as
a training-only visibility teacher rather than a runtime input.
