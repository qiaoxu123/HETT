# Official 20-step closed-loop evaluation

## Scope

This gate evaluates the stage-C unified static belief model under the frozen
HETT/CityNav protocol:

- `feedback=student`
- `max_action_len=20`
- `move_iteration=10`
- fixed altitude 50 m
- success radius 20 m
- current-pose RGB is acquired online before every action

The stage-C model intentionally does not consume RGB pixels yet. It predicts one
static goal from language, independent named-landmark instances, explicit
relative geometry, and map BEV. Motion uses the same discrete lookahead routine
as HETT stage one. A deterministic stop is emitted within 5 m of the predicted
goal. This is not yet the learned evidence stop head.

Target coordinates are absent from the inference dataset and the rollout API.
They are attached only after all actions have been generated, for the unchanged
official `CityNavBatch.eval_metrics` implementation.

## Results

| Model / split | SR | OSR | SPL | NE (m) |
|---|---:|---:|---:|---:|
| Local HETT best / Val Seen | 29.43 | 49.43 | 23.29 | 38.10 |
| Static unified belief / Val Seen | **36.92** | 43.04 | **35.56** | **35.02** |
| Local HETT best / Val Unseen | 19.54 | **35.97** | 15.73 | 50.52 |
| Static unified belief / Val Unseen | **21.10** | 29.51 | **20.34** | **48.59** |

The raw predicted-goal scores were:

| Split | Static Hit@20 | Static mean distance |
|---|---:|---:|
| Val Seen | 38.26 | 34.98 m |
| Val Unseen | 21.02 | 48.77 m |

The action conversion therefore loses 1.34 SR points on Val Seen and gains 0.08
on Val Unseen. The static localization ability is transferring to closed-loop
navigation; the controller is not the main loss at this gate.

Rollout diagnostics:

| Split | Mean actions | Mean online observations | Max actions | >20 actions |
|---|---:|---:|---:|---:|
| Val Seen | 5.59 | 5.59 | 11 | 0 |
| Val Unseen | 5.86 | 5.86 | 12 | 0 |

## Interpretation

Val Seen improves clearly, but it was used for checkpoint selection and is not
the decisive generalization result. Val Unseen SR is only +1.56 points over the
local HETT best, within the documented roughly ±1–2 point student-evaluation
variation, so it is not yet evidence of a significant SR improvement.

SPL is +4.61 and NE is 1.93 m lower on Val Unseen, showing that direct movement
to the belief maximum is efficient. OSR is 6.46 points lower, however. When the
static ranker picks the wrong region, the straight path has no mechanism to
recover or even pass near the target. The next module should therefore improve
candidate correction, not add unrestricted exploration.

## Next controlled gate

Keep altitude fixed at 50 m and generate only one-action, path-local candidates:

- toward Top-1;
- toward Top-2;
- toward their midpoint;
- small left/right offsets from the progress path.

Reject any candidate that cannot preserve an optimistic route to Top-1 within
the remaining 20-step budget. First measure a constrained teacher upper bound;
only then train the joint progress/information action head. No 24-view orbit or
30 m candidate-circle action is allowed in this gate.

## Path-local two-mode upper bound

The first constrained upper-bound test is now complete on full Val Seen. The
first action is target-independent and goes toward the midpoint of the two
spatially distinct belief modes (falling back to Top-1 if needed). Only after
that online observation does a privileged teacher choose the mode closest to
the reference target. Thus this is an upper bound, not a deployable result.

| Val Seen method | SR | OSR | SPL | NE | Mean actions | Mean path |
|---|---:|---:|---:|---:|---:|---:|
| Static Top-1 | 36.92 | 43.04 | 35.56 | 35.02 m | 5.59 | 196.47 m |
| Perfect two-mode disambiguation | **53.56** | **59.07** | **51.44** | **26.90 m** | 5.71 | 197.06 m |

The raw Top-1 Hit@20 is 38.26%; the two-mode oracle coverage is 54.74%.
Perfect selection therefore adds 16.64 closed-loop SR points while adding only
0.12 actions and 0.59 m of path length. This validates the budgeted action
geometry but does not yet show that RGB can infer the teacher choice.

The next gate is narrow: at only these path-local 50 m observations, test
whether point-cloud supervision and then RGB prediction can rank Top-1 versus
Top-2. The full action transformer should be trained only if that non-privileged
ranking is reliably better than static Top-1.
