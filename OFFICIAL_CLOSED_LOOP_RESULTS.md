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

## Path-local point-cloud and RGB transfer

That gate is now complete. The observation pose is the first budget-feasible
point on the midpoint progress path where both modes enter the official 50 m
top-down crop. It uses only the start pose and predicted modes. On Val Seen,
joint visibility is 99.72% and occurs after 3.86 actions on average.

Candidate ranking results:

| Input / split | Hit@20 | Mean error |
|---|---:|---:|
| Static / Val Seen | 38.26 | 34.98 m |
| Point-cloud teacher / Val Seen | 40.45 | 34.15 m |
| Online RGB / Val Seen | 39.27 | 34.52 m |
| Static / Val Unseen | 21.02 | 48.77 m |
| Frozen online RGB / Val Unseen | **23.40** | **48.33 m** |

The point-cloud teacher uses the language-predicted target type. Replacing it
with the true language type changes Val Seen Hit@20 only from 40.45 to 40.61,
so language type classification is not the main bottleneck. Wrong type falls to
34.41 and shuffled evidence to 36.80. For RGB, wrong type falls to 35.75 and
shuffled evidence to 37.81 on Val Seen. On frozen Val Unseen they fall to 19.73
and 20.65 versus 23.40 for correct RGB evidence.

## Strict online official result

The semantic model was finally executed inside the rollout on the current
`obs["rgb"]`, rather than replaying a cached choice. Its rank agrees with the
cached deterministic observation in 99.93% of Val Unseen episodes.

| Val Unseen model | SR | OSR | SPL | NE | Mean actions | Max actions |
|---|---:|---:|---:|---:|---:|---:|
| Local HETT best | 19.54 | 35.97 | 15.73 | 50.52 m | — | 20 |
| Static unified belief | 21.10 | 29.51 | 20.34 | 48.59 m | 5.86 | 12 |
| **Strict online path-local RGB** | **23.25** | 33.52 | **21.81** | **47.98 m** | 6.16 | 12 |

Relative to the static model, online RGB adds 2.15 SR and 4.00 OSR points.
Relative to the strongest recorded local HETT baseline, it adds 3.71 SR and
6.08 SPL points and reduces NE by 2.54 m. No episode exceeds 20 actions. The
runtime uses no point cloud and no target coordinate; point cloud served only
as a development teacher. `test_unseen` remains unopened.

This result still uses a deterministic stop within 5 m of the predicted goal.
It validates the belief/action observation loop. The two follow-up gates below
test whether a hand-designed joint action utility or learned stopping can yet
replace that rule.

## Budget-aware joint action gate

A target-free action utility was implemented over legal one-action waypoints:
Top-1, Top-2, their midpoint, and 10 m path-left/path-right offsets. It combines
expected goal progress, joint-candidate visibility and remaining-budget
shortfall. At fixed 50 m, evidence acquisition is followed by an explicit
posterior-mode commitment so that the policy cannot remain at the midpoint.

On the same first 256 Val Seen episodes:

| Method | SR | OSR | SPL | NE | Mean actions |
|---|---:|---:|---:|---:|---:|
| Static unified belief | 21.88 | 25.78 | 21.13 | 46.34 m | 6.14 |
| Cached midpoint RGB rule | **25.39** | 32.42 | **23.82** | **45.10 m** | 6.49 |
| Joint progress/information utility | 23.83 | **33.20** | 22.69 | 46.19 m | 6.88 |

The joint utility improves over static navigation but is 1.56 SR and 1.13 SPL
points below the simpler midpoint rule. In the fixed-altitude orthographic
crop, the midpoint already makes both hypotheses visible early; unsupported
lateral offsets add path cost without predicting a new appearance. This rule
is therefore rejected and is not distilled into a learned action head.

## Learned evidence-stop gate

An MLP stop head was trained on 2,048 `train_seen` episodes and selected on all
2,470 `val_seen` episodes. Runtime features contain only predicted-goal
distance, two-mode posterior/evidence, current RGB target-type statistics,
local unified-belief support and remaining budget. Ground-truth distance is
used only to create the training label.

| Val Seen stop rule | SR | Mean stop step |
|---|---:|---:|
| Deterministic predicted-goal stop | 39.109 | 5.910 |
| RGB-only learned stop | 39.231 | 5.902 |
| RGB + local-belief learned stop | 39.231 | 5.854 |
| Trajectory privileged oracle | 49.636 | — |

Both learned variants add only 0.12 SR point on the full validation set. The
local-belief version selects epoch 0 and threshold 0.83, so the apparent gain
is not considered stable enough for deployment. Zeroing either center semantics
or local belief returns SR to 39.109; shuffling them reduces SR to 38.826 and
38.745, respectively, but the head almost never has confidence to stop early.

The failure is concentrated in the two dominant target types. Center semantic
state AUC is 0.577 for Building and only 0.525 for Car; together they account
for 2,283/2,470 episodes. Of 260 trajectories that pass within 20 m and fail at
the final stop, 209 do so after candidate RGB evidence has already been
acquired. Observation timing is therefore not the main bottleneck. The next
gate should improve instance- and relation-specific visual evidence before
training a larger action head or enabling dynamic height.
