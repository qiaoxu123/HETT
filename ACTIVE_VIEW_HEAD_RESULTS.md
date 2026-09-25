# Relation-conditioned active-view head

## Protocol

The diagnostic subset contains episodes whose Top-5 includes a point within
20 m while rank 1 is wrong. Target coordinates are used only to define this
diagnostic subset and compute final metrics.

Point-cloud teacher rendering enumerates 24 cluster views: eight directions at
20, 35 and 50 m. The view head is trained on 128 `train_seen` episodes and
selected on 64 `val_seen` episodes using teacher utility only, not navigation
success. Runtime view-head inputs contain candidate geometry, independent
named-landmark geometry, per-landmark relation cues, current pose and height.
Point-cloud semantics, visibility and target coordinates are not inputs.

## Result

| View selection | val_seen diagnostic Hit@20 | mean NE |
|---|---:|---:|
| Random view | 35.94% | 26.85 m |
| Fixed 35 m / direction 0 | 39.06% | 25.54 m |
| Learned view head | **40.63%** | 26.47 m |
| Point-cloud teacher heuristic | 42.19% | 27.78 m |
| Privileged best rendered view | 79.69% | 15.92 m |

The head matches the exact teacher-best view in 20.31% of episodes. Removing
landmark geometry decreases Hit@20 to 37.50%, supporting use of per-landmark
geometry. An earlier model that aggregated relation cues reached only 37.50%.
Binding a separate cue to each landmark is therefore necessary.

The opposite-relation control remains 40.63%, so fine directional use is not
yet proven. The sample is also small: one episode is 1.56 percentage points.
This gate supports runtime integration, but not a final navigation claim.

## Scope

Observation updates in this gate still use point-cloud teacher signatures.
It isolates view selection and does not prove RGB observation transfer,
closed-loop SR/SPL, stopping, or dynamic-height navigation. Point clouds remain
training/evaluation teachers and are not allowed as runtime model inputs.

## Closed-loop logic gate

The independent runtime now composes learned view selection, altitude changes,
signed evidence, goal reselection, geometric actions and conservative stopping.
Its step interface has no target-coordinate argument. On the same 64-episode
diagnostic subset, using teacher observations only after the learned view is
reached:

| Metric | Result |
|---|---:|
| SR | 39.06% |
| OSR | 45.31% |
| SPL | 37.60% |
| mean NE | 27.15 m |
| correct-stop rate | 95.31% |

This agrees closely with the 40.63% offline candidate reranking result and
verifies that the action/stop loop preserves most of the selected-view gain.
It remains a point-cloud-teacher logic gate; replacing the teacher observation
with sufficiently discriminative RGB evidence is still required.
