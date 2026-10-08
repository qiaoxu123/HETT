# HETT Joint Goal/Trajectory: first epoch review

## Branch and implementation

- Branch: `2027-CVPR/heatmap-joint-goal-trajectory`.
- Parent remote commit before this review: `7aa31df291746a0803a33f9a9de29000d5ec046b`.
- The evaluated policy uses HETT's heatmap Top-20 as goal proposals, a newly trained HETT candidate scorer, multimodal trajectory proposals, and a learned stop head. The candidate scorer follows SBF's *idea* of scoring each candidate with language, named-landmark geometry, and observed pose history. It does not load SBF model weights, the SBF selector, or SigLIP.
- The controller executes the selected trajectory's first waypoint and replans from the next observation. Inference does not receive GT goals or future teacher states. Oracle quantities are diagnostics only.
- The experiment initializes from the existing relative-geometry Top-20 checkpoint. The trajectory, candidate-ranking, relation, and stop heads are new parameters. One training epoch was completed; no later epoch was retained for this variant.

## Paired validation result

Full validation splits were evaluated with seed 0: 2,470 `val_seen` and 2,697 `val_unseen` episodes. Test-Unseen was not used.

| Split | SR | SPL | OSR | NE | Initial Top-20 Hit@20m | Joint selected goal Hit@20m | Prior selected goal Hit@20m | Joint FDE | Prior FDE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_seen | 3.00% | 2.97% | 3.44% | 70.44m | 90.40% | 21.73% | 26.72% | 51.87m | 43.36m |
| val_unseen | 2.89% | 2.85% | 3.86% | 73.88m | 89.91% | 16.42% | 20.97% | 55.96m | 46.85m |

The reported first-epoch target was 30% `val_seen` SR. The measured 3.00% is far below it. The excellent Top-20 coverage does **not** translate into correct ranking: joint ranking lowers selected-goal hit rate relative to the heatmap-prior ranking on both splits, and selected trajectory FDE is worse than prior. The learned stop never triggered (`0` decisions); every episode timed out. This is a failed model result, not a successful candidate for five more epochs.

## Failure localization

1. Proposal recall is not the main measured failure: initial Top-20 contains a goal within 20m in about 90% of episodes.
2. Candidate/path ranking makes the proposals worse: joint selected-goal Hit@20m is 4.99pp lower than prior on seen and 4.55pp lower on unseen; selected FDE is 8.52m and 9.11m worse, respectively.
3. Closed-loop route execution and termination remain unready: final SR is below 4%, the stop policy makes no decisions, and all episodes time out.

This experiment does not show that SBF's ideas are ineffective. It shows that the current HETT-trained candidate/path/stop heads do not yet learn to use those inputs. The frozen SBF selector's earlier 40.63% seen result came from a different, bounded-waypoint pipeline and a 256-episode subset; it is not a result for this joint trajectory policy and did not transfer to unseen.

## Code review fixes before syncing

- Removed the optional frozen-SBF-selector/SigLIP runtime path so the branch's main method matches the HETT-native fusion design.
- Made the first-epoch gate use the configured policy name instead of hard-coded variant `C`.
- Made reporting work for a single configured policy and for a retry-evaluation manifest; previously it assumed A/B/C/D and could fail after the evaluation completed.
- Made checkpoint and training-argument paths environment-configurable instead of embedding this machine's absolute paths.
- Kept large checkpoints and per-episode JSON dumps local; this repository commit contains code and compact summary only.

## Verification

- Focused regression suite: 34 passed.
- Python syntax compilation: passed for agent, trajectory head, relation selector, execution helper, and experiment scripts.
- The report generator completed against the saved full-validation result after the single-policy/manifest fix.
- Existing GPU smoke artifacts show batch-size 2 and 8 checks, but no new GPU training was launched during this branch review.

## Next step

Do not continue the current checkpoint for five more epochs. Repair the ranking/stop training path first, then run one capped epoch and evaluate the same full `val_seen`/`val_unseen` splits. Continue only if the predeclared first-epoch gate is met.
