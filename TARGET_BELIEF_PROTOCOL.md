# HETT multi-landmark target belief

## Goal

Improve HETT target-position prediction without changing its action space,
altitude, stop rule, or 20-step evaluation protocol.  The isolated variable is
the target representation and its supervision.

## Baseline commit and branch

- Baseline: `51a1828f7024849606ea0d6bbdf65f63e93babb5`
- Branch: `codex/hett-multilandmark-belief`
- Worktree: `hett-experiments/48-hett-multilandmark-belief`

The released path is unchanged unless `--target_belief_head` is supplied.

## Model change

1. Preserve up to nine instruction landmarks as aligned name embeddings and
   normalized map centres.  Padding is explicitly masked and aggregation is
   permutation invariant.
2. Score a `41 x 41` normalized map grid.  At the default 410 m map extent,
   one cell is approximately 10 m wide.
3. Predict a bounded within-cell offset for metric refinement.
4. Accumulate the previous target logits through a learned gate, so later RGB
   observations update a belief instead of producing unrelated coordinates.
5. Navigate to the highest-probability cell plus its offset.  No target truth
   is an inference input.

## Training losses

The new target objective contains:

- exact target-cell cross entropy;
- negative log probability mass inside the 20 m success region;
- SmoothL1 loss for the ground-truth cell offset;
- a low-level SmoothL1 check on the selected target coordinate.

With `--normalize_rollout_loss`, every Teacher or Student rollout is divided by
its active sample-step count.  This removes the released implementation's
implicit preference for longer Teacher rollouts.

## Controls

Run all comparisons with fixed altitude 50 m, `max_action_len=20`,
`move_iteration=10`, the same split and the same seed.

The RTX PRO 6000 formal run uses an actual per-step batch size of 16 with
`grad_accum=1` (effective batch size 16).  It must not be compared as a
same-optimizer ablation against the earlier batch-2/accumulation-4 baseline
without explicitly reporting this optimisation difference.

1. Released HETT: mean landmark representation + coordinate MSE.
2. New input only: independent landmarks + original coordinate head.
3. Belief head: independent landmarks + dense belief and offset.
4. Temporal belief: configuration 3 with previous-belief accumulation.

Report per-step median/mean target error in metres, Hit@20, final SR, OSR, SPL,
NE, active steps and stop step.  Develop on `val_seen`; after freezing the
configuration, run `val_unseen`.  Do not use `test_unseen` during development.

## Anti-leakage requirements

- Target truth is used only to build losses and final metrics.
- Candidate extraction and `MultiLandmarkBeliefHead.forward` cannot receive a
  target coordinate.
- Landmark-order permutation must not change the output.
- Invalid padded landmarks must not change the output.
- A Student rollout must use `pred_goals`, never `gt_goal`, for motion.

## Completed checks

- Six CPU unit tests cover shape/range, gradient flow, landmark permutation,
  padding masks, temporal accumulation, 20 m metric labels and target-input
  leakage.
- A real-data GPU smoke used one `train_seen` episode, two rollout steps and
  one optimizer update.  Teacher and Student losses both backpropagated;
  reported peak allocated memory was 2.79 GiB on an RTX PRO 6000 Blackwell.

These checks establish implementation validity only; they are not accuracy
evidence.  No full training result has been claimed.
