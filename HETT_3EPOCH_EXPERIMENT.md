# HETT three-epoch active-altitude experiment

Date: 2026-09-24

> Priority update: the fixed-50 m run was stopped by user request before its
> first epoch completed. Its partial logs are retained but are not reportable
> results. The teacher-forced fine-altitude run was promoted to the full
> 20-epoch schedule and must not be early-stopped unless the user requests it.

## Frozen starting evidence

| Offline target grounding | `test_unseen` Hit@20 |
|---|---:|
| Original Top-1 candidate | 35.22% |
| Accepted relation + target-type fusion | 44.15% |
| Exploratory relation + type + RGB appearance | 48.49% |
| Annotation-only semantic-component upper bound | 52.41% |

The accepted improvement is +8.93 absolute Hit@20 points. The 48.49% result
is exploratory, and the 52.41% result is not deployable. At 40 m, the RGB
semantic model reaches 52.77% foreground mIoU on unseen maps versus 46.52% at
80 m, but this visual improvement has not yet produced a stable navigation
improvement.

## Worktree isolation

- Training and evaluation source: `45-active-altitude` only.
- Run output: `runs/hett_active_altitude_3ep_b16_s0_20260924`.
- The original baseline worktree `43-paper-original-20ep` remains stopped and
  its checkpoints and logs are never overwritten.
- The supervisor snapshots the exact dirty source, Git diff, dependency list,
  commands, input hashes, telemetry, checkpoints, and metrics into the run.

## Protocol

1. Train HETT for three full epochs from its normal pretrained BERT/Darknet
   initialization: seed 0, fixed 50 m, batch 16, no gradient accumulation,
   AdamW 1e-4, task interaction disabled.
2. Select the checkpoint only by `val_unseen` SR as in the existing trainer.
3. On development splits, compare the same checkpoint under fixed 50 m,
   fine-stage 50→45 m, and fine-stage 50→40 m policies.
4. Keep `test_unseen` sealed for the dynamic policy.

Dynamic height is intentionally evaluation-only in the current implementation.
Therefore this run tests whether the existing HETT representation can benefit
from lower fine-stage observations; it does not claim to train a
low-altitude-aware visual encoder. If the paired result fails, likely causes to
inspect are low-altitude distribution shift, weak target logits, a late or
incorrect refinement trigger, and the absence of the accepted offline
grounding score from HETT's online target belief.

## Dynamic-height training follow-up

The follow-up run uses a separate source snapshot and output directory. During
optimization only, ground-truth distance supplies the height label: RGB stays
at 50 m throughout coarse navigation and changes to 40 m only after HETT has
entered stage 2 and the true target is within 40 m. The three-channel global
map remains canonical at 50 m, and low-altitude RGB features are excluded from
coarse global memory. Validation never reads ground-truth distance; it uses the
deployable predicted controller. The final checkpoint is evaluated both at
fixed 50 m and with 50→40 m refinement for a paired comparison.

## 2026-09-25: training/validation altitude-distribution mismatch

The epoch-2 aggregate diagnostics expose a material mismatch:

| Path | Low-altitude frames | Altitude changes per trajectory |
|---|---:|---:|
| Training, oracle height rule | 2.691% | 0.0485 |
| `val_unseen`, predicted controller | 42.699% | 0.9867 |

These numbers do not mean that the validation set naturally contains many
more fine-navigation observations. They come from different descent rules.
During optimization, descent requires both stage 2 and a true target distance
of at most 40 m. A trajectory following an incorrect predicted coarse goal
therefore remains at 50 m. During validation, true target distance is hidden;
the controller instead checks distance to the model's own predicted goal.
After stage 1 the agent normally approaches that prediction even when the
prediction is wrong, so almost every validation trajectory descends once and
then remains at 40 m for much of the remaining decision horizon.

The approximately 15.9-fold frame-frequency gap means that the visual model
is trained on almost no 40 m imagery but is evaluated on it for about 43% of
its observations. This is a likely contributor to the weak early navigation
result and must be treated as a train/evaluation distribution shift, not as
evidence that dynamic altitude itself is ineffective.

The read-only audit script `scripts/audit_altitude_distribution_gap.py`
recomputes the comparison directly from `epoch_metrics.jsonl`. It intentionally
does not launch evaluation or modify a checkpoint. The script is recorded but
has not been run, per the current test instruction.

```bash
python scripts/audit_altitude_distribution_gap.py \
  --metrics runs/hett_teacher_altitude_20ep_b16_s0_20260924/checkpoints/epoch_metrics.jsonl \
  --epoch 2 --split val_unseen
```
