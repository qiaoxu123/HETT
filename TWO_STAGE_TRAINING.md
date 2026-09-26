# Two-stage target localisation and action training

This branch keeps the released HETT input modalities and separates target
localisation from control.

## Stage 1: target localisation

```bash
cd multiagent
./train_target_stage.sh --epochs 20 --output_dir checkpoints/two_stage_target
```

The rollout follows human trajectory prefixes. Actions are used only to visit
prefix states; the direction and progress heads are frozen. The objective is a
metre-scaled coordinate Huber loss plus coarse spatial, range, bearing and
temporal-consistency auxiliaries. Validation uses teacher prefixes and selects
`best_val_unseen` by the lowest mean target error, while also reporting
Hit@5/10/20 m.

## Stage 2: target-conditioned action fine-tuning

```bash
cd multiagent
./train_action_stage.sh \
  checkpoints/two_stage_target/best_val_unseen \
  --epochs 20 \
  --output_dir checkpoints/two_stage_action
```

The stage-1 language, vision, multimodal and coordinate-prediction parameters
are frozen. Only `target_conditioning`, `decoder_2_action_full` and
`decoder_2_progress_full` are trainable. `--reset_epoch_on_load` starts the new
stage at epoch 1 without resuming the incompatible stage-1 optimizer.

The coordinate curriculum linearly changes from
`--action_gt_coordinate_start` (default 1.0) to
`--action_gt_coordinate_end` (default 0.1). Both the action condition and the
student rollout use the same selected coordinate. Validation always uses the
frozen predictor rather than ground truth.

## Backward compatibility

`--training_stage joint` is the default and preserves the released joint loss
and control path. Existing checkpoints gain only the new target-conditioning
parameters when loaded for training.

## Required invariants

- Stage 1 receives only the current observation prefix, never future frames.
- Stage 1 produces no direct gradient for direction or progress heads.
- Stage 2 detaches target coordinates and contains no target-prediction loss.
- Stage 2 does not place frozen language/vision parameters in an optimizer.
- Target-stage model selection uses localisation error, not navigation SR.
