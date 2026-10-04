# Landmark crop altitude and encoder fine-tuning

This follow-up changes only the reference-landmark RGB crop altitude, then tests whether fine-tuning BERT and SigLIP improves static localization. It does **not** use human first-person images or Stage-2 navigation. The image source is the same orthorectified RGB map at every altitude. In `view_area_corners`, a crop at altitude *h* covers approximately `2h × 2h` meters: 80m ≈ 160m square, 50m ≈ 100m square, and 30m ≈ 60m square.

## Protocol

The frozen-encoder altitude ablation uses the same 32,326 rows, seed 0, train_seen-only learning, batch 256, 15 epochs, frozen BERT and SigLIP, predictor architecture and loss as the [initial static experiment](STATIC_GROUNDED_LOCALIZATION_REPORT.md). Every altitude has its own crop cache and fresh predictor initialized from the same seed. The best checkpoint minimizes Val Seen median error. Reference-center mean requires no image and remains 24.62% Val Unseen Hit@20m / 40.86% Test Unseen Hit@20m.

| Crop altitude | Val Seen Hit@20 / median | Val Unseen Hit@20 / median | Test Unseen Hit@20 / median |
| --- | ---: | ---: | ---: |
| 80m, frozen | 37.21% / 25.08m | 21.99% / 38.94m | **37.42% / 25.75m** |
| 50m, frozen | 36.92% / 25.41m | 22.40% / 37.77m | 36.45% / 25.99m |
| 30m, frozen | **37.61% / 24.86m** | **22.99% / 36.71m** | 35.98% / 25.92m |

The 30m crop is selected for fine-tuning because its Val Seen median is the lowest among the requested 30m/50m alternatives. Its Val Unseen gain over 80m is only 1.00 percentage point (27 net additional Hit@20 successes among 2,697 paired samples), while Test Unseen is 1.44 points worse. Thus a smaller crop helps one validation split, not consistently across unseen data.

## Trainable-encoder experiment at 30m

The fine-tuned model starts from the frozen 30m predictor checkpoint. The last two BERT encoder layers and last two SigLIP vision layers plus vision pooling are trainable; early layers remain frozen. SigLIP's text tower is not used. All 21,878 train_seen rows are used per epoch, batch 32, seed 0, three epochs. The location and weak relation losses are unchanged. A smoke optimizer step confirmed nonzero BERT and SigLIP gradient norms (0.58 and 0.14 at batch 16). The trainable pipeline recomputes embeddings with BF16; before any update, its Val Seen Hit@20/median is 37.65% / 25.15m, slightly different from the cached frozen-feature result due to numerical precision.

| 30m model | BERT/SigLIP LR; predictor LR | Val Seen Hit@20 / median | Val Unseen Hit@20 / median | Test Unseen Hit@20 / median |
| --- | --- | ---: | ---: | ---: |
| Frozen encoder | — | 37.61% / 24.86m | **22.99% / 36.71m** | **35.98% / 25.92m** |
| Fine-tuned | 1e-5; 1e-4 | **40.53% / 23.88m** | 19.65% / 38.18m | 34.99% / 26.48m |
| Fine-tuned, lower LR | 2e-6; 3e-5 | 39.76% / 24.28m | 20.62% / 38.18m | 34.82% / 26.40m |

Both fine-tuning runs improve Seen but reduce Hit@20 on both unseen splits relative to the frozen 30m model. Lower learning rates reduce the Val Unseen drop but do not remove it. **The current evidence does not support adopting encoder fine-tuning as a generalization improvement.** Selection used Val Seen only; these two learning-rate runs are exploratory, and repeated inspection of unseen numbers must not be mistaken for an untouched test protocol. None of these learned models beats the no-learning reference-center mean on unseen Hit@20.

The likely issue is cross-map overfitting: the model sees oracle reference names and absolute/normalized map geometry during training, while the extra trainable capacity can further memorize seen-map appearance and name patterns. This is an inference from the split behavior, not an isolated causal proof. A stronger next test would separate spatial-relation learning from landmark identity and use held-out maps for model development, keeping a fresh map set untouched for final evaluation.

## Reproduce

From the repository root, in the same dependency environment as the initial experiment:

```bash
PYTHONPATH=. python -m multiagent.scripts.train_grounded_static_localizer \
  --data-root /path/to/refined_citynav --rgb-dir /path/to/rgbd \
  --output-dir checkpoints/static_localization_30m \
  --top-altitude 30 --only-visual --epochs 15 --batch-size 256
PYTHONPATH=. python -m multiagent.scripts.finetune_grounded_static_localizer \
  --data-root /path/to/refined_citynav --rgb-dir /path/to/rgbd \
  --baseline-dir checkpoints/static_localization_30m \
  --output-dir checkpoints/static_localization_30m_finetune \
  --altitude 30 --epochs 3 --batch-size 32 \
  --encoder-lr 1e-5 --head-lr 1e-4
```

Run the altitude ablation by replacing `30` with `50` or `80` and using a separate output directory. Full per-epoch histories, split metrics, model checkpoints and per-row predictions are stored locally under the respective `checkpoints/` directories; they are not committed.
