# One-shot landmark-conditioned static localization

This experiment compiles each CityNav instruction once into a `GroundedInstructionMemory` (instruction, processed target phrase and attributes, weak relation tags, and oracle-matched reference landmark names). A `LandmarkRelativeGoalPredictor` then predicts a target XY offset relative to the referenced landmarks. There is no navigation rollout, Stage-2 controller, teacher trajectory, or student observation. The target XY is kept outside the compiled memory and used only as a training label or evaluation target.

## Protocol

- Source: all 32,326 refined CityNav rows: train_seen 21,878; val_seen 2,470; val_unseen 2,697; test_unseen 5,281. All rows are evaluated, including those with no resolved reference.
- Learning: train_seen labels only. Best checkpoint selected by lowest val_seen median error over 15 epochs, seed 0, batch 256. Unseen splits are evaluated after checkpoint selection.
- Inputs: frozen BERT embeddings of unique text strings; landmark name, geometry, type and map position; optional frozen SigLIP embedding of one overhead RGB crop centered on each referenced landmark. Visual features are **not** first-person observations. No target ID, target coordinate, target-centered crop, or map alias enters the predictor. Oracle `processed.landmarks` and `processed.target` remain privileged annotation inputs; this is an oracle-parsing upper-bound experiment, not an end-to-end retrieval result.
- Reference matching: 50,732 mentions; 50,701 resolved (50,230 exact, 471 fuzzy), 31 unresolved. After matching, 66 rows have no references.
- Metrics: Euclidean XY target error, Hit@10/20/30m, and median error. Reproduction commands are below.

| Method | Val Seen Hit@20 / median | Val Unseen Hit@20 / median | Test Unseen Hit@20 / median |
| --- | ---: | ---: | ---: |
| Reference-center mean (no learning) | 26.07% / 33.66m | **24.62% / 38.85m** | **40.86% / 23.90m** |
| Text + geometry | 36.76% / 25.50m | 20.13% / 39.94m | 32.66% / 27.35m |
| Text + geometry + landmark RGB | **37.21% / 25.08m** | 21.99% / 38.94m | 37.42% / 25.75m |

The visual variant improves the learned model on Val Unseen by 1.86 percentage points Hit@20 and Test Unseen by 4.75 points, but **neither learned variant beats the reference-center mean on the unseen splits**. This experiment therefore supports the claim that oracle references contain useful localization signal, while it does not establish that the present relation/offset learner generalizes across maps. The learned model may be exploiting seen-map position or name patterns; relation grounding and cross-map robustness need an isolated follow-up. Do not interpret these static Hit rates as navigation SR.

The visual variant's Val Unseen Hit@10/20/30 is 6.08% / 21.99% / 37.37%; median error is 38.94m. For the 1,428 Val Unseen rows with one resolved landmark, Hit@20 is 18.14%; for 1,075 rows with two, 25.58%; for 191 rows with three or more, 30.89%. The three zero-reference Val Unseen rows are all misses.

## Reproduce

From the repository root, in an environment with PyTorch, Transformers, rasterio, and the project dependencies:

```bash
PYTHONPATH=. python -m unittest multiagent.test_grounded_static_localization multiagent.test_landmark_multimodal_map -v
PYTHONPATH=. python -m multiagent.scripts.eval_grounded_static_baselines \
  --data-root /path/to/refined_citynav \
  --output checkpoints/static_localization/geometry_baselines.json
PYTHONPATH=. python -m multiagent.scripts.train_grounded_static_localizer \
  --data-root /path/to/refined_citynav --rgb-dir /path/to/rgbd \
  --output-dir checkpoints/static_localization --epochs 15 --batch-size 256 \
  --top-altitude 80 --device cuda:0
```

The training script saves per-epoch histories and split metrics in `static_localization_report.json`, and one prediction per source row and model variant (64,652 lines total) in `static_predictions.jsonl`. The feature cache and model checkpoints remain local under `checkpoints/`.
