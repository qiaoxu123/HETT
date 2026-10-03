# FPV goal-approach smoke report

Date: 2026-10-03  
Seed: 0  
Backbone: frozen `google/siglip-base-patch16-224`  
Training: 2 epochs, batch 64, class-weighted cross entropy

## Renderer finding

The local CityNav release has raw human Pose5D traces, CityRefer annotations,
orthophoto RGB, and DSM rasters. It has no original CityFlight/Potree renderer,
cached human FPV frames, or CityNav-compatible Unreal package. The unrelated
26 AirVLN Unreal environments on this machine cannot be substituted for the
Birmingham/Cambridge CityNav blocks.

The smoke run therefore uses `orthophoto_heightfield`, an explicitly labelled
2.5-D perspective proxy. It preserves raw yaw and pitch but is **not** an
original human FPV image and is **not** AirSim. Visual inspection found map-edge
fill and missing-facade artifacts, so these metrics only validate the pipeline
and label hypothesis. They are not evidence about real FPV performance.

## Dataset

| Split | Episodes | Samples | Match + / - | Arrival + / - |
|---|---:|---:|---:|---:|
| train_seen | 100 | 2,073 | 691 / 1,382 | 274 / 1,799 |
| val_seen | 50 | 1,050 | 350 / 700 | 154 / 896 |
| val_unseen | 50 | 876 | 292 / 584 | 112 / 764 |

There are 9,133 deduplicated rendered frames. Episode-ID intersections between
train_seen, val_seen, and val_unseen are empty.

Train distance-bin counts (`<10`, `10–20`, `20–30`, `30–40`, `40–60`, `>60`)
are `342, 480, 246, 297, 288, 420`. Train negative-type counts are 277
near-goal, 691 wrong-view, 691 wrong-language, and 414 `none` base samples.

## Two-epoch training loss

| Variant | Epoch 1 | Epoch 2 | Peak MiB |
|---|---:|---:|---:|
| current FPV only | 6.173 | 6.030 | 1,134 |
| current FPV + language | 6.158 | 5.955 | 1,136 |
| history K=4 + language | 6.146 | 5.861 | 2,132 |
| history K=4 + language + pose | 7.839 | 5.811 | 2,134 |

No run produced NaN or OOM.

## Ablation results

| Variant | Seen Match F1/AUROC | Unseen Match F1/AUROC | Seen Arrival F1/AUROC | Unseen Arrival F1/AUROC | Unseen Stop FP | Bearing acc | Phase acc |
|---|---|---|---|---|---:|---:|---:|
| current FPV only | .361/.508 | .368/.497 | .343/.681 | .288/.662 | .576 | .219 | .475 |
| current FPV + lang | .394/.580 | .473/.619 | .300/.681 | .302/.686 | .500 | .191 | .449 |
| history + lang | .272/.578 | .387/.610 | .201/.634 | .359/.718 | .212 | .264 | .449 |
| history + lang + pose | .474/.511 | .500/.508 | .380/.723 | .245/.683 | .800 | .210 | .370 |

The history-language model has the best unseen Arrival AUROC and F1. Relative
to single-frame language it gains +0.057 Arrival F1, +0.032 Arrival AUROC, and
+0.073 bearing accuracy, but its Match F1 is lower. Pose severely increases the
unseen false-positive rate and is not supported by this smoke run.

## Shortcut and hard-negative checks

For the history-language model on val_unseen:

- wrong-language shuffle: Match F1 falls from 0.387 to 0.068; Arrival F1 falls
  from 0.359 to 0.295. Language affects the model, especially match.
- frame-order reversal: Arrival F1 changes only 0.359 to 0.354 and bearing
  accuracy is unchanged. The model is not using temporal order meaningfully.
- near-goal: Match accuracy 0.322; arrival false-positive rate 0.209.
- wrong-view: Match accuracy 0.671; arrival false-positive rate 0.226.
- wrong-language: Match accuracy 0.945; arrival false-positive rate 0.233.
- 20–30 m bucket: arrival false-positive rate 0.255.

The geometry oracle (`distance <= 20 m`) has Arrival F1 0.50 and false-positive
rate 0.293 on val_unseen because geometric arrival intentionally does not imply
visual confirmation for wrong-view and wrong-language samples.

## Decision

Do **not** integrate into HETT yet. The gate is not met: unseen Match AUROC is
0.610 (<0.75) and unseen Arrival F1 is 0.359 (<0.70). Full proxy rendering and
10-epoch training were deliberately not run because the smoke set already uses
663 MB for only 200 episodes and the renderer is not real FPV.

Next evidence should come from either original CityFlight/Potree frames or a
CityNav-compatible AirSim package. With real FPV, repeat the same frozen-SigLIP
ablation. If the gate is reached, the most defensible first integration target
is near-goal/anchor verification; Stop should remain gated until the 20–30 m
false-positive rate is substantially lower.

## Direct answers

A. The 2.5-D proxy contains some distance/arrival signal (unseen Arrival AUROC
0.718), but this cannot establish that real first-person imagery does.  
B. Language participates: shuffling it sharply damages Match, but Match AUROC is
still too low.  
C. Four-frame history improves Arrival and bearing over one frame, but the
order-shuffle result shows no reliable temporal-order use.  
D. There is no simple across-the-board seen-to-unseen collapse, but metrics are
low and unstable; the pose variant collapses badly on unseen Stop FP.  
E. If confirmed with real FPV, the module is currently best suited to near-goal
refinement / anchor verification, not autonomous Stop.
