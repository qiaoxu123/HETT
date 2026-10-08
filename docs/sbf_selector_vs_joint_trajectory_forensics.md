# 为什么冻结 SBF selector 的 seen SR 40.6%，多轨迹组合却只有约 6%？

## 先校准比较对象

这两组实验不是“同一个 SBF selector 被加入轨迹模型后变差”。冻结 SBF selector 实验直接把 SBF 的已训练 selector（epoch 14，包含其 SigLIP 特征路径）应用于 HETT Top-20 坐标，再执行 bounded waypoint。新跑的四组实验使用 HETT 自己的、从头初始化的 trajectory/path/mode/stop 头，只训练一轮；**没有调用冻结 SBF selector 或 SBF 权重**。因此 40.6% → 5.8% 不是融合前后的配对因果比较。

另外，40.6% 是 `val_seen` 的固定 256 episode 子集。子集完全相同的 paired 结果如下；unseen 表现同时列出，不能只看 seen 峰值：

| Split / method | N | SR | SPL | OSR | NE |
|---|---:|---:|---:|---:|---:|
| Seen: HETT frozen checkpoint + bounded waypoint | 256 | 27.34% | 22.65% | 45.31% | 41.44m |
| Seen: same + frozen SBF selector | 256 | **40.63%** | 21.90% | 69.53% | 35.75m |
| Unseen: HETT + bounded waypoint | 256 | **25.39%** | **20.28%** | 39.84% | **43.81m** |
| Unseen: same + frozen SBF selector | 256 | 19.14% | 10.97% | 43.36% | 47.56m |

Seen selector gain is +13.28 SR points (paired episode bootstrap 95% CI [6.25,20.31], 61 gains/27 losses). Unseen loses 6.25 SR points (95% CI [-12.50,-0.39], 22 gains/38 losses). Thus the frozen selector already has a **seen/unseen transfer problem**. `40.6%` is not a general SBF system score and it comes with lower SPL even on seen.

## The full joint-trajectory run is a different method

The one-epoch run completed on the pre-update parent `26e1ff0`, using 2,470 seen and 2,697 unseen episodes. It compared one shared checkpoint under four inference settings:

| Split | A: Heatmap Top-1, legacy controller | B: prior multi-trajectory | C: joint goal/path | D: C with learned Stop off |
|---|---:|---:|---:|---:|
| Seen SR / SPL | 29.47 / 25.54 | 4.90 / 4.83 | 5.79 / 5.72 | 5.79 / 5.72 |
| Unseen SR / SPL | 18.54 / 16.25 | 5.67 / 5.60 | 5.90 / 5.78 | 5.90 / 5.78 |
| Unseen OSR / NE | 39.93 / 51.03m | 7.30 / 64.51m | 7.75 / 62.20m | 7.75 / 62.20m |

On paired full splits, B-A is -24.57 SR points seen (95% CI [-26.56,-22.55]) and -12.87 unseen ([-14.57,-11.16]). C-B adds +0.89 seen ([+0.36,+1.42]) but only +0.22 unseen ([-0.30,+0.74], inconclusive). C remains far below A. This shows the failure appears when the controller executes freshly learned paths; the learned joint ranking only recovers a small part.

A/B/C/D are not separate training runs. A is the original controller on the same jointly trained checkpoint, and B/C/D switch execution/selection behavior. Therefore this experiment isolates the **inference policy change after shared training**, not the causal value of training each objective separately.

## What failed, in order

1. **Not candidate proposal recall.** On the full unseen run, A's initial Heatmap Top-20 Hit@20 is 88.05%; B is 88.51%. At active B states it remains 88.51%, while SR falls from 18.54% for A to 5.67%. The heatmap still proposes plausible targets; executing its trajectory hypotheses fails to reach them.
2. **Trajectory candidate ranking is weak.** At unseen active states, about 55.65% have at least one predicted trajectory goal within 20m of GT (`rank_eligible_rate`). Yet the prior-selected goal hits 20m only 19.93% of the time; joint selection reaches 20.81%. The selected plan's mean FDE is 49.76m for B and 48.51m for C, while the best available proposal has mean FDE 26.41m / 26.51m. The pool often contains a substantially better plan than the one selected. At seen, C selected FDE is 43.48m versus 24.58m minimum FDE.
3. **Predicted path execution has a large reachability gap.** Unseen B/C OSR is only 7.30/7.75%, with NE 64.51/62.20m and mean path length 157/161m. That is consistent with poor proposed routes/waypoint following before the UAV reaches the goal. A has OSR 39.93% and path length 229.73m. B/C paths are much shorter while ending farther away. The trajectory hypotheses were supervised for one epoch and are compared against a teacher suffix; after student deviations this suffix is an imperfect recovery target.
4. **One epoch is a weak initialization test for new heads.** The trajectory, mode, residual, goal-ranking and Stop heads are new parameters. The initialized checkpoint supplies the older HETT backbone but none of these trained trajectory heads. Training took 7,176s for one epoch. Measured gradients/updates are finite, but this is not evidence the multi-path policies are trained well enough to control the vehicle.
5. **The joint scorer barely improves held-out selection.** C vs B changes unseen candidate hit from 19.93% to 20.81% and SR by just +0.22 points, with paired CI spanning zero. Seen SR rises by +0.89 points, far too little to account for the roughly 24 point collapse from A.
6. **Learned Stop did not cause the collapse.** B/C generated zero learned Stop decisions; D is exactly identical to C on both splits. The stop ablation could not rescue trajectories because the learned stop never fired. Conversely A's progress stop recall on unseen is only 0.24% (21 TP, 70 FP, 8,696 FN), and A has a 21.39 point OSR-SR gap with 577 entered-then-left episodes. Stop calibration remains a separate A-policy issue, but it does not explain B/C's low OSR.

## Why SBF selector's 40% does not transfer automatically

- SBF's selector is a pretrained, trained semantic/geometric Transformer coupled to frozen SigLIP candidate crops, full instruction tokens, individually resolved landmark names/positions, and candidate-agent geometry. It reranks an existing candidate and then executes that selected coordinate directly with a bounded waypoint controller.
- The old joint run did none of that. It produced 5 heatmap goals × 3 path modes and chose a predicted path, then executed its first predicted waypoint and replanned. Its supervision was HETT student/teacher rollout losses. This changes both the decision object (coordinate versus path) and the controller dynamics.
- The SBF transfer is distribution-sensitive: on the same 256 unseen episodes it reduces SR and SPL despite a modest OSR increase. The seen +13 point result did not generalize to unseen in the selector-only experiment.
- “SBF-inspired” is not equivalent to using trained SBF weights. The latest fetched commit `a812a30` adds a new `CandidateRelationSelector`, but it is a freshly initialized HETT head, not the SBF selector/checkpoint. Its zero-initialized `relation_gate` initially suppresses this evidence. The completed `26e1ff0` one-epoch run predates this code entirely and contains no per-landmark relation selector results.

## Conclusion

**The main measured bottleneck is predicted-trajectory ranking and execution, not Heatmap Top-K recall or Stop.** The selected path is about 22m worse in FDE than the best path already present in the proposal pool on unseen, and the joint head adds no statistically clear unseen SR gain. The SBF selector's 40.6% is a seen-only subset gain from a distinct coordinate-reranking plus direct-waypoint pipeline; it is not evidence that combining the selector with an undertrained path head will preserve that gain.

## Recommended next diagnostic

Do not increase model complexity yet. Using the latest `a812a30` branch and the same frozen checkpoint, first run a matched one-step offline selector analysis: for each fixed unseen state, compare prior-selected path, joint-selected path, best-in-pool path (diagnostic only), and candidate relation selector on the **same** Top-K pool; report hit@20, selected FDE, and paths. Then test coordinate choice while retaining the proven bounded waypoint controller. Only after selection improves on unseen should full multi-step trajectory execution be reconsidered. Keep oracle candidates strictly diagnostic, and report Stop separately.

## Reproducibility

The full older-branch epoch-1 report, JSON/CSV, paired comparisons, episode IDs and 12 example plots are archived under `artifacts/sbf_joint_transfer_forensics/pre_relation_selector_epoch1/`. Exact same-256 comparisons are in `artifacts/sbf_joint_transfer_forensics/same_256_episode_comparison.json`. The complete per-episode traces and 2.2GB checkpoint remain in the local archived worktree/commit `experiment/joint-goal-trajectory-1ep` (`adcb64f`); they are not copied into the updated branch.
