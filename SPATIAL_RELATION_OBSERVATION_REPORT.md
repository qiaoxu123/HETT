# 三类空间关系观测融合：单 epoch 实验

当前状态：完成一次 train_seen epoch、128 Episode 快速验证和 Seen/Unseen 全量验证。

## 模型改动

- UAV ↔ 多个已知地标：每个地标分别保留名称、位置、轮廓尺度；使用东/北位移、距离、方向和地标范围。
- 预测候选 ↔ 地标：28×28 网格中的每个候选分别与各个地标配对，结合对应名称的语言特征，判断空间关系。
- UAV ↔ 候选：距离、相对航向、最近已观测轨迹距离和累计接近量。过去历史每个 Episode 重置。
- 指令：冻结 BERT 的逐词 embedding，新增两层小卷积保留局部词序和地标名称附近的关系词；原 Goal Selector 的 contextual language 仍保留。
- RGB：当前观察的冻结 Darknet 特征，经空间均值与小型投影进入共享观测。
- 融合：新增 67,202 参数，共享关系观测修正 Heatmap 和候选特征，进入 Goal Selector、Trajectory context 及 Stop。
- 新热图和候选特征残差零初始化，训练前严格保留原最佳模型的输出。
- 固定 Waypoint Controller 执行选中并细化的目标；本轮不学习新的路径模式或改变步长规则。三类关系通过目标选择影响执行动作。

## 训练与对照

源权重：Batch16/LR2e-4 的 B_goal_soft_refined epoch01，SHA-256 `0edcd106923d5050c504c2361cb89b51bfd4321d209f89b05e2874e516649e03`。

冻结 HETT、BERT、Darknet、原 Spatial Belief；训练新的关系观测模块与已有 Compact Goal Selector。一次完整 train_seen epoch，Batch16，Adam LR2e-4，Seed0。使用已有 teacher/student sequential backward；GT 仅生成热图、Goal Soft Ranking 和当前 pose 的 20m Stop 标签。Stop BCE 权重 0.05，Heatmap 权重 0.1，Goal Soft Ranking 权重 1。

验证：训练前零残差快速复现；训练后每个验证集 128 Episode，随后 Seen 2470 / Unseen 2697 全量。固定 horizon20、50m 最大步长、20m 成功半径，官方评估计算不变。

三种预先声明设置：

1. `relations_no_stop`：融合观测，关闭 Stop，为主要导航结果。
2. `relations_stop95`：同一权重，阈值 0.95 的 Stop，为固定阈值诊断；该阈值尚未校准，不能称为安全阈值。
3. `relations_disabled`：同一训练后权重关闭新增融合，作为推理消融；它不是单独训练的无融合控制组。

## 信息边界及限制

地标是当前系统已有的 CityRefer 已知地图先验，仅来自指令地标名称匹配后的轮廓，不新增 GT 目标或未来图像。候选是模型当前热图网格，不是 GT 候选。Stop 在当前 pose 做决定，执行下一个动作之前停止。

单 seed、单 epoch 只能判断本轮效果；与原最佳方案比较时包含了继续训练已有 Goal Selector 的变化。不能把全部增益归因于关系融合，推理消融也不能替代分别训练的控制实验。

## 输出位置

配置：`configs/experiments/hett_relation_observation_1e.json`。
本地原始输出及权重：`artifacts/relation_observation/`。
可版本化的测试、汇总指标、消融和逐 Episode 诊断：`reports/relation_observation/`。

## 训练前复现检查与修复

首次尝试尚未完成 epoch 即中止：runner 的评估未统一参数冻结标志，与独立 checkpoint 收集流程发生数值分歧。受控检查确认零残差融合与关闭融合完全一致。统一评估时所有参数 requires_grad=False 后，开启零残差融合与独立 collector 在 Seen/Unseen 各 128 条上的轨迹逐点一致，最大坐标/NE 差异 0。

快速子集与历史全量末批补齐后的结果，开头少量 Episode 因重新组批可产生数值差异，因此训练前快速复现使用相同 128 条 collector 为参考；另增加训练前全量复现，检查原最佳指标与完整轨迹。失败尝试保存在本地，不作为已完成 epoch。

## 完整验证结果

| Split | 设置 | N | SR% | SPL% | OSR% | NE m |
|---|---|---:|---:|---:|---:|---:|
| val_seen | 原最佳 B_goal_soft_refined | 2470 | 29.43 | 23.33 | 45.02 | 41.15 |
| val_seen | relations_no_stop | 2470 | 30.12 | 24.48 | 46.32 | 40.11 |
| val_seen | relations_stop95 | 2470 | 30.12 | 24.48 | 46.32 | 40.11 |
| val_seen | relations_disabled | 2470 | 25.83 | 18.49 | 43.64 | 44.17 |
| val_unseen | 原最佳 B_goal_soft_refined | 2697 | 25.21 | 20.20 | 40.01 | 44.36 |
| val_unseen | relations_no_stop | 2697 | 25.03 | 20.76 | 40.90 | 43.80 |
| val_unseen | relations_stop95 | 2697 | 25.03 | 20.76 | 40.90 | 43.80 |
| val_unseen | relations_disabled | 2697 | 22.36 | 16.33 | 38.56 | 48.75 |

## 与原最佳方案逐 Episode 配对比较

- val_seen_relations_no_stop_vs_original: ΔSR +0.69pp，95% bootstrap CI [-0.6882591093117408, 2.0242914979757085]；ΔSPL +1.15pp；挽回/损失 154/137。
- val_seen_relations_stop95_vs_original: ΔSR +0.69pp，95% bootstrap CI [-0.6882591093117408, 2.0242914979757085]；ΔSPL +1.15pp；挽回/损失 154/137。
- val_seen_relations_disabled_vs_original: ΔSR -3.60pp，95% bootstrap CI [-5.303643724696356, -1.902834008097166]；ΔSPL -4.84pp；挽回/损失 190/279。
- val_unseen_relations_no_stop_vs_original: ΔSR -0.19pp，95% bootstrap CI [-1.4089729328883944, 1.0011123470522802]；ΔSPL +0.55pp；挽回/损失 136/141。
- val_unseen_relations_stop95_vs_original: ΔSR -0.19pp，95% bootstrap CI [-1.4089729328883944, 1.0011123470522802]；ΔSPL +0.55pp；挽回/损失 136/141。
- val_unseen_relations_disabled_vs_original: ΔSR -2.86pp，95% bootstrap CI [-4.338153503893214, -1.4089729328883944]；ΔSPL -3.87pp；挽回/损失 167/244。

## Stop 分类与实际停止

- val_seen：AUC 0.7287（随机 0.5），AP 0.4240（随机 0.2384）；0.5 阈值 Accuracy 0.5665，多数类基线 0.7616，Balanced Accuracy 0.6568。以上在关闭 Stop 的完整因果轨迹上离线计算。
- val_unseen：AUC 0.7150（随机 0.5），AP 0.3138（随机 0.1951）；0.5 阈值 Accuracy 0.5582，多数类基线 0.8049，Balanced Accuracy 0.6628。以上在关闭 Stop 的完整因果轨迹上离线计算。
- val_seen 固定 0.95：实际停止 0，TP/FP 0/0，Precision None，Recall 0.0；误停 Episode 比例 0.0000。Recall 按观察状态计。
- val_unseen 固定 0.95：实际停止 0，TP/FP 0/0，Precision None，Recall 0.0；误停 Episode 比例 0.0000。Recall 按观察状态计。

## 到达失败与选中目标诊断

- val_seen_relations_no_stop：成功 744，进入后最终离开 400，从未进入 1326；从未进入者中，始终没有正确选中目标 1207，曾选中正确目标 119。选中目标距离 GT≤20m 的状态比例 0.2920。这里检查的是选中目标，不是整个候选池；这些关联不等于严格的因果归因。
- val_seen_relations_stop95：成功 744，进入后最终离开 400，从未进入 1326；从未进入者中，始终没有正确选中目标 1207，曾选中正确目标 119。选中目标距离 GT≤20m 的状态比例 0.2920。这里检查的是选中目标，不是整个候选池；这些关联不等于严格的因果归因。
- val_seen_relations_disabled：成功 638，进入后最终离开 440，从未进入 1392；从未进入者中，始终没有正确选中目标 1234，曾选中正确目标 158。选中目标距离 GT≤20m 的状态比例 0.2519。这里检查的是选中目标，不是整个候选池；这些关联不等于严格的因果归因。
- val_unseen_relations_no_stop：成功 675，进入后最终离开 428，从未进入 1594；从未进入者中，始终没有正确选中目标 1483，曾选中正确目标 111。选中目标距离 GT≤20m 的状态比例 0.2457。这里检查的是选中目标，不是整个候选池；这些关联不等于严格的因果归因。
- val_unseen_relations_stop95：成功 675，进入后最终离开 428，从未进入 1594；从未进入者中，始终没有正确选中目标 1483，曾选中正确目标 111。选中目标距离 GT≤20m 的状态比例 0.2457。这里检查的是选中目标，不是整个候选池；这些关联不等于严格的因果归因。
- val_unseen_relations_disabled：成功 603，进入后最终离开 437，从未进入 1657；从未进入者中，始终没有正确选中目标 1494，曾选中正确目标 163。选中目标距离 GT≤20m 的状态比例 0.2200。这里检查的是选中目标，不是整个候选池；这些关联不等于严格的因果归因。

## 资源、权重与一致性

- 完整训练耗时 3607.1s；峰值 allocated VRAM 3.28GiB。
- 主干权重逐 tensor 检查完全不变，审计见 checkpoint_audit.json。
- Stop 在线路径必须逐点等于无 Stop 轨迹的因果前缀；检查见 stop_prefix_replay_checks.json。
- 训练前快速复现：{'val_seen': {'episodes': 128, 'max_ne_difference': 0.0, 'path_mismatches': 0, 'full_episodes': 2470, 'full_max_ne_difference': 0.0, 'full_path_mismatches': 0}, 'val_unseen': {'episodes': 128, 'max_ne_difference': 0.0, 'path_mismatches': 0, 'full_episodes': 2697, 'full_max_ne_difference': 0.0, 'full_path_mismatches': 0}}。
- 权重保存于 `/home/rental/20260922_1/Workspace/hett-arrival-aware-navigation/artifacts/relation_observation/epoch1_b16_lr2e-4_frozen_eval/checkpoints/epoch01.pt`；SHA-256 `64c7d1f81b33b9a22532c2e4a10e07b87ed95fa813f9c593de7c488dcbb520f7`。

## 本轮结论

本轮融合观测没有同时达到提高 Unseen SR 且不降低 SPL 的目标，保留原最佳方案；不自动增加网络或改变评估条件。

## Seen 校准后的 Stop 结论

继续使用此前预先声明的约束：Seen Stop Precision≥95%、误停 Episode 比例≤0.5%、Seen SR/SPL 不下降。阈值候选为固定网格和 Seen 分数分位数；只用 Seen 选择，将 selection.json 保存冻结后才对 Unseen 回放。GT 仅用于离线标签和指标；停止条件只读取当前预测概率。

Seen 分数范围 [0.0030877948738634586, 0.7929805517196655]。候选中最高非零 Precision 为 62.16%（阈值 0.75，TP/FP 23/14），没有阈值满足约束。最终阈值 1.01，状态 `no_safe_seen_threshold_disable_stop`，关闭 Stop。校准后 Seen/Unseen SR/SPL 与原始官方 evaluator 的无 Stop 结果差异均为 0。

## 对失败的解释和限制

- Unseen 进入过成功区的数量由 1079 增至 1103，但最终成功从 680 减至 675；进入后最终离开从 399 增至 428。关系融合增加了到达机会，最终成功保留仍不足。
- Stop 排序高于随机，Unseen AUC 0.715、AP 0.314（随机 AP 0.195）；0.5 阈值准确率 55.82%，低于全判未到达的 80.49%。不能把 AUC 高于随机解释成可靠的停止策略。
- 共享观测的一次训练没有形成足够精确的到达判断；经验校准未通过，不能通过降低精度要求或用 Unseen 调阈值制造提升。当前表征使用局部词序卷积与 RGB 空间均值，对目标语义和空间对应关系的表达有限；这属于待验证的解释，不是已证明的单一原因。
- 与原最佳方案比较的 SR 变化在两个 split 的配对 bootstrap 95% CI 都跨 0，单 seed 结果不能证明稳定 SR 收益。
- 停止进一步增加网络或自动追加训练；当前正式方案继续使用原 B_goal_soft_refined。

## 保存与复现

103 项 CPU 单元测试通过；GPU Batch16 梯度与参数更新检查通过；骨干逐 tensor 无变化；训练前全部 5167 Episode 与原基线逐点相同；Stop95 在线轨迹与无 Stop 因果前缀完全一致。

完整逐步观测和各策略 rollout 保存在本地 `artifacts/relation_observation/epoch1_b16_lr2e-4_frozen_eval/epoch01/`。仓库保存完整指标 CSV、压缩逐 Episode 诊断 CSV、校准表、源码/配置哈希、显存和耗时、以及 1.6MB 的全部 trajectory/relation head 权重。完整模型 checkpoint 本地保留；原冻结骨干权重不假设存在于远程。

复现：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/run_joint_goal_trajectory_experiment.py --config configs/experiments/hett_relation_observation_1e.json --output artifacts/relation_observation/new_run
python scripts/calibrate_relation_stop.py artifacts/relation_observation/new_run
python scripts/report_relation_observation.py artifacts/relation_observation/new_run
```

快速基线参考来自独立 collector；新环境可先按同样 128 Episode 协议保存为 `artifacts/relation_observation/collector_baseline_check`。完整基线参考 CSV 已版本化于 `reports/arrival_aware/joint/`。
