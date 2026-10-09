# Arrival-Aware Navigation 实验报告

## 结论

指令 + 当前 RGB 联合停止头：Unseen SR 变化 +0.000pp，SPL 变化 +0.000pp。未达到 SR 提高且 SPL 不降低的目标；不继续扩大网络或修改评估协议。
联合停止头校准状态：`no_safe_seen_threshold_disable_stop`，Seen 冻结阈值 `1.01000000`。

## 固定协议与权重

- 分支基于 `qiaoxu123/HETT` 的 `2027-CVPR/hett-compact-goal-path-soft`，起点 `d249e5e`。
- 使用本地 `b16_lr2e-4_9d88f2a/checkpoints/epoch01.pt`；三个模型 state_dict 均严格兼容，无缺失或额外参数。
- 冻结导航权重 SHA-256：`0edcd106923d5050c504c2361cb89b51bfd4321d209f89b05e2874e516649e03`。几何 head 训练 CPU Logistic Regression，联合 head 仅训练小型 MLP；HETT、BERT、视觉骨干与 Goal Selector 全部冻结。
- seed 0、batch 16、max_action_len 20、最大步长 50m、成功半径 20m。目标稳定化和路径分支关闭；Goal Soft Refined 和 bounded waypoint 函数保持原样。
- 每个验证集先用固定前 128 个 Episode 做快速检查，再评估 Seen 2470、Unseen 2697。训练使用 train_seen；阈值只在 val_seen 选择，val_unseen 不参与选择。
- 官方 SR/OSR/SPL/NE 继续调用 CityNavBatch 的原始 evaluator，未修改 `multiagent/env.py`。全量 baseline 每条 Episode 的 NE 与原最佳结果差异为 0，成功标签和 Episode 集合完全一致。

## P0：逐步诊断

- fast_eval 现在保存实际选中 cell ID、细化世界坐标、belief/selector 置信度与熵、当前/下一位置、执行位移及终止原因。记录与昂贵 teacher-path observer 分离。
- Stop 在当前观测和目标预测形成后、下一次移动前判断。History 每个 Episode 重置，仅包含最近最多 5 个当前/过去状态。
- 区分 horizon、waypoint_stagnation 和 policy_stop；同时区分离开、重新进入、最终失败。以下 exit 统计按事件计数，可在一个 Episode 中重复发生。

### val_seen

- 成功 727；从未进入成功区 1358；进入后最终失败 385。
- 从未进入的失败中，全程预测 Goal 都在 GT 20m 外 1280 条；曾选中过 GT 20m 内的 Goal、但仍未到达 78 条。后者可能涉及选择过晚、后续切换或剩余预算，不能仅据此归因于步长。
- 目标切换 11373 次，切换率 24.23%；自然终止 horizon 2470，stagnation 0。
- 离开时目标切换事件 612；预测 Goal 位于 GT 20m 外的离开事件 851；完整 50m 动作导致的离开事件 32。
- 重新进入事件 466；两端均在圈外、线段穿过圈内的宏动作 81。这些线段穿越仅作诊断，官方 OSR 仍只计算保存的宏观位置。

### val_unseen

- 成功 680；从未进入成功区 1618；进入后最终失败 399。
- 从未进入的失败中，全程预测 Goal 都在 GT 20m 外 1543 条；曾选中过 GT 20m 内的 Goal、但仍未到达 75 条。后者可能涉及选择过晚、后续切换或剩余预算，不能仅据此归因于步长。
- 目标切换 12532 次，切换率 24.46%；自然终止 horizon 2697，stagnation 0。
- 离开时目标切换事件 604；预测 Goal 位于 GT 20m 外的离开事件 874；完整 50m 动作导致的离开事件 41。
- 重新进入事件 475；两端均在圈外、线段穿过圈内的宏动作 90。这些线段穿越仅作诊断，官方 OSR 仍只计算保存的宏观位置。

离开事件的关联标记不是互斥因果分类。预测 Goal 在圈外、目标切换与缺少 Stop 可以同时发生；50m 动作事件占比较小，不能把主要损失归因于最大步长。更多失败从未到达，停止模块无法修复其目标定位/选择错误。

## P1：Oracle 上界

| Split | Policy | SR% | OSR% | SPL% | NE m |
| --- | --- | --- | --- | --- | --- |
| val_seen | A_original | 29.433 | 45.020 | 23.331 | 41.151 |
| val_seen | B_oracle_stop | 45.020 | 45.020 | 43.621 | 38.559 |
| val_seen | C_oracle_goal | 100.000 | 100.000 | 100.000 | 0.000 |
| val_seen | D_oracle_goal_stop | 100.000 | 100.000 | 100.000 | 3.914 |
| val_unseen | A_original | 25.213 | 40.007 | 20.202 | 44.357 |
| val_unseen | B_oracle_stop | 40.007 | 40.007 | 38.708 | 41.583 |
| val_unseen | C_oracle_goal | 100.000 | 100.000 | 100.000 | 0.000 |
| val_unseen | D_oracle_goal_stop | 100.000 | 100.000 | 100.000 | 4.091 |

Oracle Stop 优先截断同一条 baseline 缓存轨迹，在首次观测到 GT 20m 范围的位置停止，包含初始位置和最终预算位置，不使用线段插值。Oracle Goal 复用相同初始 Pose4D、bounded_heatmap_step、步数上限和 5 步停滞规则；此控制器没有障碍碰撞或位置裁剪，给定 GT 坐标后无需重复 backbone forward。C/D 是明确的诊断干预，其真值坐标没有接入正常 agent。
Oracle Goal 的 100% 是当前无障碍确定性控制器的诊断上界，不代表可部署模型结果。官方 SPL 的分子是起点到目标的直线距离，而不是人类轨迹长度。

## P2：几何与信念 Arrival Head

- Logistic Regression：17 个可观测特征，18 个学习参数，C=1；训练 21877 个 Episode、437540 个决策状态，其中到达正例 99952 个。训练 CPU 耗时 0.88s。
- 输入：距预测目标的距离；当前位置附近 20/40m belief 质量；heatmap/selector 概率、归一化熵、Top-1/Top-2 差；最近 5 步的 Goal 位移、离散切换和 UAV 位移。没有 GT 坐标、标签、teacher suffix、未来位置或 Episode ID。
- GT 只在离线训练标签及评估器中使用。在线 head 对 feature key 做精确 allowlist 检查；模型 forward guard 拒绝 GT-conditioned planning 输入。
- 预先声明的 Seen 校准约束：Stop Precision ≥95%，误停 Episode 比例 ≤0.5%，Seen SR/SPL 均不下降；在满足条件的阈值中按 Seen SR、SPL 选择。若没有安全阈值，阈值设为 1.01，保留 head 但关闭停止。
- 检查 21 个预声明/Seen 分数分位数阈值；Unseen 开始 head 评估前已保存并冻结阈值。

有停止动作的候选中，最高 Seen Stop Precision 仅 44.52% （TP=69，FP=86），距离预声明的 95% 安全要求较远。

| Split | Policy | SR% | SPL% | Stop Precision | Stop Recall | 误停 Episode 比例 |
| --- | --- | --- | --- | --- | --- | --- |
| val_seen | A_original | 29.433 | 23.331 | N/A | 0.00% | 0.00% |
| val_unseen | A_original | 25.213 | 20.202 | N/A | 0.00% | 0.00% |
| val_seen | distance_stop_20m | 22.105 | 21.759 | 22.11% | 85.58% | 77.89% |
| val_seen | arrival_logistic | 29.433 | 23.331 | N/A | 0.00% | 0.00% |
| val_unseen | distance_stop_20m | 20.319 | 19.964 | 20.32% | 82.78% | 79.68% |
| val_unseen | arrival_logistic | 25.213 | 20.202 | N/A | 0.00% | 0.00% |

Stop Recall 按实际访问的决策状态 TP/(TP+FN) 计算，不能代替 Episode 召回；完整 CSV 同时给出 Episode Recall、TP/FP/FN/TN 和错误停止损失的原成功 Episode 数。

### 可观测信号的区分能力

| Split | 距预测 Goal 阈值 m | 状态数 | 实际 GT 到达比例 |
| --- | --- | --- | --- |
| val_seen | 1.0 | 17273 | 30.23% |
| val_seen | 5.0 | 30297 | 29.62% |
| val_seen | 10.0 | 32239 | 29.21% |
| val_seen | 20.0 | 40358 | 27.82% |
| val_unseen | 1.0 | 18331 | 25.70% |
| val_unseen | 5.0 | 32374 | 25.09% |
| val_unseen | 10.0 | 34064 | 24.95% |
| val_unseen | 20.0 | 43452 | 23.24% |

| Split | 状态 ROC AUC | 状态 AP | 正例分数中位数 | 负例分数中位数 |
| --- | --- | --- | --- | --- |
| val_seen | 0.7463 | 0.3969 | 0.3312 | 0.1959 |
| val_unseen | 0.7418 | 0.3678 | 0.3083 | 0.1977 |

没有 Seen 阈值同时满足预声明的精度、误停率和 SR/SPL 约束，因此最终阈值 1.01 明确关闭 Stop；指标与原控制器相同不应算作提升。距预测 Goal 极近的状态中仍有大量真实未到达样本，目标信念的自洽与真实目标正确性不同；轻量线性模型未能安全区分两者。

## P2 扩展：指令与当前 RGB 联合停止

按照用户补充要求，停止判断同时输入当前指令和当前 RGB，保持原 HETT、BERT、视觉骨干、Goal Selector 和导航控制不变。

- 输入：17 个 belief/pose/history 统计 + 同一冻结 BERT 的原始词嵌入均值 768 维 + 当前冻结 Darknet 特征空间均值 512 维。词嵌入屏蔽 PAD/CLS/SEP，仅使用当前指令，不含对象 ID、真值坐标或未来状态。词均值丢失词序，RGB 空间均值丢失局部布局；联合输入不等于已经学会目标级 grounding。
- 采用一层 16 隐藏单元 ReLU MLP，1297 → 16 → 1，共 20785 个学习参数。两种模态在同一隐藏层融合；没有新增语言或视觉编码器。
- 输入选择前做了信号检查：旧 49 维指令投影在不同指令间变化约 1e-7；旧融合 visual token 更换指令仅变化 3.70e-6，而更换 RGB 变化 3.60。因此未把这些近乎不含指令差异的表示当作有效联合停止输入。诊断保存在 `reports/arrival_aware/rejected_semantic_sources/fused768_sensitivity.json`。
- 原始词嵌入能够保留不同指令的差异。复用 baseline 已保存的当前 Pose4D，按原 crop/preprocess 提取 RGB 特征；快速 Seen/Unseen 各 128 条的缓存提取特征与在线观测最大差异均为 0，原轨迹坐标也逐条相同。
- train_seen 21877 个 Episode、437540 个状态；仅 Arrival MLP 做反向传播。AdamW LR=1e-3、weight_decay=1e-4、batch=1024、固定 5 epochs；不解冻主模型、不选择 Unseen epoch。Head 拟合耗时 18.55s。导航评估 batch 仍为 16。
- 标准化仅拟合 train_seen。采用与几何 head 完全相同的 Seen 阈值候选与安全约束；保存并冻结 threshold 后才计算 Unseen head 指标。

非零停止候选的最高 Seen Precision 为 80.00% （TP=4, FP=1）。

| Split | Policy | SR% | SPL% | Stop Precision | Stop Recall | 误停 Episode 比例 |
| --- | --- | --- | --- | --- | --- | --- |
| val_seen | A_original | 29.433 | 23.331 | N/A | 0.00% | 0.00% |
| val_seen | B_oracle_stop | 45.020 | 43.621 | 100.00% | N/A | 0.00% |
| val_unseen | A_original | 25.213 | 20.202 | N/A | 0.00% | 0.00% |
| val_unseen | B_oracle_stop | 40.007 | 38.708 | 100.00% | N/A | 0.00% |
| val_seen | distance_stop_20m | 22.105 | 21.759 | 22.11% | 85.58% | 77.89% |
| val_seen | arrival_joint_mlp16 | 29.433 | 23.331 | N/A | 0.00% | 0.00% |
| val_seen | arrival_joint_mlp16_fixed_0.5_diagnostic | 26.478 | 23.248 | 44.16% | 7.85% | 20.32% |
| val_unseen | distance_stop_20m | 20.319 | 19.964 | 20.32% | 82.78% | 79.68% |
| val_unseen | arrival_joint_mlp16 | 25.213 | 20.202 | N/A | 0.00% | 0.00% |
| val_unseen | arrival_joint_mlp16_fixed_0.5_diagnostic | 24.101 | 20.391 | 34.99% | 3.80% | 17.91% |

| Split | 状态 ROC AUC | 状态 AP | 正例分数中位数 | 负例分数中位数 |
| --- | --- | --- | --- | --- |
| val_seen | 0.8158 | 0.5540 | 0.4709 | 0.0607 |
| val_unseen | 0.7391 | 0.3683 | 0.2377 | 0.0534 |

随机排序的 AUC 基线为 0.5，AP 近似等于到达正例比例：Seen 22.93%、Unseen 18.95%。联合头的排序能力高于随机，但 Unseen AUC/AP 与几何头几乎相同，跨场景收益未得到验证。

独立向量化推理核对保存于 `joint/independent_classifier_check.csv`：阈值 0.5 的状态准确率 Seen 79.66%、Unseen 79.11%；全部预测未到达的多数类基线分别为 77.07%、81.05%。向量化与逐状态 NumPy 推理最大概率差约 1.9e-15。

状态分类指标与逐 Episode 首次触发停止的精度不同。一次提前误停即可结束 Episode，不能用 AUC 高于随机或训练 loss 下降来替代实际 SR/SPL。固定 0.5 的 Unseen 对照误停 483 条（17.91%），损失原本成功的 80 条，恢复原本失败的 50 条，最终净少成功 30 条。

没有满足 Seen 精度/误停率/SR/SPL 约束的阈值，联合头阈值 1.01 关闭停止；这不是性能提升。原始词嵌入均值与 RGB 全局特征能够输入真实模态信息，但此轻量融合在冻结表示下仍未安全识别指令目标的到达；图像内容、文本描述与几何半径的对应关系未被充分学到。更多未到达失败仍来自 Goal 定位/选择，无法靠停止修复。

- 固定 0.5 是预先声明的主动停止诊断对照，不用于选择 Unseen 阈值，也不因其结果自动启用。表中同时报告其误停率与真实 SR/SPL。
- 联合头完整指标、每 Episode 结果、阈值表、训练 loss、标准化与权重：`reports/arrival_aware/joint/`。
- 实際在线快速及全量 rollout 与缓存 prefix replay 的校验：`quick_joint_head_replay.json`、`full_joint_head_replay.json`。

### 联合头复现

```bash
python scripts/collect_arrival_rollouts.py --source-run /path/to/b16_lr2e-4_9d88f2a --output artifacts/arrival/quick_joint --limit 128 --semantic
python scripts/augment_arrival_semantics.py --source-run /path/to/b16_lr2e-4_9d88f2a --baseline artifacts/arrival/full_baseline --output artifacts/arrival/full_joint --splits val_seen val_unseen train_seen
python scripts/run_arrival_analysis.py --rollouts artifacts/arrival/full_joint --output reports/arrival_aware/joint --semantic
python scripts/collect_arrival_rollouts.py --source-run /path/to/b16_lr2e-4_9d88f2a --output artifacts/arrival/quick_joint_head --limit 128 --head reports/arrival_aware/joint/arrival_head.json
python scripts/collect_arrival_rollouts.py --source-run /path/to/b16_lr2e-4_9d88f2a --output artifacts/arrival/full_joint_head --head reports/arrival_aware/joint/arrival_head.json
python scripts/verify_arrival_replay.py --baseline artifacts/arrival/full_joint --online artifacts/arrival/full_joint_head --head reports/arrival_aware/joint/arrival_head.json --output reports/arrival_aware/full_joint_head_replay.json
```

## 验证与产物

- 88 项 CPU 单元测试通过；并行独立审查实读全量 103340 个状态，pose/action/history 对齐错误为 0。完整审查：`reports/arrival_aware/independent_test_review.md`。
- 显存/耗时汇总：`reports/arrival_aware/resource_metrics.csv`；几何与联合头分类消融：`classifier_ablation.csv`。

- 单元测试覆盖：当前位置停止时机、初始停止、最后预算位置、GT/未来输入拒绝、Episode History 重置、fast_eval 切换率、离开/再进入、官方 SPL 口径和 NumPy 计数类型。
- baseline 与原最佳逐条对照：`reports/arrival_aware/baseline_reproduction.json`。
- 在线 rollout 与缓存 stop replay 的快速/全量校验记录保存于 `reports/arrival_aware/`。
- 完整指标与消融：`reports/arrival_aware/full/ablation_metrics.csv`；校准：`seen_threshold_calibration.csv`；逐 Episode 诊断：`episode_diagnostics.csv.gz` 和策略 Episode CSV；轻量权重：`arrival_head.json`。
- 所有逐步状态/动作、固定 Episode IDs、源权重兼容性与 SHA、代码哈希、显存和耗时保存在 `artifacts/arrival/`。大规模缓存留在本地；聚合报告、逐 Episode 诊断与轻量 checkpoint 随 Git 提交。

## 复现命令

在工作区建立与原项目一致的 data/weights 链接，使用 AirVLN39 环境；source-run 必须指向经过 SHA 验证的本地 batch16 实验目录。

```bash
python scripts/collect_arrival_rollouts.py --source-run /path/to/b16_lr2e-4_9d88f2a --output artifacts/arrival/quick --limit 128
python scripts/run_arrival_analysis.py --rollouts artifacts/arrival/quick --output reports/arrival_aware/quick --oracle-only
python scripts/collect_arrival_rollouts.py --source-run /path/to/b16_lr2e-4_9d88f2a --output artifacts/arrival/full_baseline --splits val_seen val_unseen train_seen
python scripts/run_arrival_analysis.py --rollouts artifacts/arrival/full_baseline --output reports/arrival_aware/full
python scripts/collect_arrival_rollouts.py --source-run /path/to/b16_lr2e-4_9d88f2a --output artifacts/arrival/quick_head --limit 128 --head reports/arrival_aware/full/arrival_head.json
python scripts/collect_arrival_rollouts.py --source-run /path/to/b16_lr2e-4_9d88f2a --output artifacts/arrival/full_head --head reports/arrival_aware/full/arrival_head.json
python -m unittest discover -s tests -p "test_*.py"
```

单 seed、单个冻结导航 checkpoint；状态间与地图内样本存在相关性。本轮不能替代跨 seed 或 Test-Unseen 验证。
