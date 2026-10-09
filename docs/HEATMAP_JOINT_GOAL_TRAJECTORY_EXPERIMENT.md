# HETT Joint Goal/Trajectory 实验报告

状态：已完成 1/1 轮；本报告汇总 epoch 1。

## 本轮结论与代码版本

2026-10-09 完成全量验证：每个策略均评估 val_seen 2,470 条和 val_unseen 2,697 条。以下详细表格对应可选路径消融 `path_ablation_9d88f2a`，代码提交为 `9d88f2ab636b1390e88abcba4476fb420e26634d`，配置为 `configs/experiments/hett_compact_optional_path_1e.json`，batch 8、学习率 1e-4、seed 0、训练 1 epoch。

- C 相对 B 的 SR 增益为 seen +0.24pp、unseen +0.07pp；两者配对 bootstrap 的 95% CI 均包含 0，尚无明确证据支持路径模块带来收益。
- C 的 SR 低于历史 refined waypoint 基线：seen -1.05pp、unseen -0.67pp。这两个差异的 CI 也包含 0，不能据此声称跨 seed 稳定退化。
- C 的 unseen 有 482 条轨迹进入成功区后离开，OSR−SR 为 17.87pp。该差距提示到达后执行值得诊断，但 Oracle 依赖真值，不能把差距当作实际可获得的提升。
- C 的 seen SR 为 27.571%，未达到配置声明的 30% 门槛；实验门控结果为 `stop_and_repair`。

同代码版本另有两组完成的 goal-only 主实验，配置为 `configs/experiments/hett_compact_goal_path_1e.json`。下表列出目标选择加坐标细化的结果：

| Run | Batch | Learning rate | Seen SR% | Unseen SR% | Seen SPL% | Unseen SPL% |
|---|---:|---:|---:|---:|---:|---:|
| epoch1_9d88f2a / B_goal_soft_refined | 8 | 1e-4 | 29.271 | 24.286 | 23.292 | 19.728 |
| b16_lr2e-4_9d88f2a / B_goal_soft_refined | 16 | 2e-4 | 29.433 | 25.213 | 23.331 | 20.202 |

batch 和学习率同时变化，不能单独归因于 batch size。两个主实验声明的门控策略均为未细化的 `B_goal_soft`，seen SR 分别为 27.854% 和 28.057%，均未达到 30% 门槛；refined 变体的 seen SR 也低于 30%。这些结果均为单 seed，尚未评估 Test-Unseen。

另在提交 `17d02db` 的 opt-in 到达锁定/切换边际扫描中，A1 基线 seen/unseen SR 为 28.583%/23.211%，lock10 为 21.660%/17.167%；lock20、lock30 同样退化。缩小 OSR−SR 差距不等于提高 SR，当前扫描不支持默认开启硬锁定策略。

三组训练实验的完整聚合指标随报告保存于 [compact_results](compact_results/)。模型权重、逐条轨迹、manifest 和日志保存在工作区 `artifacts/compact_runs/`。

各策略使用相同 checkpoint、验证 episode、seed、horizon 和成功半径。

| Split | Group | N | SR% | SPL% | OSR% | NE m | Top5 Hit@20% | Plan ADE m | Plan FDE m | Stop precision | Switch rate | Path m | Eval s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_seen | A_refined_waypoint | 2470 | 28.623 | 23.405 | 44.615 | 41.385 | N/A | N/A | N/A | N/A | 0.000 | 225.027 | 221.924 |
| val_seen | A_nms_waypoint | 2470 | 26.964 | 26.042 | 36.842 | 42.855 | N/A | N/A | N/A | N/A | 0.000 | 189.577 | 145.366 |
| val_seen | B_goal_soft | 2470 | 27.328 | 23.625 | 44.818 | 42.710 | N/A | N/A | N/A | N/A | 0.000 | 257.865 | 186.873 |
| val_seen | C_goal_path_soft | 2470 | 27.571 | 23.908 | 44.413 | 42.499 | N/A | N/A | N/A | N/A | 0.000 | 257.872 | 184.260 |
| val_unseen | A_refined_waypoint | 2697 | 23.359 | 19.405 | 38.302 | 45.385 | N/A | N/A | N/A | N/A | 0.000 | 239.013 | 232.568 |
| val_unseen | A_nms_waypoint | 2697 | 21.246 | 20.373 | 32.221 | 46.364 | N/A | N/A | N/A | N/A | 0.000 | 201.230 | 158.010 |
| val_unseen | B_goal_soft | 2697 | 22.618 | 19.596 | 40.304 | 45.548 | N/A | N/A | N/A | N/A | 0.000 | 252.635 | 188.984 |
| val_unseen | C_goal_path_soft | 2697 | 22.692 | 19.648 | 40.564 | 45.596 | N/A | N/A | N/A | N/A | 0.000 | 254.046 | 187.630 |

## 配对分析（以 episode 为单位 bootstrap 10,000 次）

- val_seen A_nms_waypoint_minus_A_refined_waypoint: ΔSR=-1.66pp, 95% CI=[-2.91497975708502, -0.44534412955465585]; ΔSPL=2.64pp; gained/lost=101/142。
- val_seen B_goal_soft_minus_A_refined_waypoint: ΔSR=-1.30pp, 95% CI=[-2.7125506072874495, 0.16194331983805668]; ΔSPL=0.22pp; gained/lost=150/182。
- val_seen C_goal_path_soft_minus_A_refined_waypoint: ΔSR=-1.05pp, 95% CI=[-2.4696356275303644, 0.4048582995951417]; ΔSPL=0.50pp; gained/lost=150/176。
- val_seen B_goal_soft_minus_A_nms_waypoint: ΔSR=0.36pp, 95% CI=[-0.97165991902834, 1.7004048582995952]; ΔSPL=-2.42pp; gained/lost=152/143。
- val_seen C_goal_path_soft_minus_A_nms_waypoint: ΔSR=0.61pp, 95% CI=[-0.728744939271255, 1.9838056680161944]; ΔSPL=-2.13pp; gained/lost=155/140。
- val_seen C_goal_path_soft_minus_B_goal_soft: ΔSR=0.24pp, 95% CI=[-0.20242914979757085, 0.728744939271255]; ΔSPL=0.28pp; gained/lost=20/14。
- val_unseen A_nms_waypoint_minus_A_refined_waypoint: ΔSR=-2.11pp, 95% CI=[-3.2628846866889134, -1.0011123470522802]; ΔSPL=0.97pp; gained/lost=92/149。
- val_unseen B_goal_soft_minus_A_refined_waypoint: ΔSR=-0.74pp, 95% CI=[-2.1134593993325916, 0.6674082313681868]; ΔSPL=0.19pp; gained/lost=174/194。
- val_unseen C_goal_path_soft_minus_A_refined_waypoint: ΔSR=-0.67pp, 95% CI=[-2.039302929180571, 0.7415647015202076]; ΔSPL=0.24pp; gained/lost=174/192。
- val_unseen B_goal_soft_minus_A_nms_waypoint: ΔSR=1.37pp, 95% CI=[0.14831294030404152, 2.632554690396737]; ΔSPL=-0.78pp; gained/lost=162/125。
- val_unseen C_goal_path_soft_minus_A_nms_waypoint: ΔSR=1.45pp, 95% CI=[0.22246941045606228, 2.6696329254727473]; ΔSPL=-0.72pp; gained/lost=162/123。
- val_unseen C_goal_path_soft_minus_B_goal_soft: ΔSR=0.07pp, 95% CI=[-0.3707823507601038, 0.5190952910641453]; ΔSPL=0.05pp; gained/lost=19/17。

## Unseen 与到达后失败

- A_refined_waypoint: OSR−SR=14.94pp；进入成功区后离开 403；停止 TP/FP/FN=None/None/None；超时 2697。
- A_nms_waypoint: OSR−SR=10.98pp；进入成功区后离开 296；停止 TP/FP/FN=None/None/None；超时 1。
- B_goal_soft: OSR−SR=17.69pp；进入成功区后离开 477；停止 TP/FP/FN=None/None/None；超时 293。
- C_goal_path_soft: OSR−SR=17.87pp；进入成功区后离开 482；停止 TP/FP/FN=None/None/None；超时 290。

## 口径与限制
- ADE/FDE：每个实际 rollout 状态的预测计划对人类剩余轨迹（按路程重采样）的误差；偏离人类轨迹时它是诊断代理，不等于最优恢复路线。
- Top-K Hit@20 同时保存 initial-state 与 all-active-step 两种口径；all-step 因策略路径不同不能当作同状态纯排序因果对比。实际轨迹候选池 K=20，热图候选另报告 K=1/5/16/20。
- Stop precision/recall/accuracy 按逐步决策状态、GT 20m 成功半径计算；Stop FN 是逐步可停止机会数，不是失败 episode 数。关闭 Stop 的 precision 为 N/A。
- Switch rate：相邻有效计划的候选 cell ID 改变次数 / 可比较相邻步数。路径长度为 evaluator 的宏观状态路径长度；不是微动作连续路径长度。
- 单 seed、单初始 checkpoint；episode bootstrap 不处理场景相关性，不代表跨 seed 稳健性。没有读取 Test-Unseen；Oracle 仅指标，无动作决策。

## 下一步建议
1. 优先诊断已有完整 C 方案：比较 B/C 的目标选择和实际动作变化，并分析进入成功区后离开的轨迹。当前实验关闭 learned stop，Stop 指标为 N/A，不能用于判断停止头有效性。
2. 以 B_goal_soft_refined 作为后续候选对照，分别检验到达判断、局部 waypoint 步长和候选视觉特征的贡献；到达判断只能使用可观测输入，真值仅作训练监督与评估。
3. 暂不默认开启路径分支或硬目标锁定。固定 batch、学习率和初始 checkpoint，逐项做消融，并用多个 seed 复核 SR/SPL 与误停率。

## 复现
配置、命令、源码哈希、初始权重哈希、每轮权重哈希、固定 episode IDs、耗时显存与梯度审计均保存在实验目录。
