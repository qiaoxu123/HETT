# HETT Joint Goal/Trajectory 实验报告

状态：已完成 0/10 轮。固定最终比较 epoch 10，当前表格为 epoch 0。

A/B/C/D 使用同一 checkpoint、相同完整验证 episode、seed、horizon 和半径；A=原 Top-1 控制器，B=prior 多轨迹，C=joint 多轨迹，D=joint 关闭 Learned Stop。

| Split | Group | N | SR% | SPL% | OSR% | NE m | Top5 Hit@20% | Plan ADE m | Plan FDE m | Stop precision | Switch rate | Path m | Eval s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_seen | A | 8 | 12.500 | 10.207 | 62.500 | 44.621 | 70.625 | N/A | N/A | N/A | 0.250 | 216.216 | 0.698 |
| val_seen | B | 8 | 0.000 | 0.000 | 0.000 | 82.969 | 55.000 | 51.464 | 51.770 | N/A | 0.112 | 157.964 | 0.716 |
| val_seen | C | 8 | 0.000 | 0.000 | 0.000 | 82.969 | 55.000 | 51.464 | 51.770 | N/A | 0.112 | 157.964 | 0.747 |
| val_seen | D | 8 | 0.000 | 0.000 | 0.000 | 82.969 | 55.000 | 51.464 | 51.770 | N/A | 0.112 | 157.964 | 0.719 |
| val_unseen | A | 8 | 0.000 | 0.000 | 12.500 | 110.281 | 0.000 | N/A | N/A | N/A | 0.322 | 217.770 | 0.765 |
| val_unseen | B | 8 | 12.500 | 12.500 | 25.000 | 63.965 | 1.250 | 50.858 | 74.652 | N/A | 0.105 | 143.207 | 0.870 |
| val_unseen | C | 8 | 12.500 | 12.500 | 25.000 | 63.965 | 1.250 | 50.858 | 74.652 | N/A | 0.105 | 143.207 | 0.849 |
| val_unseen | D | 8 | 12.500 | 12.500 | 25.000 | 63.965 | 1.250 | 50.858 | 74.652 | N/A | 0.105 | 143.207 | 0.715 |

## 配对分析（以 episode 为单位 bootstrap 10,000 次）

- val_seen B_minus_A: ΔSR=-12.50pp, 95% CI=[-37.5, 0.0]; ΔSPL=-10.21pp; gained/lost=0/1。
- val_seen C_minus_B: ΔSR=0.00pp, 95% CI=[0.0, 0.0]; ΔSPL=0.00pp; gained/lost=0/0。
- val_seen C_minus_A: ΔSR=-12.50pp, 95% CI=[-37.5, 0.0]; ΔSPL=-10.21pp; gained/lost=0/1。
- val_seen D_minus_C: ΔSR=0.00pp, 95% CI=[0.0, 0.0]; ΔSPL=0.00pp; gained/lost=0/0。
- val_unseen B_minus_A: ΔSR=12.50pp, 95% CI=[0.0, 37.5]; ΔSPL=12.50pp; gained/lost=1/0。
- val_unseen C_minus_B: ΔSR=0.00pp, 95% CI=[0.0, 0.0]; ΔSPL=0.00pp; gained/lost=0/0。
- val_unseen C_minus_A: ΔSR=12.50pp, 95% CI=[0.0, 37.5]; ΔSPL=12.50pp; gained/lost=1/0。
- val_unseen D_minus_C: ΔSR=0.00pp, 95% CI=[0.0, 0.0]; ΔSPL=0.00pp; gained/lost=0/0。

## Unseen 与到达后失败
本轮尚不能确认 Unseen SR 有可靠正向提升。

- A: OSR−SR=12.50pp；进入成功区后离开 1；停止 TP/FP/FN=0/0/1；超时 8。
- B: OSR−SR=12.50pp；进入成功区后离开 1；停止 TP/FP/FN=0/0/22；超时 8。
- C: OSR−SR=12.50pp；进入成功区后离开 1；停止 TP/FP/FN=0/0/22；超时 8。
- D: OSR−SR=12.50pp；进入成功区后离开 1；停止 TP/FP/FN=0/0/22；超时 8。

## 口径与限制
- ADE/FDE：每个实际 rollout 状态的预测计划对人类剩余轨迹（按路程重采样）的误差；偏离人类轨迹时它是诊断代理，不等于最优恢复路线。A 无实际执行多轨迹计划，selected plan ADE/FDE 为 N/A，但 JSON 保留 prior/joint/minADE/minFDE 诊断。
- Top-K Hit@20 同时保存 initial-state 与 all-active-step 两种口径；all-step 因策略路径不同不能当作同状态纯排序因果对比。轨迹池 K=5，热图候选另报告 K=1/5/16/20。
- Stop precision/recall/accuracy 按实际评估决策状态、GT 20m 成功半径计算；关闭 Stop 的 precision 为 N/A。Accuracy 可能被大量负例主导，必须同时查看 TP/FP/FN。
- Switch rate：相邻有效计划的候选 cell ID 改变次数 / 可比较相邻步数。路径长度为 evaluator 的宏观状态路径长度；不是微动作连续路径长度。
- A 为 joint-trained checkpoint 的原执行路径，不是另训一个没有轨迹辅助 loss 的纯模型；本实验分离推理策略作用，不能将 B−A 全部归因为训练目标。
- 单 seed、单初始 checkpoint；episode bootstrap 不处理场景相关性，不代表跨 seed 稳健性。没有读取 Test-Unseen；Oracle 仅指标，无动作决策。

## 下一步建议
1. 先根据 C−B 区分学习排序收益与多轨迹控制收益。
2. 根据 D−C 与 TP/FP/FN 判断 Learned Stop 是否损伤 SR；不要同时修改停止和路径头。
3. 若 OSR 高但 SPL/SR 低，优先检查目标切换、局部 waypoint 步长与到达后离开案例，再决定是否调整模型。

## 复现
配置：`configs/experiments/joint_goal_trajectory_10e.json`；命令、源码哈希、初始权重哈希、每轮权重哈希、固定 episode IDs、耗时显存与梯度审计均保存在同目录。
```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/20260922_1/miniconda3/envs/AirVLN39/bin/python scripts/run_joint_goal_trajectory_experiment.py --config "$PWD/configs/experiments/joint_goal_trajectory_10e.json" --output "$PWD/artifacts/joint_goal_trajectory/run10"
```
