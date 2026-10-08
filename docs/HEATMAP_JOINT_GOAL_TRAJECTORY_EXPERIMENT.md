# HETT Joint Goal/Trajectory 实验报告

状态：已完成 1/6 轮；本报告汇总 epoch 1。

各策略使用相同 checkpoint、验证 episode、seed、horizon 和成功半径。

| Split | Group | N | SR% | SPL% | OSR% | NE m | Top5 Hit@20% | Plan ADE m | Plan FDE m | Stop precision | Switch rate | Path m | Eval s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_seen | joint_sbf_ideas | 2470 | 2.996 | 2.973 | 3.441 | 70.438 | 61.328 | 41.800 | 51.872 | N/A | 0.088 | 135.107 | 255.691 |
| val_unseen | joint_sbf_ideas | 2697 | 2.892 | 2.854 | 3.856 | 73.880 | 55.719 | 42.351 | 55.956 | N/A | 0.091 | 143.930 | 274.920 |

## 配对分析（以 episode 为单位 bootstrap 10,000 次）


## Unseen 与到达后失败

- joint_sbf_ideas: OSR−SR=0.96pp；进入成功区后离开 26；停止 TP/FP/FN=0/0/958；超时 2697。

## 口径与限制
- ADE/FDE：每个实际 rollout 状态的预测计划对人类剩余轨迹（按路程重采样）的误差；偏离人类轨迹时它是诊断代理，不等于最优恢复路线。
- Top-K Hit@20 同时保存 initial-state 与 all-active-step 两种口径；all-step 因策略路径不同不能当作同状态纯排序因果对比。轨迹池 K=5，热图候选另报告 K=1/5/16/20。
- Stop precision/recall/accuracy 按实际评估决策状态、GT 20m 成功半径计算；关闭 Stop 的 precision 为 N/A。Accuracy 可能被大量负例主导，必须同时查看 TP/FP/FN。
- Switch rate：相邻有效计划的候选 cell ID 改变次数 / 可比较相邻步数。路径长度为 evaluator 的宏观状态路径长度；不是微动作连续路径长度。
- 单 seed、单初始 checkpoint；episode bootstrap 不处理场景相关性，不代表跨 seed 稳健性。没有读取 Test-Unseen；Oracle 仅指标，无动作决策。

## 下一步建议
1. 根据候选覆盖、排序命中与选中轨迹 FDE 分别定位候选、重排和路径执行瓶颈。
2. 根据停止 TP/FP/FN 判断 learned stop 是否工作；不要将 Stop accuracy 单独当作有效性证据。
3. 若 OSR 高但 SPL/SR 低，检查目标切换、局部 waypoint 步长与到达后离开案例。

## 复现
配置、命令、源码哈希、初始权重哈希、每轮权重哈希、固定 episode IDs、耗时显存与梯度审计均保存在实验目录。
