# HETT Joint Goal/Trajectory 实验报告

状态：已完成 1/1 轮；本报告汇总 epoch 1。

各策略使用相同 checkpoint、验证 episode、seed、horizon 和成功半径。

| Split | Group | N | SR% | SPL% | OSR% | NE m | Top5 Hit@20% | Plan ADE m | Plan FDE m | Stop precision | Switch rate | Path m | Eval s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_seen | relations_no_stop | 2470 | 30.121 | 24.480 | 46.316 | 40.110 | N/A | N/A | N/A | N/A | 0.259 | 240.604 | 174.978 |
| val_seen | relations_stop95 | 2470 | 30.121 | 24.480 | 46.316 | 40.110 | N/A | N/A | N/A | N/A | 0.259 | 240.604 | 173.236 |
| val_seen | relations_disabled | 2470 | 25.830 | 18.488 | 43.644 | 44.165 | N/A | N/A | N/A | N/A | 0.464 | 279.721 | 161.991 |
| val_unseen | relations_no_stop | 2697 | 25.028 | 20.756 | 40.897 | 43.799 | N/A | N/A | N/A | N/A | 0.226 | 249.653 | 193.331 |
| val_unseen | relations_stop95 | 2697 | 25.028 | 20.756 | 40.897 | 43.799 | N/A | N/A | N/A | N/A | 0.226 | 249.653 | 194.010 |
| val_unseen | relations_disabled | 2697 | 22.358 | 16.332 | 38.561 | 48.752 | N/A | N/A | N/A | N/A | 0.468 | 290.709 | 180.185 |

## 配对分析（以 episode 为单位 bootstrap 10,000 次）

- val_seen relations_stop95_minus_relations_no_stop: ΔSR=0.00pp, 95% CI=[0.0, 0.0]; ΔSPL=0.00pp; gained/lost=0/0。
- val_seen relations_disabled_minus_relations_no_stop: ΔSR=-4.29pp, 95% CI=[-6.0728744939271255, -2.4696356275303644]; ΔSPL=-5.99pp; gained/lost=202/308。
- val_seen relations_disabled_minus_relations_stop95: ΔSR=-4.29pp, 95% CI=[-6.0728744939271255, -2.4696356275303644]; ΔSPL=-5.99pp; gained/lost=202/308。
- val_unseen relations_stop95_minus_relations_no_stop: ΔSR=0.00pp, 95% CI=[0.0, 0.0]; ΔSPL=0.00pp; gained/lost=0/0。
- val_unseen relations_disabled_minus_relations_no_stop: ΔSR=-2.67pp, 95% CI=[-4.1527623285131625, -1.1865035224323321]; ΔSPL=-4.42pp; gained/lost=175/247。
- val_unseen relations_disabled_minus_relations_stop95: ΔSR=-2.67pp, 95% CI=[-4.1527623285131625, -1.1865035224323321]; ΔSPL=-4.42pp; gained/lost=175/247。

## Unseen 与到达后失败

- relations_no_stop: OSR−SR=15.87pp；进入成功区后离开 428；停止 TP/FP/FN=0/0/10521；超时 2697。
- relations_stop95: OSR−SR=15.87pp；进入成功区后离开 428；停止 TP/FP/FN=0/0/10521；超时 2697。
- relations_disabled: OSR−SR=16.20pp；进入成功区后离开 437；停止 TP/FP/FN=0/0/9316；超时 2697。

## 口径与限制
- ADE/FDE：每个实际 rollout 状态的预测计划对人类剩余轨迹（按路程重采样）的误差；偏离人类轨迹时它是诊断代理，不等于最优恢复路线。
- Top-K Hit@20 同时保存 initial-state 与 all-active-step 两种口径；all-step 因策略路径不同不能当作同状态纯排序因果对比。实际轨迹候选池 K=20，热图候选另报告 K=1/5/16/20。
- Stop precision/recall/accuracy 按逐步决策状态、GT 20m 成功半径计算；Stop FN 是逐步可停止机会数，不是失败 episode 数。关闭 Stop 的 precision 为 N/A。
- Switch rate：相邻有效计划的候选 cell ID 改变次数 / 可比较相邻步数。路径长度为 evaluator 的宏观状态路径长度；不是微动作连续路径长度。
- 单 seed、单初始 checkpoint；episode bootstrap 不处理场景相关性，不代表跨 seed 稳健性。没有读取 Test-Unseen；Oracle 仅指标，无动作决策。

## 下一步建议
1. 根据候选覆盖、排序命中与选中轨迹 FDE 分别定位候选、重排和路径执行瓶颈。
2. 根据停止 TP/FP/FN 判断 learned stop 是否工作；不要将 Stop accuracy 单独当作有效性证据。
3. 若 OSR 高但 SPL/SR 低，检查目标切换、局部 waypoint 步长与到达后离开案例。

## 复现
配置、命令、源码哈希、初始权重哈希、每轮权重哈希、固定 episode IDs、耗时显存与梯度审计均保存在实验目录。
