# 地标感知的 teacher 轨迹整理

## 已完成

新实现位于独立 worktree，默认不开启，不会影响原训练或 checkpoint。启用参数：

```bash
--teacher_trajectory_mode landmark_clean
```

处理流程：

1. 沿人类轨迹计算到指令地标轮廓的距离；连续两个点进入 20 m 才确认到达。
2. 在首次稳定到达处切分粗导航和局部探索。
3. 粗导航删除重复、回环和可直接连接的冗余节点，最多保留 10 个动作。
4. 局部阶段删除回环，但保留沿途分布的观察点，最多保留 10 个动作；直线路段也保留中间观察。
5. 保留并插值 MTurk 原始 yaw，不再在优化模式下随机覆盖朝向。
6. 未稳定到达地标的轨迹自动使用原 teacher，不制造错误阶段标签。

## 全量离线结果

| split | 轨迹数 | 可稳定切分 | 原始点数中位数 | 整理后点数中位数 | 粗导航动作中位数 | 局部观察动作中位数 |
|---|---:|---:|---:|---:|---:|---:|
| train_seen | 21,878 | 17,774（81.24%） | 40 | 12 | 2 | 8 |
| val_seen（只作离线审计） | 2,470 | 2,037（82.47%） | 40 | 12 | 2 | 9 |
| val_unseen（只作离线审计） | 2,697 | 2,213（82.05%） | 37 | 12 | 2 | 8 |

训练集整理后路径长度中位数是原路径的 92.92%，平均为 83.10%；目标终点最大偏差为 **0.0 m**。这里主要压缩到达前的冗余，把决策预算留给地标附近的观察，而不是把整条轨迹拉成直线。

## 同时修正的配套问题

- `stage1_step` 改成每条 episode 独立，避免 batch 内互相推进。
- teacher 与 student 的方向监督都指向整理后路径的下一节点，不再一边走人类路径、一边监督最终目标直线方向。
- 优化模式下保留人类 yaw；原模式保持旧行为。
- student 连续两步进入地标 20 m 范围后才切入第二阶段，并记录 gate 是否触发。
- 轨迹整理只作用于训练集，验证集仍使用原 reference trajectory，保证指标口径可比。
- 开发评测默认只包含 `val_seen + val_unseen`。

两条样本的训练、反向传播、保存和验证冒烟已经通过；固定后的运行没有加载 `test_unseen`。

## 三轮全量实验与复现

完整 `train_seen`、seed 0 的 3 epoch 实验已启动。每轮训练 21,878 条轨迹，随后分别验证
`val_seen` 2,470 条和 `val_unseen` 2,697 条；不加载 `test_unseen`。运行完成前不把中间进度
解释为效果结论。

移交快照（2026-09-21 22:03 +08:00）：第 1 轮训练完成 72.4%（7,918 / 10,939
batch），训练进程和 GPU 正常，尚未产生首轮 checkpoint 或验证指标。因此当前可以确认的是
实现、离线审计和训练链路有效；**不能据此声称 SR/SPL 得到提升**。本机运行目录受
`.gitignore` 管理，不随 Git 提交，跨机器复现应重新启动完整实验。

从本分支根目录运行：

```bash
/home/tenant2/miniconda3/envs/AirVLN39/bin/python scripts/supervise_experiment.py \
  --run-dir runs/teacher_cleanup_3ep_s0_repro \
  --python /home/tenant2/miniconda3/envs/AirVLN39/bin/python \
  --epochs 3 --save-every 1 --seed 0 --phase train-eval --interval 60 \
  --variant-arg=--disable_task_interaction \
  --variant-arg=--teacher_trajectory_mode \
  --variant-arg=landmark_clean
```

若另一台机器的 Python 或共享 GPU 锁路径不同，替换 `--python`，并按需传入
`--lock-file`；其余参数保持不变。`data/` 和 `weights/` 按 `SINGLE_GPU.md` 配置，不提交到 Git。

训练完成后与原始 20 epoch 基线比较：

```bash
/home/tenant2/miniconda3/envs/AirVLN39/bin/python scripts/compare_epoch_metrics.py \
  --candidate runs/teacher_cleanup_3ep_s0_repro/checkpoints/epoch_metrics.jsonl \
  --baseline runs/hett_baseline_fixed_20260911/checkpoints/epoch_metrics.jsonl \
  --output runs/teacher_cleanup_3ep_s0_repro/REPORT.md
```

对比报告逐轮比较前 3 epoch 的 SR、SPL、NE 和 loss，并将候选前三轮最佳检查点与原始
20 epoch 的最佳 `val_unseen` SR 检查点比较。短训练、单 seed 只能判断早期趋势。

## 仍需修改的部分

1. 当前阶段门是 pose＋地图轮廓确认，不能证明视觉上识别了正确地标；后续仍需要高精度、可拒答的视觉确认头。
2. 道路和大型地标轮廓过长，应该用第一阶段关系热图选择相关轮廓片段，再计算到达，避免在错误路段提前切换。
3. 当前 progress/stop 仍面向最终目标且校准较差，需要把“到达地标”“找到目标”“最终停止”拆成独立输出。
4. 本次只证明轨迹和训练链路正确，尚未进行完整训练对照，不能宣称 SR 已提升。
