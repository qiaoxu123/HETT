# HETT 原版复现基线（Baseline）

> 来源：`~/Workspace/htnav-repro` 的 HETT 复现工程（vendored `external/HETT/`，AAAI 2026 oral 论文
> [crotonyl/HETT](https://github.com/crotonyl/HETT) 的完整复现，记录于
> `HETT_REPRODUCTION.md` / `EXPERIMENTS.md` E77）。
> 用途：作为本仓库（hett-crotonyl，基于 qiaoxu123/HETT 的改进版）的**对比基线**，
> 判断我们的改动是否有效。
>
> 归档时间：2026-09-11。

## 0. 官方协议（评测/训练条件）

- 评测：`feedback student`、`move_iteration 10`、`max_action_len 20`、`grid_size 5`、
  `batch_size 2`（env 按此设计，改大会 OOM）、`altitude 50`
- 训练：单卡 RTX 5090 32GB，batch 2 × grad_accum 4、AdamW、lr 1e-4、
  teacher+student 双 rollout（ml_weight 0.2、entropy 0.01、grad clip 40）
- 损失权重：`1*direction + 0.1*progress + 2*goal_predict + 0.1*target_predict`
- 环境：conda `AirVLN39`（py3.9，torch 2.8+cu128）

## 1. 官方发布检查点（论文复现基准）

官方 26-epoch 检查点，官方协议评测（见 `valid.txt` epoch 25 块）：

| split | SR | Oracle SR | SPL | NE |
|---|---|---|---|---|
| val_seen | 29.72 | 47.77 | 25.08 | 38.35 |
| val_unseen | 18.32 | 34.30 | 15.17 | 53.23 |
| test_unseen | **26.57** | 46.94 | 21.56 | 42.70 |

论文引用行（HTNav* 25.49）复现成功。

## 2. 复现训练 best（12 epochs，单卡）——本仓库需要超过的数字

`best_val_unseen`（epoch 11），官方协议评测（见 `valid.txt` epoch 11 块）：

| split | SR | Oracle SR | SPL | NE |
|---|---|---|---|---|
| val_seen | 29.43 | 49.43 | 23.29 | 38.10 |
| val_unseen | **19.54** | 35.97 | 15.73 | 50.52 |
| test_unseen | **27.97** | 50.14 | 21.51 | 41.80 |

**结论：复现训练超过官方检查点（val_unseen +1.22，test_unseen +1.40）。**
完整 20-epoch 协议下 best 仍是 epoch 11（后 8 个 epoch 未再打破）。

## 3. 训练曲线

### 3a. 第一段（epoch 0–11，原始日志 `train_rep.log.gz`）

| epoch | IL_loss | direction | progress | goal | val_seen SR | val_unseen SR |
|---|---|---|---|---|---|---|
| 0 | 7.349 | 6.572 | 0.400 | 0.052 | 15.63 | 12.16 |
| 1 | 6.806 | 6.194 | 0.319 | 0.034 | 18.18 | 13.68 |
| 2 | 6.664 | 6.106 | 0.292 | 0.030 | 20.97 | 16.17 |
| 3 | 6.600 | 6.069 | 0.281 | 0.027 | 22.47 | 17.20 |
| 4 | 6.558 | 6.043 | 0.278 | 0.027 | 21.46 | 15.02 |
| 5 | 6.763 | 6.173 | 0.320 | 0.028 | 20.57 | 15.20 |
| 6 | 6.713 | 6.139 | 0.310 | 0.027 | 23.32 | 14.31 |
| 7 | 6.630 | 6.104 | 0.292 | 0.026 | 24.01 | 14.98 |
| 8 | 6.630 | 6.104 | 0.285 | 0.026 | 25.79 | 16.83 |
| 9 | 6.597 | 6.076 | 0.290 | 0.026 | 25.18 | 18.21 |
| 10 | 6.438 | 5.929 | 0.268 | 0.025 | 25.26 | 18.13 |
| 11 | 6.412 | 5.906 | 0.269 | 0.026 | 23.72 | 16.28 |

> 注：`train_rep.log` 是 0-11 段的原始 stdout 曲线（该轮 val_unseen 峰值 18.21@e9）。
> 最终 best（19.54@e11）的 7a 曲线在 `HETT_REPRODUCTION.md` §7a 另有记录：
> 13.68 → 16.57 → 16.91 → 17.20 → 16.35 → 17.69 → 18.21 → 19.54。
> 两条曲线出自不同轮次（详见 §5 不一致说明）。

### 3b. 第二段（resume epoch 12–19，原始日志 `train_epoch12_20.log.gz`）

| epoch | IL_loss | val_seen SR | val_unseen SR |
|---|---|---|---|
| 12 | 6.218 | 25.75 | 18.32 |
| 13 | 6.136 | 25.79 | 13.31 |
| 14 | 6.106 | 26.40 | 16.24 |
| 15 | 6.047 | 25.79 | 16.83 |
| 16 | 5.999 | 27.37 | 15.39 |
| 17 | 5.952 | 27.29 | 15.80 |
| 18 | 5.900 | 25.91 | 14.13 |
| 19 | 5.890 | 27.04 | 15.50 |

完整 20-epoch 协议结论：epoch 11 达峰后平台/过拟合，12-epoch 早停是正确决策。

## 4. 归档文件清单

| 文件 | 内容 |
|---|---|
| `valid.txt` | 官方检查点（epoch 25）+ 复现 best（epoch 11）+ 复评（epoch 11）三组完整评测原始记录 |
| `train_rep.log.gz` | 第一段训练（0-11）原始 stdout |
| `train_epoch12_20.log.gz` | 第二段训练（12-19）原始 stdout |
| `final_eval_20ep.log` | 20-epoch 完成后的 best 复评原始 stdout |
| `train.txt` | 训练记录文件（含后续 optimizer-test 轮次） |
| `HETT_REPRODUCTION.md` | htnav-repro 的权威复现文档（含 §5-§8 全部细节） |

## 5. 已知不一致与注意事项（对比时必须知道）

1. **复评波动**：同一 epoch-11 权重的多次复评结果不一致——
   权威评测 29.43 / 19.54 / 27.97（记录于 E77，两次一致）；
   后续复评（resume 轮开始、final_eval_20ep）得到 val 27.53 / 18.54、test 21.17。
   student feedback 采样评测有噪声（val 上约 ±1-2 SR）；test 的 21.17 异常偏大，
   可能还叠加了 checkpoint 被后续 optimizer-test 实验覆盖的因素。
2. **checkpoint 状态**：htnav-repro 的 `checkpoints/multi/best_val_unseen` 在
   Aug 21-22 被后续实验写入过（mtime 晚于权威评测），**不能保证仍是权威 epoch-11 权重**，
   使用前需自行复评验证。
3. **对比方法**：我们的改进跑完全量训练后，用相同的官方协议（本仓库 `multiagent/eval.sh`）
   在**全量** val_seen / val_unseen / test_unseen 上评测，与 §2 的数字对比；
   差异 ≤1-2 SR 视为评测噪声，≥3 SR 才算显著改进。
