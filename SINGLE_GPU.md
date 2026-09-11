# 单卡 RTX 5090 运行与验证

使用已有 `AirVLN39` 环境：Python 3.9、PyTorch 2.8.0+cu128（包含 sm_120）。
不要在 5090 上按旧 README 安装 torch 2.2 / CUDA 11.3。

## 默认训练配置

| 参数 | 当前默认 |
|---|---|
| 优化器 / 学习率 / epochs | AdamW / 1e-4 / 20 |
| 实际 batch / 梯度累积 | 2 / 4（train.sh） |
| 网格 / 隐藏维度 / 融合层 / heads | 5×5 / 768 / 2 / 12 |
| 目标 / 动作 / 进度损失权重 | 2.0 / 1.5 / 0.1 |
| 辅助网格分类损失权重 | 0.1（保留发布代码，可用 --target_loss_weight 0 关闭） |
| AdamW weight decay | 0.01（显式保留发布代码原先隐式使用的默认值） |
| rollout / 底层移动次数 / 高度 | 20 / 10 / 50m |
| 成功距离 | 20m |

论文参数来源：https://arxiv.org/html/2512.14222v1 ，Implementation Details。
实际 batch 2，4 次梯度平均后更新，有效 episode batch 为 8；每个 batch 包括 teacher 与 student 两次 rollout。
这并不保证与四卡 DDP 的 BatchNorm 和随机数轨迹逐位等价。
保留发布实现的 RGB 输入、目标 Sigmoid、无激活的进度回归；这些是发布代码与论文描述的既有差异，未冒充完全逐式复现。
历史基线的动作损失系数为 1.0，而且有训练实现差异；不能把新结果的变化全部归因于双向注意力。

## 运行

从工程根目录执行：

```bash
conda activate AirVLN39
bash multiagent/train.sh --output_dir checkpoints/hett_bidir
bash multiagent/eval.sh --checkpoint checkpoints/hett_bidir/best_val_unseen --output_dir checkpoints/hett_bidir
```

续训（epochs 表示总训练轮数）：

```bash
bash multiagent/train.sh --checkpoint checkpoints/hett_bidir/latest --resume_optimizer --output_dir checkpoints/hett_bidir
```

相对 checkpoint/output_dir 均按工程根目录解析。不要用现有基线目录存放新实验。
load 返回已完成轮数，下一轮不会重跑刚完成的 epoch。恢复优化器需显式传 --resume_optimizer；未保存随机数和数据迭代器状态，不宣称与不中断训练逐位一致。

快速验证：

```bash
python multiagent/check_single_gpu.py --max_episodes 10 --grad_accum 4
bash multiagent/train.sh --epochs 1 --max_episodes 10 --output_dir checkpoints/smoke_new
```

`check_single_gpu.py` 检查真实数据的 20 步双 rollout、有限 loss、所有交互参数的梯度与更新、两个交叉方向的输入依赖、四个输出头的开关差异、动态 token 长度和语言 padding，以及可精确计算的梯度累积尾批测试。

## 本次修复

- 梯度只在累积窗口开始时清零，反传损失按窗口大小平均，尾批也更新。
- teacher/student 分开反传，减少同时保留两张计算图的显存开销。
- 每一步读取最新位姿；语言 padding 同时用于融合注意力与历史相关性计算。
- attention mask 按所有模态的真实长度生成。
- 评测禁用梯度；训练中的评测复用模型，减少显存占用。
- 修复相对路径、CUDA 环境变量导出、脚本参数透传、续训轮数。
- 默认拒绝用缺少新模块参数的检查点评测，避免随机注意力层被当作已训练模型。
- 清除未计算的 gp_sr 指标，避免日志中出现无意义 NaN。

`--disable_task_interaction` 可用于同协议消融；完整公平对照需要两组分别训练。旧检查点仅在关闭交互且其他参数齐全时可评测，但仍受位姿/mask 修复影响。

## 验证记录（2026-09-11）

在本机 RTX 5090 上完成 10 条真实轨迹快速检查；微批 2、累积 4，5 个 batch 更新 2 次（含尾批）。双向交互的 24 个参数张量均收到非零梯度并更新，四个预测头均受交互开关影响。第一轮峰值 allocated 显存约 7.6 GiB；该值仅代表这批样本，不是全量训练的显存上界。

一轮训练（含 eval_first）、保存、独立加载并评测 val_seen / val_unseen / test_unseen 已通过；独立评测的两组验证指标与训练时一致。
随后恢复优化器续训一轮通过：检查点中三个模型的 completed epochs 均为 2，优化器 step 均从 2 延续到 4。
最终完整检查再次通过，峰值 allocated 为 7.620 GiB；开启/关闭交互时 direction、progress、goal、grid logits 最大绝对差分别为 0.7282、0.01808、0.01901、0.11365。合成 padding 不变性与多模态长度测试也通过。
快速训练输出位于 `multiagent/checkpoints/verify_5090_20260911/`，不是性能检查点。
快速验证只证明实现参与计算且可训练，不证明导航成功率超过历史基线。
