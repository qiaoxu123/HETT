# 给另一台机器编码代理的复现提示词

```text
你正在接手 HETT / CityNav 项目的两个串行实验。请完整执行，不要只给计划或概念方案。

仓库与提交：
- 主基线提交：51a1828
- 地标到达诊断：361445d
- 地标 teacher 清理实现：65ac5d8
- teacher 清理复现脚本与文档：3115b9f
- teacher 清理移交快照：3d09c89
- RSRefSeg2 CityRefer 微调实现：98c267a（分支 codex/finetune-rsrefseg2-cityrefer）
- 待实现分支：codex/rsrefseg2-goal-grounding

首先阅读仓库 AGENTS.md、README.md、SINGLE_GPU.md、BASELINE_RUN.md，以及待实现分支的
PROTOCOL.md 和 HANDOFF.md。以目标机器的 CUDA/PyTorch 能力为准，不要照搬旧 README 的
torch 2.2/CUDA 11.3。所有 GPU 任务必须单卡串行，启动前运行 nvidia-smi。data/ 与 weights/
只读；GB 级 checkpoint 放数据盘，worktree 只留软链。开发阶段禁止读取 test_unseen。

任务一：复现 teacher cleanup 父实验
1. 在独立 worktree 检出 3d09c89，正确配置 data/、weights/ 和 Python 环境。
2. 先运行现有单元测试和两 episode 冒烟测试。
3. 用 scripts/supervise_experiment.py 运行 train_seen、seed 0、3 epoch，参数必须包括：
   --disable_task_interaction --teacher_trajectory_mode landmark_clean
4. 每轮验证 val_seen 与 val_unseen；不要加载 test_unseen。
5. 用 scripts/compare_epoch_metrics.py 与固定 20 epoch 基线以及同轮 baseline 指标比较。
6. 输出 SR、SPL、NE、oracle SR、loss、地标 gate 触发率、提前离开率，并提交 RESULTS.md。
7. 若进程失败，先诊断并修复根因，不得把冒烟结果当完整结果。

任务二：实现 RSRefSeg2 启发的 HETT 本体改造
父实验完成后，在独立 worktree 检出 codex/rsrefseg2-goal-grounding。不要嵌入完整 SAM 或
14 亿参数 RSRefSeg2；借鉴其粗定位与精细预测解耦思路，做 HETT 原生轻量模块：

1. 修改 MapEncoder，使目标分支保留 [B,C,15,15] 空间特征，同时维持旧 checkpoint 可加载；
   动作分支尽量保持不变。
2. 使用语言 token 条件化 15x15 map tokens，输出 coarse spatial logits。
3. 预测所选 cell 内的 bounded offset，并由 cell center + offset 得到最终归一化目标坐标。
4. 地标不能只被合并成一个无身份的二值通道；至少保留实例 centroid/mask/name 的身份和
   padding mask，并做 target/landmark/relation 条件融合。
5. 先只训练/评测 grounding head，不改变停止策略。报告 Goal Hit@10 m、Hit@20 m、median、
   P90、coarse accuracy，以及同地图替换目标指令后的预测切换率。
6. 只有 val_unseen grounding 改善后，才加入独立 semantic stop head。停止标签与最终目标
   20 m 半径一致，并报告 precision、recall、premature-stop、timeout、SR、SPL。
7. 停止诊断日志必须是独立语句，绝不能用 elif 等方式短路现有停止判据；新增 AST 或运行时
   回归测试。stage1_step/stage2_step 必须保持 per-sample，不得恢复 batch 共享标量。
8. 训练只用 seed 0 做筛选，开发只用 val_seen + val_unseen。使用完全相同 episode 顺序做
   parent/候选配对比较。候选门槛：val_unseen SR 约提升 3 个百分点且 SPL 无明显退化。
9. 用 scripts/supervise_experiment.py 保存 source snapshot、provenance.json、commands.json、
   status.json、telemetry.jsonl 和 checkpoint hash。提交全部源码、测试、PROTOCOL.md、RESULTS.md；
   不提交数据、权重、runs 或 checkpoint。

必须区分以下结论边界：
- GT 20 m 停止将 baseline val_unseen SR 18.91 提高到 34.93，说明停止是已证明的主要瓶颈。
- RSRefSeg2 微调在 test_unseen 上 Acc@0.25 为 29.98、Acc@0.50 为 8.46，证明语言敏感性
  有提升但不足以直接充当可靠终点检测器。
- teacher cleanup 在移交时只有离线审计与训练链路证据，没有完整 SR/SPL 结论。
- 单 seed 仅用于筛选，不能写成论文最终收益。

执行过程中先检查当前仓库和运行状态，以文件、日志、checkpoint hash 和实际测试为准。
遇到已有用户改动必须保留。完成后给出提交哈希、精确命令、完整指标表、失败样例分类和结论。
```
