# HETT 验证状态

更新时间：2026-09-11 23:01 +08:00

## 当前结论

代码级和 CPU 数据审计已有明确进展，但所有方案的真实单卡导航效果尚未完成，因此总目标仍未完成。

| 分支 | 当前提交 | 已完成 | 待完成 |
| --- | --- | --- | --- |
| teacher fix | `d837182` | 每 episode 独立索引；12 项核心/卫生/整网测试 | 单卡短跑、完整基线 |
| recovery | `b1c55c2` | 历史失败量化；19 项测试；强制仅评估启用 | checkpoint 配对短评估、完整 unseen |
| grounding | `6630459` | 49 区域＋outside；26 项测试；region CE 非零梯度；16,110 状态审计；16 组 hard contrast | 单卡、完整训练和 checkpoint 消融 |
| combined | `bf132b0` | 两模块合并；32 项测试；region CE 非零梯度；recovery 仅评估 | 同 checkpoint 配对评估 |
| hypotheses | `55cb3bf` | 时序证据与 top-k；17 项测试；offset 监督非零梯度 | 单卡短跑、完整对照 |
| bidirectional attention | `b76f9b6` | 12 项测试；整网两方向均有非零有限梯度，四个任务头均受影响 | 真实数据单卡开/关短训与完整对照 |
| loss ablation | `9b50ad4` | 四组独立训练协议；16 项测试 | 真实数据短跑、独立完整训练 |

grounding 的监督覆盖：train_seen 10,734 个状态中可见 63.49%，val_unseen 5,376 个状态中可见 46.88%。因此 outside 类是必要项。

最新六个结构分支 CPU 回归合计 118/118 通过，loss 消融分支另有 16/16 通过。跨分支 checkpoint 审计也通过：grounding 与 combined 的参数键完全一致；关闭 grounding / 多假设时仅忽略对应新增模块；双向注意力开关不改变参数键。

辅助监督的反向路径已单独验证：region CE 同时覆盖可见 patch/outside 时，视觉输入及每个 grounding 参数都有非零有限梯度；真实 cell offset MSE 对多假设偏移头及上游语言特征也有非零有限梯度。recovery 被限制为 eval-only，避免 teacher 的 GT 轨迹改变训练分布后与纯控制收益混算。

论文参数机器审计通过：20 epochs、batch 2、1e-4、AdamW、grid 5、三项主 loss 权重 2.0/1.5/0.1 均匹配；单卡用累积 4 保持有效 episode batch 8。发布代码额外的 target-grid 权重 0.1 未写入论文三项 loss，因此已建立独立消融，不把它默认当作论文结论。

原版基线当前处于 epoch 2，最新记录为 7,500/10,939 batch（约 68.6%）；GPU、日志、磁盘均无告警。它使用启动时源码快照，不受这些 worktree 提交影响。

开发评估默认只构建 `val_seen` 和 `val_unseen`；`test_unseen` 现在必须显式传 `--include_test_unseen`，仅供方案冻结后的最终报告。旧等待链在真正训练/评估前已停止并移动到 `*.pre_val_only_20260911_2248` 归档，新链的 teacher 源码快照确认来自 `d837182`。

数据划分审计通过并保存输入哈希：train_seen 21,878、val_seen 2,470、val_unseen 2,697、test_unseen 5,281 条；跨 split 无 `(map, object, description)` 或起点重复，val_unseen 4 张地图、test_unseen 6 张地图均与训练地图分离。

## GPU 队列

- 原版基线服务：`hett-baseline-20260911.service`，当前 active。
- 串行验证服务：`hett-validation-queue-20260911.service`，当前 active/waiting。
- Grounding 消融服务：`hett-post-smoke-ablations-20260911.service`，等待短跑队列。
- 多假设消融服务：`hett-hypothesis-ablations-20260911.service`，等待 grounding 消融。
- 同 landmark 不同目标服务：`hett-grounding-contrast-20260911.service`，等待上述消融。
- 双向注意力服务：`hett-bidir-validation-20260911.service`，等待 hard contrast 完成。
- Loss 消融 smoke 服务：`hett-loss-ablation-20260911.service`，等待双向注意力验证完成。
- 修复后完整基线 seed 0：`hett-corrected-baseline-full-s0-20260911.service`，等待所有 smoke 成功完成后启动；20 epochs、全数据、`save_every=20`。
- 完整 seed-0 对照矩阵：`hett-full-seed0-matrix-20260911.service`，等待修复后完整基线成功后启动；包含 11 个预声明任务和配对分析，训练任务均为 20 epochs、全数据、`save_every=20`。
- 三种子确认矩阵：`hett-multiseed-confirmation-20260911.service`，将在 seed-0 矩阵成功后运行 seed 17、42；每个种子含 10 个完整任务，每项启动前检查至少 16 GiB 可用磁盘。完成后自动汇总 seeds 0/17/42，报告均值、样本标准差、95% 区间和逐 seed 配对变化，并生成两张比较图。
- 总监控器：`hett-chain-monitor-20260911.service`；不占 GPU，每 60 秒记录服务/训练状态，每完成一个 epoch 自动刷新图，输出在 `runs/chain_monitor_20260911/`。监控同时核验外层服务退出结果和 48 个内部子实验状态，只有状态实质变化才追加事件；跨心跳实测状态文件刷新而事件数保持为 1。
- 队列状态：`/home/tenant2/Workspace/hett-experiments/00-control/runs/gpu_validation_queue_20260911/`
- 当前等待文件：`/home/tenant2/Workspace/hett-experiments/01-teacher-fix/runs/teacher_fix_smoke_s0/status.json`

原版基线成功结束后，队列顺序为：teacher fix → recovery off → recovery on → grounding → combined → multi-hypothesis。任何一步非零退出都会停止队列，并保留失败日志，不会继续产生不可解释结果。

之后使用相同 checkpoint 继续比较 grounding 正常/关闭/打乱语言/打乱视觉、多假设正常/关闭/无时间累计，并执行 16 组 unseen hard contrast。hard contrast 中每条轨迹交换前后共享逐像素相同的 landmark mask、相同起点和环境，目标不同且相距至少 20 米；交换的只有目标指令。

最后执行四组 loss 权重真实数据 smoke：released、paper-loss-only、no-progress、neither。每组独立训练，只验证运行链路；4 episode 指标不作为效果结论。

短跑通过只表示能在 RTX 5090 上完成真实数据前向、反向、保存和评估。之后仍需依据结果启动完整训练；不能把短跑指标当最终性能。

完整修复后基线是必跑项，已设失败门：任一前置 smoke 失败，它不会启动。grounding、双向注意力、多假设和 loss 消融的完整多种子训练仍需依据真实 smoke 与修复后基线结果逐项放行，避免把 4-episode 噪声当作选型依据。

原版与修复后基线的归因边界记录在 `01-teacher-fix/BASELINE_CAUSALITY.md`：两者都关闭双向注意力；训练信号层面的核心变化只有 batch 内 teacher 轨迹计数独立化，其余是记录、保存、评估卫生和 CPU 可测性修复。
