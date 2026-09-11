# HETT 验证状态

更新时间：2026-09-11 22:38 +08:00

## 当前结论

代码级和 CPU 数据审计已有明确进展，但所有方案的真实单卡导航效果尚未完成，因此总目标仍未完成。

| 分支 | 当前提交 | 已完成 | 待完成 |
| --- | --- | --- | --- |
| teacher fix | `b01add8` | 每 episode 独立索引；10 项核心/卫生/整网测试 | 单卡短跑、完整基线 |
| recovery | `80c9909` | 历史失败量化；16 项测试 | checkpoint 配对短评估、完整 unseen |
| grounding | `428ea22` | 49 区域＋outside；22 项测试；整网 CPU 前后向；16,110 状态审计；16 组 hard contrast | 单卡、完整训练和 checkpoint 消融 |
| combined | `e21e0eb` | 两模块合并；28 项测试；整网 CPU 前后向 | 同 checkpoint 配对评估 |
| hypotheses | `af0c95b` | 时序证据与 top-k；14 项测试；整网 CPU 前后向 | 单卡短跑、完整对照 |
| bidirectional attention | `b01add8` | 10 项测试；整网两方向均有非零有限梯度，四个任务头均受影响 | 真实数据单卡开/关短训与完整对照 |
| loss ablation | `078c74c` | 四组独立训练协议；14 项测试 | 真实数据短跑、独立完整训练 |

grounding 的监督覆盖：train_seen 10,734 个状态中可见 63.49%，val_unseen 5,376 个状态中可见 46.88%。因此 outside 类是必要项。

最新六个结构分支 CPU 回归合计 101/101 通过，loss 消融分支另有 14/14 通过。跨分支 checkpoint 审计也通过：grounding 与 combined 的参数键完全一致；关闭 grounding / 多假设时仅忽略对应新增模块；双向注意力开关不改变参数键。

论文参数机器审计通过：20 epochs、batch 2、1e-4、AdamW、grid 5、三项主 loss 权重 2.0/1.5/0.1 均匹配；单卡用累积 4 保持有效 episode batch 8。发布代码额外的 target-grid 权重 0.1 未写入论文三项 loss，因此已建立独立消融，不把它默认当作论文结论。

原版基线当前处于 epoch 2，最近观测约完成 55%，GPU、日志、磁盘均无告警。它使用启动时源码快照，不受这些 worktree 提交影响。

## GPU 队列

- 原版基线服务：`hett-baseline-20260911.service`，当前 active。
- 串行验证服务：`hett-validation-queue-20260911.service`，当前 active/waiting。
- Grounding 消融服务：`hett-post-smoke-ablations-20260911.service`，等待短跑队列。
- 多假设消融服务：`hett-hypothesis-ablations-20260911.service`，等待 grounding 消融。
- 同 landmark 不同目标服务：`hett-grounding-contrast-20260911.service`，等待上述消融。
- 双向注意力服务：`hett-bidir-validation-20260911.service`，等待 hard contrast 完成。
- Loss 消融 smoke 服务：`hett-loss-ablation-20260911.service`，等待双向注意力验证完成。
- 队列状态：`/home/tenant2/Workspace/hett-experiments/00-control/runs/gpu_validation_queue_20260911/`
- 当前等待文件：`/home/tenant2/Workspace/hett-experiments/01-teacher-fix/runs/teacher_fix_smoke_s0/status.json`

原版基线成功结束后，队列顺序为：teacher fix → recovery off → recovery on → grounding → combined → multi-hypothesis。任何一步非零退出都会停止队列，并保留失败日志，不会继续产生不可解释结果。

之后使用相同 checkpoint 继续比较 grounding 正常/关闭/打乱语言/打乱视觉、多假设正常/关闭/无时间累计，并执行 16 组 unseen hard contrast。hard contrast 中每条轨迹交换前后共享逐像素相同的 landmark mask、相同起点和环境，目标不同且相距至少 20 米；交换的只有目标指令。

最后执行四组 loss 权重真实数据 smoke：released、paper-loss-only、no-progress、neither。每组独立训练，只验证运行链路；4 episode 指标不作为效果结论。

短跑通过只表示能在 RTX 5090 上完成真实数据前向、反向、保存和评估。之后仍需依据结果启动完整训练；不能把短跑指标当最终性能。
