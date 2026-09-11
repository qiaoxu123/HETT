# HETT 验证状态

更新时间：2026-09-11 22:19 +08:00

## 当前结论

代码级和 CPU 数据审计已有明确进展，但所有方案的真实单卡导航效果尚未完成，因此总目标仍未完成。

| 分支 | 当前提交 | 已完成 | 待完成 |
| --- | --- | --- | --- |
| teacher fix | `543ddfd` | 每 episode 独立索引；6 项核心/卫生测试 | 单卡短跑、完整基线 |
| recovery | `80c9909` | 历史失败量化；14 项测试 | checkpoint 配对短评估、完整 unseen |
| grounding | `a6d5d13` | 49 区域＋outside；13 项测试；16,110 状态审计 | 单卡前后向、完整训练和打乱消融 |
| combined | `7bac906` | 两模块合并；23 项测试 | 同 checkpoint 配对评估 |
| hypotheses | `8d0e57d` | 时序证据与 top-k；10 项测试 | 单卡短跑、完整对照 |

grounding 的监督覆盖：train_seen 10,734 个状态中可见 63.49%，val_unseen 5,376 个状态中可见 46.88%。因此 outside 类是必要项。

## GPU 队列

- 原版基线服务：`hett-baseline-20260911.service`，当前 active。
- 串行验证服务：`hett-validation-queue-20260911.service`，当前 active/waiting。
- 队列状态：`/home/tenant2/Workspace/hett-experiments/00-control/runs/gpu_validation_queue_20260911/`
- 当前等待文件：`/home/tenant2/Workspace/hett-experiments/01-teacher-fix/runs/teacher_fix_smoke_s0/status.json`

原版基线成功结束后，队列顺序为：teacher fix → recovery off → recovery on → grounding → combined → multi-hypothesis。任何一步非零退出都会停止队列，并保留失败日志，不会继续产生不可解释结果。

短跑通过只表示能在 RTX 5090 上完成真实数据前向、反向、保存和评估。之后仍需依据结果启动完整训练；不能把短跑指标当最终性能。
