# 本次基线训练

计划运行：`runs/hett_baseline_fixed_20260911/`。
模型为当前修复实现，`--disable_task_interaction`，从预训练 BERT/Darknet 初始化导航训练，不加载历史导航权重。
20 epochs、全量数据、seed 0、batch 2、梯度累积 4、AdamW 1e-4。

本机 systemd 用户服务 `hett-baseline-20260911` 运行监督程序；用户已开启 linger，关闭对话或退出终端不影响该服务。机器重启不会自动恢复此次 transient 服务。

```bash
systemctl --user status hett-baseline-20260911
cat runs/hett_baseline_fixed_20260911/status.json
tail -n 3 runs/hett_baseline_fixed_20260911/checkpoints/batch_metrics.jsonl
```

停止（会同时停止训练，未完成 epoch 不会保存）：

```bash
systemctl --user stop hett-baseline-20260911
```

保留数据：

- `source/`：实际执行的代码快照，后续工作区改动不会影响本轮。
- `provenance.json`：源码、标注和预训练权重 SHA-256；大幅栅格记录路径、大小和修改时间。
- `commands.json`、`pip_freeze.txt`、`git.diff`：命令、依赖和未提交改动。
- `training.log`：完整 stdout/stderr；`checkpoints/train.txt`：每轮汇总。
- `checkpoints/batch_metrics.jsonl`：每 100 batch 及首尾更新的 loss、梯度范数和显存。
- `checkpoints/epoch_metrics.jsonl`：每轮 loss 与 val_seen/val_unseen 指标。
- `checkpoints/epoch_XX.pt`：每轮完整权重和优化器状态；`latest` 是最新轮次，`best_val_unseen` 按验证集 SR 选择。
- `checkpoint_hashes.jsonl`：每轮权重及最终 best 哈希。
- `telemetry.jsonl`、`status.json`：每分钟 GPU 利用率/显存/温度/功耗、系统内存、磁盘与日志活跃度。
- `alerts.jsonl`：进程报错、30 分钟日志未更新、磁盘不足 12 GiB、GPU 温度达到 85°C。
- `evaluation/`：训练正常结束后，自动评测 best 的三个 split 指标及逐 episode 轨迹（`.pt`）。

本地监督记录异常，不会发送外部消息或盲目自动重启。进程报错时保留现场并停止，避免反复覆盖权重。预计全套 epoch 权重约占 40 GiB，启动前剩余空间约 160 GiB。
快速监督测试另存 `runs/supervisor_smoke_20260911/`，不能作为性能结果。

启动记录：首次后台启动遇到 Hugging Face 联网超时，随后回退缓存并短暂执行到首个梯度更新；尚未完成任何 epoch。该次日志完整保留在 `runs/hett_baseline_fixed_20260911_startup_retry/`。正式运行于 17:41:44 重新启动，显式使用 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，从头训练。
