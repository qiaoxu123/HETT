# 原版基线与 teacher-fix 基线的归因边界

比较对象：

- 原版运行源码快照：`/home/tenant2/Workspace/hett-crotonyl/runs/hett_baseline_fixed_20260911/source/`
- 修复后基线分支：`codex/fix-teacher-index`
- 两者都显式使用 `--disable_task_interaction`，不包含双向注意力收益。

## 会改变训练信号的修复

`stage1_step` 从整个 batch 共用的标量改成每个 episode 独立计数。
旧逻辑在 batch=2 时会让同一轮两条 teacher 轨迹分别跳到第 10、20 个底层位置，
而正确逻辑应分别到第 10、10 个位置。它会改变 teacher rollout 的 RGB、pose、history、
action/progress/goal 监督，是两条基线之间唯一应解释为训练行为变化的修复。

## 不应计入模型收益的差异

- 记录 target-grid loss：只增加日志字段，不改变总 loss。
- `save_every`：只控制不可变 epoch 副本，`latest`/`best` 保存逻辑和优化不变。
- validation 显式传 split：防止 test 环境计算 GT loss；student 动作一直由预测值生成。
- 开发 eval 默认不加载 test_unseen：改变报告范围，不改变训练或 val 轨迹。
- 两个临时张量从硬编码 `.cuda()` 改为跟随输入 device/dtype：RTX 5090 CUDA 路径上设备和
  dtype 与原实现相同，目的是支持 CPU 整网测试。
- 新增监督器、哈希、测试和文档：不进入冻结后的训练源码计算图。

因此，修复后基线相对原版基线若有变化，应表述为“teacher batch-index 修复后的整体变化”，
而不是双向注意力、grounding 或 recovery 的收益。单个 seed 的差异仍不能排除训练随机性，
最终需要重复 seed 和置信区间。
