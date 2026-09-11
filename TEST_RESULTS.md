# Teacher 索引修复验证记录

日期：2026-09-11

## 问题

旧实现用一个标量 `stage1_step` 服务整个 batch。batch=2、`move_iteration=10` 时，第一轮两个 episode 会分别读取 teacher 轨迹索引 10 和 20，而不是都读取 10。样本之间因此互相影响，且 batch 大小时序不一致。

## 修改

- 每个 episode 维护独立的 `stage1_steps[i]`。
- teacher 和 student 都按 episode 记录粗阶段步数。
- 保持原来的越界语义：索引超出轨迹时使用最后一个 pose。
- `stage1_step` 日志继续表示 batch 平均粗阶段步数。

## 已完成验证

命令：

`/home/tenant2/miniconda3/envs/AirVLN39/bin/python -m unittest discover -s tests -v`

结果：4/4 通过。

- batch=2 独立推进：两条轨迹均为 10、20。
- 某 episode 提前结束：不会推进另一个 episode 的计数器。
- 轨迹越界：返回 -1，保持读取最终 pose 的历史行为。
- 空轨迹：显式报错。
- `agent.py`、辅助模块和测试均通过 Python 语法编译。

## 尚未完成

- 真实环境单卡极短 rollout。
- 修复后完整训练与 unseen 评估。

这两项必须等当前原版基线退出后串行执行，避免在同一张 GPU 上互相影响。当前结果只能证明索引逻辑修复，不能证明导航指标提升。
