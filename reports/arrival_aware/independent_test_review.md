# Arrival Navigation 独立 CPU 审查

审查日期：2026-10-09。范围：`multiagent/arrival.py`、`multiagent/agent.py` 的 Arrival 修改，以及 rollout 收集、语义 cache 扩充、Oracle/replay 分析和阈值校准脚本。审查未启动 GPU rollout 或训练。

## 结果

- 全套 CPU 单元测试通过：**88 tests，0.973 秒**。其中新增独立测试 5 项，覆盖随机 MLP NumPy/Torch float32/float64 一致性、字典顺序无关性、跨 Episode observer 更新与 hook 幂等、空指令/屏蔽 token、GT/未来注释不影响 Stop，以及非有限数和损坏权重拒绝。
- 全量 baseline 状态核验：Seen **2,470 Episodes / 49,400 状态**，Unseen **2,697 Episodes / 53,940 状态**。当前 pose 与 `path_index` 指向轨迹坐标不一致、`next_pose - pose` 与 action 不一致、历史长度/时间索引不一致均为 **0**。
- 实际加载 `quick_semantic_exact` cache：Seen/Unseen 各 **128 Episodes / 2,560 状态**，SHA-256、float32 维度和连续 Episode-major `semantic_index` 检查均通过。
- 已保存的在线/缓存语义对照显示两集 `maximum_live_feature_difference = 0.0`；该数据来自主任务此前的 GPU 实验，本审查读取并核对其结果，没有重新执行 GPU 实验。
- baseline reproduction 保存结果：Seen/Unseen Episode ID 集合一致、success 不一致数 **0**、最大 NE 差 **0.0**；已保存 geometric head 全量在线/cache 轨迹对照两集坐标差 **0.0**。
- `git diff --check` 通过。

## 代码边界审查

1. **Stop 时机**：agent 先获取当前位置 RGB 并生成可观测特征，再在 `bounded_heatmap_step` 前检查 Stop；Stop 不增加路径长度。记录的 `path_index` 指向当前已保存位置。最后第 20 次动作后的结果仍由官方指标评估，轻量 head 不额外获得第 21 次判断机会。
2. **GT/未来隔离**：推理 head 仅接受显式几何/信念/历史特征 schema，joint head 额外接受当前指令 lexical embedding 和当前 RGB embedding；不接受 GT 坐标、标签或未来位置。GT 距离仅出现在离线监督、Oracle 和统计代码。修改 GT、teacher path、未来轨迹和额外未来注释不会改变已知状态的 Stop 分数或 Stop 截断位置。
3. **Episode 重置**：几何历史来源为当前 `traj[i]['navigation_steps']`；新 rollout 重新计算当前批指令 embedding，每次 RGB forward 前清空语义观测缓存，forward hook 以当前输出更新。独立测试覆盖批大小和指令变化后无前一 Episode 残留。
4. **训练/验证划分**：scaler 和 head 仅 fit `train_seen`；阈值仅依 `val_seen` 校准，并在独立 Unseen head 报告前写入 checkpoint。脚本此前会加载 Unseen 数据用于明确允许的 Oracle 诊断，但未将其用于 head 拟合或阈值选择。
5. **骨干冻结**：collector 和语义扩充脚本明确设置 BERT、视觉、HETT `eval()` 与 `requires_grad_(False)`；语义输出 detach，训练脚本仅创建 MLP 参数。Observer 不修改 frozen vision forward 输出。
6. **官方指标**：SR/OSR/SPL/NE 重用原 `CityNavBatch._eval_item`；Oracle Stop 仅截断实际保存的位置，不将宏动作线段穿越 GT 圆视作成功。原 evaluator 文件未修改。

## 局限

- 本次独立新增测试使用 CPU 合成数据；真实 GPU 在线策略和最终 head 提升结论须以主任务保存的实际 rollout/replay 与独立 Unseen 结果为准。
- 当前 joint descriptor 采用 masked lexical 均值和 RGB 空间均值。它保留两种模态的差异，但分别丢失词序与局部空间布局；小型 MLP 的联合输入不能单凭结构证明目标级 grounding 已充分学习。若停止精度不足，应将这一表示限制和验证指标一并报告。
- cache SHA 与连续索引能拒绝损坏/错位数据；cache 到当前真实观测的语义一致性还依赖在线/cache 对照，现有 128 Episode 对照两集均完全一致。

未发现需阻止当前实验继续的时机、泄露、cache 对齐或官方指标问题。

## 复现

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/20260922_1/miniconda3/envs/AirVLN39/bin/python \
  -m unittest discover -s tests -v
```
