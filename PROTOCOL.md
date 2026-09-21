# 第一阶段坐标头独立微调

## 目的

先验证“解耦训练”本身是否改善目标预测，不增加热图、不修改动作和停止结构。

## 固定项

- 父检查点：20 epoch 基线的 `best_val_unseen`（记录 epoch 15）。
- `--disable_task_interaction`、seed 0、原始二维坐标 goal head。
- 冻结 BERT、Darknet、共享 ET 编码器、候选头、action head、progress head；冻结模块保持 eval 模式。
- 只训练 `decoder_2_goal_full`，不恢复旧优化器。
- 训练 512 个 `train_seen` episode，增加 1 epoch（总 epoch 参数 16）。
- 开发评测只使用 val_seen 和 val_unseen，不读取 test_unseen。

## 指标和门槛

在与父检查点完全相同的 episode 上配对计算：

- 第一预测目标 Hit@20、最后预测目标 Hit@20；
- 第一/最后预测目标距离中位数；
- 导航 SR、SPL、NE 仅作副指标，不作为第一阶段机制判定。

只有 `val_unseen` 最后预测目标 Hit@20 提升至少 3 个百分点，且 val_seen 不下降超过
3 个百分点，才进入热图头实验。该 512-episode 单 seed 结果仅是筛选，不是最终证据。

