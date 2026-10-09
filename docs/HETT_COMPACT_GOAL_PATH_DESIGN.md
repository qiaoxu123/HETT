# HETT 精简版：Goal Soft First，Path Optional

分支：`2027-CVPR/hett-compact-goal-path-soft`；从 `2027-CVPR/hett-sequential-backward` (`501f25af`) 派生。

## 当前结论与优先级

1. **主线仅训练 Goal Soft Selector**：冻结已有 HETT Spatial Belief (Top-20) 与 BERT / Darknet / ET 主干，保留共享候选网络的目标排序分数。训练目标为 SBF 风格距离软标签交叉熵；对应 `trajectory_candidate_loss_weight=0`，轨迹模式不参与训练目标。
2. **路径选择降级为独立探索分支**：现有三条 Anchor 只是 ±0.30 rad/直行的 50m 转向选择；50m 内三条全部直达同一目标，C 与 B 天然非常相似。**没有障碍物、候选区域 RGB 或其他可行性证据时，轨迹选择没有充分理由超越直达 Waypoint**。不应把它视为已验证的贡献。
3. **不再把人类局部路线当成强制最优控制目标**：可选 Path Loss 仍用 `0.7×human-local-error + 0.3×goal-distance`，只是待验证的研究原型，容易被人类轨迹绕行（用户报告平均路径约为直线距离 2 倍）影响 SPL；当前主线**关闭**该损失。
4. **RGB 口径**：ET 的传统视觉编码支路使用了 Darknet 处理的图像，但当前 `Spatial Belief` 的热图主要来自地图、语言和地标几何，`CandidateRelationSelector` 直接读取的是 Heatmap 空间特征、语言 token、地标几何、位姿/历史；**没有候选位置对应的 RGB 裁剪/语义特征输入**。不能把这个 Selector 宣称为直接 RGB-aware。要真正测试 RGB 增益，须另做严格对照。
5. **历史疑似停滞错误**：检查此分支 `multiagent/agent.py`，旧轨迹执行分支采用 `old_xy.dist_to(poses[i].xy)` 和 `poses[i].yaw - old_pose.yaw`；并未找到 `old_pose.y` 被误当航向的写法。若本地出现须核对 `git rev-parse HEAD`，不要混淆旧分支。
6. **Checkpoint 未指定**：不会默认为任何权重。用户需提供已核实来源的 `HETT_INITIAL_CHECKPOINT`（绝对路径）和匹配的 `HETT_BASE_ARGUMENTS`。Runner 会记录 checkpoint SHA-256，并进行模型权重兼容检查，无法读取则失败退出。

## 模型信息流

```text
语言 + 地标 + 当前/历史地图 + 位姿
      |
      v
[冻结 HETT Spatial Belief] -> Top-20 Goals
      |
      v
[轻量 Goal Soft Selector] -> Goal*
      |
      v
[原有 bounded_heatmap_step，统一 50m / 20步] -> 更新观测 -> 重规划
```

Goal Loss：`L_goal=-sum softmax(-distance(G_i,GT)/20m) * log P(G_i)`。

选 Goal 后，使用固定有界 Waypoint 控制器。暂不训练独立 Stop、不预测人类末端方向、不让学习轨迹接管动作。

**可选探索**：`configs/experiments/hett_compact_optional_path_1e.json` 使用 Goal + Path 两种软标签，Path Loss 权重 0.5 并增加 C 路径模式评估。它不是默认模型，也不是已验证性能提升。

## 公平对照

主实验：`configs/experiments/hett_compact_goal_path_1e.json`，1 epoch，三个验证变体：

| 变体 | Goal 选择 | 执行 | 用途 |
| --- | --- | --- | --- |
| `A_refined_waypoint` | HETT Heatmap Top-1 **local soft-argmax** | 50m bounded | 对齐历史精细坐标基线 |
| `A_nms_waypoint` | HETT Heatmap Top-1 **NMS cell center** | 50m bounded | 与 B 的候选坐标一致 |
| `B_goal_soft` | 共享 Selector 在同一 NMS Top-20 内选最优 | 50m bounded | 只比较候选排序的价值 |

这里 A_refined 对齐的是**执行坐标提取方法**，并不等于先前报告的某个 SR 数字；权重与 Episode ID 仍必须一致。B−A_nms 才是更直接的候选重排差异；B−A_refined 同时含精细坐标提取差异。

可选的 C 使用与 B 相同的 Goal Selector、动作上限（每次 50m）、horizon（20 个决策步），但因为仅有 ±17.2° 的动作偏角而没有障碍证据，可能更差。把它当作否定性对照，不要误将 B/C 当成不同网络训练目标的单一消融：独立验证 Path Loss 的贡献还需同 seed、同初始化、同训练预算分别训练 Path weight 0 与 0.5 的 checkpoint。

对所有实验记录 val_seen / val_unseen 的 SR、OSR、SPL、NE、实际航程、目标距离分桶、进入成功区后的再次离开率。主判据：Seen 进步不能以 Unseen SR/SPL 明显退化为代价。

## 运行

```bash
git fetch origin
git switch 2027-CVPR/hett-compact-goal-path-soft
git pull
export HETT_INITIAL_CHECKPOINT=/absolute/path/to/verified_initial_checkpoint.pt
export HETT_BASE_ARGUMENTS=/absolute/path/to/matching_base_arguments.json
test -f "$HETT_INITIAL_CHECKPOINT" && test -f "$HETT_BASE_ARGUMENTS"

# 先两次更新的 smoke，再 1 epoch 训练与 full validation
CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_goal_trajectory_experiment.py \
  --config configs/experiments/hett_compact_goal_path_1e.json \
  --output artifacts/compact_goal_soft_smoke --smoke

CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_goal_trajectory_experiment.py \
  --config configs/experiments/hett_compact_goal_path_1e.json \
  --output artifacts/compact_goal_soft_epoch1
```

沿用 Sequential Backward。不同运行使用不同输出目录。执行结果尚待本地 GPU 验证；当前 CI 是 CPU 回归/静态检查，不等于 CityNav 实验成功。

## 再进一步的研究门槛

只有当 Goal Soft Selector 在 Seen/Unseen 上都通过验证，才考虑直接为候选加入**可观察的 RGB 局部特征**，再考虑 Path Ranking。没有额外路径信息时，不建议继续增加 Anchor 数、轨迹残差、复杂 Stop Head 或调整 Human Loss 权重，避免扩大调试变量。
