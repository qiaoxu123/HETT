# Loss 消融计划

目的：回答 progress loss 和发布代码额外的 target-grid loss 是否真正改善闭环导航。
本分支只改变 loss 权重，不加入 recovery、grounding、多假设等结构，避免归因混乱。

## 四组独立训练

| 组别 | progress weight | target-grid weight | 含义 |
| --- | ---: | ---: | --- |
| released | 0.1 | 0.1 | 修复后发布代码基线 |
| paper-loss-only | 0.1 | 0.0 | 严格使用论文列出的三项 loss |
| no-progress | 0.0 | 0.1 | 检验 progress 监督 |
| neither | 0.0 | 0.0 | 检查两项是否存在互补作用 |

其余参数固定：seed 0、AdamW、learning rate 1e-4、20 epochs、micro-batch 2、
gradient accumulation 4、mturk、student feedback、grid 5、max action 20。
短跑仅检查真实数据前后向、有限梯度、保存和加载，不用于判断性能。

## 公平性

- 每组从相同预训练初始化独立训练，不能用同一 checkpoint 在评估时切换 loss 权重。
- 首先完整跑 seed 0。只有观察到值得保留的差异，才对候选和 released 基线扩展到至少 3 seeds。
- 主指标为 val_unseen SR、SPL、NE；同时报告 stage-2 动作数、错误停止率、曾命中后丢失率。
- 保存逐项原始 loss、加权贡献、梯度范数、逐 episode 轨迹和按地图 bootstrap 区间。
- `test_unseen` 仅在方案冻结后运行，不用于选择权重。

## 判断方式

某项 loss 数值小，不等于没用。只有“去掉它后，重复训练的配对闭环指标没有稳定退化”时，
才能认为它对当前实现贡献有限。若只改变训练稳定性而不改变 SR，也要单独报告。
