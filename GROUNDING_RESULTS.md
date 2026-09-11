# 局部视觉 Grounding 验证记录

日期：2026-09-11

## 当前发现

原模型并没有直接保留 49 个空间区域。它用 49 维语言向量去注意 Darknet 的 512 个通道，再把得到的 49 维向量映射为一个视觉 token。也就是说，空间布局虽然没有完全消失，但在进入跨模态 Transformer 前已经被强烈压缩。

## 实现

新模块默认关闭；显式开启 `--enable_region_grounding` 后：

- Darknet 的 7×7 共 49 个区域分别投影成 token，并加入二维位置编码。
- 49 个区域都进入原 Cross-Modal Transformer，候选目标和动作可直接关注局部区域。
- 语言查询对编码后的区域打分并汇聚当前视觉特征。
- 新增 50 类辅助任务：49 个可见区域，加 1 个“当前画面不可见”。
- 区域标签只用于训练 loss，不传给模型输入；`test_unseen` 中标签直接设为 ignore id。
- 目标投影使用生成当前 RGB crop 的同一套 raster/perspective 坐标变换。

## CPU 与数据验证

单元测试 11/11 通过：

- 保留全部 49 个区域并输出 outside 类。
- 输出同时依赖视觉和语言，两个方向梯度均非零。
- 打乱语言会改变区域分数。
- 修改被 mask 的 padding 文本不会改变结果。
- crop 中心、边界、outside 标签以及投影可见性正确。
- teacher 独立计数回归测试继续通过。

真实 student rollout 监督分布：

| 划分 | episode | 状态 | 画面内 | 画面外 |
| --- | ---: | ---: | ---: | ---: |
| train_seen | 512 | 10,734 | 63.49% | 36.51% |
| val_unseen | 256 | 5,376 | 46.88% | 53.12% |

审计输入来自既有冻结策略 rollout，不是为了新模块挑选的目标中心 crop。输入清单和轨迹均记录 SHA256。直接世界坐标投影还与真实 raster 投影做了交叉检查；训练 10,734 个状态仅 32 个 patch 边界标签不同，正式监督采用与实际 RGB crop 完全一致的 raster 版本。

结果文件：

- `/home/tenant2/Workspace/hett-experiments/03-grounding/runs/region_supervision_audit/region_supervision_audit.json`
- `/home/tenant2/Workspace/hett-experiments/03-grounding/runs/region_supervision_audit/region_supervision_distribution.png`

## 尚未完成

- 单卡真实前向/反向与显存检查。
- 完整训练后的区域准确率、visible/outside 分组结果。
- 同地标不同目标、整句打乱、视觉打乱、关闭模块的闭环导航消融。
- unseen SR/SPL 与逐 episode 失败案例。

当前证据证明实现结构真实保留了区域、标签覆盖合理且没有把 GT 喂给推理；尚不能证明导航效果提升。
