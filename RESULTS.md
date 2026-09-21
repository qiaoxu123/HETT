# RSRefSeg 2 × CityRefer test_unseen 结果

零样本 oracle-ROI 指代定位，完整 `test_unseen` 5,281 条。ROI 构造和限制见 `PROTOCOL.md`。

| 指标 | 结果 |
|---|---:|
| Mean Box IoU | 3.17% |
| Box Acc@0.25 | 1.38% (73/5281) |
| Box Acc@0.50 | 0.11% (6/5281) |
| 预测框中心命中 | 20.66% |
| Footprint mask gIoU | 1.53% |
| Footprint mask Pr@0.5 | 0.00% |

结论：论文的 RefSegRS checkpoint 在 CityRefer 局部目标—地标区域上不能可靠找到语言所指目标；
Acc@0.50 只有 0.11%，不能直接作为 HETT 的目标检测器使用。

## 同图语言切换对照

192 个 ROI 均包含另一个有真实指令和 GT 的可见目标。把输入文本从原目标切换到该目标后：

- 新目标 Box Acc@0.25：0.52%；Acc@0.50：0.00%。
- 两条指令都在同图正确定位（Box IoU≥0.25）：0.52%。
- 换文本前后预测 mask 平均重叠：68.08%；预测框中心平均移动 8.4px。

这说明输出会随文本变化，但变化没有切换到语言指定的另一个可见目标；失败不是单纯“完全忽略文本”，
而是 RefSegRS 的短模板/对象分布无法零样本迁移到 CityRefer 的长关系指令和建筑场景。

## 数据接口限制

- 26/5,281（0.49%）指令超过 SigLIP2 的 64-token 上限，按 tokenizer 从尾部截断。
- 260/5,281（4.92%）至少一个地标名无法解析成同图具名对象；这部分单列，不影响主结果保留全测试集。
- GT 是 CityRefer 平面 footprint，不是逐像素遥感语义 mask，因此 Box 指标是主指标。
