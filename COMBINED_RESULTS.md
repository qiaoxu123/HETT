# Grounding + Recovery 组合验证记录

日期：2026-09-11

组合分支只合并两项已有 CPU 证据的实现：

- teacher 独立轨迹计数修复：`8615ec7`
- 可恢复粗细阶段控制：`d3247be`
- 49 区域语言 grounding：`eb85c19`
- GT-free 评测和 checkpoint 保存修复：`a60a4f1`

当前组合 CPU 测试 28/28 通过，包括 teacher、实验卫生、队列命令、恢复状态机、区域投影、整网 ET 前后向和语言/视觉依赖。两套参数可以同时解析，源码通过语法检查。

尚未完成真实组合收益验证。GPU 队列会先训练 grounding 短跑 checkpoint，再用同一个 checkpoint 分别关闭和开启 recovery；这样组合差异只来自控制机制，不会偷偷换权重。

通过标准不是“能运行”：需要逐 episode 配对比较 SR/SPL、曾命中后丢失、恢复次数、动作数和停止误判。完整 unseen 结果出来前不能声称互补。
