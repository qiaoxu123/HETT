# 前沿方案检索与下一版设计

## 可迁移思路

- GC-VLN：把指令表示成图约束，在3D场景图上求解；适合明确的关系和多参照物。
- OpenMap：开放词汇实例地图、指令到实例的对齐、结构与语义联合筛选。
- MapNav：在俯视语义地图上显式标注实例，再让VLM决策。
- SBFNav：保留多峰空间信念，并用语义—几何候选选择器消歧。
- HALO：无人机在线构建语言条件语义地图并探索。

来源：

- https://github.com/bagh2178/GC-VLN
- https://openmap-project.github.io/openmap.github.io/
- https://aclanthology.org/2025.acl-long.638/
- https://arxiv.org/abs/2609.05841
- https://github.com/KumarRobotics/HALO

## 本地证据

- 方向×距离候选的特权几何上限：val_unseen Hit@20 91.32%。
- 稠密热图（实验28）：val_unseen Top-1/Top-5 27.81/57.20%。
- token级地标交叉注意力（实验33）：23.68/54.58%，跨城市过拟合。
- Qwen单关系约束（实验34）：11.72/31.33%；轮廓方向修正也无明显收益。
- 已有Qwen视觉接地在160个目标可见帧上最高Hit@20为73.1%，但直接框回归存在大框和
  不可见时乱猜的问题。

## 下一版

保留实验28的Top-K多峰热图作为候选生成器，不再让语言模块直接回归坐标或大框。
对每个候选截取固定尺度、north-up地图块，叠加候选编号和地标轮廓；Qwen只输出候选编号
或`abstain`。最终分数融合几何热图分数、VLM候选分数和多视角一致性，再把选中候选还原成
20m高斯热图供动作头使用。

训练用train_seen构造一组1个正候选、4个同图难负候选；按城市留出内部开发集。先做静态
val_seen/val_unseen验证，要求unseen Top-1至少比27.81%提高3点且Top-5不低于57.20%，
并报告弃权精度。通过后才接动作头；仍不运行test_unseen。
