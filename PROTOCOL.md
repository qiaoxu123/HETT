# RSRefSeg 2 × CityRefer 局部指代定位评测协议

## 问题定义

本实验不评估整图搜索或导航。对 CityNav/CityRefer 的每条语言指令，构造一个包含
目标轮廓及指令中具名地标轮廓的局部正射影像 ROI，只把 ROI 和完整英文指令交给
RSRefSeg 2。模型输出二值掩码；掩码外接框与 CityRefer 目标轮廓外接框比较，用来回答
“在目标—地标局部区域内，语言能否找到目标”。

这是 oracle-ROI 指代定位评测：ROI 的构造使用标注，因而结果不能解释成未知区域中的
端到端搜索能力，也不能解释成 HETT 导航 SR。

## 冻结的 ROI 规则

1. 从 `processed_descriptions.json` 读取该指令的具名地标；同图同名对象取轮廓面积最大者。
2. 对跨度不超过 80 m 的地标，ROI 包围其完整轮廓；道路等跨度超过 80 m 的大地标只纳入
   其轮廓上离目标最近的局部段。这样避免一条贯穿地图的道路把“局部区域”扩成整张城市图。
3. ROI 为正方形，边长至少 160 m，并在联合包围盒四周至少留 20 m；不设置上限，避免
   为了方便而裁掉远处/大型地标。
4. ROI 超出正射图边界的部分填黑；输出为 512×512 RGB。模型看不到 GT 轮廓或中心点。
5. GT mask 是目标的 CityRefer 平面轮廓。它是建筑/物体 footprint，不是逐像素遥感语义标注。
6. RSRefSeg 2 代码把 tokenizer 上限写成 128，但其 SigLIP2 文本编码器只能接受 64 token。
   输入按同一 SigLIP2 tokenizer 从尾部标准截断到 64 token；保留原文并报告截断比例。

## 指标

- 主指标：预测 mask 外接框的 mean Box IoU、Acc@0.25、Acc@0.50、Acc@0.75、预测框中心落入
  GT 框比例、非空预测比例。
- 辅助指标：footprint mask gIoU、Pr@0.5。由于 CityRefer 只提供平面轮廓，mask 指标仅作辅助。
- 分组：解析到的地标数量、是否含 surroundings、指令长度、ROI 尺度。
- 语言依赖对照：在同一 ROI 中存在另一个带指令的标注目标时，交换目标指令并分别评分；
  该对照单独报告，不混入主测试集指标。

## 数据使用纪律

- 用 `val_unseen` 的固定小样本做格式、坐标和显存检查，不根据其结果调模型权重。
- 上述规则冻结后，完整运行 `test_unseen` 5,281 条。
- 使用论文官方 RefSegRS checkpoint，零样本迁移到 CityRefer，不在 CityRefer 上训练或微调。
