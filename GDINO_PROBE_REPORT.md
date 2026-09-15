# Grounding DINO 固定 30 帧测试

## 结论

Grounding DINO Tiny 可以在当前 RTX 5090 上运行并检测部分物体，但这个小样本中不能可靠区分指定目标。尚不能直接替代目标预测器。

## 设置

- 官方模型 IDEA-Research/grounding-dino-tiny，revision a2bb814dd30d776dcf7e30523b00659f4f141c71；模型说明：https://huggingface.co/IDEA-Research/grounding-dino-tiny 。
- transformers 4.44.2、torch 2.8.0+cu128、FP32、官方 processor 默认图像预处理（原图 224×224，内部缩放不增加真实细节）。独立 .venv，原训练环境 transformers 仍为 4.32.1。
- 复用轮廓实验固定 30 帧：val_unseen 四地图，10 个不同目标，各取人工轨迹起中末。不是模型 rollout，不是随机独立 30 个目标。
- 简短类别词从现有指令解析的 target 文本中用固定词表提取；另一组使用该 target 短语。没有使用真实 object_type 标签。15 帧两种提示恰好相同，因此不能将本实验当作充分的语言消融。
- 检测阈值固定 box=0.35，text=0.25，均未针对结果调参。选最高分框，不以 GT 选择候选。不调用 SAM、不训练、不读取全局视觉特征、不调用全图检测缓存。
- GT 轮廓只在推理后投影用于评分。GT 多边形与视野相交面积至少 25 像素且至少保留原面积 25% 归为 in_view；面积 <1 像素归为 out_of_view；其余单列。几何覆盖不等于人工确认可见，未标注遮挡。
- 以视野内 GT 多边形的外接矩形评价 IoU，最高分预测框 IoU>=0.5 为命中。部分可见的大建筑可能得到偏乐观的外接框评分，因此另报基本完整可见子集。

## 结果

17 帧 in_view，12 帧 out_of_view，1 帧 partial_small 不计入这两组。

|指标|类别词|目标短语|
|---|---:|---:|
|in_view 最高分框命中|4/17 (23.5%)|4/17 (23.5%)|
|in_view 任一候选框命中（仅诊断）|4/17|4/17|
|GT 面积至少 90% 在画面的子集命中|2/14|2/14|
|out_of_view 仍输出候选|9/12|10/12|
|GPU forward 平均耗时|0.079 秒|0.048 秒|

耗时不含预处理、加载、后处理，也不是严谨吞吐测试；类别词含首次 warm-up，不能据此比较速度。
目标不在视野时输出同类物体对普通检测器不一定是错误，但如果直接作为导航目标，就是误引导风险。
未验证全文空间关系理解、实际可见性误报率、SAM 轮廓质量、世界坐标精度或导航成功率。

## 图与复现

`runs/gdino_fixed30/predictions.json` 保存全部提示、框、分数与评分；summary.json 和 additional_checks.json 保存统计。
examples.png 展示短语组 IoU 最高、中位、最低的 in_view 帧；all_frames.png 保留全部；comparison.png 显示主结果。
绿色是 GT 外接框（只用于评分），红色为最高分预测，橙色为其他候选（图中最多五个，评分使用所有候选）。
运行：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/test_gdino_frames.py`；随后运行 scripts/report_gdino.py。
4 项评分/文本检查通过。独立 worktree 10-grounding-dino，分支 codex/verify-grounding-dino；通过共享 GPU 锁运行，没有改动或恢复基线训练。

下一步应先针对漏检小物体、相似物体混淆做更充分的固定样本测试，而不是立即接入导航或声称方法无效。
