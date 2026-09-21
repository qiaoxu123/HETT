# HETT BERT pooler 候选选择结果

实验保持显式几何候选、20米概率损失和训练预算不变，只把实验29的从头训练 GRU 换成
20 epoch HETT 最佳检查点的冻结 BERT `pooler_output`。

| 划分 | Top-1 Hit@20 | Top-5 Recall@20 | 中位误差 |
|---|---:|---:|---:|
| val_seen | 23.12±0.00% | 51.10±0.10% | 34.67 m |
| val_unseen | 22.69±0.00% | 52.69±0.49% | 39.32 m |

结果未通过门槛。它比 GRU 选择器的 unseen Top-1 20.95%略高，但低于其 Top-5
54.41%，也明显低于稠密热图27.73% / 58.00%。

## 关键诊断

对缓存中的64,044条不同指令表示检查发现：

- 768维各特征跨文本的平均标准差：`1.62e-8`；
- 最大特征标准差：`1.17e-5`；
- 同一指令不同参考地标的余弦相似度：1.0；
- 随机不同文本的余弦相似度：1.0。

即该 HETT 检查点的 BERT `pooler_output` 已塌缩为几乎固定向量。原代码把它作为
`cls_hidden`，用于视觉注意力和候选语言调制，因此这些路径很可能没有获得有效的句子级区别。
这不代表 BERT 的 token 序列输出也塌缩；下一项应在同一权重上改用 masked mean token
embedding，保持选择器其他部分不变。

未运行 test_unseen。汇总见 `runs/HETT_BERT_SELECTOR_AGGREGATE.md`，表示诊断见
`runs/hett_bert_embedding_diagnostic.json`。
