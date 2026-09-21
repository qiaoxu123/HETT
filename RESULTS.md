# HETT token-mean 候选选择结果

实验保持候选选择器不变，只将已塌缩的 BERT `pooler_output` 替换为 masked mean
`last_hidden_state`。

| 划分 | Top-1 Hit@20 | Top-5 Recall@20 |
|---|---:|---:|
| val_seen | 23.12±0.00% | 50.97±0.21% |
| val_unseen | 22.69±0.00% | 52.55±0.71% |

结果与 pooler 实验几乎完全相同，未通过门槛。进一步用 FP32 直接检查四条不同指令：

- token 均值特征标准差均值 `1.44e-8`，最大 `3.00e-7`；
- pooler 标准差均值 `1.47e-8`；
- 两条左右相反指令的 token 均值最大绝对差只有 `4.77e-7`。

FP16缓存会把这些微小差异量化为完全相同，但FP32结果已经表明它们只是数值噪声。该20 epoch
HETT检查点的整个BERT输出已发生表示塌缩，并非只坏在pooler。因此共享这个语言编码器无法完成
目标关系选择，也可能解释原HETT对语言和地标语义利用不足。

下一项应使用未被HETT训练破坏的原始 `bert-base-uncased` token均值做同协议对照。如果原始BERT
明显更好，应修复HETT语言训练/冻结策略；若仍差，则需要token级交叉注意力而非单向量分类。

未运行 test_unseen。三 seed 汇总在 `runs/HETT_TOKEN_SELECTOR_AGGREGATE.md`。
