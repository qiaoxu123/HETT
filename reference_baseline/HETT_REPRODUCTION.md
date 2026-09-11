# HETT 复现指南（HETT Reproduction Guide）

复现对象：[crotonyl/HETT](https://github.com/crotonyl/HETT)（AAAI 2026 oral，
History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language
Navigation）。本文档记录了在本仓库 `external/HETT/`（vendored 原仓库）上
从零完成"官方检查点评测 + 训练流程"复现的全部步骤，所有数字与
`POINTCLOUD.md` / `EXPERIMENTS.md` E77 一致。

复现结果摘要：官方检查点 test_unseen SR **26.57**（论文引用 25.49）；
我们的训练（12 epochs，单卡）test_unseen SR **27.97**，超过官方检查点。

---

## 0. 前置条件

- **硬件**：单卡 NVIDIA RTX 5090 32GB（官方为 4× RTX A5000 24GB；本指南用
  梯度累积做单卡等效）
- **软件**：`AirVLN39` conda 环境（Python 3.9, torch 2.8+cu128, CUDA 12.8——
  官方代码是 torch 2.2/CUDA 11.3，不能在 sm_120 上直接运行，权重可加载）
- **数据**（本仓库已具备）：
  - `data/cityrefer/`（objects.json + processed_descriptions.json）
  - `data_refined/refined_citynav/processed_citynav/`（HETT 发布的 refined 标注）
  - `data/rgbd/`（0.1 m 正射栅格，PNG+TIFF）
- **官方发布物**（需下载一次，~2.6 GB）：
  - Dropbox（README 中链接）包含 `checkpoint/best_val_unseen`、
    `darknet/{best.pt, yolo_v3.cfg}`、`refined_citynav/`

## 1. 下载并解压官方发布物

```bash
mkdir -p /home/tenant2/Workspace/DATA/hett
curl -L -o /home/tenant2/Workspace/DATA/hett/hett_release.zip \
  "https://www.dropbox.com/scl/fo/ua3kn6mw3adn2hcsdikt3/AEdoRoo-OeXnbrpeOpk4BcQ?rlkey=v5rlqiqa1k9ml8isinpv7glgk&st=ahseq0r2&dl=1"
cd /home/tenant2/Workspace/DATA/hett && unzip -q hett_release.zip
# 注意：checkpoint/best_val_unseen 本身又是一个 zip（老 DDP 格式），见下一步
```

## 2. 解码官方检查点（老 DDP 格式 → 标准 torch 格式）

官方检查点是 torch 分布式检查点（`data.pkl` 元数据 + `data/0..N` 分片，
`version: 3`）。torch 2.8 无法直接加载，用自定义 `persistent_load`
逐分片重建：

```python
import pickle, torch, os
p = '/path/to/checkpoint/best_val_unseen'
SIZES = {torch.FloatStorage: 4, torch.DoubleStorage: 8, torch.LongStorage: 8,
         torch.IntStorage: 4, torch.HalfStorage: 2, torch.BFloat16Storage: 2,
         torch.ByteStorage: 1, torch.CharStorage: 1, torch.BoolStorage: 1,
         torch.ShortStorage: 2}

def persistent_load(pid):
    # pid = ('storage', torch.FloatStorage, [shard_index, 'cuda:0', numel])
    if pid[0] == 'storage':
        cls, meta = pid[1], pid[2]
        shard, numel = meta[0], (meta[-1] if isinstance(meta[-1], int) else None)
        path = f'{p}/data/{shard}'
        if numel is None:
            numel = os.path.getsize(path) // SIZES.get(cls, 4)
        return cls.from_file(path, shared=False, size=numel)
    raise NotImplementedError

u = pickle.Unpickler(open(f'{p}/data.pkl', 'rb'))
u.persistent_load = persistent_load
state = u.load()          # {'lang_model': {...}, 'vision_model': {...}, 'vln_model': {...}}
torch.save(state, f'{p}/../hett_best_val_unseen.pt')
```

结果：三个子模型 **键名 100% 匹配** vendored 代码
（`lang_model` = CustomBERTModel，`vision_model` = Darknet，
`vln_model` = ET，各含 epoch/state_dict/optimizer）。

## 3. torch 2.2 → 2.8 适配补丁（external/HETT 内）

| 文件 | 修改 |
|---|---|
| `multiagent/models/vln_model.py` | BERT 路径 `'/cver/xcding/...'` → `'bert-base-uncased'` |
| `multiagent/main.py` | 同上（tokenizer 路径） |
| `multiagent/agent.py` | 同上 |
| `multiagent/dataset/episode.py` | `from direction.cityreferobject` → `from multiagent.cityreferobject`；`direction.space` → `multiagent.space` |
| `multiagent/defaultpaths.py` | `PROJECT_ROOT = Path("..")` → `Path(__file__).resolve().parent.parent`（绝对路径，避免 cwd 依赖） |
| AirVLN39 环境 | `pip install 'affine==2.4.0'`（affine 3.0 与 rasterio 1.4.3 不兼容，`raster.index` 全报 Invalid inputs）；`pip install rasterio shapely` |

## 4. 数据接线（符号链接农场）

在 `external/HETT/` 下建 `data/` 和 `weights/`：

```bash
mkdir -p data weights
ln -sfn <repo>/data/cityrefer data/cityrefer
ln -sfn <repo>/data_refined/refined_citynav/processed_citynav data/processed_citynav
ln -sfn <repo>/data/rgbd data/rgbd
cp /path/to/hett/darknet/{best.pt,yolo_v3.cfg} weights/
# 注意：data/cityrefer/processed_descriptions.json 在原仓库里是二级相对符号链接，
# 经本符号链接农场访问会断链——用绝对链接重建：
ln -sfn <repo>/data_refined/refined_citynav/cityrefer/processed_descriptions.json \
      data/cityrefer/processed_descriptions.json
mkdir -p checkpoints/multi   # 训练保存目录
```

## 5. 官方协议评测（零训练复现）

```bash
cd external/HETT/multiagent
python main.py --mode eval \
  --checkpoint <repo>/../DATA/hett/checkpoint/hett_best_val_unseen.pt \
  --feedback student --batch_size 2 --grid_size 5 \
  --move_iteration 10 --max_action_len 20 --world_size 1 \
  --altitude 50 --seed 0
```

结果（官方协议：feedback student / move_iteration 10 / max_action_len 20 /
grid 5）：

| split | SR | OSR | SPL | NE |
|---|---|---|---|---|
| val_seen | 29.72 | 47.77 | 25.08 | 38.35 |
| val_unseen | 18.32 | 34.30 | 15.17 | 53.23 |
| test_unseen | **26.57** | 46.94 | 21.56 | 42.70 |

论文引用数字（HTNav* 行）为 25.49——**论文数字复现成功**。

## 6. 训练复现（单卡等效官方 4 卡）

论文训练配置：4× A5000、20 epochs、每卡 batch 2（有效 batch 8）、
AdamW、lr 1e-4、grid size 5、teacher+student 双 rollout。

单卡适配：batch 2 × **梯度累积 4**（等效 batch 8，且 env 每步行为与官方
单卡 batch 2 完全一致）。补丁：

- `multiagent/agent.py` 的 `train()`：每 batch 累积梯度，每 `grad_accum`
  步执行 clip_grad_norm_(40) + 三个优化器 step
- `multiagent/parser.py`：新增 `--grad_accum`（默认 1）

```bash
cd external/HETT/multiagent
python main.py --mode train --feedback student --altitude 50 \
  --learning_rate 1e-4 --batch_size 2 --grad_accum 4 --optim adamW \
  --train_trajectory_type mturk --epochs 20 --save_every 1 \
  --log_every 1 --eval_every 1 --grid_size 5 \
  --move_iteration 10 --max_action_len 20 --world_size 1 --seed 0 \
  --log_dir log
```

- 损失权重保持发布代码原值（`ml_loss = 1*direction + 0.1*progress +
  2*goal_predict + 0.1*target_predict`——官方检查点即此配置训出）
- 每 epoch 保存 `checkpoints/multi/latest`；val 提升时自动保存
  `checkpoints/multi/best_val_unseen`
- **断点续训**：被杀/重启后加 `--checkpoint checkpoints/multi/latest`
  即可（加载权重+优化器+epoch 号，从下一 epoch 继续）
- 速度：单卡 ~6-7 小时/epoch；官方 4 卡约 1.5-2 小时/epoch

## 7. 训练结果与最终评测

### 7a. 第一次运行（epoch 0-11，best-val 自动选择）

曲线（val_unseen SR）：13.68 → 16.57 → 16.91 → 17.20 → 16.35 →
17.69 → 18.21 → 19.54（epoch 11 达峰，best 19.54）。

### 7b. 完整 20 epoch（论文协议，2026-08-20 完成）

从 `checkpoints/multi/latest` 断点续训至 epoch 19（与 7a 合起来正好
20 个 epoch 迭代）。resume 后曲线（val_unseen SR）：

```
epoch: 11   12   13   14   15   16   17   18   19
SR:    18.54 18.32 13.31 16.24 16.83 15.39 15.80 14.13 15.50
```

**best（19.54，epoch 11）全程未被打破**——模型在 epoch 11 达峰后
平台/过拟合，证明 12-epoch 提前收尾在性能上是正确决策；跑满 20
epoch 的价值是严格对齐论文协议。

### 7c. 最终评测（best 检查点 = epoch 11，官方协议）

```bash
python main.py --mode eval \
  --checkpoint checkpoints/multi/best_val_unseen \
  --feedback student --batch_size 2 --grid_size 5 \
  --move_iteration 10 --max_action_len 20 --world_size 1 \
  --altitude 50 --seed 0
```

| split | 我们训练（epoch 11 best，20-epoch 完整） | 官方检查点（epoch 25/26） |
|---|---|---|
| val_seen SR | 29.43 | 29.72 |
| val_unseen SR | **19.54** | 18.32 |
| test_unseen SR | **27.97** | 26.57 |
| test_unseen NE / OSR | 41.80 / **50.14** | 42.70 / 46.94 |

**完整 20-epoch 协议下，我们的检查点仍以 12 个 epoch 的权重超过官方
26 个 epoch 的检查点（val_unseen +1.22，test +1.40）。**

## 8. 踩坑记录

1. 老 DDP 检查点：`torch.load` 不支持目录；`pickle.load` 的
   `persistent_load` 关键字在 Python 3.14 被移除（用 3.9 的 Unpickler）
2. 分片文件是**裸存储**（numel × itemsize == 文件大小），不是
   torch.save 的 tensor
3. affine 3.0 / rasterio 1.4.3 不兼容：所有 `raster.index` 报
   "Invalid inputs"，降级 affine==2.4.0
4. 二级相对符号链接（processed_descriptions.json）经符号链接农场会断链
5. 官方 eval 的 batch_size 必须为 2（env 按此设计）；batch 8 会 OOM
   （32GB 单卡），梯度累积是正确等效
6. 训练长任务防杀：`--checkpoint latest` 断点续训 + 每 epoch 保存
