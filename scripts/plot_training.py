"""Plot archived HETT runs separately and snapshot live training telemetry (CPU only)."""
import csv
import gzip
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'runs' / ('plots_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(parents=True)
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'figure.facecolor': '#fafbfc',
                     'axes.facecolor': '#ffffff', 'savefig.facecolor': '#fafbfc'})
C = ['#2563eb', '#e58b21', '#169c78', '#9c50bf']


def archived(name):
    with gzip.open(ROOT / 'reference_baseline' / name, 'rt') as stream:
        text = stream.read().replace('\r', '\n')
    pattern = r'^IL_loss ([\d.]+) direction_loss ([\d.]+) progress_loss ([\d.]+) goal_predict_loss ([\d.]+)'
    matches = list(re.finditer(pattern, text, re.M))
    rows = []
    for i, match in enumerate(matches):
        block = text[match.end():matches[i+1].start() if i+1 < len(matches) else len(text)]
        epoch = re.search(r'^\d+m \d+s[^\n]*\n+epoch (\d+)', block, re.M)
        if not epoch:
            raise ValueError('Missing primary epoch after loss in ' + name)
        total, direction, progress, goal = map(float, match.groups())
        target_weighted = total - direction - .1*progress - 2*goal
        assert target_weighted >= -0.001
        row = dict(epoch=int(epoch[1]), total=total, direction=direction, progress=progress,
                   goal=goal, target=target_weighted/.1,
                   direction_weighted=direction, progress_weighted=.1*progress,
                   goal_weighted=2*goal, target_weighted=target_weighted)
        # Only the first evaluation after this epoch, not repeated BEST RESULT blocks.
        for split in ['val_seen', 'val_unseen']:
            line = re.search(r'^' + split + r' , (.+)$', block[epoch.end():], re.M)
            assert line, (name, row['epoch'], split)
            for key, value in re.findall(r'(\w+): (-?[\d.]+)', line[1]):
                row[split + '_' + key] = float(value)
        rows.append(row)
    assert len({r['epoch'] for r in rows}) == len(rows)
    return rows


def export(name, rows):
    with (OUT / (name + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name, note):
    fig.text(.035, .018, note, fontsize=9, color='#4b5563')
    fig.tight_layout(rect=(.015, .05, .99, .94))
    fig.savefig(OUT / (name + '.png'), dpi=160)
    fig.savefig(OUT / (name + '.pdf'))
    plt.close(fig)


def curves(rows, name, title):
    x = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=17, fontweight='bold')
    for ax, key, label, color in zip(axes.flat, ['total','direction','goal','progress','target'],
        ['总 IL loss','方向 loss','目标坐标 loss','进度 loss','网格分类 loss（反推）'],
        ['#263445', C[0], C[2], C[1], C[3]]):
        y = [r[key] for r in rows]
        ax.plot(x,y,'o-',color=color,lw=2,ms=4)
        ax.set_title(label)
        ax.set_xlabel('日志 epoch（从 0 计）')
        ax.set_ylabel('日志尺度；分项未乘任务权重')
        ax.grid(alpha=.15)
        ax.annotate(f'{y[-1]:.4f}',(x[-1],y[-1]),xytext=(-45,12),textcoords='offset points',color=color)
    ax=axes.flat[-1]
    shares=np.array([[100*r[k+'_weighted']/r['total'] for r in rows]
                     for k in ['direction','progress','goal','target']])
    ax.stackplot(x,shares,labels=['方向 ×1','进度 ×0.1','坐标 ×2','网格 ×0.1'],colors=C,alpha=.85)
    ax.set(title='加权后占总 loss 的比例',xlabel='日志 epoch（从 0 计）',ylabel='占比（%）',ylim=(0,100))
    ax.legend(loc='lower left',fontsize=9)
    save(fig,name,'来源：'+('train_rep.log.gz' if name.endswith('a') else 'train_epoch12_20.log.gz')+
         '。网格损失由总量减去其余加权项反推，有四位小数舍入误差；数值占比不等于梯度贡献。')


run_a=archived('train_rep.log.gz')
run_b=archived('train_epoch12_20.log.gz')
assert len(run_a)==12 and len(run_b)==9, (len(run_a),len(run_b))
export('historical_a',run_a)
export('historical_b',run_b)
curves(run_a,'loss_a','历史记录 A：各项 loss 与加权占比（epoch 0–11）')
curves(run_b,'loss_b','历史记录 B：续训各项 loss（独立记录，epoch 11–19）')

fig,axes=plt.subplots(2,2,figsize=(13,8))
fig.suptitle('验证集表现：loss 下降不等于导航效果持续提升',fontsize=17,fontweight='bold')
for column,(rows,label) in enumerate([(run_a,'记录 A'),(run_b,'记录 B：续训')]):
    x=[r['epoch'] for r in rows]
    ax=axes[0,column]
    for key,title,color in [('sr','最终 SR',C[0]),('sr1','Stage 1 末端 SR',C[1]),('oracle_sr','Oracle SR',C[2])]:
        ax.plot(x,[r['val_unseen_'+key] for r in rows],'o-',ms=4,label=title,color=color)
    ax.set(title=label+' · Val Unseen 成功率',ylabel='成功率（%）',xlabel='日志 epoch（从 0 计）')
    ax.legend(fontsize=9)
    ax.grid(alpha=.15)
    ax=axes[1,column]
    for key,title,color in [('ne','最终 NE',C[0]),('stage1_ne','Stage 1 末端 NE',C[1])]:
        ax.plot(x,[r['val_unseen_'+key] for r in rows],'o-',ms=4,label=title,color=color)
    ax.set(title=label+' · 距离误差（越低越好）',ylabel='距离（米）',xlabel='日志 epoch（从 0 计）')
    ax.legend(fontsize=9)
    ax.grid(alpha=.15)
save(fig,'validation','两份日志分开绘制，不拼接为同一次运行。Stage 1 末端与最终指标是同轨迹比较，不是单阶段独立消融。')

live=ROOT/'runs/hett_baseline_fixed_20260911/checkpoints/batch_metrics.jsonl'
rows=[]
for line in live.read_text().splitlines():
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass  # Ignore a partially written final record.
export('current_batches',rows)
x=np.array([r['batch'] for r in rows])
y=np.array([r['recent_il_loss'] for r in rows])
epoch=rows[-1]['epoch']
assert len({r['epoch'] for r in rows})==1, 'Update live plot grouping after next epoch'
fig,axes=plt.subplots(2,2,figsize=(13,8))
snapshot=datetime.fromtimestamp(rows[-1]['time']).strftime('%Y-%m-%d %H:%M:%S')
fig.suptitle(f'当前原版结构基线 · 第 {epoch} 轮 · 截至 {snapshot}',fontsize=15,fontweight='bold')
ax=axes[0,0]
ax.plot(x,y,color=C[0],alpha=.25,lw=1,label='采样总 loss')
window=5
smooth=np.convolve(y,np.ones(window)/window,mode='valid')
ax.plot(x[window-1:],smooth,color=C[0],lw=2,label='最近 5 个采样点均值')
ax.set(title='总 loss：波动中看趋势',ylabel='加权 IL loss')
ax.legend(fontsize=9)
ax=axes[0,1]
g=np.array([r['grad_norm'] for r in rows])
ax.plot(x,g,color=C[3],lw=1.5)
ax.axhline(40,color='#dc5353',ls='--',label='裁剪阈值 40')
ax.set(title='ET 梯度范数（裁剪前）',ylabel='梯度范数',yscale='log')
ax.legend(fontsize=9)
ax=axes[1,0]
ax.plot(x,[r['peak_gpu_gib'] for r in rows],color=C[2],lw=2)
ax.set(title='本进程历史峰值显存',ylabel='PyTorch allocated（GiB）',ylim=(0,10))
ax=axes[1,1]
times=np.array([r['time'] for r in rows])
rate=np.diff(times)/np.diff(x)
ax.plot(x[1:],rate,color=C[1],lw=1.5)
ax.set(title='相邻记录之间的平均训练耗时',ylabel='秒 / 微批（batch=2）')
for ax in axes.flat:
    ax.set_xlabel('第 1 轮内的微批序号')
    ax.grid(alpha=.15)
save(fig,'current','双向注意力关闭。每约 100 微批采样一次；loss 是最近两个 rollout 的均值，不是整段均值。\n当前尚无完整轮次分项 loss；不能与历史逐轮 loss 直接比较。显存非整卡占用，梯度范数也不是各任务梯度贡献。')
(OUT/'README.txt').write_text('Historical logs are independent runs. Target loss is inferred using original weights 1/.1/2/.1.\n'
    'Current run is the fixed baseline WITHOUT bidirectional interaction. No full-epoch per-task losses yet.\n'
    'Snapshot: '+snapshot+'\nCSV files contain extracted data. PNG and PDF are standalone exports.\n')
print(json.dumps({'output_dir':str(OUT),'historical_epochs':[len(run_a),len(run_b)],
    'current_samples':len(rows),'last_batch':int(x[-1]),'total_batches':rows[-1]['batches'],
    'early_loss_mean':float(y[:5].mean()),'recent_loss_mean':float(y[-5:].mean()),
    'peak_gpu_gib':rows[-1]['peak_gpu_gib']},ensure_ascii=False))
