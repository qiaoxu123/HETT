#!/usr/bin/env python3
"""Aggregate paired metrics and draw final trajectories; never changes a policy."""
import sys,json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def paired(a,b):
    aa={tuple(e['episode_id']):e for e in a};bb={tuple(e['episode_id']):e for e in b}
    if aa.keys()!=bb.keys():raise ValueError('Episode sets differ across variants')
    out={};rng=np.random.default_rng(0)
    for metric,scale in [('success',100),('spl',100),('osr',100),('ne',1),('path_length_m',1)]:
        d=np.array([bb[i][metric]-aa[i][metric] for i in aa])*scale
        # Bounded working memory, 10k paired bootstrap replicates.
        means=np.concatenate([rng.choice(d,(1000,len(d)),replace=True).mean(1) for _ in range(10)])
        out[metric]={'delta':float(d.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}
    out['gained']=sum(bb[i]['success']>aa[i]['success'] for i in aa)
    out['lost']=sum(bb[i]['success']<aa[i]['success'] for i in aa)
    return out

def plots(out,epoch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2,rasterio,textwrap
    data=ROOT/'data';objects=json.loads((data/'cityrefer/objects.json').read_text())
    dest=out/'figures';dest.mkdir(exist_ok=True);index=[]
    for split in ('val_seen','val_unseen'):
        allp={v:{tuple(e['episode_id']):e for e in json.loads((out/f'epoch{epoch:02d}'/f'{split}_{v}.json').read_text())['episodes']} for v in 'ABCD'}
        categories={
            'joint_rescue':lambda a,b,c,d:not a['success'] and c['success'],
            'joint_regression':lambda a,b,c,d:a['success'] and not c['success'],
            'entered_then_left':lambda a,b,c,d:c['entered_then_left'],
            'stop_difference':lambda a,b,c,d:c['success']!=d['success'],
        }
        selected=[]
        for label,fn in categories.items():
            candidates=[i for i in allp['A'] if fn(*(allp[v][i] for v in 'ABCD'))]
            rng=np.random.default_rng(0)
            if candidates:
                for n in rng.choice(len(candidates),min(2,len(candidates)),replace=False):selected.append((label,candidates[n]))
        for j,(label,eid) in enumerate(selected):
            mapname,obj,desc=eid;instruction=objects[mapname][str(obj)]['descriptions'][desc]
            with rasterio.open(data/'rgbd'/f'{mapname}.tif') as f:bounds=f.bounds
            rgb=cv2.imread(str(data/'rgbd'/f'{mapname}.png'))
            rgb=cv2.cvtColor(cv2.resize(rgb,(1600,1600)),cv2.COLOR_BGR2RGB)
            fig,axes=plt.subplots(1,4,figsize=(16,5),constrained_layout=True)
            for ax,v in zip(axes,'ABCD'):
                e=allp[v][eid];xy=np.asarray(e['path_xy']);human=np.asarray(e['teacher_xy']);goal=e['goal_xy']
                ax.imshow(rgb,extent=(bounds.left,bounds.right,bounds.bottom,bounds.top),origin='upper')
                ax.plot(human[:,0],human[:,1],color='white',ls='--',lw=1,label='teacher (reference)')
                ax.plot(xy[:,0],xy[:,1],color='cyan',lw=2,label='executed')
                ax.scatter(*xy[0],c='lime',marker='o',s=35);ax.scatter(*xy[-1],c='red',marker='x',s=45)
                ax.add_patch(plt.Circle(goal,20,fill=False,color='yellow',lw=2))
                ax.scatter(*goal,marker='*',c='yellow',s=70)
                cloud=np.concatenate([xy,human,np.asarray(goal)[None]],axis=0)
                lo=cloud.min(0)-45;hi=cloud.max(0)+45
                ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal')
                ax.set_title(f"{v}: SR={int(e['success'])}, SPL={100*e['spl']:.1f}%\nNE={e['ne']:.1f}m; {e['termination']}")
                ax.tick_params(labelsize=6)
            fig.suptitle(f'{split} | {label} | {eid}\n'+textwrap.fill(instruction,140),fontsize=9)
            fn=f'{split}_{j:02d}_{label}.png';fig.savefig(dest/fn,dpi=130);plt.close(fig)
            index.append({'split':split,'category':label,'episode_id':list(eid),'instruction':instruction,'figure':fn})
    write(dest/'index.json',index)

def main():
    out=Path(sys.argv[1]).resolve();rows=json.loads((out/'results.json').read_text())
    manifest_path=out/'manifest.json'
    if not manifest_path.exists():
        manifest_path=out/'retry_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    budget=manifest.get('config',{}).get('epochs',max(r['epoch'] for r in rows))
    epoch=max(r['epoch'] for r in rows);current=[r for r in rows if r['epoch']==epoch]
    with (out/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    variants=list(dict.fromkeys(r['variant'] for r in current))
    comparisons={}
    for split in ('val_seen','val_unseen'):
        if len(variants)<2:
            continue
        es={v:json.loads((out/f'epoch{epoch:02d}'/f'{split}_{v}.json').read_text())['episodes'] for v in variants}
        comparisons[split]={f'{b}_minus_{a}':paired(es[a],es[b])
                            for i,a in enumerate(variants) for b in variants[i+1:]}
    write(out/'paired_comparisons.json',{'epoch':epoch,'splits':comparisons})
    lines=['# HETT Joint Goal/Trajectory 实验报告','',f'状态：已完成 {epoch}/{budget} 轮；本报告汇总 epoch {epoch}。',
        '', '各策略使用相同 checkpoint、验证 episode、seed、horizon 和成功半径。',
        '', '| Split | Group | N | SR% | SPL% | OSR% | NE m | Top5 Hit@20% | Plan ADE m | Plan FDE m | Stop precision | Switch rate | Path m | Eval s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'N/A' if v is None else f'{v:.3f}'
    for r in current:
        values=[r['split'],r['variant'],str(r['episodes'])]+[fmt(r[k]) for k in ['sr','spl','oracle_sr','ne']]+[fmt(r['heatmap_top5_hit20']*100)]+[fmt(r[k]) for k in ['selected_plan_ADE_m','selected_plan_FDE_m','stop_precision','goal_switch_rate','path_length_m','seconds']]
        lines.append('| '+' | '.join(values)+' |')
    lines+=['','## 配对分析（以 episode 为单位 bootstrap 10,000 次）','']
    for split,c in comparisons.items():
        for name,r in c.items():
            lines.append(f"- {split} {name}: ΔSR={r['success']['delta']:.2f}pp, 95% CI={r['success']['ci95']}; ΔSPL={r['spl']['delta']:.2f}pp; gained/lost={r['gained']}/{r['lost']}。")
    u={r['variant']:r for r in current if r['split']=='val_unseen'}
    lines+=['','## Unseen 与到达后失败','']
    for v,r in u.items():lines.append(f"- {v}: OSR−SR={r['osr_sr_gap_pp']:.2f}pp；进入成功区后离开 {r['entered_then_left']}；停止 TP/FP/FN={r['stop_TP']}/{r['stop_FP']}/{r['stop_FN']}；超时 {r['horizon_timeouts']}。")
    lines+=['','## 口径与限制',
        '- ADE/FDE：每个实际 rollout 状态的预测计划对人类剩余轨迹（按路程重采样）的误差；偏离人类轨迹时它是诊断代理，不等于最优恢复路线。',
        '- Top-K Hit@20 同时保存 initial-state 与 all-active-step 两种口径；all-step 因策略路径不同不能当作同状态纯排序因果对比。轨迹池 K=5，热图候选另报告 K=1/5/16/20。',
        '- Stop precision/recall/accuracy 按实际评估决策状态、GT 20m 成功半径计算；关闭 Stop 的 precision 为 N/A。Accuracy 可能被大量负例主导，必须同时查看 TP/FP/FN。',
        '- Switch rate：相邻有效计划的候选 cell ID 改变次数 / 可比较相邻步数。路径长度为 evaluator 的宏观状态路径长度；不是微动作连续路径长度。',
        '- 单 seed、单初始 checkpoint；episode bootstrap 不处理场景相关性，不代表跨 seed 稳健性。没有读取 Test-Unseen；Oracle 仅指标，无动作决策。',
        '', '## 下一步建议',
        '1. 根据候选覆盖、排序命中与选中轨迹 FDE 分别定位候选、重排和路径执行瓶颈。',
        '2. 根据停止 TP/FP/FN 判断 learned stop 是否工作；不要将 Stop accuracy 单独当作有效性证据。',
        '3. 若 OSR 高但 SPL/SR 低，检查目标切换、局部 waypoint 步长与到达后离开案例。',
        '', '## 复现',
        '配置、命令、源码哈希、初始权重哈希、每轮权重哈希、固定 episode IDs、耗时显存与梯度审计均保存在实验目录。']
    text='\n'.join(lines)+'\n';(out/'REPORT.md').write_text(text)
    (ROOT/'docs/HEATMAP_JOINT_GOAL_TRAJECTORY_EXPERIMENT.md').write_text(text)
    if epoch==budget and (out/'COMPLETE.json').exists() and all(v in variants for v in 'ABCD'):
        plots(out,epoch)
if __name__=='__main__':main()
