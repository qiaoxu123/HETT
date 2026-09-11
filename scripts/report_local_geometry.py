"""Summarize a completed, fixed-protocol probe; no fitting or model selection."""
import json
from pathlib import Path
import argparse
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_local_geometry import Cropper, SIDE, SEEDS
import numpy as np


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from multiagent.space import Pose4D
    plt.rcParams['font.family']='Noto Sans CJK JP'
    scores=json.loads((out/'metrics.json').read_text())
    results=scores['results'];coverage=scores['coverage']
    meta=json.loads((out/'evaluation_manifest.json').read_text())
    tm=json.loads((out/'train_seen_manifest.json').read_text())
    pred=dict(np.load(out/'predictions.npz'))
    target=np.load(out/'evaluation_labels.npy')
    rows=[]
    for mode in ['position_text','rgb_text','rgb_geo_text']:
        measures=[results[f'{mode}/normal/seed{s}'] for s in SEEDS]
        rows.append(dict(mode=mode, **{k:float(np.mean([r[k] for r in measures]))
                      for k in ['r5','r10','r20','mean_error_m','abstain_pct','inside_r20']},
                    forced_inside_r20=float(np.mean([r['forced_local']['inside_r20'] for r in measures]))))
    # Cases fixed by collection order, never selected by outcome.
    cropper=Cropper()
    count=min(4,len(target))
    fig,axes=plt.subplots(count,3,figsize=(11,3.2*count),squeeze=False,constrained_layout=True)
    for i in range(count):
        sample=meta['samples'][i]
        image,geom=cropper.get(sample['map'],Pose4D(*sample['pose']))
        rgb_pred=pred[f'rgb_text/normal/seed17/prediction'][i]
        geo_pred=pred[f'rgb_geo_text/normal/seed17/prediction'][i]
        for col in range(3):
            ax=axes[i,col]
            if col==0:ax.imshow(image,extent=[-SIDE/2,SIDE/2,SIDE/2,-SIDE/2])
            else:
                feature=geom[:,0 if col==1 else 3].reshape(7,7)
                im=ax.imshow(feature,extent=[-SIDE/2,SIDE/2,SIDE/2,-SIDE/2],cmap='viridis')
                fig.colorbar(im,ax=ax,shrink=.7)
            ax.scatter(0,0,c='white',edgecolors='black',marker='o',s=30,label='当前地点')
            ax.scatter(*rgb_pred,c='#2288ff',marker='x',s=80,label='RGB预测')
            ax.scatter(*geo_pred,c='#ff9900',marker='^',s=70,label='加几何预测')
            if (np.abs(target[i])<SIDE/2).all():
                ax.scatter(*target[i],facecolors='none',edgecolors='#00dd33',s=160,linewidths=2,label='真值（仅评分/展示）')
            ax.set(xlim=(-SIDE/2,SIDE/2),ylim=(SIDE/2,-SIDE/2),xlabel='东向相对位置（米）',ylabel='南向相对位置（米）')
            title=['局部 RGB','相对地面平均高度（米）','平均 log(1＋点密度)'][col]
            ax.set_title(f'{sample["map"]} · 状态 {sample["step"]}\n{title}')
        note='目标在图外' if not (np.abs(target[i])<SIDE/2).all() else '目标在图内'
        axes[i,0].text(.02,.02,note,transform=axes[i,0].transAxes,color='white',bbox={'facecolor':'black','alpha':.7})
    legend={}
    for ax in axes[:,0]:
        handles,labels=ax.get_legend_handles_labels()
        legend.update(zip(labels,handles))
    fig.legend(list(legend.values()),list(legend),loc='outside lower center',ncol=4)
    fig.suptitle('固定取前四个验证案例，不按成败挑选｜种子 17',fontsize=14)
    fig.savefig(out/'cases.png',dpi=140,bbox_inches='tight');plt.close(fig)
    group=np.array([s['map'] for s in meta['samples']]); names=sorted(set(group))
    permap=[]
    for name in names:
        mask=group==name
        a=np.stack([pred[f'rgb_text/normal/seed{s}'][mask]<=20 for s in SEEDS]).mean()*100
        b=np.stack([pred[f'rgb_geo_text/normal/seed{s}'][mask]<=20 for s in SEEDS]).mean()*100
        permap.append(dict(map=name,n=int(mask.sum()),rgb_r20=float(a),geo_r20=float(b),delta_pp=float(b-a)))
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,per_map=permap),indent=2)+'\n')
    lines=['# 局部点云几何：预注册小样本离线测试', '',
           '本次不是 Text2Loc/VLM-Loc 完整复现，不是闭环导航，不声称提升真实到达率。', '',
           '## 固定方案', '',
           f'- 固定 HETT 第1轮权重；{tm["episodes"]} 条训练任务、{coverage["train_states"]} 个训练状态。',
           f'- {len(target)} 条验证任务，{len(names)} 张未见地图；训练和验证地图不重叠。',
           '- 冒烟测试用过的验证任务按ID排除，排除规则与目标位置/成败无关。',
           '- 按地图均衡抽样，非整个验证集的自然比例。训练状态取决策后12/15/18步；验证取15步，提前停止的保留其终止状态。',
           '- 裁剪中心来自学生轨迹；49m×49m北向局部区域完全落在当前相机地面覆盖范围内。',
           '- RGB与BERT编码器冻结；三组小模型均训练30轮，种子17/29/43；固定末轮，不用验证结果挑模型。',
           '- 新定位模块不输入地标库、目标ID、地图ID、世界坐标、语义真值；训练真值仅用于监督，验证真值仅用于计分。',
           '- 用于生成轨迹的HETT仍保留其原有地标先验；这里只限制新增定位模块的输入。',
           '- 几何是预建点云地图的高度/密度栅格，不是实时LiDAR，也不是完整点级3D网络。密度包含同一平面位置的点，并未做传感器可见性过滤。',
           '- 输出49个候选位置或区域外；区域外总概率>0.5时保持观察位置，否则选局部最高分。本次不执行任何新飞行动作。', '',
           '## 结果（三个种子平均）', '',
           '| 模型 | R@5m | R@10m | R@20m | 平均误差m | 保持原位置/拒绝定位% |',
           '|---|---:|---:|---:|---:|---:|']
    base=results['stay_at_observation']
    lines.append(f'| 原地不改 | {base["r5"]:.2f} | {base["r10"]:.2f} | {base["r20"]:.2f} | {base["mean_error_m"]:.2f} | 100 |')
    for r in rows:
        lines.append(f'| {r["mode"]} | {r["r5"]:.2f} | {r["r10"]:.2f} | {r["r20"]:.2f} | {r["mean_error_m"]:.2f} | {r["abstain_pct"]:.2f} |')
    low,high=coverage['map_bootstrap_95ci_pp']
    lines += ['',f'几何相对RGB的 R@20m 差值：{coverage["paired_geo_minus_rgb_r20_pp"]:+.2f} 个百分点；按地图成组配对bootstrap的95%区间 [{low:+.2f}, {high:+.2f}]。',
              f'训练/验证目标在裁剪区域内比例：{coverage["train_inside_pct"]:.2f}% / {coverage["val_inside_pct"]:.2f}%。',
              f'验证状态已开始第二阶段：{np.mean([s["stage2_started"] for s in meta["samples"]])*100:.2f}%。', '',
              '## 分地图结果', '', '| 地图 | 样本数 | RGB R@20 | 加几何 R@20 | 差值pp |','|---|---:|---:|---:|---:|']
    for r in permap:lines.append(f'| {r["map"]} | {r["n"]} | {r["rgb_r20"]:.2f} | {r["geo_r20"]:.2f} | {r["delta_pp"]:+.2f} |')
    lines += ['', '## 解释边界', '',
              '- 小样本、早期HETT权重和有限场景；不能推断训练充分后的上限。',
              '- 区间是少量地图上的探索性不确定性估计，不是强显著性证明。',
              '- 真值在区域内的子集结果仅作诊断，不用于挑样本训练/发布主结果。',
              '- 不把“某种轻量几何输入没提升”推广成“点云或对象关系推理没用”。',
              '- 后续若调参，当前验证集即成为开发集，不能继续当作未触碰测试集。', '',
              '审计和全部分种子结果见 data_audit.json、gt_perturbation_audit.json、metrics.json；协议与文件哈希见 protocol.json、fit_protocol.json。', '',
              f'![比较]({out.resolve()}/comparison.png)', '', f'![固定案例]({out.resolve()}/cases.png)', '']
    geometry_path=out/'geometry_usage_audit.json'
    if geometry_path.exists():
        audit=json.loads(geometry_path.read_text())
        lines += ['## 补充诊断', '',
                  f'- 目标位于原相机完整地面覆盖范围内：{audit["full_camera_ground_footprint_target_coverage_pct"]:.2f}%；位于本次49m局部区内：{coverage["val_inside_pct"]:.2f}%。后者不等于相机的全部可见范围。',
                  '- 几何分支权重确实更新，移除几何会改变模型分数；负结果不是分支没接通。',
                  '- 目标在局部区内时，强制输出局部位置的R@20（仅诊断）：' + ', '.join(f'{r["mode"]}={r["forced_inside_r20"]:.2f}%' for r in rows),
                  '- 增强模型打乱语言后，三个种子预测位置改变数：' + str([r['positions_changed_by_shuffled_text'] for r in audit['checks']]), '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(rows=rows,per_map=permap),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    report(p.parse_args().out)
