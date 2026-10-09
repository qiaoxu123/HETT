#!/usr/bin/env python3
"""Report fixed one-epoch relation fusion; all GT access is offline diagnostic."""
import argparse,csv,gzip,json,hashlib,shutil,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from report_joint_goal_trajectory import paired

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def baseline(split):
    with gzip.open(ROOT/'reports/arrival_aware/joint'/f'{split}_A_original_episodes.csv.gz','rt') as f:
        return [dict(episode_id=json.loads(e['episode_id']),success=float(e['success']),spl=float(e['spl']),
                     osr=float(e['oracle_success']),ne=float(e['ne']),path_length_m=float(e['trajectory_lengths']),
                     path_xy=json.loads(e['path_xy'])) for e in csv.DictReader(f)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);opt=ap.parse_args()
    run=opt.run.resolve();dest=ROOT/'reports/relation_observation';dest.mkdir(exist_ok=True,parents=True)
    if not (run/'COMPLETE.json').exists(): raise RuntimeError('Full epoch/evaluations not complete')
    for name in ['manifest.json','checkpoint_load.json','data.json','train_epoch01.json','results.json','results.csv','paired_comparisons.json','COMPLETE.json']:
        shutil.copyfile(run/name,dest/name)
    rows=json.loads((run/'results.json').read_text());comparison={};classifiers=[];initial={};replay_checks={};failure_groups={}
    for split in ['val_seen','val_unseen']:
        base=baseline(split);base_by_id={tuple(e['episode_id']):e for e in base}
        initial_eps=json.loads((run/'initial_quick/epoch00'/f'{split}_relations_no_stop.json').read_text())['episodes']
        with gzip.open(run.parent/'collector_baseline_check'/f'{split}.json.gz','rt') as f:
            quick_base={tuple(e['episode_id']):e for e in json.load(f)['episodes']}
        diffs=[abs(e['ne']-quick_base[tuple(e['episode_id'])]['ne']) for e in initial_eps]
        path_errors=sum(not np.array_equal(np.array(e['path_xy']),np.array(quick_base[tuple(e['episode_id'])]['path_xy'])) for e in initial_eps)
        full_initial=json.loads((run/'initial_full/epoch00'/f'{split}_relations_no_stop.json').read_text())['episodes']
        full_ne=max(abs(e['ne']-base_by_id[tuple(e['episode_id'])]['ne']) for e in full_initial)
        full_path_errors=sum(not np.array_equal(np.array(e['path_xy']),np.array(base_by_id[tuple(e['episode_id'])]['path_xy'])) for e in full_initial)
        initial[split]=dict(episodes=len(initial_eps),max_ne_difference=max(diffs),path_mismatches=path_errors,
                            full_episodes=len(full_initial),full_max_ne_difference=full_ne,full_path_mismatches=full_path_errors)
        if path_errors or full_path_errors: raise RuntimeError('Initial baseline trajectories do not reproduce')
        reference=json.loads((run/'epoch01'/f'{split}_relations_no_stop.json').read_text())['episodes']
        by_id={tuple(e['episode_id']):e for e in reference}
        stop_eps=json.loads((run/'epoch01'/f'{split}_relations_stop95.json').read_text())['episodes']
        prefix_errors=[]
        for e in stop_eps:
            other=by_id[tuple(e['episode_id'])]
            if not np.array_equal(np.asarray(e['path_xy']),np.asarray(other['path_xy'])[:len(e['path_xy'])]):
                prefix_errors.append(e['episode_id'])
        replay_checks[split]=dict(episodes=len(stop_eps),stop_prefix_mismatches=prefix_errors)
        if prefix_errors: raise RuntimeError('Stop online path differs from no-stop causal prefix')
        for variant in ['relations_no_stop','relations_stop95','relations_disabled']:
            src=run/'epoch01'/f'{split}_{variant}.json';payload=json.loads(src.read_text())
            comparison[f'{split}_{variant}_vs_original']=paired(base,payload['episodes'])
            groups=dict(success=0,entered_then_left=0,never_entered=0,
                        never_entered_no_correct_selected_goal=0,never_entered_despite_correct_selected_goal=0,
                        selected_goal_hits=0,selected_goal_states=0,episodes_with_reentry=0)
            for e in payload['episodes']:
                target=np.asarray(e['goal_xy'])
                inside=np.linalg.norm(np.asarray(e['path_xy'])-target,axis=1)<=20.
                selected=[np.linalg.norm(np.asarray(s['predicted_goal_xy'])-target)<=20.
                          for s in e['navigation_steps']]
                groups['selected_goal_states']+=len(selected)
                groups['selected_goal_hits']+=sum(selected)
                if e['success']: groups['success']+=1
                elif inside.any(): groups['entered_then_left']+=1
                else:
                    groups['never_entered']+=1
                    groups['never_entered_despite_correct_selected_goal' if any(selected)
                           else 'never_entered_no_correct_selected_goal']+=1
                entries=sum(not inside[i-1] and inside[i] for i in range(1,len(inside)))
                groups['episodes_with_reentry']+=int(entries>int(not inside[0]))
            failure_groups[f'{split}_{variant}']=groups
            # Complete step records stay in the local run/epoch01/*.json.
            # Version compact per-episode CSVs rather than duplicate large rollouts.
            compact=[]
            for e in payload['episodes']:
                nav=e['navigation_steps'];target=np.asarray(e['goal_xy'])
                selected=[float(np.linalg.norm(np.asarray(step['predicted_goal_xy'])-target)) for step in nav]
                compact.append(dict(episode_id=json.dumps(e['episode_id']),termination=e['termination'],
                    success=e['success'],osr=e['osr'],spl=e['spl'],ne=e['ne'],path_length_m=e['path_length_m'],
                    goal_xy=json.dumps(e['goal_xy']),path_xy=json.dumps(e['path_xy']),
                    arrival_diagnostics=json.dumps(e['arrival_diagnostics']),
                    selected_goal_distances_m=json.dumps(selected),
                    selected_goal_ids=json.dumps([step['goal_id'] for step in nav]),
                    predicted_goal_xy=json.dumps([step['predicted_goal_xy'] for step in nav]),
                    stop_probabilities=json.dumps([step['stop_probability'] for step in nav]),
                    goal_switches=sum(step['goal_switch'] for step in nav)))
            with gzip.open(dest/f'{split}_{variant}_episodes.csv.gz','wt') as f:
                writer=csv.DictWriter(f,fieldnames=list(compact[0]),lineterminator='\n')
                writer.writeheader();writer.writerows(compact)
            if variant=='relations_no_stop':
                labels=[];scores=[]
                for e in payload['episodes']:
                    for step in e['navigation_steps']:
                        labels.append(np.linalg.norm(np.asarray(step['pose'][:2])-e['goal_xy'])<=20.)
                        scores.append(step['stop_probability'])
                y=np.array(labels);p=np.array(scores)
                classifiers.append(dict(split=split,states=len(y),positive_rate=float(y.mean()),auc=float(roc_auc_score(y,p)),
                    ap=float(average_precision_score(y,p)),accuracy05=float(accuracy_score(y,p>=.5)),
                    majority_accuracy=float(max(y.mean(),1-y.mean())),balanced_accuracy05=float(balanced_accuracy_score(y,p>=.5)),
                    random_auc=.5,random_ap=float(y.mean())))
        for folder in ['quick','initial_quick']:
            for f in (run/folder).rglob('*.json'):
                rel=f.relative_to(run);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(target.with_suffix('.json.gz'),'wt') as out: out.write(f.read_text())
    dump(dest/'baseline_comparisons.json',comparison);dump(dest/'initial_baseline_reproduction.json',initial)
    dump(dest/'stop_classifier_metrics.json',classifiers)
    dump(dest/'stop_prefix_replay_checks.json',replay_checks)
    dump(dest/'failure_groups.json',failure_groups)
    manifest=json.loads((run/'manifest.json').read_text());checkpoint=run/'checkpoints/epoch01.pt'
    before=torch.load(manifest['config']['initial_checkpoint'],map_location='cpu',weights_only=False)
    after=torch.load(checkpoint,map_location='cpu',weights_only=False)
    changed={}
    for key in ['lang_model','vision_model','vln_model']:
        a=before[key]['state_dict'];b=after[key]['state_dict']
        changed[key]=[name for name in a if not torch.equal(a[name],b[name])]
    inherited_backbone_changes=[name for name in changed['vln_model'] if not name.startswith('trajectory_head.')]
    if changed['lang_model'] or changed['vision_model'] or inherited_backbone_changes:
        raise RuntimeError('Frozen backbone weights changed')
    torch.save({k:v for k,v in after['vln_model']['state_dict'].items() if k.startswith('trajectory_head.')},dest/'trajectory_relation_head.pt')
    dump(dest/'checkpoint_audit.json',dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),
        portable_head_sha256=sha(dest/'trajectory_relation_head.pt'),changed_parameters=changed,
        frozen_backbone_changes=inherited_backbone_changes))
    for smoke in ['smoke_b16']:
        for name in ['gpu_batch16.json','checkpoint_load.json','COMPLETE.json']:
            shutil.copyfile(run.parent/smoke/name,dest/f'{smoke}_{name}')
    # Keep the implementation/training description written before the run.
    report=ROOT/'SPATIAL_RELATION_OBSERVATION_REPORT.md';text=report.read_text().split('\n## 完整验证结果',1)[0]
    text=text.replace('当前状态：代码、99 项单元测试和全量基线复现完成；一个完整 epoch 正在训练，尚无训练后性能结论。',
                      '当前状态：完成一次 train_seen epoch、128 Episode 快速验证和 Seen/Unseen 全量验证。')
    text += '\n## 完整验证结果\n\n| Split | 设置 | N | SR% | SPL% | OSR% | NE m |\n|---|---|---:|---:|---:|---:|---:|\n'
    for split in ['val_seen','val_unseen']:
        base=baseline(split)
        text+=f"| {split} | 原最佳 B_goal_soft_refined | {len(base)} | {100*np.mean([e['success'] for e in base]):.2f} | {100*np.mean([e['spl'] for e in base]):.2f} | {100*np.mean([e['osr'] for e in base]):.2f} | {np.mean([e['ne'] for e in base]):.2f} |\n"
        for row in rows:
            if row['split']==split:
                text+=f"| {split} | {row['variant']} | {row['episodes']} | {row['sr']:.2f} | {row['spl']:.2f} | {row['oracle_sr']:.2f} | {row['ne']:.2f} |\n"
    text+='\n## 与原最佳方案逐 Episode 配对比较\n\n'
    for key,value in comparison.items():
        text+=f"- {key}: ΔSR {value['success']['delta']:+.2f}pp，95% bootstrap CI {value['success']['ci95']}；ΔSPL {value['spl']['delta']:+.2f}pp；挽回/损失 {value['gained']}/{value['lost']}。\n"
    text+='\n## Stop 分类与实际停止\n\n'
    for m in classifiers:
        text+=f"- {m['split']}：AUC {m['auc']:.4f}（随机 0.5），AP {m['ap']:.4f}（随机 {m['random_ap']:.4f}）；0.5 阈值 Accuracy {m['accuracy05']:.4f}，多数类基线 {m['majority_accuracy']:.4f}，Balanced Accuracy {m['balanced_accuracy05']:.4f}。以上在关闭 Stop 的完整因果轨迹上离线计算。\n"
    for row in rows:
        if row['variant']=='relations_stop95':
            text+=f"- {row['split']} 固定 0.95：实际停止 {row['policy_stops']}，TP/FP {row['stop_TP']}/{row['stop_FP']}，Precision {row['stop_precision']}，Recall {row['stop_recall']}；误停 Episode 比例 {row['stop_FP']/row['episodes']:.4f}。Recall 按观察状态计。\n"
    text+='\n## 到达失败与选中目标诊断\n\n'
    for key,g in failure_groups.items():
        text+=f"- {key}：成功 {g['success']}，进入后最终离开 {g['entered_then_left']}，从未进入 {g['never_entered']}；从未进入者中，始终没有正确选中目标 {g['never_entered_no_correct_selected_goal']}，曾选中正确目标 {g['never_entered_despite_correct_selected_goal']}。选中目标距离 GT≤20m 的状态比例 {g['selected_goal_hits']/max(1,g['selected_goal_states']):.4f}。这里检查的是选中目标，不是整个候选池；这些关联不等于严格的因果归因。\n"
    train=json.loads((run/'train_epoch01.json').read_text())
    text+=f"\n## 资源、权重与一致性\n\n- 完整训练耗时 {train['train_seconds']:.1f}s；峰值 allocated VRAM {train['peak_vram_allocated_bytes']/2**30:.2f}GiB。\n- 主干权重逐 tensor 检查完全不变，审计见 checkpoint_audit.json。\n- Stop 在线路径必须逐点等于无 Stop 轨迹的因果前缀；检查见 stop_prefix_replay_checks.json。\n- 训练前快速复现：{initial}。\n- 权重保存于 `{checkpoint}`；SHA-256 `{sha(checkpoint)}`。\n"
    u=next(r for r in rows if r['split']=='val_unseen' and r['variant']=='relations_no_stop')
    base_sr=25.213199851687058;base_spl=20.20241392295586
    conclusion=('本轮融合观测同时提高 Unseen SR 且未降低 SPL，值得进一步做独立训练控制和多 seed 复核。'
                if u['sr']>base_sr and u['spl']>=base_spl else
                '本轮融合观测没有同时达到提高 Unseen SR 且不降低 SPL 的目标，保留原最佳方案；不自动增加网络或改变评估条件。')
    text+='\n## 本轮结论\n\n'+conclusion+'\n'
    calibration=run/'calibrated_stop/selection.json'
    if calibration.exists():
        cal_dest=dest/'calibrated_stop';cal_dest.mkdir(exist_ok=True)
        for name in ['selection.json','results.json','seen_threshold_calibration.csv']:
            shutil.copyfile(run/'calibrated_stop'/name,cal_dest/name)
        selected=json.loads(calibration.read_text())
        text+='\n## Seen Stop 阈值校准\n\n'
        text+=f"固定约束：Stop Precision≥95%、误停 Episode≤0.5%、Seen SR/SPL 不下降。只用 Seen 选择并先保存冻结 selection.json，再回放 Unseen。状态 `{selected['status']}`，阈值 {selected['threshold']}；Seen 分数范围 {selected['seen_score_range']}。\n"
    for csv_path in dest.rglob('*.csv'):
        csv_path.write_bytes(csv_path.read_bytes().replace(b'\r\n',b'\n'))
    report.write_text(text)
    print(json.dumps(dict(results=rows,initial_reproduction=initial,frozen_backbone_changes=inherited_backbone_changes),indent=2))
if __name__=='__main__': main()
