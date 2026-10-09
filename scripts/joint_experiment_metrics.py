"""Read-only, per-episode diagnostics for the fixed joint-trajectory experiment."""
import numpy as np
import torch


def observe(agent, obs, poses, trajectories, ended, t, goals, heatmap_ids,
            proposals, ade, fde, prior_ids, joint_ids, stop_probs,
            selected_goal_ids=None):
    if agent.feedback != 'student' or agent.env_name not in ('val_seen','val_unseen'):
        raise RuntimeError('Read-only observer only supports validation student rollouts')
    scale=agent.args.map_meters;radius=agent.args.success_dist
    ids=heatmap_ids.cpu().numpy();g=agent.args.heatmap_grid_size
    selected_ids=(selected_goal_ids.cpu().numpy() if selected_goal_ids is not None
                  else ids[:, 0])
    cand=np.stack(((ids%g+.5)/g,(ids//g+.5)/g),-1)
    pool=proposals.goal_xy.cpu().numpy()
    ade=ade.cpu().numpy()*scale;fde=fde.cpu().numpy()*scale
    pi=prior_ids.cpu().numpy();ji=joint_ids.cpu().numpy()
    modes=proposals.mode_logits.shape[-1]
    for i,ob in enumerate(obs):
        steps=trajectories[i]['experiment_steps']
        # Do not keep counting ended episodes while other batch members run.
        if steps and steps[-1].get('ended', steps[-1]['stopped']):
            continue
        gt=np.asarray(ob['normalized_goal']);dist=np.linalg.norm((cand[i]-gt)*scale,axis=-1)
        pd=np.linalg.norm((pool[i]-gt)*scale,axis=-1)
        endpoint=agent.env.unnormalize_position(goals[i],ob['map_name'],scale)
        pre=ob['pose'].xy.dist_to(ob['goal'])
        reason=trajectories[i].get('stop_reason', [])
        stop=bool(ended[i] and reason and reason[-1] in ('learned_stop','progress_stop')); near=bool(pre<=radius)
        idx=int(ji[i] if agent.args.trajectory_selector_mode=='joint' else pi[i])
        plan_control=agent.args.trajectory_use_for_control
        selected_goal_id=(int(proposals.goal_ids[i,idx//modes]) if plan_control
                          else int(selected_ids[i]))
        macro_moved_m=float(np.linalg.norm(
            np.asarray(poses[i][:2], dtype=float) - np.asarray(ob['pose'][:2], dtype=float)))
        macro_yaw_delta=float(np.arctan2(
            np.sin(poses[i].yaw - ob['pose'].yaw),
            np.cos(poses[i].yaw - ob['pose'].yaw)))
        steps.append(dict(t=t,pose=list(ob['pose']),next_pose=list(poses[i]),
            macro_displacement_m=macro_moved_m, macro_yaw_delta_rad=macro_yaw_delta,
            gt_distance_m=float(pre),
            endpoint_xy=list(endpoint),goal_id=selected_goal_id,
            goal_switch=bool(steps and steps[-1]['goal_id']!=selected_goal_id),
            heatmap_hits={str(k):bool(dist[:min(k,len(dist))].min()<=radius) for k in (1,5,16,20)},
            trajectory_pool_hit=bool(pd.min()<=radius),
            prior_hit=bool(pd[pi[i]//modes]<=radius),joint_hit=bool(pd[ji[i]//modes]<=radius),
            prior_ADE_m=float(ade[i,pi[i]]),prior_FDE_m=float(fde[i,pi[i]]),
            joint_ADE_m=float(ade[i,ji[i]]),joint_FDE_m=float(fde[i,ji[i]]),
            minADE_m=float(ade[i].min()),minFDE_m=float(fde[i].min()),
            selected_plan_ADE_m=float(ade[i,idx]) if plan_control else None,
            selected_plan_FDE_m=float(fde[i,idx]) if plan_control else None,
            stopped=stop,ended=bool(ended[i]),stop_correct=bool(stop and near),in_success_radius=near,
            stop_probability=float(stop_probs[i]) if stop_probs is not None else None))


def summarize(env,predictions,variant,epoch,seconds):
    official,_=env.eval_metrics(predictions)
    episodes=[]
    for eid,p in predictions.items():
        path=[q.xy for q in p['trajectory']];gt=[q.xy for q in p['gt_trajectory']]
        m=env._eval_item(gt,path,p['goal']);steps=p.get('experiment_steps',[])
        d=[q.dist_to(p['goal']) for q in path]
        # Read the controller's actual end reason; a horizon or a
        # stagnation termination must never count as a learned stop.
        reasons=p.get('stop_reason', [])
        termination=reasons[-1] if reasons else 'unreported'
        episodes.append(dict(episode_id=list(eid),success=float(m['success']),osr=float(m['oracle_success']),
            spl=float(m['spl']),ne=float(m['ne']),path_xy=[list(q) for q in path],
            teacher_xy=[list(q) for q in gt],goal_xy=list(p['goal']),steps=steps,
            entered_then_left=bool(min(d)<=env.args.success_dist and d[-1]>env.args.success_dist),
            path_length_m=float(sum(a.dist_to(b) for a,b in zip(path[:-1],path[1:]))),
            termination=termination,initial_distance_m=float(d[0]),
            progress_m=float(d[0]-d[-1]),
            success_from_outside=bool(d[0]>env.args.success_dist and d[-1]<=env.args.success_dist),
            reached_from_outside=bool(d[0]>env.args.success_dist and min(d)<=env.args.success_dist)))
    steps=[s for e in episodes for s in e['steps']]
    action_count=max(1, sum(max(0, len(e['path_xy'])-1) for e in episodes))
    def mean(key):
        v=[s[key] for s in steps if s[key] is not None]
        return float(np.mean(v)) if v else None
    tp=sum(s['stop_correct'] for s in steps);fp=sum(s['stopped'] and not s['in_success_radius'] for s in steps)
    fn=sum(not s['stopped'] and s['in_success_radius'] for s in steps);tn=len(steps)-tp-fp-fn
    summary={k:float(v) if np.isfinite(v) else None for k,v in official.items()}
    summary.update(epoch=epoch,variant=variant,split=env.split,episodes=len(episodes),active_steps=len(steps) or action_count,
        seconds=seconds,episode_seconds=seconds/len(episodes),step_seconds=seconds/max(1,len(steps) or action_count),
        entered_then_left=sum(e['entered_then_left'] for e in episodes),
        path_length_m=float(np.mean([e['path_length_m'] for e in episodes])),
        initial_distance_m=float(np.mean([e['initial_distance_m'] for e in episodes])),
        mean_goal_progress_m=float(np.mean([e['progress_m'] for e in episodes])),
        initial_already_successful=sum(e['initial_distance_m']<=env.args.success_dist for e in episodes),
        reached_from_outside=sum(e['reached_from_outside'] for e in episodes),
        success_from_outside=sum(e['success_from_outside'] for e in episodes),
        macro_zero_translation_rate=float(np.mean([
            s['macro_displacement_m']<1e-4 for s in steps if not s['stopped']
        ])) if steps and any(not s['stopped'] for s in steps) else None,
        stop_TP=tp if steps else None,stop_FP=fp if steps else None,
        stop_FN=fn if steps else None,stop_TN=tn if steps else None,
        stop_precision=tp/(tp+fp) if tp+fp else None,stop_recall=tp/(tp+fn) if tp+fn else None,
        stop_accuracy=(tp+tn)/len(steps) if steps else None,
        goal_switch_count=sum(s['goal_switch'] for s in steps),
        goal_switch_rate=sum(s['goal_switch'] for s in steps)/max(1,len(steps)-len(episodes)),
        rank_eligible_rate=mean('trajectory_pool_hit'),
        policy_stops=sum(e['termination'] in ('learned_stop','progress_stop') for e in episodes),
        horizon_timeouts=sum(e['termination']=='horizon' for e in episodes),
        trajectory_stagnations=sum(e['termination']=='trajectory_stagnation' for e in episodes),
        unreported_terminations=sum(e['termination']=='unreported' for e in episodes))
    for k in (1,5,16,20):
        summary[f'heatmap_top{k}_hit20']=(
            float(np.mean([s['heatmap_hits'][str(k)] for s in steps])) if steps else None
        )
        initial=[e['steps'][0]['heatmap_hits'][str(k)] for e in episodes if e['steps']]
        summary[f'initial_heatmap_top{k}_hit20']=(float(np.mean(initial)) if initial else None)
    for key in ('prior_hit','joint_hit','minADE_m','minFDE_m','prior_ADE_m','prior_FDE_m','joint_ADE_m','joint_FDE_m','selected_plan_ADE_m','selected_plan_FDE_m'):
        summary[key]=mean(key)
    summary['osr_sr_gap_pp']=summary['oracle_sr']-summary['sr']
    return dict(summary=summary,episodes=episodes)
