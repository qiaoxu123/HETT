"""Diagnostic replay only: official metrics, Oracle bounds and failure events."""
from collections import Counter
from types import SimpleNamespace

import numpy as np

from multiagent.heatmap_execution import bounded_heatmap_step
from multiagent.space import Point2D, Pose4D
from multiagent.arrival import arrival_inputs, SEMANTIC_NAMES


def load_rollout(path):
    """Load observable arrays separately from annotation-bearing episode JSON."""
    import gzip
    import hashlib
    import json
    from pathlib import Path
    path=Path(path)
    with gzip.open(path,'rt') as f:data=json.load(f)
    if 'semantic_cache' in data:
        meta=data['semantic_cache']
        if tuple(meta['feature_names'])!=SEMANTIC_NAMES:
            raise ValueError('Frozen semantic cache schema mismatch')
        cache_path=path.parent/meta['file']
        h=hashlib.sha256()
        with cache_path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=meta['sha256']:
            raise ValueError('Semantic cache SHA-256 mismatch')
        cache=np.load(cache_path,mmap_mode='r',allow_pickle=False)
        if cache.shape!=(meta['states'],len(SEMANTIC_NAMES)) or cache.dtype!=np.float32:
            raise ValueError('Semantic cache dimensions or dtype mismatch')
        index=0
        for e in data['episodes']:
            for s in e['navigation_steps']:
                if s['semantic_index']!=index:
                    raise ValueError('Semantic cache state alignment mismatch')
                s['semantic_features']=cache[index];index+=1
        if index!=len(cache):raise ValueError('Semantic state count mismatch')
    return data


def official_item(episode, path=None, radius=20.):
    # Reuse the upstream evaluator, including its straight-line SPL numerator.
    from multiagent.env import CityNavBatch
    path = episode['path_xy'] if path is None else path
    env = SimpleNamespace(args=SimpleNamespace(success_dist=radius))
    return CityNavBatch._eval_item(env,
        [Point2D(*q) for q in episode['teacher_xy']],
        [Point2D(*q) for q in path], Point2D(*episode['goal_xy']))


def metrics(episodes, radius=20.):
    rows = [official_item(e, radius=radius) for e in episodes]
    return dict(episodes=len(rows), sr=100 * np.mean([s['success'] for s in rows]),
                oracle_sr=100 * np.mean([s['oracle_success'] for s in rows]),
                spl=100 * np.mean([s['spl'] for s in rows]),
                ne=np.mean([s['ne'] for s in rows]),
                path_length_m=np.mean([s['trajectory_lengths'] for s in rows]))


def first_entry(path, goal, radius=20.):
    distances = np.linalg.norm(np.asarray(path) - goal, axis=-1)
    indices = np.flatnonzero(distances <= radius)
    return int(indices[0]) if len(indices) else None


def oracle_stop(episode):
    """Counterfactual stop at an observed pose, never a segment interpolation."""
    index = first_entry(episode['path_xy'], episode['goal_xy'])
    result = dict(episode)
    if index is not None:
        result['path_xy'] = episode['path_xy'][:index + 1]
        result['termination'] = 'oracle_stop'
    return result


def oracle_goal(episode, stop=False, max_actions=20, max_step_m=50., stagnation_steps=5):
    """Replay the identical deterministic controller with a diagnostic GT goal.

    The controller has no obstacle/pose clipping or belief-dependent execution;
    this intervention needs no backbone forward. It is not a deployable policy.
    """
    pose = Pose4D(*episode['initial_pose'])
    goal = Point2D(*episode['goal_xy'])
    path = [list(pose.xy)]
    stagnant = 0
    reason = 'horizon'
    for _ in range(max_actions):
        if stop and pose.xy.dist_to(goal) <= 20.:
            reason = 'oracle_stop'
            break
        new_pose = bounded_heatmap_step(pose, goal, max_step_m=max_step_m)
        moved = new_pose.xy.dist_to(pose.xy)
        stagnant = stagnant + 1 if moved < 1e-4 else 0
        # Terminal zero displacement can be omitted, as in should_record_pose.
        path.append(list(new_pose.xy))
        pose = new_pose
        if stagnant >= stagnation_steps:
            reason = 'waypoint_stagnation'
            break
    result = dict(episode, path_xy=path, termination=reason)
    return result


def episode_diagnostics(episode, radius=20.):
    path = np.asarray(episode['path_xy'])
    goal = np.asarray(episode['goal_xy'])
    distance = np.linalg.norm(path - goal, axis=-1)
    inside = distance <= radius
    entries = np.flatnonzero(inside & np.r_[True, ~inside[:-1]])
    exits = np.flatnonzero(~inside & np.r_[False, inside[:-1]])
    nav = episode.get('navigation_steps', [])
    exit_actions = [s for s in nav if s['path_index'] + 1 in exits]
    # A 50 m macro action can cross the circle without landing inside it.
    # Official OSR deliberately uses only saved macro positions.
    crossings = 0
    for a, b in zip(path[:-1], path[1:]):
        v = b - a
        ratio = np.clip(np.dot(goal - a, v) / max(np.dot(v, v), 1e-12), 0, 1)
        if min(np.linalg.norm(a-goal), np.linalg.norm(b-goal)) > radius:
            crossings += int(np.linalg.norm(a + ratio*v - goal) <= radius)
    reached = bool(inside.any())
    failed = not bool(inside[-1])
    goal_errors=[float(np.linalg.norm(np.asarray(s['predicted_goal_xy'])-goal)) for s in nav]
    goal_hits=sum(error<=radius for error in goal_errors)
    return dict(episode_id=episode['episode_id'], reached=reached, final_success=not failed,
        first_entry_index=int(entries[0]) if len(entries) else None,
        entries=len(entries), exits=len(exits), reentries=max(0, len(entries)-1),
        entered_then_left=bool(len(exits)), reached_but_final_failed=reached and failed,
        failure_group='success' if not failed else 'entered_then_left' if reached else 'never_entered',
        exit_goal_switches=sum(s['goal_switch'] for s in exit_actions),
        exit_wrong_predicted_goals=sum(np.linalg.norm(np.asarray(s['predicted_goal_xy'])-goal)>radius for s in exit_actions),
        exit_large_actions=sum(np.linalg.norm(s.get('action_xy', [0.,0.]))>radius for s in exit_actions),
        exit_full_50m_actions=sum(np.linalg.norm(s.get('action_xy', [0.,0.]))>=49.999 for s in exit_actions),
        skipped_success_circle_crossings=crossings,
        goal_switches=sum(s['goal_switch'] for s in nav),
        predicted_goal_hit_states=goal_hits,
        never_entered_no_correct_goal=bool(failed and not reached and nav and goal_hits==0),
        never_entered_despite_correct_goal=bool(failed and not reached and goal_hits>0),
        termination=episode.get('termination', 'unreported'),
        predicted_goal_error_min_m=min(goal_errors) if goal_errors else None,
        predicted_goal_error_mean_m=float(np.mean(goal_errors)) if goal_errors else None)


def stop_replay(episode, policy):
    """Exact replay for a pure stopping intervention on unchanged baseline moves."""
    nav = episode['navigation_steps']
    first_stop = None
    probabilities = []
    for s in nav:
        stopped, probability = policy(arrival_inputs(s, policy))
        probabilities.append(probability)
        if stopped:
            first_stop = s
            break
    result = dict(episode)
    if first_stop is not None:
        result['path_xy'] = episode['path_xy'][:first_stop['path_index']+1]
        result['termination'] = 'policy_stop'
    visited = nav[:len(probabilities)]
    labels = [np.linalg.norm(np.asarray(s['pose'][:2])-episode['goal_xy']) <= 20. for s in visited]
    stop_positive = bool(first_stop is not None and labels[-1])
    stop_negative = bool(first_stop is not None and not labels[-1])
    result['stop_diagnostics'] = dict(tp=int(stop_positive), fp=int(stop_negative),
        fn=int(sum(labels))-int(stop_positive), tn=len(labels)-int(sum(labels))-int(stop_negative),
        eligible_episode=int(any(np.linalg.norm(np.asarray(s['pose'][:2])-episode['goal_xy']) <= 20.
                                 for s in nav)),
        false_stop_replaced_success=int(stop_negative and official_item(episode)['success']),
        policy_stop=int(first_stop is not None), evaluated_states=len(visited),
        stop_probability=probabilities[-1] if first_stop is not None else None,
        stop_path_index=first_stop['path_index'] if first_stop is not None else None)
    return result


def stop_metrics(episodes):
    rows = [e['stop_diagnostics'] for e in episodes]
    totals = Counter()
    for r in rows:
        totals.update({k:int(v) for k,v in r.items() if isinstance(v, (int,np.integer))})
    tp, fp, fn = (totals[k] for k in ('tp','fp','fn'))
    return dict(stop_tp=tp, stop_fp=fp, stop_fn=fn, stop_tn=totals['tn'],
        stop_precision=tp/(tp+fp) if tp+fp else None,
        stop_recall=tp/(tp+fn) if tp+fn else None,
        stop_episode_recall=tp/totals['eligible_episode'] if totals['eligible_episode'] else None,
        false_stop_episode_rate=fp/len(episodes),
        false_positive_state_rate=fp/(fp+totals['tn']) if fp+totals['tn'] else None,
        false_stop_replaced_success=totals['false_stop_replaced_success'],
        policy_stops=tp+fp)
