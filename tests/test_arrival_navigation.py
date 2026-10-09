"""Timing, causal features, episode reset and upstream metric regressions."""
import unittest
import tempfile
import json
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from multiagent.arrival import (FEATURE_NAMES, JOINT_FEATURE_NAMES, SEMANTIC_NAMES,
    LogisticArrivalPolicy, navigation_record, semantic_descriptor, arrival_inputs,
    install_semantic_observer, lexical_instruction_features, MLPArrivalPolicy)
from scripts.arrival_analysis import (official_item, oracle_stop, oracle_goal,
                                     stop_replay, stop_metrics, episode_diagnostics, load_rollout)


def episode(path):
    return dict(episode_id=['map',1,1], path_xy=path, teacher_xy=[[0.,0.],[100.,0.]],
                goal_xy=[100.,0.], initial_pose=[path[0][0],path[0][1],50.,0.],
                termination='horizon', navigation_steps=[])


def record(t=0, previous=None, goal_id=0):
    return navigation_record(t=t, path_index=t, pose=[t*10.,0.,50.,0.],
        normalized_pose=np.array([0.,0.]), predicted_goal=[100.,0.],
        goal_id=goal_id, heatmap=np.ones(4)/4,
        selector_probabilities=np.array([.7,.3]),selector_probability=.7,
        previous=previous or [],map_meters=410.)


class ArrivalNavigationTest(unittest.TestCase):
    def test_joint_descriptor_uses_both_instruction_and_current_rgb(self):
        import torch
        torch.manual_seed(0)
        h=torch.randn(2,768,requires_grad=True)
        rgb=torch.randn(2,512,7,7,requires_grad=True)
        a=semantic_descriptor(h,rgb)
        changed_instruction=semantic_descriptor(-h,rgb)
        changed_rgb=semantic_descriptor(h,-rgb)
        self.assertEqual(a.shape,(2,len(SEMANTIC_NAMES)))
        self.assertFalse(a.requires_grad)
        self.assertFalse(torch.allclose(a,changed_instruction))
        self.assertFalse(torch.allclose(a,changed_rgb))

    def test_joint_schema_rejects_missing_modalities_and_gt(self):
        payload=dict(feature_names=list(JOINT_FEATURE_NAMES),mean=[0.]*len(JOINT_FEATURE_NAMES),
            scale=[1.]*len(JOINT_FEATURE_NAMES),coef=[0.]*len(JOINT_FEATURE_NAMES),
            intercept=0.,threshold=.9)
        policy=LogisticArrivalPolicy(payload)
        r=record()
        with self.assertRaises(ValueError):
            arrival_inputs(r,policy)
        r['semantic_features']=[0.]*len(SEMANTIC_NAMES)
        inputs=arrival_inputs(r,policy)
        self.assertEqual(set(inputs),set(JOINT_FEATURE_NAMES))
        self.assertEqual(policy.probability(inputs),.5)
        with self.assertRaises(ValueError):
            policy.probability(dict(inputs,gt_distance=0.))
        with self.assertRaises(ValueError):
            semantic_descriptor(None,None)

    def test_semantic_observer_preserves_frozen_output_and_current_frame(self):
        import torch
        model=torch.nn.Conv2d(3,512,1).eval()
        for p in model.parameters():p.requires_grad_(False)
        agent=SimpleNamespace(vision_model=model,arrival_instruction_features=torch.randn(1,768))
        frame=torch.randn(1,3,7,7)
        before=model(frame).clone()
        install_semantic_observer(agent)
        after=model(frame)
        first=agent.arrival_semantic_features.copy()
        self.assertTrue(torch.equal(before,after))
        model(-frame)
        self.assertFalse(np.array_equal(first,agent.arrival_semantic_features))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_lexical_features_ignore_padding_and_special_tokens(self):
        import torch
        embedding=torch.nn.Embedding(200,768)
        model=SimpleNamespace(bert=SimpleNamespace(embeddings=SimpleNamespace(word_embeddings=embedding)))
        ids=torch.tensor([[101,3,4,102,0],[101,5,6,102,0]])
        a=lexical_instruction_features(model,ids,ids!=0)
        expected=embedding(ids[:,1:3]).mean(1)
        self.assertTrue(torch.equal(a,expected))
        self.assertFalse(a.requires_grad)
        self.assertFalse(torch.equal(a[0],a[1]))
        padded=torch.nn.functional.pad(ids,(0,3))
        b=lexical_instruction_features(model,padded,padded!=0)
        self.assertTrue(torch.equal(a,b))

    def test_mlp_numpy_torch_parity_and_joint_modality_effect(self):
        import torch
        d=len(JOINT_FEATURE_NAMES)
        w1=np.zeros((16,d));w1[0,len(FEATURE_NAMES)]=1.;w1[0,len(FEATURE_NAMES)+768]=1.
        w2=np.zeros(16);w2[0]=1.
        payload=dict(feature_names=list(JOINT_FEATURE_NAMES),mean=np.zeros(d).tolist(),
            scale=np.ones(d).tolist(),w1=w1.tolist(),b1=[0.]*16,w2=w2.tolist(),b2=0.,threshold=.9)
        policy=MLPArrivalPolicy(payload)
        features=dict.fromkeys(JOINT_FEATURE_NAMES,0.)
        baseline=policy.probability(features)
        features[SEMANTIC_NAMES[0]]=1.
        language=policy.probability(features)
        features[SEMANTIC_NAMES[0]]=0.;features[SEMANTIC_NAMES[768]]=1.
        rgb=policy.probability(features)
        self.assertGreater(language,baseline);self.assertGreater(rgb,baseline)
        x=torch.tensor([features[n] for n in JOINT_FEATURE_NAMES],dtype=torch.float64)
        expected=torch.sigmoid(torch.tensor(w2)@torch.relu(torch.tensor(w1)@x)).item()
        self.assertAlmostEqual(rgb,expected,places=10)
        with self.assertRaises(ValueError):policy.probability(dict(features,gt_label=1.))

    def test_semantic_cache_alignment_hash_and_causal_joint_stop(self):
        e=episode([[0.,0.],[85.,0.],[135.,0.]])
        a,b=record(),record(1)
        a['semantic_index']=0;b['semantic_index']=1
        b['pose']=[85.,0.,50.,0.]
        e['navigation_steps']=[a,b]
        cache=np.zeros((2,len(SEMANTIC_NAMES)),dtype=np.float32)
        cache[1,0]=1.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);array=root/'features.npy';np.save(array,cache,allow_pickle=False)
            payload=dict(episodes=[e],semantic_cache=dict(file=array.name,states=2,
                feature_names=list(SEMANTIC_NAMES),sha256=hashlib.sha256(array.read_bytes()).hexdigest()))
            path=root/'val_seen.json.gz'
            def write():
                with gzip.open(path,'wt') as f:json.dump(payload,f)
            write();loaded=load_rollout(path)['episodes'][0]
            model=dict(feature_names=list(JOINT_FEATURE_NAMES),mean=[0.]*len(JOINT_FEATURE_NAMES),
                scale=[1.]*len(JOINT_FEATURE_NAMES),coef=[0.]*len(JOINT_FEATURE_NAMES),
                intercept=-5.,threshold=.9)
            model['coef'][len(FEATURE_NAMES)]=10.
            policy=LogisticArrivalPolicy(model)
            result=stop_replay(loaded,policy)
            self.assertEqual(result['path_xy'],[[0.,0.],[85.,0.]])
            # Changing annotations cannot change the stop decision or its score.
            loaded['goal_xy']=[1000.,1000.]
            other=stop_replay(loaded,policy)
            self.assertEqual(other['path_xy'],result['path_xy'])
            self.assertEqual(other['stop_diagnostics']['stop_probability'],result['stop_diagnostics']['stop_probability'])
            payload['episodes'][0]['navigation_steps'][1]['semantic_index']=0
            write()
            with self.assertRaises(ValueError):load_rollout(path)
            array.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):load_rollout(path)

    def test_stop_uses_current_pose_before_next_action(self):
        e=episode([[0.,0.],[85.,0.],[135.,0.]])
        a,b=record(),record(1)
        a['pose']=[0.,0.,50.,0.]; b['pose']=[85.,0.,50.,0.]
        a['features']['predicted_distance_m']=100
        b['features']['predicted_distance_m']=15
        e['navigation_steps']=[a,b]
        r=stop_replay(e,lambda f:(f['predicted_distance_m']<=20,None))
        self.assertEqual(r['path_xy'],[[0.,0.],[85.,0.]])
        self.assertEqual(r['stop_diagnostics']['tp'],1)
        self.assertEqual(official_item(r)['success'],1.)

    def test_initial_stop_and_no_future_interpolation(self):
        e=episode([[90.,0.],[140.,0.]])
        self.assertEqual(oracle_stop(e)['path_xy'],[[90.,0.]])
        crossing=episode([[70.,0.],[130.,0.]])
        self.assertEqual(oracle_stop(crossing)['path_xy'],crossing['path_xy'])
        self.assertEqual(episode_diagnostics(crossing)['skipped_success_circle_crossings'],1)

    def test_oracle_final_budget_pose_is_scored(self):
        e=episode([[0.,0.],[1000.,0.]])
        e['goal_xy']=[1000.,0.]
        self.assertEqual(official_item(oracle_goal(e))['success'],1.)
        self.assertEqual(len(oracle_goal(e)['path_xy']),21)

    def test_exit_reentry_and_final_failure_are_distinct(self):
        e=episode([[0.,0.],[90.,0.],[130.,0.],[95.,0.],[140.,0.]])
        d=episode_diagnostics(e)
        self.assertEqual((d['entries'],d['exits'],d['reentries']),(2,2,1))
        self.assertTrue(d['reached_but_final_failed'])

    def test_history_reset_and_goal_switch(self):
        a=record(); b=record(1,[a],goal_id=1)
        self.assertTrue(b['goal_switch'])
        self.assertEqual(b['features']['history_count'],2)
        reset=record(goal_id=1)
        self.assertFalse(reset['goal_switch'])
        self.assertEqual(reset['features']['history_count'],1)
        self.assertEqual(reset['features']['goal_shift_max_m'],0.)

    def test_features_ignore_annotations_and_future(self):
        a=record(); before=dict(a['features'])
        future=record(1,[a])
        self.assertEqual(before,a['features'])
        with self.assertRaises(ValueError):
            record(0,[future])
        self.assertEqual(set(a['features']),set(FEATURE_NAMES))
        with self.assertRaises(TypeError):
            navigation_record(gt_goal=[100,0])
        payload=dict(feature_names=list(FEATURE_NAMES),mean=[0]*len(FEATURE_NAMES),
            scale=[1]*len(FEATURE_NAMES),coef=[0]*len(FEATURE_NAMES),intercept=0,threshold=.9)
        policy=LogisticArrivalPolicy(payload)
        with self.assertRaises(ValueError):
            policy(dict(a['features'],gt_distance_m=0))
        self.assertEqual(policy(a['features']),(False,.5))

    def test_upstream_metrics_and_spl_numerator_are_unchanged(self):
        e=episode([[0.,0.],[0.,100.],[100.,0.]])
        m=official_item(e)
        self.assertEqual(m['success'],1.)
        self.assertEqual(m['oracle_success'],1.)
        self.assertEqual(m['ne'],0.)
        self.assertAlmostEqual(m['spl'],100/(100+np.sqrt(20000)))
        failed=episode([[0.,0.],[90.,0.],[140.,0.]])
        self.assertEqual(official_item(failed)['success'],0.)
        self.assertEqual(official_item(failed)['oracle_success'],1.)
        self.assertEqual(official_item(failed)['spl'],0.)
        self.assertEqual(official_item(oracle_stop(failed))['spl'],1.)

    def test_fast_eval_switches_and_termination_accounting(self):
        from scripts.joint_experiment_metrics import summarize
        from multiagent.space import Pose4D, Point2D
        env=SimpleNamespace(args=SimpleNamespace(success_dist=20.),split='val_seen',
            eval_metrics=lambda preds:({'sr':0.,'oracle_sr':0.,'spl':0.,'ne':50.},{}),
            _eval_item=lambda gt,path,goal:dict(success=0.,oracle_success=0.,spl=0.,ne=50.))
        a=record();b=record(1,[a],goal_id=1)
        for s in [a,b]:
            s.update(action='move',action_xy=[10.,0.])
        p=dict(trajectory=[Pose4D(0.,0.,50.,0.),Pose4D(10.,0.,50.,0.)],
            gt_trajectory=[Pose4D(0.,0.,50.,0.)],goal=Point2D(100.,0.),
            stop_reason=['waypoint_stagnation'],navigation_steps=[a,b])
        result=summarize(env,{('map',1,1):p},'B',1,1.)['summary']
        self.assertEqual(result['goal_switch_count'],1)
        self.assertEqual(result['goal_switch_rate'],1.)
        self.assertEqual(result['waypoint_stagnations'],1)
        self.assertEqual(result['policy_stops'],0)

    def test_training_label_is_current_not_next_pose(self):
        from scripts.run_arrival_analysis import feature_dataset
        e=episode([[0.,0.],[100.,0.]])
        s=record();s['next_pose']=[100.,0.,50.,0.]
        e['navigation_steps']=[s]
        x,y,w=feature_dataset([e])
        self.assertEqual(y.tolist(),[0])
        self.assertEqual(x.shape,(1,len(FEATURE_NAMES)))
        self.assertEqual(w.tolist(),[1.])

    def test_calibration_cache_matches_exact_replay(self):
        from scripts.run_arrival_analysis import CalibrationCache
        e=episode([[0.,0.],[85.,0.],[135.,0.]])
        a,b=record(),record(1)
        a['pose']=[0.,0.,50.,0.]; b['pose']=[85.,0.,50.,0.]
        e['navigation_steps']=[a,b]
        cache=CalibrationCache([e],np.array([.1,.9]))
        m=cache.evaluate(.8)
        self.assertEqual(m['sr'],100.)
        self.assertEqual(m['spl'],100.)
        self.assertEqual((m['stop_tp'],m['stop_fp'],m['stop_fn']),(1,0,0))
        self.assertEqual(cache.evaluate(1.01)['sr'],0.)

    def test_numpy_labels_preserve_negative_and_missed_counts(self):
        e=episode([[0.,0.],[85.,0.],[135.,0.]])
        a,b=record(),record(1)
        a['pose']=[0.,0.,50.,0.];b['pose']=[85.,0.,50.,0.]
        e['navigation_steps']=[a,b]
        result=stop_metrics([stop_replay(e,lambda f:(False,None))])
        self.assertEqual((result['stop_fn'],result['stop_tn']),(1,1))
        self.assertEqual(result['stop_recall'],0.)

    def test_numpy_diagnostic_counts_are_json_serializable(self):
        from scripts.run_arrival_analysis import save
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'counts.json'
            save(p,dict(count=np.int64(3),rate=np.float64(.5)))
            self.assertEqual(json.loads(p.read_text()),dict(count=3,rate=.5))

    def test_wrong_goal_and_unreached_correct_goal_are_separate(self):
        e=episode([[0.,0.],[40.,0.]])
        s=record();s['predicted_goal_xy']=[300.,0.]
        e['navigation_steps']=[s]
        d=episode_diagnostics(e)
        self.assertTrue(d['never_entered_no_correct_goal'])
        self.assertFalse(d['never_entered_despite_correct_goal'])
        s['predicted_goal_xy']=[100.,0.]
        d=episode_diagnostics(e)
        self.assertFalse(d['never_entered_no_correct_goal'])
        self.assertTrue(d['never_entered_despite_correct_goal'])


if __name__=='__main__':
    unittest.main()
