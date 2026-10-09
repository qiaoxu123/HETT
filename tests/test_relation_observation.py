import unittest
import torch
from multiagent.models.relation_observation import RelationObservation

class RelationObservationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5); torch.set_num_threads(1)
        self.model = RelationObservation(feature_dim=8, language_dim=12, hidden=16).eval()
        self.args = dict(spatial=torch.randn(2,8,5,5), current_xy=torch.tensor([[.2,.3],[.4,.2]]),
                        heading_sc=torch.tensor([[0.,1.],[1.,0.]]), language_tokens=torch.randn(2,7,12),
                        language_mask=torch.tensor([[1,1,1,1,1,0,0],[1,1,1,1,1,1,0]],dtype=torch.bool),
                        rgb_features=torch.randn(2,512), landmark_xy=torch.rand(2,3,2),
                        landmark_extent=torch.ones(2,3,2)*.1, landmark_valid=torch.tensor([[1,1,1],[0,0,0]],dtype=torch.bool),
                        landmark_text_mask=torch.zeros(2,3,7,dtype=torch.bool), history_xy=torch.rand(2,3,2))
        self.args['landmark_text_mask'][:,0,1]=True
        self.args['landmark_text_mask'][:,1,3]=True
    def test_zero_residual_preserves_baseline(self):
        logits,features,fused,geom=self.model(**self.args)
        self.assertEqual(tuple(logits.shape),(2,5,5)); self.assertTrue(torch.equal(logits,torch.zeros_like(logits)))
        self.assertTrue(torch.equal(features,torch.zeros_like(features)))
        self.assertTrue(torch.isfinite(fused).all()); self.assertEqual(tuple(geom.shape),(2,25,7))
    def test_landmark_permutation_keeps_name_geometry_binding(self):
        a=self.model(**self.args)[2]; changed=dict(self.args)
        for key in ['landmark_xy','landmark_extent','landmark_valid','landmark_text_mask']: changed[key]=changed[key][:,[2,0,1]]
        self.assertTrue(torch.allclose(a,self.model(**changed)[2],atol=1e-6))
    def test_current_pose_rgb_instruction_and_anchor_change_observation(self):
        a=self.model(**self.args)[2]
        for key in ['current_xy','rgb_features','language_tokens','landmark_xy','history_xy']:
            altered=dict(self.args); altered[key]=altered[key]+.3
            self.assertGreater(float((a-self.model(**altered)[2]).detach().abs().max()),1e-6,key)
    def test_zero_distance_and_no_landmarks_finite(self):
        altered=dict(self.args, landmark_valid=torch.zeros_like(self.args['landmark_valid']))
        self.assertTrue(all(torch.isfinite(x).all() for x in self.model(**altered)))
    def test_padding_text_does_not_change_observation(self):
        a=self.model(**self.args)[2]; altered=dict(self.args)
        altered['language_tokens']=self.args['language_tokens'].clone()
        altered['language_tokens'][~self.args['language_mask']]=999
        self.assertTrue(torch.equal(a,self.model(**altered)[2]))
    def test_stop_gradients_and_shapes(self):
        logits,features,fused,geom=self.model(**self.args)
        probs=torch.softmax(logits.flatten(1),-1).reshape_as(logits)
        stop=self.model.stop(fused,geom,torch.tensor([0,24]),probs)
        self.assertEqual(tuple(stop.shape),(2,)); self.assertTrue(torch.isfinite(stop).all())
        (logits.square().mean()+features.sum()+stop.sum()).backward()
        self.assertGreater(float(self.model.edges[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(self.model.visual.weight.grad.abs().sum()),0)
    def test_invalid_anchors_have_no_effect(self):
        a=self.model(**self.args)[2]; changed=dict(self.args)
        changed['landmark_xy']=self.args['landmark_xy'].clone()
        changed['landmark_xy'][~self.args['landmark_valid']]=1000
        self.assertTrue(torch.equal(a,self.model(**changed)[2]))
    def test_dynamic_observation_metric_bearing(self):
        from multiagent.static_observation import build_uav_landmark_observation
        import math
        refs=[dict(name='A',center_xy=[.7,.3]),dict(name='B',center_xy=[.5,.5])]
        out=build_uav_landmark_observation(refs,[.5,.5],0.,100.)
        self.assertAlmostEqual(out[0]['distance_m'],math.sqrt(800))
        self.assertAlmostEqual(out[0]['relative_bearing_rad'],math.pi/4)
        self.assertEqual(out[1]['distance_m'],0.)
    def test_relation_stop_occurs_before_waypoint_action_without_gt_condition(self):
        import ast
        from pathlib import Path
        tree=ast.parse((Path(__file__).resolve().parents[1]/'multiagent/agent.py').read_text())
        branches=[node for node in ast.walk(tree) if isinstance(node,ast.If)
                  and 'selected_stop_probs[i]' in ast.unparse(node.test)
                  and 'trajectory_relation_observation' in ast.unparse(node.test)]
        self.assertEqual(len(branches),1)
        condition=ast.unparse(branches[0].test)
        self.assertNotIn("obs[i]['goal']",condition)
        self.assertNotIn('gt_',condition)
        body=ast.unparse(ast.Module(body=branches[0].body,type_ignores=[]))
        self.assertIn('continue',body)
        self.assertNotIn('bounded_heatmap_step',body)
        self.assertNotIn('self.move',body)
    def test_episode_history_has_no_module_state(self):
        # Reprocessing an episode after an unrelated episode must be identical.
        a=self.model(**self.args)[2].detach().clone()
        unrelated=dict(self.args,current_xy=self.args['current_xy']+.4,
                       history_xy=self.args['history_xy']+.4)
        self.model(**unrelated)
        self.assertTrue(torch.equal(a,self.model(**self.args)[2]))
    def test_goal_labels_not_in_forward_schema(self):
        import inspect
        names=set(inspect.signature(self.model.forward).parameters)
        self.assertTrue(names.isdisjoint({'goal','normalized_goal','target','trajectory','teacher_goal','future'}))

if __name__=='__main__': unittest.main()
