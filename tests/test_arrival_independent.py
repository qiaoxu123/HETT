"""Independent CPU checks of semantic arrival inference and causal boundaries."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from multiagent.arrival import (
    FEATURE_NAMES, JOINT_FEATURE_NAMES, SEMANTIC_NAMES, MLPArrivalPolicy,
    install_semantic_observer, lexical_instruction_features,
)
from scripts.arrival_analysis import stop_replay


class IndependentArrivalReview(unittest.TestCase):
    def payload(self, seed=13):
        rng = np.random.default_rng(seed)
        d = len(JOINT_FEATURE_NAMES)
        return dict(feature_names=list(JOINT_FEATURE_NAMES),
                    mean=rng.normal(size=d).tolist(),
                    scale=rng.uniform(.2, 2., size=d).tolist(),
                    w1=rng.normal(0., .015, (16, d)).tolist(),
                    b1=rng.normal(0., .1, 16).tolist(),
                    w2=rng.normal(0., .2, 16).tolist(), b2=.17,
                    threshold=.9, head_type='mlp16')

    def test_random_mlp_numpy_torch_fp32_and_fp64_parity(self):
        p = self.payload()
        policy = MLPArrivalPolicy(p)
        rng = np.random.default_rng(5)
        for _ in range(20):
            x = rng.normal(size=len(JOINT_FEATURE_NAMES))
            # Reversed dictionary insertion order must not alter schema ordering.
            features = dict(reversed(list(zip(JOINT_FEATURE_NAMES, x))))
            actual = policy.probability(features)
            for dtype, tolerance in ((torch.float32, 2e-7), (torch.float64, 1e-12)):
                z = ((torch.tensor(x, dtype=dtype) - torch.tensor(p['mean'], dtype=dtype))
                     / torch.tensor(p['scale'], dtype=dtype))
                h = torch.relu(torch.tensor(p['w1'], dtype=dtype) @ z
                               + torch.tensor(p['b1'], dtype=dtype))
                expected = torch.sigmoid(torch.tensor(p['w2'], dtype=dtype) @ h + p['b2'])
                self.assertLess(abs(actual - expected.item()), tolerance)

    def test_semantic_observer_episode_reset_and_idempotence(self):
        model = torch.nn.Conv2d(3, 512, 1).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        agent = SimpleNamespace(vision_model=model,
                                arrival_instruction_features=torch.ones(2, 768))
        install_semantic_observer(agent)
        hook = agent._arrival_semantic_hook
        install_semantic_observer(agent)
        self.assertIs(agent._arrival_semantic_hook, hook)
        self.assertEqual(len(model._forward_hooks), 1)
        with torch.no_grad():
            model(torch.ones(2, 3, 7, 7))
            first = agent.arrival_semantic_features.copy()
            # Simulate next episode batch with a different instruction and size.
            agent.arrival_instruction_features = torch.full((1, 768), -3.)
            agent.arrival_semantic_features = None
            model(torch.ones(1, 3, 7, 7))
        latest = agent.arrival_semantic_features
        self.assertEqual(latest.shape, (1, len(SEMANTIC_NAMES)))
        np.testing.assert_array_equal(latest[0, :768], np.full(768, -3.))
        np.testing.assert_array_equal(first[0, 768:], latest[0, 768:])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_empty_instruction_and_masked_words_cannot_leak(self):
        embedding = torch.nn.Embedding(200, 768)
        model = SimpleNamespace(bert=SimpleNamespace(
            embeddings=SimpleNamespace(word_embeddings=embedding)))
        ids = torch.tensor([[101, 102, 0], [101, 7, 102]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 1]])
        result = lexical_instruction_features(model, ids, mask)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertFalse(result.requires_grad)
        self.assertIsNone(embedding.weight.grad)

    def test_annotations_and_future_states_do_not_change_joint_stop(self):
        p = self.payload()
        p['threshold'] = 0.
        policy = MLPArrivalPolicy(p)
        state = dict(t=0, path_index=0, pose=[0., 0., 50., 0.],
                     features=dict.fromkeys(FEATURE_NAMES, 0.),
                     semantic_features=np.zeros(len(SEMANTIC_NAMES)))
        episode = dict(goal_xy=[100., 0.], path_xy=[[0., 0.], [50., 0.]],
                       teacher_xy=[[0., 0.], [100., 0.]],
                       navigation_steps=[state], termination='horizon')
        first = stop_replay(episode, policy)
        other = copy.deepcopy(episode)
        other['goal_xy'] = [999., 999.]
        other['teacher_xy'] = [[888., 888.]]
        other['path_xy'][1] = [-5000., -5000.]
        other['navigation_steps'][0]['future_gt_arrival'] = True
        second = stop_replay(other, policy)
        self.assertEqual(first['path_xy'], second['path_xy'])
        self.assertEqual(first['stop_diagnostics']['stop_probability'],
                         second['stop_diagnostics']['stop_probability'])

    def test_mlp_rejects_nonfinite_input_and_corrupt_dimensions(self):
        p = self.payload()
        model = MLPArrivalPolicy(p)
        features = dict.fromkeys(JOINT_FEATURE_NAMES, 0.)
        features[SEMANTIC_NAMES[-1]] = np.inf
        with self.assertRaises(ValueError):
            model.probability(features)
        p['w1'] = p['w1'][:-1]
        with self.assertRaises(ValueError):
            MLPArrivalPolicy(p)


if __name__ == '__main__':
    unittest.main()
