"""Causal observable features and portable logistic/MLP arrival heads.

This module intentionally has no environment/annotation imports. Supervision
and Oracle policies live in the diagnostic script, never in this policy.
"""
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    'predicted_distance_m', 'belief_mass20', 'belief_mass40',
    'selected_heatmap_probability', 'heatmap_top1_probability',
    'heatmap_entropy', 'heatmap_top12_gap', 'selector_probability',
    'selector_entropy', 'selector_top12_gap', 'goal_shift_mean_m',
    'goal_shift_max_m', 'goal_spread_m', 'switch_fraction',
    'pose_displacement_m', 'pose_travel_m', 'history_count',
)

# Frozen lexical BERT embeddings retain instruction differences lost by this
# checkpoint's contextual output. RGB is the current frozen Darknet activation.
SEMANTIC_NAMES = (tuple(f'instruction_word_embedding_{i}' for i in range(768))
                  + tuple(f'current_rgb_embedding_{i}' for i in range(512)))
JOINT_FEATURE_NAMES = FEATURE_NAMES + SEMANTIC_NAMES


def lexical_instruction_features(model, input_ids, attention_mask):
    """Frozen lookup only; exclude padding and BERT special tokens."""
    import torch
    with torch.no_grad():
        words=model.bert.embeddings.word_embeddings(input_ids).float()
        valid=attention_mask.bool() & (input_ids!=101) & (input_ids!=102) & (input_ids!=0)
        result=(words*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
    return result.detach()


def semantic_descriptor(instruction, current_rgb):
    """Current lexical instruction and current RGB, without GT or future input."""
    import torch
    if (not isinstance(instruction,torch.Tensor) or not isinstance(current_rgb,torch.Tensor)
            or instruction.ndim!=2 or instruction.shape[-1]!=768
            or current_rgb.ndim!=4 or current_rgb.shape[:2]!=(instruction.shape[0],512)):
        raise ValueError('Frozen instruction Bx768 and current RGB Bx512x7x7 required')
    result=torch.cat((instruction.detach().float(),current_rgb.detach().float().mean((2,3))),-1)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite instruction/RGB features')
    return result


def install_semantic_observer(agent):
    """Observe a forward already required by navigation; add no encoder pass."""
    if getattr(agent, '_arrival_semantic_hook', None) is not None:
        return
    def observe(module, inputs, outputs):
        agent.arrival_semantic_features = semantic_descriptor(
            agent.arrival_instruction_features,outputs).cpu().numpy()
    agent._arrival_semantic_hook = agent.vision_model.register_forward_hook(observe)


def arrival_inputs(record, policy):
    """Merge the explicit observable schemas only when a joint head needs them."""
    features = record['features']
    if getattr(policy, 'requires_semantic', False):
        values = record.get('semantic_features')
        if values is None or len(values) != len(SEMANTIC_NAMES):
            raise ValueError('Joint arrival head requires current instruction and RGB')
        return dict(features, **dict(zip(SEMANTIC_NAMES, values)))
    return features


def probability_stats(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    sorted_p = np.sort(p)[::-1]
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(max(2, len(p)))
    return float(sorted_p[0]), float(entropy), float(sorted_p[0] - sorted_p[1])


def navigation_record(*, t, path_index, pose, normalized_pose, predicted_goal,
                      goal_id, heatmap, selector_probabilities, selector_probability,
                      previous, map_meters, window=5):
    """Build a record using the current state and at most four previous states.

    `previous` contains only earlier observable records from THIS episode.
    A GT coordinate, label, future pose or teacher path is not accepted.
    """
    pose = np.asarray(pose, dtype=float)
    goal = np.asarray(predicted_goal, dtype=float)
    probs = np.asarray(heatmap, dtype=float)
    g = int(round(np.sqrt(len(probs))))
    yy, xx = np.mgrid[:g, :g]
    cells = np.stack(((xx.ravel() + .5) / g, (yy.ravel() + .5) / g), -1)
    distances = np.linalg.norm((cells - normalized_pose) * map_meters, axis=-1)
    top1, entropy, gap = probability_stats(probs)
    _, sel_entropy, sel_gap = probability_stats(selector_probabilities)
    recent = previous[-(window - 1):]
    if any(s['t']>=t for s in recent):
        raise ValueError('Arrival history must contain only earlier decision states')
    positions = np.asarray([s['pose'][:2] for s in recent] + [pose[:2].tolist()])
    goals = np.asarray([s['predicted_goal_xy'] for s in recent] + [goal.tolist()])
    ids = [s['goal_id'] for s in recent] + [int(goal_id)]
    shifts = np.linalg.norm(np.diff(goals, axis=0), axis=-1)
    travelled = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    values = [
        np.linalg.norm(pose[:2] - goal), probs[distances <= 20].sum(),
        probs[distances <= 40].sum(), probs[int(goal_id)], top1, entropy, gap,
        selector_probability, sel_entropy, sel_gap,
        shifts.mean() if len(shifts) else 0., shifts.max() if len(shifts) else 0.,
        np.linalg.norm(goals - goals.mean(0), axis=-1).mean(),
        np.mean(np.diff(ids) != 0) if len(ids) > 1 else 0.,
        np.linalg.norm(positions[-1] - positions[0]), travelled.sum(), len(ids),
    ]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite observable arrival features')
    return dict(t=int(t), path_index=int(path_index), pose=pose.tolist(),
                goal_id=int(goal_id), predicted_goal_xy=goal.tolist(),
                goal_switch=bool(previous and previous[-1]['goal_id'] != goal_id),
                features=dict(zip(FEATURE_NAMES, map(float, values))))


class LogisticArrivalPolicy:
    """Small NumPy inference head; exact feature allowlist prevents leakage."""
    def __init__(self, payload, threshold=None):
        self.feature_names = tuple(payload['feature_names'])
        if self.feature_names not in (FEATURE_NAMES, JOINT_FEATURE_NAMES):
            raise ValueError('Arrival feature schema mismatch')
        self.requires_semantic = self.feature_names == JOINT_FEATURE_NAMES
        self.mean = np.asarray(payload['mean'])
        self.scale = np.asarray(payload['scale'])
        self.coef = np.asarray(payload['coef'])
        self.intercept = float(payload['intercept'])
        self.threshold = float(payload['threshold'] if threshold is None else threshold)
        shape=(len(self.feature_names),)
        if (any(a.shape!=shape for a in (self.mean,self.scale,self.coef))
                or np.any(self.scale<=0)
                or not all(np.isfinite(a).all() for a in (self.mean,self.scale,self.coef))
                or not np.isfinite([self.intercept,self.threshold]).all()
                or not 0<=self.threshold<=1.01):
            raise ValueError('Invalid arrival checkpoint dimensions')

    @classmethod
    def load(cls, path, threshold=None):
        payload=json.loads(Path(path).read_text())
        return MLPArrivalPolicy(payload,threshold) if payload.get('head_type')=='mlp16' else cls(payload,threshold)

    def probability(self, features):
        if set(features) != set(self.feature_names):
            raise ValueError('Only observable feature schema is allowed')
        x = np.asarray([features[n] for n in self.feature_names], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite arrival input')
        score = float(((x - self.mean) / self.scale) @ self.coef + self.intercept)
        return float(1. / (1. + np.exp(-np.clip(score, -60, 60))))

    def __call__(self, features):
        probability = self.probability(features)
        return probability >= self.threshold, probability


class MLPArrivalPolicy:
    """One hidden layer, portable NumPy inference, jointly uses text and RGB."""
    requires_semantic=True
    feature_names=JOINT_FEATURE_NAMES

    def __init__(self,payload,threshold=None):
        if tuple(payload['feature_names'])!=JOINT_FEATURE_NAMES:
            raise ValueError('Joint arrival feature schema mismatch')
        self.mean=np.asarray(payload['mean']);self.scale=np.asarray(payload['scale'])
        self.w1=np.asarray(payload['w1']);self.b1=np.asarray(payload['b1'])
        self.w2=np.asarray(payload['w2']);self.b2=float(payload['b2'])
        self.threshold=float(payload['threshold'] if threshold is None else threshold)
        d=len(JOINT_FEATURE_NAMES)
        arrays=(self.mean,self.scale,self.w1,self.b1,self.w2)
        if (self.mean.shape!=(d,) or self.scale.shape!=(d,) or self.w1.shape!=(16,d)
                or self.b1.shape!=(16,) or self.w2.shape!=(16,) or np.any(self.scale<=0)
                or not all(np.isfinite(a).all() for a in arrays)
                or not np.isfinite([self.b2,self.threshold]).all() or not 0<=self.threshold<=1.01):
            raise ValueError('Invalid joint arrival checkpoint')

    def probability(self,features):
        if set(features)!=set(self.feature_names):raise ValueError('Observable joint schema required')
        x=np.asarray([features[n] for n in self.feature_names],dtype=float)
        if not np.isfinite(x).all():raise ValueError('Nonfinite arrival features')
        hidden=np.maximum(self.w1@((x-self.mean)/self.scale)+self.b1,0.)
        score=float(self.w2@hidden+self.b2)
        return float(1/(1+np.exp(-np.clip(score,-60,60))))

    __call__=LogisticArrivalPolicy.__call__
