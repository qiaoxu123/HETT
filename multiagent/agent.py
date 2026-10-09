import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms

# from r2r.agent_cmt import Seq2SeqCMTAgent
from multiagent.actions import Action
from multiagent.defaultpaths import GOAL_PREDICTOR_CHECKPOINT_DIR
from multiagent.models.dark_net import Darknet
from multiagent.models.CLIP import CLIP
# from direction.models.ddppo.resenet_encoders import TorchVisionResNet50
from multiagent.models.goal_predictor import GoalPredictor, MapEncoder
from multiagent.models.spatial_belief import metric_gaussian_target, greedy_nms_topk, local_soft_argmax_xy
from multiagent.models.multi_landmark import build_landmark_batch
from multiagent.models.heatmap_trajectory import (
    TrajectoryProposals, resample_teacher_suffix,
    trajectory_imitation_loss, stop_supervision_loss,
    candidate_ranking_loss, candidate_trajectory_imitation_loss,
    select_goal_mode_indices, arrival_stop_targets,
    goal_distance_soft_ranking_loss, local_path_soft_ranking_loss,
)
from multiagent.heatmap_execution import bounded_heatmap_step
from multiagent.trajectory_rollout_utils import should_record_pose
from multiagent.rollout_backward import backward_rollout_pair
from multiagent.mapdata import MAP_BOUNDS
from multiagent.observation import cropclient
from multiagent.space import Pose4D, Point2D, Point3D
from multiagent.teacher.algorithm.lookahead import lookahead_discrete_action
from multiagent.teacher.trajectory import _moved_pose
from models.vln_model import CustomBERTModel
from models.ET_haa import ET
from transformers import AutoModel, BertTokenizerFast
# import clip
import cv2
import shapely
import shapely.geometry
from shapely.geometry import Polygon, MultiPoint
from logger import write_to_record_file, print_progress, timeSince


def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    tensors = tensors.items()
    for line in tensors:
        print('{}\t{}'.format(*line))


# https://programmerah.com/using-shapely-geometry-polygon-to-calculate-the-iou-of-any-two-quadrilaterals-28395/
def compute_iou(a, b):
    a = np.array(a)  # quadrilateral two-dimensional coordinate representation
    poly1 = Polygon(
        a).convex_hull  # python quadrilateral object, will automatically calculate four points, the last four points in the order of: top left bottom right bottom right top left top
    # print(Polygon(a).convex_hull)  # you can print to see if this is the case

    b = np.array(b)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # Merge two box coordinates to become 8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # contains the smallest polygon point of the two quadrilaterals
    if not poly1.intersects(poly2):  # If the two quadrilaterals do not intersect
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # intersection area
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area)/(union_area-inter_area)  #wrong
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # The source code gives two ways to calculate IOU, the first one is: intersection part / area of the smallest polygon containing two quadrilaterals
            # The second one: intersection/merge (common way to calculate IOU of rectangular box)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def masked_navigation_losses(pred_direction, pred_progress, pred_goals,
                             gt_direction, gt_progress, gt_goal, active):
    """Compute the original per-sample losses with batched tensor operations."""
    true_sin_cos = torch.stack(
        (torch.sin(gt_direction), torch.cos(gt_direction)), dim=-1
    )
    active = active.to(dtype=pred_direction.dtype)

    direction_per_sample = (pred_direction - true_sin_cos).square().sum(dim=-1)
    progress_per_sample = (
        pred_progress.reshape(pred_progress.shape[0], -1)
        - gt_progress.reshape(gt_progress.shape[0], -1)
    ).square().sum(dim=-1)
    goal_per_sample = (pred_goals - gt_goal).square().reshape(
        pred_goals.shape[0], -1
    ).mean(dim=-1)

    return (
        (direction_per_sample * active).sum(),
        (progress_per_sample * active).sum(),
        (goal_per_sample * active).sum(),
    )


def advance_teacher_stage1_index(step_counts, episode_index, move_iteration, trajectory_length):
    """Advance one teacher trajectory without using other batch members' steps."""
    step_counts[episode_index] += 1
    index = int(step_counts[episode_index]) * move_iteration
    return index if index < trajectory_length else -1


def is_default_gpu(opts) -> bool:
    return opts.local_rank == -1 or dist.get_rank() == 0


def get_direction(start, end):
    vec = np.array(end) - np.array(start)
    _angle = 0
    #          90
    #      135    45
    #     180  .    0
    #      225   -45 
    #          270
    if vec[1] > 0:  # lng is postive
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle + 90) % 360
    return _angle


class NavCMTAgent:
    def __init__(self, args, allow_ngpus=True, rank=0):
        self.results = {}
        self.losses = []  # For learning agents
        self.args = args
        self.env = []
        self.env_name = ''
        random.seed(1)

        # RGB normalization values
        self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
        self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Models

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.lang_model = CustomBERTModel().cuda()

        # self.img_tensor = transforms.ToTensor()

        self.vision_model = Darknet(self.args.darknet_model_file, 224).cuda()

        new_state = torch.load(self.args.darknet_weight_file)
        state = self.vision_model.state_dict()
        model_keys = set(state.keys())
        state_dict = {k: v for k, v in new_state['model'].items() if k in model_keys}
        state.update(state_dict)
        self.vision_model.load_state_dict(state)



        # create the et model
        self.vln_model = ET(self.args).cuda()
        # self.map_encoder = MapEncoder(240)
        # self.goal_predictpr = GoalPredictor(240, 7)
        self.progress_regression = nn.MSELoss(reduction='sum')

        if self.args.world_size > 1 and allow_ngpus:
            self.lang_model = DDP(self.lang_model, broadcast_buffers=False, find_unused_parameters=True,
                                  device_ids=[self.args.local_rank], output_device=self.args.local_rank)
            self.vision_model = DDP(self.vision_model, broadcast_buffers=False, find_unused_parameters=True,
                                    device_ids=[self.args.local_rank], output_device=self.args.local_rank)
            self.vln_model = DDP(self.vln_model, broadcast_buffers=False, find_unused_parameters=True,
                                 device_ids=[self.args.local_rank], output_device=self.args.local_rank)

            # self.lang_model = nn.DataParallel(self.lang_model).cuda()
            # self.vision_model = nn.DataParallel(self.vision_model).cuda()
            # self.vln_model = nn.DataParallel(self.vln_model).cuda()
            self.lang_model_without_ddp = self.lang_model.module
            self.vision_model_without_ddp = self.vision_model.module
            self.vln_model_without_ddp = self.vln_model.module


        else:
            self.lang_model_without_ddp = self.lang_model
            self.vision_model_without_ddp = self.vision_model
            self.vln_model_without_ddp = self.vln_model

        # self.vln_model = ViT_LSTM(
        #     self.args, 
        #     self.vision_model).cuda()

        # optimizer        
        assert args.optim in ("adam", "adamW")
        OptimizerClass = torch.optim.Adam if args.optim == "adam" else torch.optim.AdamW
        self.et_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.vln_model.parameters()),
                                           lr=args.learning_rate)
        self.lang_model_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.lang_model.parameters()),
                                                   lr=self.args.learning_rate)
        self.vision_model_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.vision_model.parameters()),
                                                     lr=self.args.learning_rate)
        self.optimizers = (self.et_optimizer, self.lang_model_optimizer, self.vision_model_optimizer)
        # self.optimizers = (self.et_optimizer, self.lang_model_optimizer)

        #         # Optimizers
        # if self.args.optim == 'rms':
        #     optimizer = torch.optim.RMSprop
        # elif self.args.optim == 'adam':
        #     optimizer = torch.optim.Adam
        # elif self.args.optim == 'adamW':
        #     optimizer = torch.optim.AdamW
        # elif self.args.optim == 'sgd':
        #     optimizer = torch.optim.SGD
        # else:
        #     assert False
        # if self.default_gpu:
        #     print('Optimizer: %s' % self.args.optim)

        # self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        # self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        # self.lang_model_optimizer = optimizer(filter(lambda p: p.requires_grad, self.lang_model.parameters()), lr=self.args.lr)
        # self.vln_model_optimizer = optimizer(filter(lambda p: p.requires_grad, self.vln_model.parameters()), lr=self.args.lr)
        # self.optimizers = (self.lang_model_optimizer, self.vln_model_optimizer)

        # Evaluations
        self.losses = []
        self.progress_regression = nn.MSELoss(reduction='sum')
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction='sum')
        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def get_results(self):

        return self.results

    def visualize(self, loader, env_name='no_name_provided', feedback='student', not_in_train=False, **kwargs):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        self.env_name = env_name

        self.vln_model.eval()
        self.lang_model.eval()
        self.vision_model.eval()

        self.losses = []
        self.results = {}
        self.loss = 0
        idx = 0
        start = time.time()
        for l in loader:
            idx += 1
            for traj in self.rollout(visualize=True):  # loop for #batch times
                self.loss = 0
                self.results[traj['instr_id']] = traj
                # print(traj)
            tot = loader.dataset.size() / self.env.batch_size
            print_progress(idx, tot,
                           prefix='Progress:', suffix='%s (%d/%d)' % (
                    timeSince(start, float(idx) / tot), idx, tot), bar_length=80)


    @torch.inference_mode()
    def test(self, loader, env_name='no_name_provided', feedback='student', not_in_train=False, **kwargs):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        self.env_name = env_name

        self.vln_model.eval()
        self.lang_model.eval()
        self.vision_model.eval()

        self.losses = []
        self.results = {}
        self.loss = 0
        idx = 0
        start = time.time()
        for l in loader:
            idx += 1
            for traj in self.rollout(visualize=False):  # loop for #batch times
                self.loss = 0
                self.results[traj['instr_id']] = traj
                # print(traj)
            tot = loader.dataset.size() / self.env.batch_size
            print_progress(idx, tot,
                           prefix='Progress:', suffix='%s (%d/%d)' % (
                    timeSince(start, float(idx) / tot), idx, tot), bar_length=80)

    def train(self, loader, n_epochs, feedback='student', nss_w_weighting=1, **kwargs):
        ''' Train for a given number of epochs '''
        self.feedback = feedback

        self.lang_model.train()
        self.vln_model.train()
        self.vision_model.train()
        if getattr(self.args, 'trajectory_compact_mode', False):
            # One trainable selector. Frozen upstream weights stay in the
            # loaded checkpoint; their optimizer steps see no gradients.
            for model in (self.lang_model, self.vision_model):
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                model.eval()
            for name, parameter in self.vln_model_without_ddp.named_parameters():
                active_selector = (
                    name.startswith('trajectory_head.')
                    and not name.startswith(('trajectory_head.residual.',
                                             'trajectory_head.stop.'))
                )
                if not active_selector:
                    parameter.requires_grad_(False)
            # Disable dropout in the frozen ET backbone, retain dropout in
            # the trainable candidate selector.
            for name, module in self.vln_model_without_ddp.named_children():
                if name != 'trajectory_head':
                    module.eval()

        self.losses = []
        for epoch in range(1, n_epochs + 1):
            idx = 0
            start = time.time()
            # print('?')
            for _, l in enumerate(loader):
                idx += 1
                if self.args.benchmark_batches and idx > self.args.benchmark_batches:
                    break
                # if idx >= 100:
                #     break
                # train_loop_start_time = time.time()
                self.lang_model_optimizer.zero_grad(set_to_none=True)
                self.vision_model_optimizer.zero_grad(set_to_none=True)
                self.et_optimizer.zero_grad(set_to_none=True)
                self.loss = 0

                profile_backward = bool(getattr(self.args, 'profile_rollout', False))
                sequential = (feedback == 'student' and
                              bool(getattr(self.args, 'trajectory_sequential_backward', False)))

                def apply_backward(loss):
                    if profile_backward:
                        torch.cuda.synchronize()
                        backward_started = time.perf_counter()
                    loss.backward()
                    if profile_backward:
                        torch.cuda.synchronize()
                        duration = time.perf_counter() - backward_started
                        self.logs['profile_backward_seconds'].append(duration)
                        self.logs[f'profile_{self.feedback}_backward_seconds'].append(duration)

                if sequential:
                    # The teacher and student rollouts construct independent
                    # graphs, so there is no reason to retain both until the
                    # last backward. We preserve both losses/gradients and
                    # perform exactly ONE optimizer step afterwards.
                    def teacher_forward():
                        self.feedback = 'teacher'
                        self.loss = 0
                        self.rollout(train_ml=self.args.ml_weight)
                        return self.loss

                    def student_forward():
                        self.feedback = 'student'
                        self.loss = 0
                        self.rollout(train_ml=self.args.ml_weight)
                        return self.loss

                    self.loss = backward_rollout_pair(
                        teacher_forward, student_forward,
                        backward=apply_backward,
                    )
                else:
                    if feedback == 'teacher':
                        self.feedback = 'teacher'
                        self.rollout(train_ml=self.args.teacher_weight)
                    elif feedback == 'student':
                        self.feedback = 'teacher'
                        self.rollout(train_ml=self.args.ml_weight)
                        self.feedback = 'student'
                        self.rollout(train_ml=self.args.ml_weight)
                    else:
                        raise ValueError(f'Unsupported feedback: {feedback}')
                    apply_backward(self.loss)
                if hasattr(self, 'experiment_gradient_callback') and self.experiment_gradient_callback:
                    self.experiment_gradient_callback(self, idx)
                # print('suc')

                torch.nn.utils.clip_grad_norm_(self.vln_model.parameters(), 40.)

                self.lang_model_optimizer.step()
                self.vision_model_optimizer.step()
                self.et_optimizer.step()
                # print("---------- One iter takes %s seconds ---" % (time.time() - train_loop_start_time))

                if self.default_gpu:
                    tot = n_epochs * loader.dataset.size() / self.env.batch_size
                    # print('is')
                    print_progress(idx, tot,
                                   prefix='Progress:', suffix='%s (%d/%d)' % (
                            timeSince(start, float(idx) / tot), idx, tot), bar_length=80)
            if self.args.benchmark_batches:
                elapsed = time.time() - start
                print('\nBENCHMARK batches=%d seconds=%.3f seconds_per_batch=%.6f' % (
                    min(idx, self.args.benchmark_batches), elapsed,
                    elapsed / max(min(idx, self.args.benchmark_batches), 1)
                ), flush=True)
                break

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def rollout(self, train_ml=None, visualize=False):

        # Profiling is opt-in: synchronizing CUDA on each boundary would
        # otherwise make the normal training path significantly slower.
        profile_enabled = bool(getattr(self.args, 'profile_rollout', False))
        phase_seconds = defaultdict(float)
        def phase_clock():
            if profile_enabled:
                torch.cuda.synchronize()
                return time.perf_counter()
            return 0.0

        start_reset = phase_clock()
        if (getattr(self.args, 'trajectory_use_for_control', False)
                and not getattr(self.args, 'heatmap_trajectory_enabled', False)):
            raise ValueError("trajectory control requires an enabled trajectory head")
        if (getattr(self.args, 'trajectory_use_for_control', False)
                and getattr(self.args, 'heatmap_execution', 'two_stage') == 'waypoint'):
            raise ValueError(
                "trajectory_use_for_control and heatmap_execution=waypoint "
                "are mutually exclusive: waypoint execution otherwise "
                "silently bypasses the trajectory planner")
        obs = self.env._get_obs(random_direction=(self.feedback == 'teacher'))
        phase_seconds['observation_reset'] += phase_clock() - start_reset if profile_enabled else 0.0
        batch_size = len(obs)

        # Language input
        lang_inputs = []
        for i, ob in enumerate(obs):
            # if self.args.vision_only:
            #     lang_inputs.append('')
            # else:
            lang_inputs.append(ob['instruction'])
        encoding = self.tokenizer(lang_inputs, padding=True, return_tensors="pt")
        input_ids = encoding['input_ids'].cuda()
        attention_mask = encoding['attention_mask'].cuda()
        if train_ml is not None and getattr(self.args, 'trajectory_compact_mode', False):
            with torch.no_grad():
                lang_features, linear_cls, cls_hidden = self.lang_model(input_ids, attention_mask)
        else:
            lang_features, linear_cls, cls_hidden = self.lang_model(input_ids, attention_mask)
        # Compute static landmark/text alignment ONCE per episode batch.
        # Dynamic UAV pose still updates at every rollout step.
        multi_landmarks = None
        if getattr(self.args, 'heatmap_multi_landmark', False):
            multi_landmarks = build_landmark_batch(
                obs, self.tokenizer, input_ids,
                max_landmarks=self.args.heatmap_max_landmarks,
            )
            matched = multi_landmarks['landmark_text_mask'].any(-1)
            real = multi_landmarks['landmark_valid']
            self.logs['landmark_refs_total'].append(float(real.sum().item()))
            self.logs['landmark_refs_name_matched'].append(
                float((matched & real).sum().item())
            )
            self.logs['landmark_refs_truncated'].append(
                float(multi_landmarks['landmark_truncated'])
            )

        # lang_features --> 768
        # linear_cls --> 49 (used to attend to img features)
        # c_0 = cls_hidden

        # print(lang_features.size()) # batch_size*sequence_length*768

        # Record starting points of the current batch
        current_directions = [np.array(ob['pose'].yaw, dtype=np.float32) for ob in obs]
        current_positions = [np.array(ob['position'], dtype=np.float32) for ob in obs]
        poses = [ob['pose'] for ob in obs]
        direction_t = torch.from_numpy(np.array(current_directions, dtype=np.float32))
        position_t = torch.from_numpy(np.array(current_positions, dtype=np.float32))
        traj = [defaultdict(list) for ob in obs]

        global_position = np.stack([np.array([i, j], dtype=np.float32)
                                    for i in range(self.args.grid_size)
                                    for j in range(self.args.grid_size)]) / self.args.grid_size

        global_positions = torch.from_numpy(np.stack([global_position
                                                      for _ in range(batch_size)
                                                      ])).cuda()

        for i, ob in enumerate(obs):
            traj[i]['goal'] = ob['goal']
            traj[i]['instr_id'] = ob['id']
            # rounds = lang_inputs[i].split('[QUE]')
            # remove = 0
            # for r in rounds:
            #     if 'Yes' in r[0:5]:
            #         remove += 1
            # traj[i]['num_dia'] = len(rounds) - remove
            # traj[i]['path_corners'] = [(np.array(ob['gt_path_corners'][0]), ob['starting_angle'])]
            traj[i]['gt_trajectory'] = ob['trajectory']
            traj[i]['trajectory'] = [poses[i]]
            traj[i]['stage1_trajectory'] = [poses[i]]
        # print(np.array([len(ob['trajectory']) for ob in obs]))

        # Initialization the finishing status
        ended = np.array([False] * batch_size)
        waypoint_stagnant_steps = np.zeros(batch_size, dtype=np.int32)

        # Init the logs
        # ml_loss = 0.
        direction_loss = torch.tensor(0.).cuda()
        progress_loss = 0.
        goal_predict_loss = 0.
        heatmap_loss = torch.tensor(0.).cuda()
        trajectory_loss = torch.tensor(0.).cuda()
        trajectory_stop_loss = torch.tensor(0.).cuda()
        trajectory_ranking_loss = torch.tensor(0.).cuda()
        trajectory_candidate_loss = torch.tensor(0.).cuda()
        trajectory_ranking_valid_count = torch.zeros((), device='cuda')
        trajectory_candidate_eval_count = torch.zeros((), device='cuda')
        trajectory_prior_goal_hits = torch.zeros((), device='cuda')
        trajectory_joint_goal_hits = torch.zeros((), device='cuda')
        trajectory_oracle_goal_hits = torch.zeros((), device='cuda')
        trajectory_prior_fde_sum_m = torch.zeros((), device='cuda')
        trajectory_joint_fde_sum_m = torch.zeros((), device='cuda')
        trajectory_stop_decisions = 0
        trajectory_stop_correct = 0
        trajectory_stop_positive_sum = torch.zeros((), device='cuda')
        trajectory_stop_active_sum = torch.zeros((), device='cuda')
        trajectory_arrival_gate_blocked = 0
        trajectory_plan_steps = 0
        trajectory_goal_switches = 0
        trajectory_travel_distance_m = 0.0
        previous_plan_goal_id = np.full(batch_size, -1, dtype=np.int64)
        trajectory_supervision_count = torch.zeros((), device='cuda')
        trajectory_minade_sum = torch.zeros((), device='cuda')
        trajectory_minfde_sum = torch.zeros((), device='cuda')
        trajectory_eval_count = torch.zeros((), device='cuda')
        heatmap_diag_count = torch.zeros((), device='cuda')
        heatmap_coverage_hits = {
            k: torch.zeros((), device='cuda') for k in (1, 4, 5, 8, 16, 20)
        }
        heatmap_top1_distance_sum = torch.zeros((), device='cuda')
        heatmap_refined_top1_distance_sum = torch.zeros((), device='cuda')
        heatmap_top16_nearest_distance_sum = torch.zeros((), device='cuda')
        heatmap_top16_candidate_distance_sum = torch.zeros((), device='cuda')

        stage1_step = 0
        teacher_stage1_steps = np.zeros(batch_size, dtype=np.int32)
        stage2_step = 0
        stage2_rotate = 0

        input = {
            'directions': torch.zeros((batch_size, 0, 4)).cuda(),
            'grid_fts': torch.zeros(batch_size, 0, 768).cuda(),
            'grid_index': torch.zeros(batch_size, 0).cuda(),
            'frames': torch.zeros(batch_size, 0, 512, 49).cuda(),
            'lenths': [0 for _ in range(batch_size)],
            'lang': lang_features,
            'lang_mask': attention_mask.bool(),
            'candidates': global_positions,
            'centroids': torch.zeros((batch_size, 0, 2)).cuda(),
            'lang_cls': linear_cls,
            'map_fts': torch.zeros(batch_size, 0, 512, 49).cuda(),
        }

        if multi_landmarks is not None:
            for key in ('landmark_xy', 'landmark_extent', 'landmark_valid',
                        'landmark_text_mask'):
                input[key] = multi_landmarks[key]

        stage1_ended = np.array([False] * batch_size)
        stage2_recover_count = np.zeros(batch_size, dtype=np.int32)
        stage2_recoveries = 0
        stage2_entry_count = 0
        stage2_entry_gt_distance_sum = 0.0
        stage2_entry_gt_bins = {
            'le20': 0,
            '20_30': 0,
            '30_40': 0,
            'gt40': 0,
        }

        # Only previously observed student/teacher poses; never future states.
        # Keep a short causal window to support revisits/progress evidence.
        trajectory_pose_history = torch.empty((batch_size, 0, 2), device='cuda')
        # ADE/FDE/oracle statistics are diagnostics, not training losses.
        # Their old per-step .item() calls serialized the GPU. Keep them for
        # full validation or explicit train-diagnostics runs only.
        compact_mode = bool(getattr(self.args, 'trajectory_compact_mode', False))
        collect_traj_diagnostics = (
            (train_ml is None and not getattr(self.args, 'trajectory_fast_eval', False))
            or (train_ml is not None and getattr(self.args, 'trajectory_train_diagnostics', False))
        )
        for t in range(self.args.max_action_len):
            step_start = phase_clock()

            # print("- action rollingout takes %s seconds ---" % (time.time() - rollingout_action_start_time))
            # rollingout_action_start_time = time.time()
            images = []
            for i in range(len(obs)):
                images.append(obs[i]['rgb'].copy())
            images = np.stack(images)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # W x H x C to C x W x H
            images = np.ascontiguousarray(images, dtype=np.float32)
            images -= self.rgb_mean
            images /= self.rgb_std
            if compact_mode and train_ml is not None:
                with torch.no_grad():
                    im_feature = self.vision_model(torch.from_numpy(images).cuda())
            else:
                im_feature = self.vision_model(torch.from_numpy(images).cuda())
            im_feature = im_feature.view(im_feature.size(0), im_feature.size(1), -1)
            after_vision = phase_clock()
            if profile_enabled:
                phase_seconds['visual_preprocess_and_darknet'] += after_vision - step_start






            current_direct = direction_t.view(-1, 1).cuda()
            current_pos = position_t.view(-1, 2).cuda()
            trajectory_pose_history = torch.cat(
                (trajectory_pose_history, current_pos[:, None, :].detach()), dim=1
            )[:, -10:]
            input['trajectory_history_xy'] = trajectory_pose_history
            direction = torch.concat(
                (torch.sin(current_direct), torch.cos(current_direct), current_pos), axis=1)

            # if self.args.no_direction:
            #     input['directions'] = torch.hstack((input['directions'], torch.zeros_like(direction.view(-1, 1, 2))))
            # else:
            input['directions'] = direction.view(-1, 1, 4)
            # if self.args.language_only:
            #     input['frames'] = torch.hstack((input['frames'], torch.zeros_like(im_feature.view(-1, 1, 512, 49))))
            # else:
            # print(input['frames'].shape, im_feature.shape)
            input['frames'] = im_feature.view(-1, 1, 512, 49)
            input['maps'] = torch.from_numpy(np.array([ob['maps'] for ob in obs], dtype=np.float32)).cuda()


            centroid_lens = np.array(len(ob['centroids']) for ob in obs)
            input['centroids'] = torch.from_numpy(np.array([ob['centroids'] for ob in obs], dtype=np.float32)).cuda()
            # input['directions'] = direction.view(-1,1,2)
            # input['frames'] = im_feature.view(-1,1, 512,49)

            for i in range(len(obs)):
                if not ended[i]:
                    input['lenths'][i] += 1

            # print('.')





            # The true target is a teacher-only conditioning signal for
            # imitation loss; it is NEVER present in student inference.
            teacher_goal_tensor = None
            if train_ml is not None and getattr(self.args, 'heatmap_trajectory_enabled', False):
                teacher_goal_tensor = torch.as_tensor(
                    np.asarray([ob['normalized_goal'] for ob in obs], dtype=np.float32),
                    device=input['directions'].device
                )
            model_outputs = self.vln_model(
                directions=input['directions'],
                frames=input['frames'],
                lenths=input['lenths'],
                grid_fts=input['grid_fts'],
                grid_index=input['grid_index'],
                maps=input['maps'],
                lang=input['lang'],
                lang_mask=input['lang_mask'],
                trajectory_history_xy=input['trajectory_history_xy'],
                candidates=input['candidates'],
                centroids=input['centroids'],
                lang_cls=input['lang_cls'],
                **({key: input[key] for key in (
                    'landmark_xy', 'landmark_extent', 'landmark_valid',
                    'landmark_text_mask'
                )} if multi_landmarks is not None else {}),
                **({'trajectory_teacher_goal': teacher_goal_tensor}
                   if teacher_goal_tensor is not None else {})
            )
            after_model = phase_clock()
            if profile_enabled:
                phase_seconds['et_belief_and_planning'] += after_model - after_vision
            trajectory_predictions = None
            teacher_trajectory_predictions = None
            if len(model_outputs) == 6:
                pred_direction, pred_progress, pred_goals, pred_logits, grid_ft, traj_pair = model_outputs
                proposal_tensors, teacher_trajectory_predictions = traj_pair
                trajectory_predictions = TrajectoryProposals(*proposal_tensors)
            else:
                pred_direction, pred_progress, pred_goals, pred_logits, grid_ft = model_outputs

            # Dense Stage-1 spatial belief: softmax -> greedy NMS Top-K.
            heatmap_probs = torch.softmax(pred_logits, dim=1)
            heatmap_topk_ids = greedy_nms_topk(
                heatmap_probs.reshape(
                    -1,
                    self.args.heatmap_grid_size,
                    self.args.heatmap_grid_size,
                ),
                top_k=self.args.heatmap_top_k,
                kernel_size=self.args.heatmap_nms_kernel,
            )
            heatmap_goal_ids = heatmap_topk_ids[:, 0]
            heatmap_goal_rows = torch.div(
                heatmap_goal_ids,
                self.args.heatmap_grid_size,
                rounding_mode='floor',
            ).float()
            heatmap_goal_cols = (
                heatmap_goal_ids % self.args.heatmap_grid_size
            ).float()
            # Refine the Top-1 cell to a continuous coordinate using only its
            # local probability mass; this preserves the selected mode while
            # avoiding a hard jump to the cell center.
            heatmap_goals = local_soft_argmax_xy(
                heatmap_probs.reshape(
                    -1,
                    self.args.heatmap_grid_size,
                    self.args.heatmap_grid_size,
                ),
                heatmap_goal_ids,
                window_size=self.args.heatmap_local_window,
            )

            input['grid_fts'] = torch.cat((input['grid_fts'], grid_ft), dim=1)
            grid_index = torch.tensor(np.array([ob['cur_grid'] for ob in obs])).unsqueeze(1).cuda()
            # print(input['grid_index'], grid_index)

            input['grid_index'] = torch.cat((input['grid_index'], grid_index), dim=1)
            # print("- model prediction takes %s seconds ---" % (time.time() - rollingout_action_start_time))
            # pred_direction = output
            # pred_progress = progress

            # Transfer all values used by the CPU simulator in one synchronization.
            nt_direct = torch.atan2(pred_direction[:, 0], pred_direction[:, 1])
            host_predictions = torch.cat((
                pred_progress.reshape(batch_size, -1)[:, :1],
                nt_direct.unsqueeze(1),
                heatmap_goals,
                heatmap_goal_ids.unsqueeze(1),
                heatmap_probs.gather(1, heatmap_goal_ids.unsqueeze(1)),
            ), dim=1).detach().cpu().numpy()
            selected_trajectory_paths = None
            selected_trajectory_goals = None
            compact_local_waypoints = None
            if (compact_mode and trajectory_predictions is not None and
                    self.feedback == 'student'):
                # Goal-first selection. The mode probability must never
                # change which geographic goal wins the ranking.
                goal_scores = torch.logsumexp(
                    trajectory_predictions.joint_logits, dim=-1)
                if getattr(self.args, 'trajectory_selector_mode', 'joint') == 'prior':
                    goal_scores = heatmap_probs.gather(
                        1, trajectory_predictions.goal_ids).clamp_min(1e-8).log()
                goal_index = goal_scores.argmax(-1)
                rows = torch.arange(batch_size, device=goal_index.device)
                if getattr(self.args, 'trajectory_compact_refine_goal', False):
                    # Execute the SELECTED candidate with the same local
                    # soft-argmax refinement as the historical baseline:
                    # NMS cell centres cost up to ~10 m of the 20 m radius.
                    selected_ids = trajectory_predictions.goal_ids[rows, goal_index]
                    selected_trajectory_goals = local_soft_argmax_xy(
                        heatmap_probs.reshape(
                            -1, self.args.heatmap_grid_size,
                            self.args.heatmap_grid_size),
                        selected_ids,
                        window_size=self.args.heatmap_local_window,
                    ).detach().cpu().numpy()
                else:
                    selected_trajectory_goals = trajectory_predictions.goal_xy[
                        rows, goal_index].detach().cpu().numpy()
                if getattr(self.args, 'trajectory_compact_enable_path', False):
                    mode_index = trajectory_predictions.mode_logits[
                        rows, goal_index].argmax(-1)
                    compact_local_waypoints = trajectory_predictions.trajectories[
                        rows, goal_index, mode_index, -1, :].detach().cpu().numpy()
            selected_stop_probs = None
            selected_plan_goal_ids = None
            if (trajectory_predictions is not None and self.feedback == 'student'
                    and getattr(self.args, 'trajectory_use_for_control', False)):
                if getattr(self.args, 'trajectory_selector_mode', 'prior') == 'joint':
                    scores = trajectory_predictions.joint_logits.flatten(1)
                else:
                    # Original behavior: heatmap goal prior + mode probability.
                    candidate_prior = heatmap_probs.gather(
                        1, trajectory_predictions.goal_ids
                    ).clamp_min(1e-8).log()
                    scores = (candidate_prior[:, :, None] + F.log_softmax(
                        trajectory_predictions.mode_logits, dim=-1
                    )).flatten(1)
                selected_ids = select_goal_mode_indices(scores.view(batch_size, *trajectory_predictions.mode_logits.shape[1:]))
                plans = trajectory_predictions.trajectories.flatten(1, 2)
                chosen = plans[torch.arange(batch_size, device=plans.device), selected_ids]
                selected_trajectory_paths = chosen.detach().cpu().numpy()
                selected_trajectory_goals = selected_trajectory_paths[:, -1]
                selected_goal_index = torch.div(
                    selected_ids, trajectory_predictions.mode_logits.shape[-1],
                    rounding_mode='floor'
                )
                selected_plan_goal_ids = trajectory_predictions.goal_ids.gather(
                    1, selected_goal_index.unsqueeze(1)
                ).squeeze(1).detach().cpu().numpy()
                selected_stop_probs = torch.sigmoid(
                    trajectory_predictions.stop_logits).detach().cpu().numpy()
            pred_progress_t = host_predictions[:, 0]
            at_direction = host_predictions[:, 1]
            # for i in range(len(a_t_next_pos_ratio)):
            #     max_of_a_t_next_pos_i = max(abs(a_t_next_pos_ratio[i][0]), abs(a_t_next_pos_ratio[i][1]), 1)
            #     a_t_next_pos_ratio[i][0] /= max_of_a_t_next_pos_i
            #     a_t_next_pos_ratio[i][1] /= max_of_a_t_next_pos_i

            # Predicted altitude
            # a_t_altitude = pred_altitude.cpu().detach().numpy()
            #
            # # Clip the prediction to (0,1)
            # for i in range(len(a_t_altitude)):
            #     a_t_altitude[i] = min(1., max(0., a_t_altitude[i]))
            # for i in range(len(pred_progress_t)):
            #     pred_progress_t[i] = min(1., max(0., pred_progress_t[i]))
            gt_direction = np.array([ob['direction'] for ob in obs], dtype=np.float32)
            gt_goal_np = np.array([ob['normalized_goal'] for ob in obs], dtype=np.float32)
            gt_progress_np = np.array([ob['progress'] for ob in obs], dtype=np.float32)
            gt_goal = torch.from_numpy(gt_goal_np)
            gt_progress = torch.from_numpy(gt_progress_np)
            gt_target = torch.from_numpy(np.array([ob['grid_goal'] for ob in obs], dtype=np.int64))
            # there is no ground truth in unseen_test set
            if (not 'test' in self.env_name and
                    (train_ml is None or not compact_mode) and
                    (train_ml is not None or not getattr(self.args, 'trajectory_fast_eval', False))):
                # Expensive supervised heatmap diagnostics are unnecessary
                # for official-metric-only fast evaluation.
                # Get ground truth
                # print(t, target, gt_progress)

                # Compute loss

                device = pred_direction.device
                gt_values = torch.as_tensor(
                    np.concatenate((
                        gt_direction[:, None],
                        gt_progress_np[:, None],
                        gt_goal_np,
                    ), axis=1),
                    device=device,
                )
                step_direction_loss, step_progress_loss, step_goal_loss = masked_navigation_losses(
                    pred_direction,
                    pred_progress,
                    pred_goals,
                    gt_values[:, 0],
                    gt_values[:, 1],
                    gt_values[:, 2:4],
                    torch.as_tensor(~ended, device=device),
                )
                direction_loss = direction_loss + step_direction_loss
                progress_loss = progress_loss + step_progress_loss
                goal_predict_loss = goal_predict_loss + step_goal_loss
                # print(at_direction, gt_direction, ml_loss)
                gt_heatmap = metric_gaussian_target(
                    torch.as_tensor(
                        gt_goal_np,
                        dtype=torch.float32,
                        device=pred_logits.device,
                    ),
                    field_size=self.args.heatmap_grid_size,
                    sigma_m=self.args.heatmap_sigma_m,
                    map_meters=self.args.map_meters,
                ).flatten(1)
                per_sample_heatmap_loss = -(
                    gt_heatmap * F.log_softmax(pred_logits, dim=1)
                ).sum(dim=1)
                active_bool = torch.as_tensor(~ended, device=pred_logits.device)
                active_mask = active_bool.float()
                heatmap_loss += (per_sample_heatmap_loss * active_mask).sum()

                # Evaluate the retained NMS proposals against the metric GT.
                # Coverage uses the same success radius as navigation metrics.
                candidate_rows = torch.div(
                    heatmap_topk_ids,
                    self.args.heatmap_grid_size,
                    rounding_mode='floor',
                ).float()
                candidate_cols = (
                    heatmap_topk_ids % self.args.heatmap_grid_size
                ).float()
                # Convert [row=y, col=x] belief indices back to normalized
                # world (x, y) before computing metric candidate distances.
                candidate_xy = torch.stack(
                    (
                        (candidate_cols + 0.5) / self.args.heatmap_grid_size,
                        (candidate_rows + 0.5) / self.args.heatmap_grid_size,
                    ),
                    dim=-1,
                )
                gt_xy = torch.as_tensor(
                    gt_goal_np, dtype=candidate_xy.dtype, device=candidate_xy.device
                )
                candidate_distances_m = torch.linalg.vector_norm(
                    (candidate_xy - gt_xy.unsqueeze(1)) * self.args.map_meters,
                    dim=-1,
                )
                active_count = active_bool.sum()
                heatmap_diag_count += active_count
                for k in (1, 4, 5, 8, 16, 20):
                    coverage_k = min(k, candidate_distances_m.shape[1])
                    covered = (
                        candidate_distances_m[:, :coverage_k].min(dim=1).values
                        <= self.args.success_dist
                    )
                    heatmap_coverage_hits[k] += (covered & active_bool).sum()
                heatmap_top1_distance_sum += (
                    candidate_distances_m[:, 0] * active_mask
                ).sum()
                refined_top1_distance_m = torch.linalg.vector_norm(
                    (heatmap_goals - gt_xy) * self.args.map_meters,
                    dim=-1,
                )
                heatmap_refined_top1_distance_sum += (
                    refined_top1_distance_m * active_mask
                ).sum()
                heatmap_top16_nearest_distance_sum += (
                    candidate_distances_m.min(dim=1).values * active_mask
                ).sum()
                heatmap_top16_candidate_distance_sum += (
                    candidate_distances_m.mean(dim=1) * active_mask
                ).sum()
                # print(pred_logits.shape)
            # Evaluate trajectories against teacher suffixes without feeding
            # future teacher states to the policy. A teacher-forced GT endpoint
            # supplies the imitation target only, not the inference candidates.
            if (trajectory_predictions is not None and 'test' not in self.env_name
                    and (train_ml is not None or collect_traj_diagnostics)):
                valid_trajectory = torch.as_tensor(~ended, device=pred_logits.device)
                # Compact goal-only training never reads the teacher suffix:
                # skip the per-sample CPU resampling when path loss is off.
                need_targets = not (
                    train_ml is not None and compact_mode
                    and float(self.args.trajectory_candidate_loss_weight) == 0.0
                    and not collect_traj_diagnostics)
                targets = None if not need_targets else torch.stack([
                    resample_teacher_suffix(
                        ob['trajectory'], ob['position'],
                        map_name=ob['map_name'], bounds=MAP_BOUNDS,
                        map_meters=self.args.map_meters,
                        steps=self.args.trajectory_waypoints,
                        goal_xy=ob['normalized_goal'],
                    ) for ob in obs
                ]).to(device=pred_logits.device)
                if train_ml is not None and compact_mode:
                    target_xy = torch.as_tensor(
                        gt_goal_np, dtype=pred_logits.dtype, device=pred_logits.device)
                    active_count = valid_trajectory.sum()
                    trajectory_ranking_loss += goal_distance_soft_ranking_loss(
                        trajectory_predictions, target_xy,
                        map_meters=self.args.map_meters,
                        temperature_m=self.args.trajectory_goal_soft_temperature_m,
                        active=valid_trajectory) * active_count
                    nearest_goal_distance_m = (
                        trajectory_predictions.goal_xy.detach() - target_xy[:, None]
                    ).norm(dim=-1).min(-1).values * self.args.map_meters
                    rank_mask = (nearest_goal_distance_m <= self.args.success_dist) & valid_trajectory
                    trajectory_ranking_valid_count += rank_mask.sum()
                    if targets is not None:
                        path_loss, path_valid_count = local_path_soft_ranking_loss(
                            trajectory_predictions, target_xy, current_pos, targets[:, 0],
                            map_meters=self.args.map_meters,
                            temperature_m=self.args.trajectory_path_soft_temperature_m,
                            positive_radius_m=self.args.success_dist,
                            local_step_m=self.args.heatmap_waypoint_step_m,
                            active=valid_trajectory)
                        trajectory_candidate_loss += path_loss * path_valid_count
                    trajectory_supervision_count += active_count
                elif train_ml is not None and teacher_trajectory_predictions is not None:
                    trajectory_loss = trajectory_loss + trajectory_imitation_loss(
                        teacher_trajectory_predictions, targets,
                        active=valid_trajectory
                    ) * valid_trajectory.sum()
                    # Positive only when close to target AND near the end of
                    # remaining teacher supervision; avoid early-stop collapse.
                    goal_distance_m = (
                        torch.as_tensor(
                            np.asarray([ob['normalized_goal'] for ob in obs], dtype=np.float32),
                            device=pred_logits.device
                        ) - current_pos
                    ).norm(dim=-1) * self.args.map_meters
                    # Align stop labels to the actual evaluation success
                    # criterion. Teacher suffix distance was an unrelated
                    # extra constraint that suppressed valid positive labels.
                    stop_target = arrival_stop_targets(
                        current_pos,
                        torch.as_tensor(gt_goal_np, dtype=current_pos.dtype,
                                        device=current_pos.device),
                        map_meters=self.args.map_meters,
                        success_radius_m=self.args.success_dist,
                    )
                    trajectory_stop_positive_sum += (
                        stop_target.detach() * valid_trajectory).sum()
                    trajectory_stop_active_sum += valid_trajectory.sum()
                    trajectory_stop_loss = trajectory_stop_loss + stop_supervision_loss(
                        trajectory_predictions.stop_logits, stop_target,
                        active=valid_trajectory, pos_weight=2.,
                    ) * valid_trajectory.sum()
                    trajectory_supervision_count += valid_trajectory.sum()
                    # Predicted-goal supervision: a sample is labeled only if
                    # its predicted candidate pool has a GT-nearby proposal.
                    # Neither GT positions nor teacher paths enter the scorer.
                    target_xy = torch.as_tensor(
                        gt_goal_np, dtype=pred_logits.dtype, device=pred_logits.device)
                    nearest_candidate_m = (
                        trajectory_predictions.goal_xy - target_xy[:, None]
                    ).norm(dim=-1).min(dim=-1).values * self.args.map_meters
                    rank_valid = (nearest_candidate_m <= self.args.success_dist) & valid_trajectory
                    rank_count = rank_valid.sum()
                    trajectory_ranking_valid_count += rank_count
                    trajectory_ranking_loss = trajectory_ranking_loss + candidate_ranking_loss(
                        trajectory_predictions, target_xy,
                        map_meters=self.args.map_meters,
                        positive_radius_m=self.args.success_dist,
                        active=valid_trajectory,
                    ) * rank_count
                    trajectory_candidate_loss = trajectory_candidate_loss + (
                        candidate_trajectory_imitation_loss(
                            trajectory_predictions, target_xy, targets,
                            map_meters=self.args.map_meters,
                            positive_radius_m=self.args.success_dist,
                            active=valid_trajectory,
                        ) * rank_count
                    )
                # Oracle-best of all generated proposals: diagnostic ONLY.
                # Skip this entire block in training by default.
                if collect_traj_diagnostics:
                    with torch.no_grad():
                        candidate_paths = trajectory_predictions.trajectories.reshape(
                            batch_size, -1, self.args.trajectory_waypoints, 2
                        )
                        ade = (candidate_paths - targets[:, None]).norm(dim=-1).mean(-1)
                        fde = (candidate_paths[:, :, -1] - targets[:, None, -1]).norm(dim=-1)
                        trajectory_minade_sum += (
                            (ade.min(-1).values * valid_trajectory).sum()
                        ) * self.args.map_meters
                        trajectory_minfde_sum += (
                            (fde.min(-1).values * valid_trajectory).sum()
                        ) * self.args.map_meters
                        trajectory_eval_count += valid_trajectory.sum()
                        # Fair prior-vs-joint per-step candidate diagnostic. Both
                        # use the same predicted proposals and never GT selection.
                        # Oracle only checks if any candidate is within the radius.
                        d_goal = (trajectory_predictions.goal_xy - torch.as_tensor(
                            gt_goal_np, dtype=pred_logits.dtype, device=pred_logits.device
                        )[:, None]).norm(dim=-1) * self.args.map_meters
                        p_log = heatmap_probs.gather(
                            1, trajectory_predictions.goal_ids
                        ).clamp_min(1e-8).log()
                        prior_scores = p_log[:, :, None] + F.log_softmax(
                            trajectory_predictions.mode_logits, dim=-1)
                        prior_indices = select_goal_mode_indices(prior_scores)
                        joint_indices = select_goal_mode_indices(
                            trajectory_predictions.joint_logits)
                        modes = trajectory_predictions.mode_logits.shape[-1]
                        prior_goals = torch.div(prior_indices, modes, rounding_mode='floor')
                        joint_goals = torch.div(joint_indices, modes, rounding_mode='floor')
                        prior_dist = d_goal.gather(1, prior_goals[:, None]).squeeze(1)
                        joint_dist = d_goal.gather(1, joint_goals[:, None]).squeeze(1)
                        trajectory_prior_goal_hits += (
                            ((prior_dist <= self.args.success_dist) & valid_trajectory).sum())
                        trajectory_joint_goal_hits += (
                            ((joint_dist <= self.args.success_dist) & valid_trajectory).sum())
                        trajectory_oracle_goal_hits += (
                            ((d_goal.min(-1).values <= self.args.success_dist) & valid_trajectory).sum())
                        rows = torch.arange(batch_size, device=pred_logits.device)
                        trajectory_prior_fde_sum_m += (
                            fde[rows, prior_indices] * valid_trajectory
                        ).sum() * self.args.map_meters
                        trajectory_joint_fde_sum_m += (
                            fde[rows, joint_indices] * valid_trajectory
                        ).sum() * self.args.map_meters
                        trajectory_candidate_eval_count += valid_trajectory.sum()

            # Log the trajectory
            # print(at_direction, gt_direction)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['actions'].append(at_direction[i])
                    if not 'test' in self.env_name:
                        traj[i]['gt_actions'].append(gt_direction[i])
                        traj[i]['gt_progress'].append(gt_progress[i].item())
                        traj[i]['gt_goal'].append(gt_goal[i])
                    traj[i]['progress'].append(float(host_predictions[i, 0]))
                    traj[i]['heatmap_goal_id'].append(int(host_predictions[i, 4]))
                    traj[i]['heatmap_confidence'].append(float(host_predictions[i, 5]))
                    traj[i]['heatmap_topk_ids'].append(
                        heatmap_topk_ids[i].detach().cpu().tolist()
                    )

            if self.feedback == 'teacher':
                # print('teacher', at_goal.shape)
                a_t = gt_direction
                pred_progress_t = gt_progress
                cpu_goal = gt_goal.numpy()
            elif self.feedback == 'student':  # student
                a_t = at_direction
                # Use the highest-probability heatmap cell as the coarse Stage-1 goal.
                cpu_goal = (selected_trajectory_goals
                            if selected_trajectory_goals is not None
                            else host_predictions[:, 2:4])

                # _, at_goal = pred_logits.max(1)
                # at_goal = at_goal.squeeze(1)
                # at_goal = gt_goal
                # print('student', at_goal.shape)
            else:
                sys.exit('Invalid feedback option')

            # print(cpu_goal)

            # Interact with the simulator with actions
            for i in range(len(obs)):

                dst = self.env.unnormalize_position(cpu_goal[i], obs[i]['map_name'],
                                                    self.args.map_meters)

                # gt_center = self.env.unnormalize_position(global_position[gt_goal.cpu().detach().numpy()[i]], obs[i]['map_name'],
                #                                     self.args.map_meters)
                # dst = Point2D(obs[i]['centroid_goal'][0], obs[i]['centroid_goal'][1])
                if ended[i]:
                    continue

                if (self.feedback == 'student'
                        and getattr(self.args, 'heatmap_execution', 'two_stage') == 'waypoint'):
                    old_pose = poses[i]
                    # The compact path head supplies a LOCAL waypoint. Keep
                    # bounded execution and the original global goal separate.
                    action_waypoint = (self.env.unnormalize_position(
                        compact_local_waypoints[i], obs[i]['map_name'],
                        self.args.map_meters)
                        if compact_local_waypoints is not None else dst)
                    poses[i] = bounded_heatmap_step(
                        poses[i], action_waypoint,
                        # ONE action budget for A/B/C. Compact paths only
                        # change heading, never impose a 20m-only cap.
                        max_step_m=self.args.heatmap_waypoint_step_m,
                    )
                    moved = poses[i].xy.dist_to(old_pose.xy)
                    waypoint_stagnant_steps[i] = (
                        waypoint_stagnant_steps[i] + 1 if moved < 1e-4 else 0
                    )
                    traj[i]['pred_goal'].append(dst)
                    traj[i]['stage1_trajectory'].append(poses[i])
                    stage1_step += 1
                    if t >= self.args.max_action_len - 1:
                        ended[i] = True
                        traj[i]['stop_reason'].append('horizon')
                    if waypoint_stagnant_steps[i] >= getattr(
                            self.args, 'heatmap_waypoint_stagnation_steps', 5):
                        ended[i] = True
                        traj[i]['stop_reason'].append('waypoint_stagnation')
                    continue

                if selected_trajectory_paths is not None:
                    trajectory_plan_steps += 1
                    goal_id = int(selected_plan_goal_ids[i])
                    if previous_plan_goal_id[i] != -1 and previous_plan_goal_id[i] != goal_id:
                        trajectory_goal_switches += 1
                    previous_plan_goal_id[i] = goal_id
                    # Fully trajectory-based option: no Stage-1/Stage-2
                    # switching. Execute only a short local waypoint and
                    # replan from the next observation. Never use GT here.
                    endpoint_dist = dst.dist_to(poses[i].xy)
                    # GT is diagnostics only and never changes the action.
                    if ('test' not in self.env_name
                            and poses[i].xy.dist_to(obs[i]['goal']) <= self.args.success_dist
                            and endpoint_dist > self.args.success_dist):
                        trajectory_arrival_gate_blocked += 1
                    if (not getattr(self.args, 'trajectory_disable_learned_stop', False)
                            and selected_stop_probs[i] >= self.args.trajectory_stop_threshold
                            and endpoint_dist <= self.args.success_dist):
                        trajectory_stop_decisions += 1
                        if poses[i].xy.dist_to(obs[i]['goal']) <= self.args.success_dist:
                            trajectory_stop_correct += 1
                        ended[i] = True
                        traj[i]['stop_reason'].append('learned_stop')
                        continue
                    # Short arclength-spaced paths often put the first point
                    # inside the discrete lookahead controller's 5m STOP
                    # radius. Skip already reached points and use a farther
                    # valid waypoint; fall back to the predicted endpoint.
                    local_xy = selected_trajectory_paths[i, -1]
                    for candidate_xy in selected_trajectory_paths[i]:
                        candidate_dst = self.env.unnormalize_position(
                            candidate_xy, obs[i]['map_name'], self.args.map_meters)
                        if candidate_dst.dist_to(poses[i].xy) >= 5.0:
                            local_xy = candidate_xy
                            break
                    local_dst = self.env.unnormalize_position(
                        local_xy, obs[i]['map_name'], self.args.map_meters)
                    remaining = local_dst.dist_to(poses[i].xy)
                    steps = min(self.args.move_iteration,
                                max(1, int(math.ceil(remaining / 5.0))))
                    old_pose = poses[i]
                    old_xy = old_pose.xy
                    poses[i] = self.move(poses[i], local_dst, steps)
                    moved_m = old_xy.dist_to(poses[i].xy)
                    rotated_rad = abs(np.arctan2(
                        np.sin(poses[i].yaw - old_pose.yaw),
                        np.cos(poses[i].yaw - old_pose.yaw)))
                    trajectory_travel_distance_m += moved_m
                    # Rotation-only actions are genuine controller progress
                    # and must not count as a frozen simulator.
                    no_action_progress = moved_m < 1e-4 and rotated_rad < 1e-4
                    waypoint_stagnant_steps[i] = (
                        waypoint_stagnant_steps[i] + 1 if no_action_progress else 0
                    )
                    if waypoint_stagnant_steps[i] >= getattr(
                            self.args, 'heatmap_waypoint_stagnation_steps', 5):
                        ended[i] = True
                        traj[i]['stop_reason'].append('trajectory_stagnation')
                    elif t >= self.args.max_action_len - 1:
                        ended[i] = True
                        traj[i]['stop_reason'].append('horizon')
                    traj[i]['pred_goal'].append(dst)
                    traj[i]['trajectory_local_waypoint_xy'].append(
                        [float(local_xy[0]), float(local_xy[1])])
                    traj[i]['stage1_trajectory'].append(poses[i])
                    stage1_step += 1
                    continue

                coarse_goal_dist = dst.dist_to(poses[i].xy)

                # Stage-2 recovery is student-only: if newly predicted coarse
                # goals stay far away, return to Stage 1 instead of remaining
                # permanently locked in fine navigation. Hysteresis
                # (25 m enter / 40 m recover by default) avoids boundary chatter.
                if self.feedback == 'student' and stage1_ended[i]:
                    if coarse_goal_dist > self.args.stage2_recover_dist:
                        stage2_recover_count[i] += 1
                    else:
                        stage2_recover_count[i] = 0

                    if stage2_recover_count[i] >= self.args.stage2_recover_patience:
                        stage1_ended[i] = False
                        stage2_recover_count[i] = 0
                        stage2_recoveries += 1

                if pred_progress_t[i] > 0.95 and self.feedback == 'student' and stage1_ended[i]:
                    ended[i] = True
                    traj[i]['stop_reason'].append('progress_stop')
                    continue
                elif t >= self.args.max_action_len - 1:
                    ended[i] = True
                    traj[i]['stop_reason'].append('horizon')
                    continue

                # Stage 1 only needs to enter the coarse target neighborhood;
                # fine localization is delegated to Stage 2.
                if coarse_goal_dist > self.args.stage1_switch_dist and not stage1_ended[i]:
                    stage1_step += 1
                    traj[i]['pred_goal'].append(dst)
                    if self.feedback == 'teacher':
                        cur_step = advance_teacher_stage1_index(
                            teacher_stage1_steps, i, self.args.move_iteration,
                            len(obs[i]['trajectory']),
                        )
                        poses[i] = obs[i]['trajectory'][cur_step]
                    else:
                        poses[i] = self.move(
                            poses[i],
                            dst,
                            self.args.move_iteration,
                        )
                    if not ended[i]:
                        traj[i]['stage1_trajectory'].append(poses[i])

                elif abs(a_t[i]) < np.pi / 12:
                    if not stage1_ended[i]:
                        gt_entry_dist = poses[i].xy.dist_to(ob['goal'])
                        stage2_entry_count += 1
                        stage2_entry_gt_distance_sum += float(gt_entry_dist)
                        if gt_entry_dist <= 20:
                            stage2_entry_gt_bins['le20'] += 1
                        elif gt_entry_dist <= 30:
                            stage2_entry_gt_bins['20_30'] += 1
                        elif gt_entry_dist <= 40:
                            stage2_entry_gt_bins['30_40'] += 1
                        else:
                            stage2_entry_gt_bins['gt40'] += 1
                    stage1_ended[i] = True
                    stage2_step += 1
                    poses[i] = _moved_pose(poses[i], *Action(5, 0, 0))
                    if len(traj[i]['stage2_trajectory']) == 0:
                        traj[i]['stage2_trajectory'].append(traj[i]['stage1_trajectory'][-1])
                    if not ended[i]:
                        traj[i]['stage2_trajectory'].append(poses[i])
                else:
                    if not stage1_ended[i]:
                        gt_entry_dist = poses[i].xy.dist_to(ob['goal'])
                        stage2_entry_count += 1
                        stage2_entry_gt_distance_sum += float(gt_entry_dist)
                        if gt_entry_dist <= 20:
                            stage2_entry_gt_bins['le20'] += 1
                        elif gt_entry_dist <= 30:
                            stage2_entry_gt_bins['20_30'] += 1
                        elif gt_entry_dist <= 40:
                            stage2_entry_gt_bins['30_40'] += 1
                        else:
                            stage2_entry_gt_bins['gt40'] += 1
                    stage1_ended[i] = True
                    stage2_rotate += 1
                    poses[i] = _moved_pose(poses[i], *Action(0, a_t[i], 0))
                    if len(traj[i]['stage2_trajectory']) == 0:
                        traj[i]['stage2_trajectory'].append(traj[i]['stage1_trajectory'][-1])
                    if not ended[i]:
                        traj[i]['stage2_trajectory'].append(poses[i])

            observer = getattr(self, 'experiment_step_callback', None)
            if observer is not None and trajectory_predictions is not None and 'test' not in self.env_name:
                observer(
                    self, obs, poses, traj, ended, t,
                    selected_trajectory_goals if selected_trajectory_goals is not None else heatmap_goals,
                    heatmap_topk_ids, trajectory_predictions, ade, fde,
                    prior_indices, joint_indices, selected_stop_probs,
                    selected_goal_ids=heatmap_goal_ids,
                )

            # Save every executed displacement, including a terminal action.
            # Previously an action on the final time step could move the UAV,
            # set ended=True and then be omitted from official SR/NE/SPL/OSR.
            for i, ob in enumerate(obs):
                if should_record_pose(traj[i]['trajectory'][-1], poses[i], ended=ended[i]):
                    traj[i]['trajectory'].append(poses[i])
            # Refresh the environment first, then use the resulting pose/state
            # as the spatial input for the next navigation step.
            before_obs = phase_clock()
            if profile_enabled:
                phase_seconds['loss_diagnostics_and_control'] += before_obs - after_model
            obs = self.env._get_obs(poses, random_direction=(self.feedback == 'teacher'))
            if profile_enabled:
                phase_seconds['dynamic_observation'] += phase_clock() - before_obs
            current_directions = [np.array(ob['pose'].yaw, dtype=np.float32) for ob in obs]
            current_positions = [np.array(ob['position'], dtype=np.float32) for ob in obs]
            direction_t = torch.from_numpy(np.array(current_directions, dtype=np.float32))
            position_t = torch.from_numpy(np.array(current_positions, dtype=np.float32))
            # current_view_corners = [np.array(ob['gt_path_corners'][0]) for ob in obs]

            # Early exit if all ended
            if ended.all():
                break
        if profile_enabled:
            for phase, seconds in phase_seconds.items():
                self.logs[f'profile_{phase}_seconds'].append(float(seconds))
                self.logs[f'profile_{self.feedback}_{phase}_seconds'].append(float(seconds))
        # print(visualize)
        if visualize:
            for i, ob in enumerate(obs):
                print('visualization/%s_%d_%d' % (
                                                ob['id'][0], ob['id'][1], ob['id'][2]))
                dist = traj[i]['trajectory'][-1].xy.dist_to(traj[i]['gt_trajectory'][-1].xy)
                # print(dist)
                # print(dist)
                # if dist > 20:
                #     continue
                landmarks = self.env.nav_maps[i].landmark_map.get_contours()
                ladnmark_names = self.env.nav_maps[i].landmark_map.landmark_names
                # print(self.env.split)
                # print(obs[i]['grid_goal'])
                # print(ob['id'])
                cropclient.save_grids(ob['id'][0], traj[i]['stage1_trajectory'], traj[i]['stage2_trajectory'],
                                     traj[i]['gt_trajectory'], landmarks, ladnmark_names,
                                     os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR,
                                                  'visualization/%s_%d_%d' % (
                                                ob['id'][0], ob['id'][1], ob['id'][2])),
                                     traj[i]['pred_goal'])
        if train_ml is not None:
            # print(ml_loss)
            # ml_loss = direction_loss + progress_loss
            if compact_mode:
                # Only the shared goal/path selector is trained. The heatmap
                # and unrelated legacy auxiliary heads are held fixed.
                ml_loss = (self.args.trajectory_ranking_loss_weight * trajectory_ranking_loss
                           + self.args.trajectory_candidate_loss_weight * trajectory_candidate_loss)
            else:
                ml_loss = (direction_loss + 0.1 * progress_loss + 2 * goal_predict_loss
                       + self.args.heatmap_loss_weight * heatmap_loss
                       + self.args.trajectory_loss_weight * trajectory_loss
                       + self.args.trajectory_stop_weight * trajectory_stop_loss
                       + self.args.trajectory_ranking_loss_weight * trajectory_ranking_loss
                       + self.args.trajectory_candidate_loss_weight * trajectory_candidate_loss)
            # ml_loss = progress_loss + goal_predict_loss
            self.loss += ml_loss * train_ml / batch_size

            # self.logs['ml_loss'].append((ml_loss * train_ml / batch_size).item())

            self.logs['direction_loss'].append(float(direction_loss) * train_ml / batch_size)
            self.logs['progress_loss'].append(float(progress_loss) * train_ml / batch_size)
            self.logs['goal_predict_loss'].append(float(goal_predict_loss) * train_ml / batch_size)
            self.logs['heatmap_loss'].append(float(heatmap_loss) * train_ml / batch_size)
            if train_ml is not None and getattr(self.args, 'heatmap_trajectory_enabled', False):
                self.logs['trajectory_loss'].append(
                    (trajectory_loss * train_ml / batch_size).item())
                self.logs['trajectory_stop_loss'].append(
                    (trajectory_stop_loss * train_ml / batch_size).item())
                self.logs['trajectory_ranking_loss'].append(
                    (trajectory_ranking_loss * train_ml / batch_size).item())
                self.logs['trajectory_candidate_loss'].append(
                    (trajectory_candidate_loss * train_ml / batch_size).item())
            self.logs['IL_loss'].append(float(ml_loss) * train_ml / batch_size)

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        # if t==0:
        #     self.logs
        self.logs['stage1_step'].append(float(stage1_step) / batch_size)
        self.logs['stage2_step'].append(float(stage2_step) / batch_size)
        self.logs['stage2_rotate'].append(float(stage2_rotate) / batch_size)
        self.logs['stage2_recoveries'].append(float(stage2_recoveries) / batch_size)
        self.logs['stage2_entry_count'].append(float(stage2_entry_count))
        self.logs['stage2_entry_gt_distance_sum_m'].append(float(stage2_entry_gt_distance_sum))
        for key, value in stage2_entry_gt_bins.items():
            self.logs[f'stage2_entry_gt_{key}'].append(float(value))

        self.logs['trajectory_minade_sum_m'].append(trajectory_minade_sum)
        self.logs['trajectory_minfde_sum_m'].append(trajectory_minfde_sum)
        self.logs['trajectory_eval_count'].append(float(trajectory_eval_count))
        self.logs['trajectory_teacher_samples'].append(float(trajectory_supervision_count))
        self.logs['trajectory_ranking_valid_count'].append(float(trajectory_ranking_valid_count))
        self.logs['trajectory_candidate_eval_count'].append(float(trajectory_candidate_eval_count))
        self.logs['trajectory_prior_goal_hits'].append(float(trajectory_prior_goal_hits))
        self.logs['trajectory_joint_goal_hits'].append(float(trajectory_joint_goal_hits))
        self.logs['trajectory_oracle_goal_hits'].append(float(trajectory_oracle_goal_hits))
        self.logs['trajectory_prior_fde_sum_m'].append(trajectory_prior_fde_sum_m)
        self.logs['trajectory_joint_fde_sum_m'].append(trajectory_joint_fde_sum_m)
        self.logs['trajectory_stop_decisions'].append(float(trajectory_stop_decisions))
        self.logs['trajectory_stop_correct'].append(float(trajectory_stop_correct))
        self.logs['trajectory_stop_positive_count'].append(float(trajectory_stop_positive_sum.item()))
        self.logs['trajectory_stop_supervised_count'].append(float(trajectory_stop_active_sum.item()))
        self.logs['trajectory_arrival_gate_blocked'].append(float(trajectory_arrival_gate_blocked))
        self.logs['trajectory_plan_steps'].append(float(trajectory_plan_steps))
        self.logs['trajectory_goal_switches'].append(float(trajectory_goal_switches))
        self.logs['trajectory_travel_distance_m'].append(float(trajectory_travel_distance_m))
        diagnostic_values = torch.stack((
            heatmap_diag_count,
            *(heatmap_coverage_hits[k] for k in (1, 4, 5, 8, 16, 20)),
            heatmap_top1_distance_sum,
            heatmap_refined_top1_distance_sum,
            heatmap_top16_nearest_distance_sum,
            heatmap_top16_candidate_distance_sum,
        )).detach().cpu().tolist()
        diagnostic_names = (
            'heatmap_diag_count',
            'heatmap_coverage_1_hits',
            'heatmap_coverage_4_hits',
            'heatmap_coverage_5_hits',
            'heatmap_coverage_8_hits',
            'heatmap_coverage_16_hits',
            'heatmap_coverage_20_hits',
            'heatmap_top1_distance_sum_m',
            'heatmap_refined_top1_distance_sum_m',
            'heatmap_top16_nearest_distance_sum_m',
            'heatmap_top16_candidate_distance_sum_m',
        )
        for name, value in zip(diagnostic_names, diagnostic_values):
            self.logs[name].append(value)

        # print('[3]')
        # debug_memory()
        # print()
        return traj



    def move(self, pose: Pose4D, dst: Point2D, iterations: int):
        dst = Point3D(dst.x, dst.y, pose.z)

        for _ in range(iterations):
            action = lookahead_discrete_action(pose, [dst])
            pose = _moved_pose(pose, *action.value)
            # if(action.name == 'STOP'):

        return pose

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

        all_tuple = [("lang_model", self.lang_model_without_ddp, self.lang_model_optimizer),
                     ("vision_model", self.vision_model_without_ddp, self.vision_model_optimizer),
                     ("vln_model", self.vln_model_without_ddp, self.et_optimizer),
                     ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys == load_keys:
                print("NOTICE: LOADing ALL KEYS IN THE ", name)
                state_dict = states[name]['state_dict']
            else:
                print("NOTICE: DIFFERENT KEYS IN THE ", name)
                # if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                #     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                state_dict = {k: v for k, v in states[name]['state_dict'].items() if k in model_keys}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])

            def count_parameters(mo):
                return sum(p.numel() for p in mo.parameters() if p.requires_grad)

            print('Model parameters: ', count_parameters(model))

        all_tuple = [("lang_model", self.lang_model_without_ddp, self.lang_model_optimizer),
                     ("vision_model", self.vision_model_without_ddp, self.vision_model_optimizer),
                     ("vln_model", self.vln_model_without_ddp, self.et_optimizer),
                     ]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_model']['epoch'] - 1
