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


def build_gaussian_heatmap(target_xy, grid_size, sigma, device):
    """Build a Gaussian spatial target around continuous normalized goal xy."""
    target_xy = target_xy.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    target_rows = target_xy[:, 0] * grid_size
    target_cols = target_xy[:, 1] * grid_size

    coords = torch.arange(
        grid_size,
        device=device,
        dtype=torch.float32,
    ) + 0.5
    grid_rows, grid_cols = torch.meshgrid(coords, coords, indexing='ij')
    dist_sq = (
        (grid_rows.unsqueeze(0) - target_rows[:, None, None]) ** 2
        + (grid_cols.unsqueeze(0) - target_cols[:, None, None]) ** 2
    )
    heatmap = torch.exp(-dist_sq / (2 * sigma ** 2))
    heatmap = heatmap / heatmap.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return heatmap.view(-1, grid_size ** 2)


def build_dense_anchor_trajectories(current_xy, grid_size, num_waypoints):
    """Build straight trajectory anchors from current xy to every dense cell."""
    device = current_xy.device
    dtype = current_xy.dtype

    coords = (
        torch.arange(grid_size, device=device, dtype=dtype) + 0.5
    ) / grid_size
    rows, cols = torch.meshgrid(coords, coords, indexing='ij')
    endpoints = torch.stack((rows, cols), dim=-1).reshape(1, -1, 1, 2)

    fractions = torch.linspace(
        1.0 / num_waypoints,
        1.0,
        num_waypoints,
        device=device,
        dtype=dtype,
    ).view(1, 1, num_waypoints, 1)

    current_xy = current_xy.view(-1, 1, 1, 2)
    return current_xy + fractions * (endpoints - current_xy)


def select_dense_proposals(
    heatmap_probs,
    grid_size,
    top_k,
    nms_kernel,
):
    """Greedy NMS Top-K with local soft-argmax endpoint refinement."""
    if nms_kernel % 2 == 0:
        raise ValueError("trajectory_nms_kernel must be odd")

    batch_size = heatmap_probs.shape[0]
    heatmap_2d = heatmap_probs.view(batch_size, grid_size, grid_size)
    working = heatmap_2d.clone()
    k = min(top_k, grid_size * grid_size)
    suppress_radius = nms_kernel // 2

    selected_ids = []
    selected_scores = []
    for _ in range(k):
        flat = working.view(batch_size, -1)
        scores, ids = flat.max(dim=1)
        selected_ids.append(ids)
        selected_scores.append(scores)

        for b in range(batch_size):
            peak_id = int(ids[b].item())
            row = peak_id // grid_size
            col = peak_id % grid_size
            r0 = max(0, row - suppress_radius)
            r1 = min(grid_size, row + suppress_radius + 1)
            c0 = max(0, col - suppress_radius)
            c1 = min(grid_size, col + suppress_radius + 1)
            working[b, r0:r1, c0:c1] = -1.0

    top_ids = torch.stack(selected_ids, dim=1)
    top_scores = torch.stack(selected_scores, dim=1)

    refined = torch.zeros(
        batch_size,
        k,
        2,
        device=heatmap_probs.device,
        dtype=heatmap_probs.dtype,
    )
    refine_radius = 1

    for b in range(batch_size):
        for j in range(k):
            peak_id = int(top_ids[b, j].item())
            row = peak_id // grid_size
            col = peak_id % grid_size
            r0 = max(0, row - refine_radius)
            r1 = min(grid_size, row + refine_radius + 1)
            c0 = max(0, col - refine_radius)
            c1 = min(grid_size, col + refine_radius + 1)

            patch = heatmap_2d[b, r0:r1, c0:c1]
            patch_sum = patch.sum().clamp_min(1e-8)

            patch_rows = (
                torch.arange(r0, r1, device=patch.device, dtype=patch.dtype) + 0.5
            ) / grid_size
            patch_cols = (
                torch.arange(c0, c1, device=patch.device, dtype=patch.dtype) + 0.5
            ) / grid_size
            rr, cc = torch.meshgrid(patch_rows, patch_cols, indexing='ij')

            refined[b, j, 0] = (patch * rr).sum() / patch_sum
            refined[b, j, 1] = (patch * cc).sum() / patch_sum

    return top_ids, top_scores, refined


def build_future_trajectory_target(env, ob, current_pose, num_waypoints, map_meters):
    """Sample future GT waypoints along the remaining teacher path.

    The current pose is projected to the nearest point on the teacher trajectory.
    Waypoints are then sampled uniformly in arc length over the remaining path
    and normalized to the same map coordinates used by the heatmap.
    """
    path_xy = np.array([[p.x, p.y] for p in ob['trajectory']], dtype=np.float32)
    goal_xy = np.array(ob['goal'], dtype=np.float32)

    if path_xy.shape[0] == 0:
        path_xy = goal_xy[None, :]
    elif np.linalg.norm(path_xy[-1] - goal_xy) > 1e-4:
        path_xy = np.concatenate([path_xy, goal_xy[None, :]], axis=0)

    current_xy = np.array([current_pose.x, current_pose.y], dtype=np.float32)
    nearest_idx = int(np.linalg.norm(path_xy - current_xy[None, :], axis=1).argmin())
    remaining = path_xy[nearest_idx:]
    remaining = np.concatenate([current_xy[None, :], remaining], axis=0)

    segment_lengths = np.linalg.norm(remaining[1:] - remaining[:-1], axis=1)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)]
    )
    total_length = float(cumulative[-1])

    if total_length < 1e-6:
        sampled_xy = np.repeat(goal_xy[None, :], num_waypoints, axis=0)
    else:
        sample_distances = np.linspace(
            total_length / num_waypoints,
            total_length,
            num_waypoints,
            dtype=np.float32,
        )
        sampled_xy = []
        for distance in sample_distances:
            upper = int(np.searchsorted(cumulative, distance, side='right'))
            upper = min(max(upper, 1), len(cumulative) - 1)
            lower = upper - 1
            denom = max(float(cumulative[upper] - cumulative[lower]), 1e-6)
            ratio = float((distance - cumulative[lower]) / denom)
            sampled_xy.append(
                remaining[lower] + ratio * (remaining[upper] - remaining[lower])
            )
        sampled_xy = np.asarray(sampled_xy, dtype=np.float32)

    normalized = [
        env.normalize_position(Point2D(float(x), float(y)), ob['map_name'], map_meters)
        for x, y in sampled_xy
    ]
    return np.asarray(normalized, dtype=np.float32)


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

        self.tokenizer = BertTokenizerFast.from_pretrained('/cver/xcding/code/tokenizer_files/bert-base-uncase')
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

        self.losses = []
        for epoch in range(1, n_epochs + 1):
            idx = 0
            start = time.time()
            # print('?')
            for _, l in enumerate(loader):
                idx += 1
                # if idx >= 100:
                #     break
                # train_loop_start_time = time.time()
                self.lang_model_optimizer.zero_grad()
                self.vision_model_optimizer.zero_grad()
                self.et_optimizer.zero_grad()
                self.loss = 0

                if feedback == 'teacher':
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.teacher_weight)
                elif feedback == 'student':  # agents in teacher and student separately

                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight)  # self.args.nss_w*nss_w_weighting, **kwargs)
                    # if epoch_train > 10000:
                    self.feedback = 'student'
                    self.rollout(train_ml=self.args.ml_weight)
                else:
                    assert False

                # print("--- One rollout takes %s seconds ---" % (time.time() - train_loop_start_time))

                # print(self.rank, epoch, self.loss)
                # torch.autograd.set_detect_anomaly(True)

                self.loss.backward()
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

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def rollout(self, train_ml=None, visualize=False):

        # rollout_start_time = time.time()

        obs = self.env._get_obs(random_direction=(self.feedback == 'teacher'))
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
        lang_features, linear_cls, cls_hidden = self.lang_model(input_ids, attention_mask)

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

        # Init the logs
        # ml_loss = 0.
        direction_loss = torch.tensor(0.).cuda()
        progress_loss = 0.
        goal_predict_loss = 0.
        heatmap_loss = torch.tensor(0.).cuda()
        trajectory_loss = torch.tensor(0.).cuda()

        trajectory_step = 0

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

        for t in range(self.args.max_action_len):

            # print("- action rollingout takes %s seconds ---" % (time.time() - rollingout_action_start_time))
            # rollingout_action_start_time = time.time()
            images = []
            for i in range(len(obs)):
                images.append(obs[i]['rgb'].copy())
            images = np.stack(images)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # W x H x C to C x W x H
            images = np.ascontiguousarray(images, dtype=np.float32)
            images -= self.rgb_mean
            images /= self.rgb_std
            im_feature = self.vision_model(torch.from_numpy(images).cuda())
            im_feature = im_feature.view(im_feature.size(0), im_feature.size(1), -1)






            current_direct = direction_t.view(-1, 1).cuda()
            current_pos = position_t.view(-1, 2).cuda()
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





            pred_direction, pred_progress, pred_goals, pred_logits, trajectory_residuals, grid_ft = self.vln_model(
                directions=input['directions'],
                frames=input['frames'],
                lenths=input['lenths'],
                grid_fts=input['grid_fts'],
                grid_index=input['grid_index'],
                maps=input['maps'],
                lang=input['lang'],
                lang_mask=input['lang_mask'],
                candidates=input['candidates'],
                centroids=input['centroids'],
                lang_cls=input['lang_cls']
            )

            # Dense spatial belief is decoupled from HETT's original 7x7
            # history grid. NMS keeps only a small set of meaningful modes.
            heatmap_probs = torch.softmax(pred_logits, dim=1)
            proposal_ids, proposal_scores, proposal_endpoints = select_dense_proposals(
                heatmap_probs,
                self.args.belief_grid_size,
                self.args.trajectory_top_k,
                self.args.trajectory_nms_kernel,
            )

            current_xy = input['directions'][:, -1, 2:4]

            gather_index = proposal_ids[:, :, None, None].expand(
                -1,
                -1,
                self.args.trajectory_steps,
                2,
            )
            proposal_residuals = torch.gather(
                trajectory_residuals,
                dim=1,
                index=gather_index,
            )

            fractions = torch.linspace(
                1.0 / self.args.trajectory_steps,
                1.0,
                self.args.trajectory_steps,
                device=pred_logits.device,
                dtype=pred_logits.dtype,
            ).view(1, 1, self.args.trajectory_steps, 1)
            proposal_anchors = (
                current_xy[:, None, None, :]
                + fractions * (
                    proposal_endpoints[:, :, None, :]
                    - current_xy[:, None, None, :]
                )
            )
            proposal_trajectories = (
                proposal_anchors + proposal_residuals
            ).clamp(0.0, 1.0)

            # Proposals are sorted by belief score. The first trajectory is
            # executed now; all K hypotheses are retained for logging/analysis.
            selected_trajectories = proposal_trajectories[:, 0]
            selected_first_waypoint = selected_trajectories[:, 0, :]

            input['grid_fts'] = torch.cat((input['grid_fts'], grid_ft), dim=1)
            grid_index = torch.tensor(np.array([ob['cur_grid'] for ob in obs])).unsqueeze(1).cuda()
            # print(input['grid_index'], grid_index)

            input['grid_index'] = torch.cat((input['grid_index'], grid_index), dim=1)
            # print("- model prediction takes %s seconds ---" % (time.time() - rollingout_action_start_time))
            # pred_direction = output
            # pred_progress = progress

            # Predicted progress
            pred_progress_t = pred_progress.cpu().detach().numpy()

            # Predicted waypoint
            nt_direct = torch.atan2(pred_direction[:, 0], pred_direction[:, 1])
            at_direction = nt_direct.cpu().detach().numpy()
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
            gt_goal = torch.from_numpy(np.array([ob['normalized_goal'] for ob in obs], dtype=np.float32))
            gt_progress = torch.from_numpy(np.array([ob['progress'] for ob in obs], dtype=np.float32))
            # there is no ground truth in unseen_test set
            if not 'test' in self.env_name:
                # Get ground truth
                # print(t, target, gt_progress)

                # Compute loss

                for i in range(len(obs)):
                    true_direction = torch.tensor(gt_direction[i])

                    true_sin = torch.sin(true_direction)
                    true_cos = torch.cos(true_direction)
                    true_sin_cos = torch.stack([true_sin, true_cos], dim=-1).cuda()
                    # gt_progress = torch.tensor(obs[i]['progress']).cuda()
                    # print(pred_direction[i].view(-1).shape, pred_progress[i].view(-1).shape, true_sin_cos.shape, gt_progress[i].view(-1).shape)
                    # cuda_gt_next_pos_ratio = torch.from_numpy(target[i][0]).cuda()
                    # print(pred_direction[i].view(-1), true_sin_cos)
                    if not ended[i]:
                        # if stage1_ended[i]:
                        direction_loss += self.progress_regression(pred_direction[i].view(-1), true_sin_cos)

                        progress_loss += self.progress_regression(pred_progress[i].view(-1),
                                                                  gt_progress[i].view(-1).cuda())
                        goal_predict_loss += F.mse_loss(pred_goals[i].view(-1), gt_goal[i].view(-1).cuda())
                        # print(pred_goals[i], gt_goal[i], goal_predict_loss)

                    # ml_loss += direction_loss
                    # ml_loss += progress_loss

                    # print(ml_loss)
                    if direction_loss != direction_loss:  # debug for nan loss
                        print('0', direction_loss)
                    if progress_loss != progress_loss:  # debug for nan loss
                        print('0', progress_loss)
                # print(at_direction, gt_direction, ml_loss)
                dense_sigma = (
                    self.args.heatmap_sigma
                    * self.args.belief_grid_size
                    / self.args.grid_size
                )
                gt_heatmap = build_gaussian_heatmap(
                    gt_goal,
                    self.args.belief_grid_size,
                    dense_sigma,
                    pred_logits.device,
                )
                per_sample_heatmap_loss = -(
                    gt_heatmap * F.log_softmax(pred_logits, dim=1)
                ).sum(dim=1)
                active_mask = torch.from_numpy((~ended).astype(np.float32)).to(pred_logits.device)
                heatmap_loss += (per_sample_heatmap_loss * active_mask).sum()

                gt_future_waypoints = torch.from_numpy(
                    np.stack([
                        build_future_trajectory_target(
                            self.env,
                            ob,
                            poses[i],
                            self.args.trajectory_steps,
                            self.args.map_meters,
                        )
                        for i, ob in enumerate(obs)
                    ])
                ).to(
                    device=trajectory_residuals.device,
                    dtype=trajectory_residuals.dtype,
                )

                dense_anchor_trajectories = build_dense_anchor_trajectories(
                    current_xy,
                    self.args.belief_grid_size,
                    self.args.trajectory_steps,
                )
                pred_trajectories = (
                    dense_anchor_trajectories + trajectory_residuals
                ).clamp(0.0, 1.0)

                trajectory_targets = gt_future_waypoints[:, None, :, :].expand_as(
                    pred_trajectories
                )
                per_candidate_trajectory_loss = F.smooth_l1_loss(
                    pred_trajectories,
                    trajectory_targets,
                    reduction='none',
                ).mean(dim=(2, 3))
                per_sample_trajectory_loss = (
                    per_candidate_trajectory_loss * gt_heatmap
                ).sum(dim=1)
                trajectory_loss += (
                    per_sample_trajectory_loss * active_mask
                ).sum()
                # print(pred_logits.shape)
            # Log the trajectory
            # print(at_direction, gt_direction)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['actions'].append(at_direction[i])
                    if not 'test' in self.env_name:
                        traj[i]['gt_actions'].append(gt_direction[i])
                        traj[i]['gt_progress'].append(gt_progress[i].item())
                        traj[i]['gt_goal'].append(gt_goal[i])
                    traj[i]['progress'].append(pred_progress[i].item())
                    traj[i]['heatmap_goal_id'].append(int(proposal_ids[i, 0].item()))
                    traj[i]['heatmap_confidence'].append(
                        float(proposal_scores[i, 0].item())
                    )
                    traj[i]['trajectory_proposal_ids'].append(
                        proposal_ids[i].detach().cpu().tolist()
                    )
                    traj[i]['trajectory_proposal_scores'].append(
                        proposal_scores[i].detach().cpu().tolist()
                    )
                    traj[i]['trajectory_proposal_endpoints'].append(
                        proposal_endpoints[i].detach().cpu().tolist()
                    )
                    traj[i]['predicted_trajectory'].append(
                        selected_trajectories[i].detach().cpu().tolist()
                    )

            if self.feedback == 'teacher':
                # Teacher follows the first waypoint of the remaining GT path.
                at_goal = gt_future_waypoints[:, 0, :]
                pred_progress_t = gt_progress
            elif self.feedback == 'student':
                # HOME/MultiPath-style receding horizon: select one trajectory
                # hypothesis from the heatmap and execute only its first waypoint.
                at_goal = selected_first_waypoint
            else:
                sys.exit('Invalid feedback option')

            cpu_goal = at_goal.cpu().detach().numpy()
            # print(cpu_goal)

            # Interact with the simulator with actions
            for i in range(len(obs)):

                dst = self.env.unnormalize_position(cpu_goal[i], obs[i]['map_name'],
                                                    self.args.map_meters)

                if ended[i]:
                    continue

                if pred_progress_t[i] > 0.95 and self.feedback == 'student':
                    ended[i] = True
                    continue
                elif t == self.args.max_action_len:
                    ended[i] = True
                    continue

                # There is no hard coarse/fine switch. The heatmap chooses a
                # trajectory mode, and only the first waypoint is executed.
                # The next outer step observes again and predicts a new field.
                trajectory_step += 1
                traj[i]['pred_goal'].append(dst)
                poses[i] = self.move(
                    poses[i],
                    dst,
                    self.args.move_iteration,
                )
                if not ended[i]:
                    traj[i]['stage1_trajectory'].append(poses[i])

            # Save trajectory output
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(poses[i])
                    # Update the status
            # Refresh the environment first, then use the resulting pose/state
            # as the spatial input for the next navigation step.
            obs = self.env._get_obs(poses, random_direction=(self.feedback == 'teacher'))
            current_directions = [np.array(ob['pose'].yaw, dtype=np.float32) for ob in obs]
            current_positions = [np.array(ob['position'], dtype=np.float32) for ob in obs]
            direction_t = torch.from_numpy(np.array(current_directions, dtype=np.float32))
            position_t = torch.from_numpy(np.array(current_positions, dtype=np.float32))
            # current_view_corners = [np.array(ob['gt_path_corners'][0]) for ob in obs]

            # Early exit if all ended
            if ended.all():
                break
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
            ml_loss = (
                1 * direction_loss
                + 0.1 * progress_loss
                + 2 * goal_predict_loss
                + self.args.heatmap_loss_weight * heatmap_loss
                + self.args.trajectory_loss_weight * trajectory_loss
            )
            # ml_loss = progress_loss + goal_predict_loss
            self.loss += ml_loss * train_ml / batch_size

            # self.logs['ml_loss'].append((ml_loss * train_ml / batch_size).item())

            self.logs['direction_loss'].append((direction_loss * train_ml / batch_size).item())
            self.logs['progress_loss'].append((progress_loss * train_ml / batch_size).item())
            self.logs['goal_predict_loss'].append((goal_predict_loss * train_ml / batch_size).item())
            self.logs['heatmap_loss'].append((heatmap_loss * train_ml / batch_size).item())
            self.logs['trajectory_loss'].append((trajectory_loss * train_ml / batch_size).item())
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        # if t==0:
        #     self.logs
        self.logs['trajectory_step'].append(float(trajectory_step) / batch_size)

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
