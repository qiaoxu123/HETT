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
from torch.nn.utils.rnn import pad_sequence

from torchvision import transforms

# from r2r.agent_cmt import Seq2SeqCMTAgent
from multiagent.actions import Action
from multiagent.defaultpaths import GOAL_PREDICTOR_CHECKPOINT_DIR
from multiagent.models.dark_net import Darknet
from multiagent.models.CLIP import CLIP
# from direction.models.ddppo.resenet_encoders import TorchVisionResNet50
from multiagent.models.goal_predictor import GoalPredictor, MapEncoder
from multiagent.models.spatial_belief import metric_gaussian_target, greedy_nms_topk, local_soft_argmax_xy
from multiagent.models.candidate_selector import (
    candidate_ranking_loss,
    heatmap_ids_to_normalized_xy,
    select_candidate_with_abstention,
)
from multiagent.mapdata import GROUND_LEVEL
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

        # Optional diagnostic training mode: keep the already-trained HETT
        # heatmap/controller/backbones fixed and optimize only the new visual
        # candidate discriminator. This prevents a randomly initialized
        # selector from changing the proposal distribution while it learns.
        if (
            getattr(self.args, 'candidate_selector', False)
            and getattr(self.args, 'candidate_selector_freeze_base', False)
        ):
            for parameter in self.lang_model.parameters():
                parameter.requires_grad = False
            for parameter in self.vision_model.parameters():
                parameter.requires_grad = False
            for name, parameter in self.vln_model.named_parameters():
                parameter.requires_grad = name.startswith('candidate_visual_selector.')
        # self.map_encoder = MapEncoder(240)
        # self.goal_predictpr = GoalPredictor(240, 7)
        self.progress_regression = nn.MSELoss(reduction='sum')

        if self.args.world_size > 1 and allow_ngpus:
            def maybe_wrap_ddp(model):
                if not any(parameter.requires_grad for parameter in model.parameters()):
                    return model
                return DDP(
                    model,
                    broadcast_buffers=False,
                    find_unused_parameters=True,
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank,
                )

            self.lang_model = maybe_wrap_ddp(self.lang_model)
            self.vision_model = maybe_wrap_ddp(self.vision_model)
            self.vln_model = maybe_wrap_ddp(self.vln_model)

        self.lang_model_without_ddp = (
            self.lang_model.module if isinstance(self.lang_model, DDP) else self.lang_model
        )
        self.vision_model_without_ddp = (
            self.vision_model.module if isinstance(self.vision_model, DDP) else self.vision_model
        )
        self.vln_model_without_ddp = (
            self.vln_model.module if isinstance(self.vln_model, DDP) else self.vln_model
        )

        # self.vln_model = ViT_LSTM(
        #     self.args, 
        #     self.vision_model).cuda()

        # optimizer        
        assert args.optim in ("adam", "adamW")
        OptimizerClass = torch.optim.Adam if args.optim == "adam" else torch.optim.AdamW

        def make_optimizer(model):
            parameters = [p for p in model.parameters() if p.requires_grad]
            if not parameters:
                return None
            return OptimizerClass(parameters, lr=self.args.learning_rate)

        self.et_optimizer = make_optimizer(self.vln_model)
        self.lang_model_optimizer = make_optimizer(self.lang_model)
        self.vision_model_optimizer = make_optimizer(self.vision_model)
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

        if (
            getattr(self.args, 'candidate_selector', False)
            and getattr(self.args, 'candidate_selector_freeze_base', False)
        ):
            self.lang_model.eval()
            self.vision_model.eval()
            self.vln_model.eval()
            self.vln_model_without_ddp.candidate_visual_selector.train()
        else:
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
                if self.args.benchmark_batches and idx > self.args.benchmark_batches:
                    break
                # if idx >= 100:
                #     break
                # train_loop_start_time = time.time()
                for optimizer in (
                    self.lang_model_optimizer,
                    self.vision_model_optimizer,
                    self.et_optimizer,
                ):
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
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

                for optimizer in (
                    self.lang_model_optimizer,
                    self.vision_model_optimizer,
                    self.et_optimizer,
                ):
                    if optimizer is not None:
                        optimizer.step()
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
            if optimizer is not None:
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

        # Build referenced-landmark tokens once per rollout.  This uses exactly
        # the same landmark names/centroids already used by the heatmap mask;
        # no target IDs or GT coordinates are introduced.
        selector_landmark_features = None
        selector_landmark_xy = None
        selector_landmark_mask = None
        if getattr(self.args, 'candidate_selector', False):
            landmark_name_lists = [
                list(ob.get('referenced_landmark_names', ())) for ob in obs
            ]
            landmark_xy_lists = [
                np.asarray(
                    ob.get('referenced_landmark_xy', np.zeros((0, 2), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1, 2)
                for ob in obs
            ]
            counts = [
                min(len(names), len(xys))
                for names, xys in zip(landmark_name_lists, landmark_xy_lists)
            ]
            flat_names = [
                landmark_name_lists[b][j]
                for b, count in enumerate(counts)
                for j in range(count)
            ]

            if flat_names:
                landmark_encoding = self.tokenizer(
                    flat_names, padding=True, return_tensors="pt"
                )
                landmark_ids = landmark_encoding['input_ids'].cuda()
                landmark_attention = landmark_encoding['attention_mask'].cuda()
                _, _, flat_landmark_features = self.lang_model(
                    landmark_ids, landmark_attention
                )
            else:
                flat_landmark_features = lang_features.new_zeros(
                    (0, lang_features.shape[-1])
                )

            feature_rows = []
            xy_rows = []
            offset = 0
            for b, count in enumerate(counts):
                if count:
                    feature_rows.append(
                        flat_landmark_features[offset:offset + count]
                    )
                    xy_rows.append(
                        torch.as_tensor(
                            landmark_xy_lists[b][:count],
                            dtype=lang_features.dtype,
                            device=lang_features.device,
                        )
                    )
                    offset += count
                else:
                    feature_rows.append(
                        lang_features.new_zeros((1, lang_features.shape[-1]))
                    )
                    xy_rows.append(lang_features.new_zeros((1, 2)))

            selector_landmark_features = pad_sequence(
                feature_rows, batch_first=True
            )
            selector_landmark_xy = pad_sequence(
                xy_rows, batch_first=True
            )
            selector_landmark_mask = torch.zeros(
                selector_landmark_xy.shape[:2],
                dtype=torch.bool,
                device=selector_landmark_xy.device,
            )
            for b, count in enumerate(counts):
                if count:
                    selector_landmark_mask[b, :count] = True

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
        heatmap_diag_count = torch.zeros((), device='cuda')
        heatmap_coverage_hits = {
            k: torch.zeros((), device='cuda') for k in (1, 4, 8, 16)
        }
        heatmap_top1_distance_sum = torch.zeros((), device='cuda')
        heatmap_refined_top1_distance_sum = torch.zeros((), device='cuda')
        heatmap_top16_nearest_distance_sum = torch.zeros((), device='cuda')
        heatmap_top16_candidate_distance_sum = torch.zeros((), device='cuda')

        candidate_selector_loss = torch.zeros((), device='cuda')
        candidate_selector_hard_loss = torch.zeros((), device='cuda')
        candidate_selector_list_loss = torch.zeros((), device='cuda')
        candidate_selector_train_eligible = torch.zeros((), device='cuda')
        candidate_selector_decision_count = torch.zeros((), device='cuda')
        candidate_selector_trigger_count = torch.zeros((), device='cuda')
        candidate_selector_change_count = torch.zeros((), device='cuda')
        candidate_selector_rescue_count = torch.zeros((), device='cuda')
        candidate_selector_regression_count = torch.zeros((), device='cuda')
        candidate_selector_visible_sum = torch.zeros((), device='cuda')
        candidate_selector_selected_distance_sum = torch.zeros((), device='cuda')
        candidate_selector_eval_eligible = torch.zeros((), device='cuda')
        candidate_selector_visual_top1_hits = torch.zeros((), device='cuda')
        candidate_selector_raw_top1_hits = torch.zeros((), device='cuda')
        candidate_selector_visual_distance_sum = torch.zeros((), device='cuda')
        candidate_selector_raw_distance_sum = torch.zeros((), device='cuda')

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
        if getattr(self.args, 'candidate_selector', False):
            selector_cells = self.args.heatmap_grid_size ** 2
            input['candidate_visual_memory'] = torch.zeros(
                batch_size, selector_cells, 512, device='cuda'
            )
            input['candidate_visual_count'] = torch.zeros(
                batch_size, selector_cells, device='cuda'
            )
            input['selector_landmark_features'] = selector_landmark_features
            input['selector_landmark_xy'] = selector_landmark_xy
            input['selector_landmark_mask'] = selector_landmark_mask

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





            view_radius_m = torch.as_tensor(
                [
                    max(
                        1.0,
                        float(ob['pose'].z) - float(GROUND_LEVEL[ob['map_name']]),
                    )
                    for ob in obs
                ],
                dtype=torch.float32,
                device='cuda',
            )

            (
                pred_direction,
                pred_progress,
                pred_goals,
                pred_logits,
                grid_ft,
                selector_candidate_ids,
                selector_logits,
                selector_visible,
                selector_candidate_xy,
                selector_current_visual,
                selector_current_visible,
            ) = self.vln_model(
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
                lang_cls=input['lang_cls'],
                view_radius_m=view_radius_m,
                **(
                    {
                        'candidate_visual_memory': input['candidate_visual_memory'],
                        'candidate_visual_count': input['candidate_visual_count'],
                        'selector_landmark_features': input['selector_landmark_features'],
                        'selector_landmark_xy': input['selector_landmark_xy'],
                        'selector_landmark_mask': input['selector_landmark_mask'],
                    }
                    if getattr(self.args, 'candidate_selector', False)
                    else {}
                ),
            )

            # Store only actually observed candidate-specific RGB evidence.
            # The memory is keyed by 28x28 heatmap cell and detached across
            # navigation steps; unobserved hypotheses never receive image data.
            if selector_current_visual is not None:
                memory_index = selector_candidate_ids.unsqueeze(-1).expand(
                    -1, -1, selector_current_visual.shape[-1]
                )
                observed_visual = (
                    selector_current_visual.detach()
                    * selector_current_visible.unsqueeze(-1).to(
                        selector_current_visual.dtype
                    )
                )
                input['candidate_visual_memory'].scatter_add_(
                    1, memory_index, observed_visual
                )
                input['candidate_visual_count'].scatter_add_(
                    1,
                    selector_candidate_ids,
                    selector_current_visible.to(
                        input['candidate_visual_count'].dtype
                    ),
                )

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
            raw_heatmap_goal_ids = heatmap_topk_ids[:, 0]
            heatmap_goal_ids = raw_heatmap_goal_ids
            selector_decision = None
            selector_applied = torch.zeros(
                batch_size, dtype=torch.bool, device=pred_logits.device
            )
            if (
                getattr(self.args, 'candidate_selector', False)
                and selector_logits is not None
            ):
                selector_decision = select_candidate_with_abstention(
                    raw_heatmap_goal_ids,
                    selector_candidate_ids,
                    selector_logits,
                    selector_visible,
                    min_visible_candidates=self.args.candidate_selector_min_visible,
                    min_confidence=self.args.candidate_selector_min_confidence,
                    min_margin=self.args.candidate_selector_min_margin,
                )
                # Never let a randomly initialized/learning selector alter
                # student training trajectories. It is trained from GT only
                # and becomes an inference-time discriminator after rollout.
                if self.feedback == 'student' and train_ml is None:
                    selector_applied = selector_decision.triggered
                    heatmap_goal_ids = torch.where(
                        selector_applied,
                        selector_decision.chosen_ids,
                        raw_heatmap_goal_ids,
                    )

            # Keep a raw-heatmap refinement for apples-to-apples diagnostics,
            # then refine the actually selected mode for navigation.
            raw_heatmap_goals = local_soft_argmax_xy(
                heatmap_probs.reshape(
                    -1,
                    self.args.heatmap_grid_size,
                    self.args.heatmap_grid_size,
                ),
                raw_heatmap_goal_ids,
                window_size=self.args.heatmap_local_window,
            )
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
            pred_progress_t = host_predictions[:, 0]
            at_direction = host_predictions[:, 1]
            selector_host = None
            if selector_decision is not None:
                selector_host = torch.stack(
                    (
                        selector_decision.confidence,
                        selector_decision.margin,
                        selector_decision.visible_count.to(torch.float32),
                        selector_applied.to(torch.float32),
                        heatmap_goal_ids.to(torch.float32),
                        raw_heatmap_goal_ids.to(torch.float32),
                    ),
                    dim=1,
                ).detach().cpu().numpy()
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
            if not 'test' in self.env_name:
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
                for k in (1, 4, 8, 16):
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
                    (raw_heatmap_goals - gt_xy) * self.args.map_meters,
                    dim=-1,
                )
                heatmap_refined_top1_distance_sum += (
                    refined_top1_distance_m * active_mask
                ).sum()

                if (
                    getattr(self.args, 'candidate_selector', False)
                    and selector_logits is not None
                ):
                    selector_distances_m = torch.linalg.vector_norm(
                        (
                            selector_candidate_xy
                            - gt_xy.unsqueeze(1).to(selector_candidate_xy)
                        ) * self.args.map_meters,
                        dim=-1,
                    )
                    ranking = candidate_ranking_loss(
                        selector_logits,
                        selector_distances_m,
                        selector_visible,
                        active_bool,
                        good_radius_m=self.args.candidate_selector_good_radius_m,
                        list_temperature_m=self.args.candidate_selector_list_temperature_m,
                        list_weight=self.args.candidate_selector_list_weight,
                        min_visible_candidates=self.args.candidate_selector_min_visible,
                    )
                    candidate_selector_loss = (
                        candidate_selector_loss + ranking.total
                    )
                    candidate_selector_hard_loss = (
                        candidate_selector_hard_loss + ranking.hard
                    )
                    candidate_selector_list_loss = (
                        candidate_selector_list_loss + ranking.listwise
                    )
                    candidate_selector_train_eligible += ranking.eligible_count

                    # GT is used only for diagnostics/training supervision.
                    # Compare visual Top-1 and raw heatmap Top-1 on exactly the
                    # same steps where the visible candidate set contains a
                    # supervised good hypothesis.
                    selector_pred_distance_m = selector_distances_m.gather(
                        1, selector_decision.selected_rank[:, None]
                    ).squeeze(1)
                    eligible_eval = ranking.eligible_mask
                    raw_distance_m = candidate_distances_m[:, 0]
                    candidate_selector_eval_eligible += eligible_eval.sum()
                    candidate_selector_visual_top1_hits += (
                        eligible_eval
                        & (selector_pred_distance_m <= self.args.success_dist)
                    ).sum()
                    candidate_selector_raw_top1_hits += (
                        eligible_eval
                        & (raw_distance_m <= self.args.success_dist)
                    ).sum()
                    candidate_selector_visual_distance_sum += (
                        selector_pred_distance_m
                        * eligible_eval.to(selector_pred_distance_m)
                    ).sum()
                    candidate_selector_raw_distance_sum += (
                        raw_distance_m * eligible_eval.to(raw_distance_m)
                    ).sum()

                    decision_mask = (
                        active_bool
                        & (
                            selector_decision.visible_count
                            >= self.args.candidate_selector_min_visible
                        )
                    )
                    candidate_selector_decision_count += decision_mask.sum()
                    candidate_selector_visible_sum += (
                        selector_decision.visible_count.to(active_mask)
                        * decision_mask.to(active_mask)
                    ).sum()
                    applied_active = selector_applied & active_bool
                    candidate_selector_trigger_count += applied_active.sum()
                    changed = (
                        applied_active
                        & (heatmap_goal_ids != raw_heatmap_goal_ids)
                    )
                    candidate_selector_change_count += changed.sum()

                    selected_xy = heatmap_ids_to_normalized_xy(
                        heatmap_goal_ids[:, None],
                        field_size=self.args.heatmap_grid_size,
                    ).squeeze(1).to(gt_xy)
                    selected_distance_m = torch.linalg.vector_norm(
                        (selected_xy - gt_xy) * self.args.map_meters,
                        dim=-1,
                    )
                    candidate_selector_selected_distance_sum += (
                        selected_distance_m * applied_active.to(selected_distance_m)
                    ).sum()
                    candidate_selector_rescue_count += (
                        changed
                        & (raw_distance_m > self.args.success_dist)
                        & (selected_distance_m <= self.args.success_dist)
                    ).sum()
                    candidate_selector_regression_count += (
                        changed
                        & (raw_distance_m <= self.args.success_dist)
                        & (selected_distance_m > self.args.success_dist)
                    ).sum()

                heatmap_top16_nearest_distance_sum += (
                    candidate_distances_m.min(dim=1).values * active_mask
                ).sum()
                heatmap_top16_candidate_distance_sum += (
                    candidate_distances_m.mean(dim=1) * active_mask
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
                    traj[i]['progress'].append(float(host_predictions[i, 0]))
                    traj[i]['heatmap_goal_id'].append(int(host_predictions[i, 4]))
                    traj[i]['heatmap_raw_goal_id'].append(
                        int(raw_heatmap_goal_ids[i].detach().cpu())
                    )
                    traj[i]['heatmap_confidence'].append(float(host_predictions[i, 5]))
                    if selector_host is not None:
                        traj[i]['candidate_selector_confidence'].append(
                            float(selector_host[i, 0])
                        )
                        traj[i]['candidate_selector_margin'].append(
                            float(selector_host[i, 1])
                        )
                        traj[i]['candidate_selector_visible'].append(
                            int(selector_host[i, 2])
                        )
                        traj[i]['candidate_selector_applied'].append(
                            bool(selector_host[i, 3])
                        )
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
                cpu_goal = host_predictions[:, 2:4]

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
                    continue
                elif t == self.args.max_action_len:
                    ended[i] = True
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
                + self.args.candidate_selector_loss_weight * candidate_selector_loss
            )
            # ml_loss = progress_loss + goal_predict_loss
            self.loss += ml_loss * train_ml / batch_size

            # self.logs['ml_loss'].append((ml_loss * train_ml / batch_size).item())

            self.logs['direction_loss'].append((direction_loss * train_ml / batch_size).item())
            self.logs['progress_loss'].append((progress_loss * train_ml / batch_size).item())
            self.logs['goal_predict_loss'].append((goal_predict_loss * train_ml / batch_size).item())
            self.logs['heatmap_loss'].append((heatmap_loss * train_ml / batch_size).item())
            self.logs['candidate_selector_loss'].append(
                (candidate_selector_loss * train_ml / batch_size).item()
            )
            self.logs['candidate_selector_hard_loss'].append(
                (candidate_selector_hard_loss * train_ml / batch_size).item()
            )
            self.logs['candidate_selector_list_loss'].append(
                (candidate_selector_list_loss * train_ml / batch_size).item()
            )
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

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

        diagnostic_values = torch.stack((
            heatmap_diag_count,
            *(heatmap_coverage_hits[k] for k in (1, 4, 8, 16)),
            heatmap_top1_distance_sum,
            heatmap_refined_top1_distance_sum,
            heatmap_top16_nearest_distance_sum,
            heatmap_top16_candidate_distance_sum,
        )).detach().cpu().tolist()
        diagnostic_names = (
            'heatmap_diag_count',
            'heatmap_coverage_1_hits',
            'heatmap_coverage_4_hits',
            'heatmap_coverage_8_hits',
            'heatmap_coverage_16_hits',
            'heatmap_top1_distance_sum_m',
            'heatmap_refined_top1_distance_sum_m',
            'heatmap_top16_nearest_distance_sum_m',
            'heatmap_top16_candidate_distance_sum_m',
        )
        for name, value in zip(diagnostic_names, diagnostic_values):
            self.logs[name].append(value)

        selector_values = torch.stack((
            candidate_selector_train_eligible,
            candidate_selector_decision_count,
            candidate_selector_trigger_count,
            candidate_selector_change_count,
            candidate_selector_rescue_count,
            candidate_selector_regression_count,
            candidate_selector_visible_sum,
            candidate_selector_selected_distance_sum,
            candidate_selector_eval_eligible,
            candidate_selector_visual_top1_hits,
            candidate_selector_raw_top1_hits,
            candidate_selector_visual_distance_sum,
            candidate_selector_raw_distance_sum,
        )).detach().cpu().tolist()
        selector_names = (
            'candidate_selector_train_eligible',
            'candidate_selector_decision_count',
            'candidate_selector_trigger_count',
            'candidate_selector_change_count',
            'candidate_selector_rescue_count',
            'candidate_selector_regression_count',
            'candidate_selector_visible_sum',
            'candidate_selector_selected_distance_sum_m',
            'candidate_selector_eval_eligible',
            'candidate_selector_visual_top1_hits',
            'candidate_selector_raw_top1_hits',
            'candidate_selector_visual_distance_sum_m',
            'candidate_selector_raw_distance_sum_m',
        )
        for name, value in zip(selector_names, selector_values):
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
                'optimizer': None if optimizer is None else optimizer.state_dict(),
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
            if (
                self.args.resume_optimizer
                and optimizer is not None
                and states[name].get('optimizer') is not None
            ):
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
