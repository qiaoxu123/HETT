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
from multiagent.dataset.episode import reverse_waypoint_direction
from multiagent.models.dark_net import Darknet
from multiagent.models.CLIP import CLIP
# from direction.models.ddppo.resenet_encoders import TorchVisionResNet50
from multiagent.models.goal_predictor import GoalPredictor, MapEncoder
from multiagent.models.multilandmark_belief import target_cell_ids, target_region_mask
from multiagent.navigation_control import should_replan_stage2, should_stop_navigation
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

        # --------------- 0. 模型初始化：语言编码器 / 视觉编码器 / ET 策略网络 -----------------
        # 其中：
        # - lang_model: 把自然语言指令编码成 token 表示
        # - vision_model: 把 RGB 图像编码成 visual feature
        # - vln_model: 即 ET，负责把语言、视觉、地图和历史记忆融合，输出动作与目标预测
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.lang_model = CustomBERTModel().cuda()

        # self.img_tensor = transforms.ToTensor()

        self.vision_model = Darknet(self.args.darknet_model_file, 224).cuda()

        # 视觉编码器加载预训练权重
        new_state = torch.load(self.args.darknet_weight_file, map_location='cpu', weights_only=False)
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

        # --------------- 1. 分布式训练包裹：DDP -----------------
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

        # --------------- 2. 优化器：分别训练语言 / 视觉 / ET 模型 -----------------
        # 这里的三个优化器对应了三部分参数：
        # - lang_model_optimizer: 训练语言编码器
        # - vision_model_optimizer: 训练视觉编码器
        # - et_optimizer: 训练 ET 及其 Transformer + decoder heads
        assert args.optim in ("adam", "adamW")
        OptimizerClass = torch.optim.Adam if args.optim == "adam" else torch.optim.AdamW
        self.et_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.vln_model.parameters()),
                                           lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lang_model_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.lang_model.parameters()),
                                                   lr=self.args.learning_rate, weight_decay=args.weight_decay)
        self.vision_model_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.vision_model.parameters()),
                                                     lr=self.args.learning_rate, weight_decay=args.weight_decay)
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


    @torch.no_grad()
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
        # --------------- 0. 训练入口：控制三套模型同步更新 -----------------
        # 在本函数中，语言模型 / 视觉模型 / ET 策略模型都会被设为 train 模式，并且梯度会被统一回传。
        self.feedback = feedback

        self.lang_model.train()
        self.vln_model.train()
        self.vision_model.train()

        self.losses = []
        # 单卡训练时的梯度累积步数（--grad_accum，默认 1 表示不累积）
        grad_accum = getattr(self.args, 'grad_accum', 1)
        if grad_accum < 1:
            raise ValueError('grad_accum must be positive')
        for epoch in range(1, n_epochs + 1):
            idx = 0
            start = time.time()
            acc = 0
            num_batches = math.ceil(loader.dataset.size() / self.env.batch_size)
            # print('?')
            for _, l in enumerate(loader):
                idx += 1
                acc += 1
                # if idx >= 100:
                #     break
                # train_loop_start_time = time.time()

                # 每个累积窗口只清零一次，尾部窗口按实际 batch 数归一化。
                if (acc - 1) % grad_accum == 0:
                    for optimizer in self.optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    window_size = min(grad_accum, num_batches - acc + 1)
                self.loss = 0

                # --------------- 2. 进行一个 rollout：构造当前 batch 的观测与动作 -----------------
                # rollout 内部会执行：
                # - tokenize instruction
                # - encode vision
                # - build map / candidates / direction input
                # - self.vln_model(...) 触发 ET.forward
                # - 计算方向、进度、目标、候选目标损失
                if feedback == 'teacher':
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.teacher_weight)
                elif feedback == 'student':  # agents in teacher and student separately
                    if getattr(self.args, 'reverse_human_teacher', False):
                        # Training-only inverse task: begin at the final human
                        # observation, follow the recorded poses backwards, and
                        # navigate to the original route start from a synthetic
                        # metric instruction.  The following student rollout is
                        # reinitialized, so no reverse visual/history state leaks.
                        self.feedback = 'teacher'
                        self.env.set_reverse_teacher_mode(True)
                        try:
                            self.rollout(train_ml=self.args.reverse_teacher_weight)
                            (self.loss / window_size).backward()
                        finally:
                            self.env.set_reverse_teacher_mode(False)
                        self.loss = 0
                    else:
                        # Checkpoint-compatible released HETT teacher rollout.
                        self.feedback = 'teacher'
                        self.rollout(train_ml=self.args.ml_weight)  # self.args.nss_w*nss_w_weighting, **kwargs)
                        (self.loss / window_size).backward()
                        self.loss = 0
                    # if epoch_train > 10000:
                    self.feedback = 'student'
                    self.rollout(train_ml=self.args.ml_weight)
                else:
                    assert False

                # print("--- One rollout takes %s seconds ---" % (time.time() - train_loop_start_time))

                # print(self.rank, epoch, self.loss)
                # torch.autograd.set_detect_anomaly(True)

                # --------------- 3. 反向传播：梯度从 loss 回到整个模型 -----------------
                # 这里 self.loss 是在 rollout 中累积的总损失，最终会流回：
                # - language encoder
                # - vision encoder
                # - ET transformer + decoder heads
                if not torch.isfinite(self.loss):
                    raise FloatingPointError('Non-finite training loss')
                (self.loss / window_size).backward()
                # print('suc')

                # --------------- 4. 梯度裁剪与更新参数 -----------------
                # 对 ET 的参数做裁剪，避免爆炸梯度；随后三个优化器分别更新对应模块。
                # 单卡模式：每 grad_accum 个 batch 更新一次（与原多卡代码的更新节奏保持一致）
                if acc % grad_accum == 0 or acc == num_batches:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.vln_model.parameters(), 40., error_if_nonfinite=True)

                    self.lang_model_optimizer.step()
                    self.vision_model_optimizer.step()
                    self.et_optimizer.step()
                    if self.default_gpu and hasattr(self.args, 'output_dir') and (acc <= grad_accum or acc % 100 == 0 or acc == num_batches):
                        with open(os.path.join(self.args.output_dir, 'batch_metrics.jsonl'), 'a') as stream:
                            stream.write(json.dumps(dict(time=time.time(), batch=acc, batches=num_batches,
                                epoch=getattr(self, 'current_epoch', None),
                                recent_optimization_loss=float(np.sum(self.logs['IL_loss'][-2:])),
                                grad_norm=float(grad_norm),
                                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)) + '\n')
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

        # --------------- 0. 一个 batch 的 rollout 入口 -----------------
        # 这里的核心想法是：
        # 1. 先从环境中取出一批观测 obs
        # 2. 对 instruction 做 tokenization + language encoding
        # 3. 对 RGB / map / pose / candidate 组织成 ET 的输入
        # 4. 调用 self.vln_model(...) -> 进入 ET.forward
        # 5. 计算损失并输出动作/目标/进度更新环境状态
        # rollout_start_time = time.time()

        reverse_teacher = bool(getattr(self.env, 'reverse_teacher_mode', False))
        obs = self.env._get_obs(
            random_direction=(self.feedback == 'teacher' and not reverse_teacher)
        )
        batch_size = len(obs)

        # --------------- 1. 语言输入：instruction -> token ids -> BERT embedding -----------------
        # lang_inputs: list[str]，长度= batch_size
        # encoding: {'input_ids': [B, L], 'attention_mask': [B, L]}
        # lang_features: [B, L, 768]
        # linear_cls: [B, 49]，用于视觉特征的注意力加权
        # cls_hidden: [B, hidden]，保留句子级特征
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

        visual_alignment_loss = torch.tensor(0., device=input_ids.device)
        visual_alignment_positive_cosine = torch.tensor(0., device=input_ids.device)
        visual_alignment_accuracy = torch.tensor(0., device=input_ids.device)
        if reverse_teacher and self.args.reverse_visual_align_weight > 0:
            # Original terminal camera orientations are a separate visual
            # supervision stream; reverse flight/action RGB remains re-rendered
            # with motion-consistent headings.
            target_views = self.env.get_reverse_target_views(self.args.reverse_target_views)
            target_views = target_views[:, :, :, :, ::-1].transpose(0, 1, 4, 2, 3)
            target_views = np.ascontiguousarray(target_views, dtype=np.float32)
            target_views -= self.rgb_mean[None, None]
            target_views /= self.rgb_std[None, None]
            view_batch, view_count = target_views.shape[:2]
            target_view_features = self.vision_model(
                torch.from_numpy(target_views).cuda().flatten(0, 1)
            )
            # Darknet exposes the same flattened 512 x 7 x 7 tensor used by
            # the navigation path, but its native layout is 3584 x 7.
            target_view_features = target_view_features.view(
                view_batch, view_count, 512, -1
            )
            projected_target_views = self.vln_model_without_ddp.project_target_views(
                target_view_features
            )

            target_texts = [ob['visual_alignment_text'] for ob in obs]
            target_encoding = self.tokenizer(
                target_texts,
                padding=True,
                return_tensors='pt',
            )
            target_ids = target_encoding['input_ids'].cuda()
            target_attention = target_encoding['attention_mask'].cuda()
            if getattr(self.args, 'reverse_freeze_text_targets', False):
                with torch.no_grad():
                    _, _, target_cls = self.lang_model(target_ids, target_attention)
                target_cls = target_cls.detach()
            else:
                _, _, target_cls = self.lang_model(target_ids, target_attention)
            visual_embeddings = F.normalize(projected_target_views, dim=-1)
            text_embeddings = F.normalize(target_cls, dim=-1)
            contrastive_logits = visual_embeddings @ text_embeddings.T
            contrastive_logits = contrastive_logits / self.args.reverse_visual_temperature
            normalized_texts = [text.strip().casefold() for text in target_texts]
            positive_mask = torch.tensor(
                [[left == right for right in normalized_texts] for left in normalized_texts],
                dtype=torch.bool, device=input_ids.device,
            )
            positive_logits = contrastive_logits.masked_fill(
                ~positive_mask, -float('inf')
            )
            visual_alignment_positive_cosine = (
                visual_embeddings * text_embeddings
            ).sum(dim=-1).mean()
            if view_batch == 1:
                # InfoNCE has no negatives for a one-item smoke-test batch.
                # Keep the alignment branch trainable in that case.
                visual_alignment_loss = 1.0 - visual_alignment_positive_cosine
            else:
                visual_alignment_loss = -(
                    torch.logsumexp(positive_logits, dim=-1)
                    - torch.logsumexp(contrastive_logits, dim=-1)
                ).sum()
            visual_alignment_accuracy = (
                positive_mask[
                    torch.arange(view_batch, device=input_ids.device),
                    contrastive_logits.argmax(dim=-1),
                ]
            ).float().mean()

        landmark_centers = None
        landmark_name_features = None
        landmark_valid = None
        landmark_confidence = None
        if getattr(self.args, 'target_belief_head', False):
            max_landmarks = self.args.max_landmarks
            centers_array = np.zeros((batch_size, max_landmarks, 2), dtype=np.float32)
            valid_array = np.zeros((batch_size, max_landmarks), dtype=np.bool_)
            confidence_array = np.zeros((batch_size, max_landmarks), dtype=np.float32)
            landmark_texts = []
            for batch_index, ob in enumerate(obs):
                names = ob.get('landmark_names', [])
                centers = ob.get('landmark_centers', [])
                confidences = ob.get('landmark_confidences', [1.0] * len(names))
                count = min(len(names), len(centers), len(confidences), max_landmarks)
                if count:
                    centers_array[batch_index, :count] = np.asarray(centers[:count], dtype=np.float32)
                    valid_array[batch_index, :count] = True
                    confidence_array[batch_index, :count] = np.asarray(
                        confidences[:count], dtype=np.float32
                    )
                landmark_texts.extend(list(names[:count]) + [''] * (max_landmarks - count))

            landmark_encoding = self.tokenizer(
                landmark_texts, padding=True, return_tensors='pt'
            )
            landmark_ids = landmark_encoding['input_ids'].cuda()
            landmark_attention = landmark_encoding['attention_mask'].cuda()
            _, _, landmark_cls = self.lang_model(landmark_ids, landmark_attention)
            landmark_centers = torch.from_numpy(centers_array).cuda()
            landmark_name_features = landmark_cls.view(batch_size, max_landmarks, -1)
            landmark_valid = torch.from_numpy(valid_array).cuda()
            landmark_confidence = torch.from_numpy(confidence_array).cuda()

        # lang_features --> 768
        # linear_cls --> 49 (used to attend to img features)
        # c_0 = cls_hidden

        # print(lang_features.size()) # batch_size*sequence_length*768

        # --------------- 2. 当前时刻的 pose / position / trajectory 记录 -----------------
        # current_directions: [B]，表示当前 yaw 角
        # current_positions: [B, 2]，表示当前 agent 位置
        # direction_t: [B]，position_t: [B, 2]
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
        reverse_goal_direction_loss = torch.tensor(0.).cuda()
        progress_loss = torch.tensor(0.).cuda()
        goal_predict_loss = torch.tensor(0.).cuda()
        target_predict_loss = torch.tensor(0.).cuda()
        belief_region_loss = torch.tensor(0.).cuda()
        belief_offset_loss = torch.tensor(0.).cuda()
        valid_step_count = 0

        stage1_step = 0
        stage2_step = 0
        stage2_rotate = 0

        input = {
            'directions': torch.zeros((batch_size, 0, 4)).cuda(),
            'grid_fts': torch.zeros(batch_size, 0, 768).cuda(),
            'grid_index': torch.zeros(batch_size, 0).cuda(),
            'frames': torch.zeros(batch_size, 0, 512, 49).cuda(),
            'lenths': [0 for _ in range(batch_size)],
            'lang': lang_features,
            'candidates': global_positions,
            'centroids': torch.zeros((batch_size, 0, 2)).cuda(),
            'lang_cls': linear_cls,
            'map_fts': torch.zeros(batch_size, 0, 512, 49).cuda(),
        }
        if getattr(self.args, 'target_belief_head', False):
            input.update({
                'landmark_centers': landmark_centers,
                'landmark_name_features': landmark_name_features,
                'landmark_valid': landmark_valid,
                'landmark_confidence': landmark_confidence,
                'previous_belief': None,
            })

        stage1_ended = np.array([False] * batch_size)
        stage2_anchor_goals = [None] * batch_size
        previous_predicted_goals = [None] * batch_size
        goal_stable_steps = np.zeros(batch_size, dtype=np.int64)
        stop_steps = np.full(batch_size, self.args.max_action_len, dtype=np.int64)
        replan_count = 0

        target_error_sum = 0.0
        target_hit5_count = 0
        target_hit20_count = 0
        target_metric_count = 0
        belief_entropy_sum = 0.0
        belief_peak_shift_sum = 0.0
        belief_diagnostic_count = 0
        belief_peak_shift_count = 0

        for t in range(self.args.max_action_len):
            direction_t = torch.tensor([ob['pose'].yaw for ob in obs], dtype=torch.float32)
            position_t = torch.tensor(np.array([ob['position'] for ob in obs]), dtype=torch.float32)

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

            # --------------- 3. 把环境中的 pose / frames / maps 组装成 ET 输入 -----------------
            # if self.args.no_direction:
            #     input['directions'] = torch.hstack((input['directions'], torch.zeros_like(direction.view(-1, 1, 2))))
            # else:
            input['directions'] = direction.view(-1, 1, 4)    # [B, 1, 4]
            # if self.args.language_only:
            #     input['frames'] = torch.hstack((input['frames'], torch.zeros_like(im_feature.view(-1, 1, 512, 49))))
            # else:
            # print(input['frames'].shape, im_feature.shape)
            input['frames'] = im_feature.view(-1, 1, 512, 49) # [B, 1, 512, 49]
            input['maps'] = torch.from_numpy(np.array([ob['maps'] for ob in obs], dtype=np.float32)).cuda() # [B, H, W, C] or [B, C, H, W]

            centroid_lens = np.array(len(ob['centroids']) for ob in obs)
            input['centroids'] = torch.from_numpy(np.array([ob['centroids'] for ob in obs], dtype=np.float32)).cuda() # [B, N_centroid, 2]
            # input['directions'] = direction.view(-1,1,2)
            # input['frames'] = im_feature.view(-1,1, 512,49)

            for i in range(len(obs)):
                if not ended[i]:
                    input['lenths'][i] += 1

            # --------------- 4. 调用 ET 模型：触发 ET.forward -----------------
            # 输出：
            # - pred_direction: [B, 2]
            # - pred_progress: [B, 1]
            # - pred_goals: [B, 2]
            # - pred_logits: [B, N_cand, 1]
            # - grid_ft: [B, N_hist+1, 768]
            model_inputs = dict(
                directions=input['directions'],     # [B, 1, 4]
                frames=input['frames'],             # [B, T_frame, 512, 49]
                lenths=input['lenths'],             # [B]
                grid_fts=input['grid_fts'],         # [B, N_hist, 768]
                grid_index=input['grid_index'],     # [B, N_hist]
                maps=input['maps'],                 # [B, ...]
                lang=input['lang'],                 # [B, L_lang, 768]
                candidates=input['candidates'],     # [B, N_cand, 2]
                centroids=input['centroids'],       # [B, N_centroid, 2]
                lang_cls=input['lang_cls'],          # [B, 49]
                lang_mask=attention_mask,
                task_ids=torch.full(
                    (batch_size,), int(reverse_teacher), dtype=torch.long,
                    device=input['lang'].device,
                ),
            )
            if getattr(self.args, 'target_belief_head', False):
                model_inputs.update({
                    'landmark_centers': input['landmark_centers'],
                    'landmark_name_features': input['landmark_name_features'],
                    'landmark_valid': input['landmark_valid'],
                    'landmark_confidence': input['landmark_confidence'],
                    'previous_belief': input['previous_belief'],
                })
            model_outputs = self.vln_model(**model_inputs)
            if getattr(self.args, 'target_belief_head', False):
                previous_belief = input['previous_belief']
                (pred_direction, pred_progress, pred_goals, pred_logits,
                 grid_ft, belief_offsets, pred_reverse_goal_direction) = model_outputs
                input['previous_belief'] = pred_logits.squeeze(-1)
                belief_probs = torch.softmax(input['previous_belief'], dim=-1)
                belief_entropy = -(
                    belief_probs * torch.log(belief_probs.clamp_min(1e-12))
                ).sum(dim=-1)
                active_tensor = torch.from_numpy(~ended).to(belief_entropy.device)
                if active_tensor.any():
                    belief_entropy_sum += belief_entropy[active_tensor].sum().item()
                    belief_diagnostic_count += int(active_tensor.sum().item())
                    if previous_belief is not None:
                        grid = self.vln_model_without_ddp.target_belief_head.grid_centers
                        old_peak = grid[previous_belief.argmax(dim=-1)]
                        new_peak = grid[input['previous_belief'].argmax(dim=-1)]
                        shifts = torch.linalg.vector_norm(new_peak - old_peak, dim=-1)
                        belief_peak_shift_sum += (
                            shifts[active_tensor].sum().item() * self.args.map_meters
                        )
                        belief_peak_shift_count += int(active_tensor.sum().item())
            else:
                (pred_direction, pred_progress, pred_goals, pred_logits,
                 grid_ft, pred_reverse_goal_direction) = model_outputs
                belief_offsets = None

            # --------------- 5. 更新历史网格记忆：grid_fts / grid_index -----------------
            # 这里把当前 step 的输出特征追加到历史记忆里，供下一步 rollout 使用，从而形成 historical grid map。
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
            gt_goal_direction = np.array([ob['direction'] for ob in obs], dtype=np.float32)
            if reverse_teacher:
                gt_direction = np.array([
                    reverse_waypoint_direction(
                        ob['trajectory'], step=t, stride=self.args.move_iteration
                    )
                    for ob in obs
                ], dtype=np.float32)
            else:
                gt_direction = gt_goal_direction
            gt_goal = torch.from_numpy(np.array([ob['normalized_goal'] for ob in obs], dtype=np.float32))
            gt_progress = torch.from_numpy(np.array([ob['progress'] for ob in obs], dtype=np.float32))
            gt_target = torch.from_numpy(np.array([ob['grid_goal'] for ob in obs], dtype=np.int64))
            # there is no ground truth in unseen_test set
            if not 'test' in self.env_name:
                # Get ground truth
                # print(t, target, gt_progress)

                # Compute loss

                active_mask = torch.from_numpy(~ended).to(device=pred_goals.device)
                valid_step_count += int(active_mask.sum().item())
                gt_goal_device = gt_goal.to(device=pred_goals.device)

                if not reverse_teacher and active_mask.any():
                    target_errors = torch.linalg.vector_norm(
                        pred_goals - gt_goal_device, dim=-1
                    ) * self.args.map_meters
                    active_errors = target_errors[active_mask].detach()
                    target_error_sum += active_errors.sum().item()
                    target_hit5_count += int((active_errors <= 5.0).sum().item())
                    target_hit20_count += int((active_errors <= 20.0).sum().item())
                    target_metric_count += int(active_errors.numel())

                for i in range(len(obs)):
                    true_direction = torch.tensor(gt_direction[i])

                    true_sin = torch.sin(true_direction)
                    true_cos = torch.cos(true_direction)
                    true_sin_cos = torch.stack([true_sin, true_cos], dim=-1).cuda()
                    goal_direction = torch.tensor(gt_goal_direction[i])
                    goal_sin_cos = torch.stack(
                        [torch.sin(goal_direction), torch.cos(goal_direction)], dim=-1
                    ).cuda()
                    # gt_progress = torch.tensor(obs[i]['progress']).cuda()
                    # print(pred_direction[i].view(-1).shape, pred_progress[i].view(-1).shape, true_sin_cos.shape, gt_progress[i].view(-1).shape)
                    # cuda_gt_next_pos_ratio = torch.from_numpy(target[i][0]).cuda()
                    # print(pred_direction[i].view(-1), true_sin_cos)
                    if not ended[i]:
                        # if stage1_ended[i]:
                        direction_loss += self.progress_regression(pred_direction[i].view(-1), true_sin_cos)

                        if reverse_teacher:
                            reverse_goal_direction_loss += self.progress_regression(
                                pred_reverse_goal_direction[i].view(-1), goal_sin_cos
                            )

                        progress_loss += self.progress_regression(pred_progress[i].view(-1),
                                                                  gt_progress[i].view(-1).cuda())
                        if reverse_teacher:
                            pass
                        elif getattr(self.args, 'target_belief_head', False):
                            goal_predict_loss += F.smooth_l1_loss(
                                pred_goals[i].view(-1), gt_goal_device[i].view(-1), reduction='sum'
                            )
                        else:
                            goal_predict_loss += F.mse_loss(
                                pred_goals[i].view(-1), gt_goal_device[i].view(-1)
                            )
                        # print(pred_goals[i], gt_goal[i], goal_predict_loss)

                    # ml_loss += direction_loss
                    # ml_loss += progress_loss

                    # print(ml_loss)
                    if direction_loss != direction_loss:  # debug for nan loss
                        print('0', direction_loss)
                    if progress_loss != progress_loss:  # debug for nan loss
                        print('0', progress_loss)
                # print(at_direction, gt_direction, ml_loss)
                if getattr(self.args, 'target_belief_head', False) and not reverse_teacher:
                    belief_logits = pred_logits.squeeze(-1)
                    belief_grid_size = self.args.belief_grid_size
                    belief_targets = target_cell_ids(gt_goal_device, belief_grid_size)
                    if active_mask.any():
                        target_predict_loss += F.cross_entropy(
                            belief_logits[active_mask], belief_targets[active_mask], reduction='sum'
                        )

                        grid_centers = self.vln_model_without_ddp.target_belief_head.grid_centers
                        region_mask = target_region_mask(
                            gt_goal_device,
                            grid_centers.to(device=gt_goal_device.device, dtype=gt_goal_device.dtype),
                            radius_m=self.args.belief_radius_m,
                            map_meters=self.args.map_meters,
                        )
                        log_probs = F.log_softmax(belief_logits, dim=-1)
                        region_log_mass = torch.logsumexp(
                            log_probs.masked_fill(~region_mask, -float('inf')), dim=-1
                        )
                        belief_region_loss += (-region_log_mass[active_mask]).sum()

                        batch_ids = torch.arange(batch_size, device=pred_goals.device)
                        predicted_offsets = belief_offsets[batch_ids, belief_targets]
                        target_offsets = gt_goal_device - grid_centers[belief_targets]
                        per_sample_offset = F.smooth_l1_loss(
                            predicted_offsets * belief_grid_size,
                            target_offsets * belief_grid_size,
                            reduction='none',
                        ).sum(dim=-1)
                        belief_offset_loss += per_sample_offset[active_mask].sum()
                elif getattr(self.args, 'normalize_rollout_loss', False) and not reverse_teacher:
                    logits = pred_logits.squeeze(-1)
                    target_predict_loss += F.cross_entropy(
                        logits[active_mask], gt_target.to(logits.device)[active_mask], reduction='sum'
                    )
                elif not reverse_teacher:
                    target_predict_loss += self.criterion(pred_logits, gt_target.unsqueeze(1).cuda())
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

            if self.feedback == 'teacher':
                at_goal = gt_goal
                # print('teacher', at_goal.shape)
                a_t = gt_direction
                pred_progress_t = gt_progress
            elif self.feedback == 'student':  # student
                a_t = at_direction
                at_goal = pred_goals

                # _, at_goal = pred_logits.max(1)
                # at_goal = at_goal.squeeze(1)
                # at_goal = gt_goal
                # print('student', at_goal.shape)
            else:
                sys.exit('Invalid feedback option')

            cpu_goal = at_goal.cpu().detach().numpy()
            # print(cpu_goal)

            # Interact with the simulator with actions
            for i in range(len(obs)):

                dst = self.env.unnormalize_position(cpu_goal[i], obs[i]['map_name'],
                                                    self.args.map_meters)

                if self.feedback == 'student':
                    previous_goal = previous_predicted_goals[i]
                    if (previous_goal is not None
                            and dst.dist_to(previous_goal) <= self.args.goal_stability_distance_m):
                        goal_stable_steps[i] += 1
                    else:
                        goal_stable_steps[i] = 1
                    previous_predicted_goals[i] = dst

                    anchor = stage2_anchor_goals[i]
                    if should_replan_stage2(
                            stage1_ended[i], anchor, dst,
                            self.args.stage2_replan_distance_m):
                        stage1_ended[i] = False
                        stage2_anchor_goals[i] = None
                        replan_count += 1

                # gt_center = self.env.unnormalize_position(global_position[gt_goal.cpu().detach().numpy()[i]], obs[i]['map_name'],
                #                                     self.args.map_meters)
                # dst = Point2D(obs[i]['centroid_goal'][0], obs[i]['centroid_goal'][1])
                if ended[i]:
                    continue
                # if dst.dist_to(poses[i].xy) < 10:
                #     ended[i] = True
                #     continue


                elif (self.feedback == 'student' and should_stop_navigation(
                        stage1_ended[i], pred_progress_t[i],
                        dst.dist_to(poses[i].xy), goal_stable_steps[i], self.args)):
                    # Updated 'ended' list and make environment action
                    ended[i] = True
                    stop_steps[i] = t
                    continue

                # print(cpu_goal[i], global_position[cpu_goal[i]])
                # dst = self.env.unnormalize_position(global_position[cpu_goal[i]], obs[i]['map_name'],
                #                                     self.args.map_meters)
                # dst = Point2D(obs[i]['centroid_goal'][0], obs[i]['centroid_goal'][1])

                # if pred_progress_t[i] < 0.75 and dst.dist_to(poses[i].xy) > 20 and not stage1_ended[i]:
                # if pred_progress_t[i] < 0.75 and dst.dist_to(poses[i].xy) > 10 and not stage1_ended[i]:

                # if pred_progress_t[i] > 0.9 and not stage1_ended[i]:
                #     stage1_ended[i] = True
                if (dst.dist_to(poses[i].xy) > self.args.stage1_arrival_distance_m
                        and not stage1_ended[i]):
                    stage1_step += 1
                    traj[i]['pred_goal'].append(dst)
                    # pred_goal_xys = [
                    #     unnormalize_position(global_position[goal_id] / args.grid_size, eps.map_name, args.map_meters)
                    #     for eps, goal_id in zip(episodes_batch, goal_ids)]
                    # dst = Point2D(obs[i]['centroid_goal'][0], obs[i]['centroid_goal'][1])
                    # dst = self.env.unnormalize_position(global_position[cpu_goal[i]], obs[i]['map_name'], self.args.map_meters)
                    if self.feedback == 'teacher':
                        # Reverse supervision must traverse each demonstration
                        # in order.  The released forward path keeps its old
                        # global counter for checkpoint-compatible controls.
                        cur_step = ((t + 1) * self.args.move_iteration
                                    if reverse_teacher else
                                    stage1_step * self.args.move_iteration)
                        cur_step = cur_step if cur_step < len(obs[i]['trajectory']) else -1
                        poses[i] = obs[i]['trajectory'][cur_step]
                    else:
                        poses[i] = self.move(poses[i], dst,
                                         self.args.move_iteration)
                    if not ended[i]:
                        traj[i]['stage1_trajectory'].append(poses[i])

                elif abs(a_t[i]) < np.pi / 12:
                    stage1_ended[i] = True
                    if stage2_anchor_goals[i] is None:
                        stage2_anchor_goals[i] = dst
                    stage2_step += 1
                    poses[i] = _moved_pose(poses[i], *Action(5, 0, 0))
                    if len(traj[i]['stage2_trajectory']) == 0:
                        traj[i]['stage2_trajectory'].append(traj[i]['stage1_trajectory'][-1])
                    if not ended[i]:
                        traj[i]['stage2_trajectory'].append(poses[i])
                else:
                    stage1_ended[i] = True
                    if stage2_anchor_goals[i] is None:
                        stage2_anchor_goals[i] = dst
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
            obs = self.env._get_obs(
                poses,
                random_direction=(self.feedback == 'teacher' and not reverse_teacher),
            )  # get gt_obs
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
            if reverse_teacher:
                # Original-start supervision must never update the forward
                # landmark-conditioned target/belief heads.
                visual_alignment_scale = (
                    max(valid_step_count, 1) / batch_size
                    if getattr(self.args, 'normalize_rollout_loss', False)
                    else 1.0
                )
                ml_loss = (
                    self.args.direction_loss_weight * direction_loss
                    + self.args.progress_loss_weight * progress_loss
                    + self.args.reverse_goal_direction_weight * reverse_goal_direction_loss
                    + self.args.reverse_visual_align_weight
                    * visual_alignment_scale * visual_alignment_loss
                )
            else:
                ml_loss = (self.args.direction_loss_weight * direction_loss
                           + self.args.progress_loss_weight * progress_loss
                           + self.args.goal_loss_weight * goal_predict_loss
                           + self.args.target_loss_weight * target_predict_loss)
            if getattr(self.args, 'target_belief_head', False) and not reverse_teacher:
                ml_loss = (ml_loss
                           + self.args.belief_region_loss_weight * belief_region_loss
                           + self.args.belief_offset_loss_weight * belief_offset_loss)
            # ml_loss = progress_loss + goal_predict_loss
            if getattr(self.args, 'normalize_rollout_loss', False):
                loss_normalizer = max(valid_step_count, 1)
            else:
                loss_normalizer = batch_size
            self.loss += ml_loss * train_ml / loss_normalizer

            # self.logs['ml_loss'].append((ml_loss * train_ml / batch_size).item())

            self.logs['direction_loss'].append((direction_loss * train_ml / loss_normalizer).item())
            self.logs['reverse_goal_direction_loss'].append(
                (reverse_goal_direction_loss * train_ml / loss_normalizer).item()
            )
            self.logs['progress_loss'].append((progress_loss * train_ml / loss_normalizer).item())
            self.logs['goal_predict_loss'].append((goal_predict_loss * train_ml / loss_normalizer).item())
            self.logs['target_predict_loss'].append((target_predict_loss * train_ml / loss_normalizer).item())
            self.logs['belief_region_loss'].append(
                (belief_region_loss * train_ml / loss_normalizer).detach().item()
                if torch.is_tensor(belief_region_loss)
                else float(belief_region_loss * train_ml / loss_normalizer)
            )
            self.logs['belief_offset_loss'].append(
                (belief_offset_loss * train_ml / loss_normalizer).detach().item()
                if torch.is_tensor(belief_offset_loss)
                else float(belief_offset_loss * train_ml / loss_normalizer)
            )
            self.logs['visual_alignment_loss'].append(
                (visual_alignment_loss * train_ml / loss_normalizer).detach().item()
            )
            self.logs['visual_alignment_positive_cosine'].append(
                visual_alignment_positive_cosine.detach().item()
            )
            self.logs['visual_alignment_accuracy'].append(
                visual_alignment_accuracy.detach().item()
            )
            rollout_name = 'reverse' if reverse_teacher else 'forward'
            self.logs[f'{rollout_name}_direction_loss'].append(
                (direction_loss * train_ml / loss_normalizer).detach().item()
            )
            self.logs[f'{rollout_name}_progress_loss'].append(
                (progress_loss * train_ml / loss_normalizer).detach().item()
            )
            self.logs[f'{rollout_name}_IL_loss'].append(
                (ml_loss * train_ml / loss_normalizer).detach().item()
            )
            self.logs['IL_loss'].append((ml_loss * train_ml / loss_normalizer).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        # if t==0:
        #     self.logs
        self.logs['stage1_step'].append(float(stage1_step) / batch_size)
        self.logs['stage2_step'].append(float(stage2_step) / batch_size)
        self.logs['stage2_rotate'].append(float(stage2_rotate) / batch_size)
        if not reverse_teacher:
            self.logs['stop_step'].append(float(stop_steps.mean()))
            self.logs['stopped_rate'].append(float((stop_steps < self.args.max_action_len).mean()))
            self.logs['replan_count'].append(float(replan_count) / batch_size)
            if target_metric_count:
                self.logs['target_error_mean_m'].append(
                    target_error_sum / target_metric_count
                )
                self.logs['target_hit5'].append(target_hit5_count / target_metric_count)
                self.logs['target_hit20'].append(target_hit20_count / target_metric_count)
            if belief_diagnostic_count:
                self.logs['belief_entropy'].append(
                    belief_entropy_sum / belief_diagnostic_count
                )
                if belief_peak_shift_count:
                    self.logs['belief_peak_shift_m'].append(
                        belief_peak_shift_sum / belief_peak_shift_count
                    )

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
        states['training_state'] = {
            'python_random': random.getstate(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
            'cuda_random': torch.cuda.get_rng_state_all(),
        }
        temporary = path + '.tmp'
        torch.save(states, temporary)
        os.replace(temporary, path)

    def load(self, path):
        '''Load model parameters and, when present, reproducibility state.'''
        states = torch.load(path, map_location='cpu', weights_only=False)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys == load_keys:
                print("NOTICE: LOADing ALL KEYS IN THE ", name)
                state_dict = states[name]['state_dict']
            else:
                print("NOTICE: DIFFERENT KEYS IN THE ", name)
                missing = sorted(model_keys - load_keys)
                required_missing = [k for k in missing if not (
                    self.args.disable_task_interaction and k.startswith('task_interaction.'))]
                if required_missing and self.args.mode != 'train':
                    raise ValueError(f'{name}: checkpoint is missing parameters: {required_missing}')
                if missing:
                    print('New parameters initialized randomly:', missing)
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
        training_state = states.get('training_state')
        if training_state is not None:
            random.setstate(training_state['python_random'])
            np.random.set_state(training_state['numpy_random'])
            torch.set_rng_state(training_state['torch_random'])
            torch.cuda.set_rng_state_all(training_state['cuda_random'])
        return states['vln_model']['epoch']
